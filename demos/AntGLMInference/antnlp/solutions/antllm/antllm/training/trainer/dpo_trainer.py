import os
import random
import shutil
import warnings
from typing import Dict, Tuple, Optional, Any, Union, List, Literal
from collections import defaultdict

import numpy as np
import torch.nn.functional as F

import torch
import torch.nn as nn
from transformers import Trainer
from transformers.utils import logging
from transformers.trainer import nested_detach
from transformers.trainer import (
    OPTIMIZER_NAME,
    SCALER_NAME,
    SCHEDULER_NAME,
    TRAINER_STATE_NAME,
    TRAINING_ARGS_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
)
from transformers.trainer_pt_utils import reissue_pt_warnings
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, ShardedDDPOption
from transformers.utils import (
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    is_peft_available,
    is_safetensors_available,
)
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from collections import OrderedDict

if is_peft_available():
    from peft import PeftModel

if is_safetensors_available():
    import safetensors.torch


logger = logging.get_logger(__name__)

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp


class DPOTrainer(Trainer):
    def __init__(
        self,
        reference_model: torch.nn.Module,
        beta: float,
        reference_free: bool = False,
        *args,
        no_save_deepspeed_checkpoint=False,
        save_pytorch_model_bin_checkpoint=True,
        rank=0,
        max_shard_size="30GB",
        train_peft=False,
        no_save_base_model=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reference_model = reference_model
        self.beta = beta
        self.reference_free = reference_free
        self.no_save_deepspeed_checkpoint = no_save_deepspeed_checkpoint
        self.save_pytorch_model_bin_checkpoint = save_pytorch_model_bin_checkpoint
        self.max_shard_size = max_shard_size
        self.rank = rank
        self.train_peft = train_peft
        self.kwargs = kwargs
        self.no_save_base_model = no_save_base_model
        logger.info(f"self.rank: {self.rank}")
        self.test_key = "accuracy"
        self.best_metrics = OrderedDict(
            {
                "best_epoch": 0,
                f"best_eval_{self.test_key}": 0,
            }
        )
        # Since we inherit from trainer we always have access to an accelerator
        if hasattr(self, "accelerator"):
            self.reference_model = self.accelerator.prepare_model(
                self.reference_model, evaluation_mode=True
            )
        else:
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        if self.deepspeed and not self.no_save_deepspeed_checkpoint:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_16bit_weights_on_model_save` is True
            self.deepspeed.save_checkpoint(output_dir)
        if not self.save_pytorch_model_bin_checkpoint:
            return
        if not self.no_save_base_model:
            self.save_model(output_dir, _internal_call=True)

        # Save optimizer and scheduler
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            self.optimizer.consolidate_state_dict()

        if is_torch_tpu_available():
            xm.rendezvous("saving_optimizer_states")
            xm.save(
                self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME)
            )
            with warnings.catch_warnings(record=True) as caught_warnings:
                xm.save(
                    self.lr_scheduler.state_dict(),
                    os.path.join(output_dir, SCHEDULER_NAME),
                )
                reissue_pt_warnings(caught_warnings)
        elif is_sagemaker_mp_enabled():
            opt_state_dict = self.optimizer.local_state_dict(gather_if_shard=False)
            smp.barrier()
            if smp.rdp_rank() == 0 or smp.state.cfg.shard_optimizer_state:
                smp.save(
                    opt_state_dict,
                    os.path.join(output_dir, OPTIMIZER_NAME),
                    partial=True,
                    v3=smp.state.cfg.shard_optimizer_state,
                )
            if self.args.should_save:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    torch.save(
                        self.lr_scheduler.state_dict(),
                        os.path.join(output_dir, SCHEDULER_NAME),
                    )
                reissue_pt_warnings(caught_warnings)
                if self.do_grad_scaling:
                    torch.save(
                        self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME)
                    )
        elif self.args.should_save and not self.deepspeed:
            # deepspeed.save_checkpoint above saves model/optim/sched
            torch.save(
                self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME)
            )
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(
                    self.lr_scheduler.state_dict(),
                    os.path.join(output_dir, SCHEDULER_NAME),
                )
            reissue_pt_warnings(caught_warnings)
            if self.do_grad_scaling:
                torch.save(
                    self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME)
                )

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.local_rank == -1:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        if is_torch_tpu_available():
            rng_states["xla"] = xm.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)

        if self.args.world_size <= 1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(
                rng_states,
                os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"),
            )

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

        if self.rank == 0:
            dir_to_remove = os.path.dirname(output_dir)
            logger.info(f"Delete old deepspeed checkpoint in {dir_to_remove}")
            for file_or_dir in os.listdir(dir_to_remove):
                if file_or_dir.startswith(f"{PREFIX_CHECKPOINT_DIR}"):
                    if (
                        file_or_dir
                        != f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                    ):
                        ab_file_or_dir = os.path.join(
                            os.path.join(dir_to_remove, file_or_dir)
                        )
                        for sub_file_or_dir in os.listdir(ab_file_or_dir):
                            if sub_file_or_dir.startswith("global_step"):
                                shutil.rmtree(
                                    os.path.join(ab_file_or_dir, sub_file_or_dir),
                                    ignore_errors=True,
                                )

    def _maybe_log_save_evaluate(
        self, tr_loss, model, trial, epoch, ignore_keys_for_eval
    ):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            self.log(logs)
        
        eval_metrics = None
        if self.control.should_evaluate:
            predictions = self.predict(self.eval_dataset).predictions
            chosen_rewards, rejected_rewards = predictions
            acc = sum(chosen_rewards > rejected_rewards) / len(chosen_rewards)
            eval_metrics = {"eval_accuracy": acc}
            self._report_to_hp_search(trial, epoch, eval_metrics)

            self.best_metrics["eval_accuracy"] = acc
            if (
                eval_metrics["eval_" + self.test_key]
                > self.best_metrics["best_eval_" + self.test_key]
            ):
                self.best_metrics["best_epoch"] = epoch
                self.best_metrics["best_steps"] = self.state.global_step
                self.best_metrics["best_eval_" + self.test_key] = eval_metrics[
                    "eval_" + self.test_key
                ]

            logger.info(f"***** Epoch {epoch}: Best results *****")
            for key, value in self.best_metrics.items():
                logger.info(f"{key} = {value}")
            self.log(self.best_metrics)

            self.control = self.callback_handler.on_evaluate(
                self.args,
                self.state,
                self.control,
                eval_metrics,
            )

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=eval_metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )

    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        with torch.no_grad():
            reference_chosen_logits = self.reference_model(
                input_ids=batch["input_ids_chosen"],
                attention_mask=batch["attention_mask_chosen"],
                position_ids=batch["position_ids_chosen"],
            ).logits
            reference_rejected_logits = self.reference_model(
                input_ids=batch["input_ids_rejected"],
                attention_mask=batch["attention_mask_rejected"],
                position_ids=batch["position_ids_rejected"],
            ).logits

        policy_chosen_outputs = model(
            input_ids=batch["input_ids_chosen"],
            attention_mask=batch["attention_mask_chosen"],
            position_ids=batch["position_ids_chosen"],
        )
        policy_chosen_logits = policy_chosen_outputs.logits
        policy_rejected_logits = model(
            input_ids=batch["input_ids_rejected"],
            attention_mask=batch["attention_mask_rejected"],
            position_ids=batch["position_ids_rejected"],
        ).logits

        policy_chosen_logps = _get_batch_logps(
            policy_chosen_logits, batch["labels_chosen"], average_log_prob=False
        )
        policy_rejected_logps = _get_batch_logps(
            policy_rejected_logits, batch["labels_rejected"], average_log_prob=False
        )
        reference_chosen_logps = _get_batch_logps(
            reference_chosen_logits, batch["labels_chosen"], average_log_prob=False
        )
        reference_rejected_logps = _get_batch_logps(
            reference_rejected_logits, batch["labels_rejected"], average_log_prob=False
        )

        losses, chosen_rewards, rejected_rewards = dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            beta=self.beta,
            reference_free=self.reference_free,
        )

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()

        return losses.mean(), metrics

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, metrics = self.get_batch_metrics(model, inputs, train_eval="train")

        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss
    
    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, metrics = self.get_batch_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_rewards/chosen": metrics["eval_rewards/chosen"].to(self.accelerator.device),
            "eval_rewards/rejected": metrics["eval_rewards/rejected"].to(self.accelerator.device),
        }
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]
        labels = None

        return (loss.detach(), logits, labels)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (
            (PreTrainedModel,)
            if not is_peft_available()
            else (PreTrainedModel, PeftModel)
        )
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(unwrap_model(self.model), supported_classes):
                unwrap_model(self.model).save_pretrained(
                    output_dir,
                    state_dict=state_dict,
                    safe_serialization=self.args.save_safetensors,
                    max_shard_size=self.max_shard_size,
                )
            else:
                logger.info(
                    "Trainer.model is not a `PreTrainedModel`, only saving its state dict."
                )
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME)
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=self.args.save_safetensors,
                max_shard_size="30GB",
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
    
    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)


def _get_batch_logps(
    logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False
) -> torch.FloatTensor:
    # 这里logits对应的原始token id 和labels已经错1位了
    assert logits.shape[:-1] == labels.shape

    labels = labels.clone()

    loss_mask = labels != -100

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def dpo_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    beta: float,
    reference_free: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = (
        beta * (policy_rejected_logps - reference_rejected_logps).detach()
    )

    return losses, chosen_rewards, rejected_rewards
