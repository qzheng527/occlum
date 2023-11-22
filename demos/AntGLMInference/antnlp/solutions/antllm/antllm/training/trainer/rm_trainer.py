import warnings

import os
import random
import shutil
import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from packaging import version

import datasets
from transformers import Trainer, __version__ as transformers_version
from transformers.utils import logging, is_datasets_available
from transformers.trainer import (
    OPTIMIZER_NAME,
    SCALER_NAME,
    SCHEDULER_NAME,
    TRAINER_STATE_NAME,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    ShardedDDPOption,
    seed_worker,
    has_length
)
from transformers.modeling_utils import unwrap_model
from transformers.utils import is_sagemaker_mp_enabled, is_torch_tpu_available
from transformers.trainer_pt_utils import reissue_pt_warnings, IterableDatasetShard, LengthGroupedSampler

from collections import OrderedDict
from typing import Dict, Optional
import evaluate
from peft import PeftModel

IS_TRANSFORMERS_4_31 = version.parse(transformers_version) >= version.parse("4.31.0")

logger = logging.get_logger(__name__)


if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp


# TODO：需要一个 小的 用于测试的 antglm 方便代码重构之后做单测
class RMBaseTrainer(Trainer):
    def __init__(self, model, test_key: str, *args, predict_dataset=None, **kwargs):
        super().__init__(model=model, *args, **kwargs)
        self.model = model
        self.predict_dataset = predict_dataset
        self.test_key = test_key
        if test_key == "mse":
            self.best_metrics = OrderedDict(
                {
                    "best_epoch": 0,
                    f"best_eval_{self.test_key}": 100,
                }
            )
        else:
            self.best_metrics = OrderedDict(
                {
                    "best_epoch": 0,
                    f"best_eval_{self.test_key}": 0,
                }
            )

        logger.setLevel(self.args.get_process_log_level())

    def _get_train_sampler_new_version(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )
        elif self.args.no_shuffle_dataloader:
            return SequentialSampler(self.train_dataset)
        else:
            return RandomSampler(self.train_dataset)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )
        
        if IS_TRANSFORMERS_4_31:
            logger.info(f"trainsformers version: {transformers_version}")
            dataloader_params = {
                "batch_size": self._train_batch_size,
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
            }

            if not isinstance(train_dataset, torch.utils.data.IterableDataset):
                dataloader_params["sampler"] = self._get_train_sampler_new_version()
                dataloader_params["drop_last"] = self.args.dataloader_drop_last
                dataloader_params["worker_init_fn"] = seed_worker

            return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
        else:
            if isinstance(train_dataset, torch.utils.data.IterableDataset):
                if self.args.world_size > 1:
                    train_dataset = IterableDatasetShard(
                        train_dataset,
                        batch_size=self._train_batch_size,
                        drop_last=self.args.dataloader_drop_last,
                        num_processes=self.args.world_size,
                        process_index=self.args.process_index,
                    )

                return DataLoader(
                    train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    collate_fn=data_collator,
                    num_workers=self.args.dataloader_num_workers,
                    pin_memory=self.args.dataloader_pin_memory,
                )

        train_sampler = self._get_train_sampler()
        if self.args.no_shuffle_dataloader:
            train_sampler.shuffle = False
        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        transformer_state_dict = self.model.model.model.state_dict()
        head_state_dict = self.model.model.value_head.state_dict()

        unwrap_base_model = unwrap_model(self.model.model.model)
        if isinstance(unwrap_base_model, PeftModel):
            unwrap_base_model.save_pretrained(output_dir, max_shard_size="30GB")
        else:
            unwrap_base_model.save_pretrained(
                output_dir, state_dict=transformer_state_dict, max_shard_size="30GB"
            )
        # torch.save(transformer_state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        for k, v in head_state_dict.items():
            head_state_dict[k] = v.cpu()
        torch.save(head_state_dict, os.path.join(output_dir, "head.bin"))

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def log_best_metrics(self):
        self.log_metrics("best", self.best_metrics)
        self.save_metrics("best", self.best_metrics, combined=False)

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
        self.save_model(output_dir, _internal_call=True)
        if self.deepspeed and not self.args.no_save_deepspeed_checkpoint:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_16bit_weights_on_model_save` is True
            self.deepspeed.save_checkpoint(output_dir)

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

        # 清除之前的 deepspeed checkpoint
        if self.args.process_index == 0:
            dir_to_remove = os.path.dirname(output_dir)
            logger.info(f"delete old deepspeed checkpoint in {dir_to_remove}")
            for file_or_dir in os.listdir(dir_to_remove):
                if file_or_dir.startswith(f"{PREFIX_CHECKPOINT_DIR}"):
                    if file_or_dir != f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}":
                        ab_file_or_dir = os.path.join(
                            os.path.join(dir_to_remove, file_or_dir))
                        for sub_file_or_dir in os.listdir(ab_file_or_dir):
                            if sub_file_or_dir.startswith('global_step'):
                                shutil.rmtree(os.path.join(ab_file_or_dir, sub_file_or_dir), ignore_errors=True)


class RMTrainer(RMBaseTrainer):
    def __init__(
        self, model, *args, predict_dataset=None, test_key="accuracy", **kwargs
    ):
        super(RMTrainer, self).__init__(
            model=model,
            *args,
            predict_dataset=predict_dataset,
            test_key=test_key,
            **kwargs,
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

        if self.control.should_evaluate or self.control.should_save:
            predictions = self.predict(self.eval_dataset).predictions
            # RewardModelForPairWise输出包括logits, chosen_reward, rejected_reward
            logits, chosen_rewards, rejected_rewards = predictions
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

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=eval_metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )


class RMTrainerForPointWise(RMBaseTrainer):
    def __init__(
        self,
        model,
        num_head: int,
        *args,
        predict_dataset=None,
        test_key="mse",
        **kwargs,
    ):
        if num_head == 1:
            test_key = "auc"
        roc_auc_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../..",
            "evaluation/metrics/evaluate_factory/roc_auc.py",
        )
        self.roc_auc_score_func = evaluate.load(roc_auc_path)

        super(RMTrainerForPointWise, self).__init__(
            model=model,
            *args,
            predict_dataset=predict_dataset,
            test_key=test_key,
            **kwargs,
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

        if self.control.should_evaluate or self.control.should_save:
            predictions = self.predict(self.eval_dataset).predictions
            reward_engaged, labels_engaged = predictions
            if self.test_key == "auc":
                results = self.roc_auc_score_func.compute(
                    references=torch.tensor(labels_engaged),
                    prediction_scores=torch.tensor(reward_engaged),
                )
                acc = results["roc_auc"]
                eval_metrics = {"eval_auc": acc}
                self._report_to_hp_search(trial, epoch, eval_metrics)

                self.best_metrics["eval_auc"] = acc
                if (
                    eval_metrics["eval_" + self.test_key]
                    > self.best_metrics["best_eval_" + self.test_key]
                ):
                    self.best_metrics["best_epoch"] = epoch
                    self.best_metrics["best_steps"] = self.state.global_step
                    self.best_metrics["best_eval_" + self.test_key] = eval_metrics[
                        "eval_" + self.test_key
                    ]
            else:
                acc = sum(np.square(reward_engaged - labels_engaged)) / len(
                    reward_engaged
                )
                eval_metrics = {"eval_mse": acc}
                self._report_to_hp_search(trial, epoch, eval_metrics)

                self.best_metrics["eval_mse"] = acc
                if (
                    eval_metrics["eval_" + self.test_key]
                    < self.best_metrics["best_eval_" + self.test_key]
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

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=eval_metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )
