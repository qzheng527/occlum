#!/usr/bin/env python
# coding=utf-8
# @Author: tianxuan.jl
# @Date: Fri 17 Mar 2023 09:22:56 PM CST

import os
import random
import shutil
import warnings
from typing import Optional

import numpy as np
import torch
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.trainer import (OPTIMIZER_NAME, SCALER_NAME, SCHEDULER_NAME,
                                  TRAINER_STATE_NAME, TRAINING_ARGS_NAME)
from transformers.trainer_pt_utils import (DistributedLengthGroupedSampler,
                                           DistributedSamplerWithLoop,
                                           LengthGroupedSampler,
                                           reissue_pt_warnings)
from transformers.trainer_utils import (PREFIX_CHECKPOINT_DIR,
                                        ShardedDDPOption, has_length)
from transformers.training_args import ParallelMode
from transformers.utils import (WEIGHTS_NAME, is_datasets_available,
                                is_sagemaker_mp_enabled,
                                is_torch_tpu_available)

from solutions.antllm.antllm.data.dataset.shard_distributed_sampler import \
    ShardDistributedSampler
from solutions.antllm.antllm.models.glm.modeling_glm import \
    GLMForConditionalGeneration
from solutions.antllm.antllm.models.peft.modeling_peft import (  # noqa
    AntPeftForCausalLM, PeftModel)

if is_datasets_available():
    import datasets

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp


class SFTTrainer(Trainer):
    def __init__(self,
                 *args,
                 no_save_deepspeed_checkpoint=False,
                 save_pytorch_model_bin_checkpoint=True,
                 rank=0,
                 max_shard_size='30GB',
                 train_peft=False,
                 no_save_base_model=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.no_save_deepspeed_checkpoint = no_save_deepspeed_checkpoint
        self.save_pytorch_model_bin_checkpoint = save_pytorch_model_bin_checkpoint
        self.max_shard_size = max_shard_size
        self.rank = rank
        self.train_peft = train_peft
        self.kwargs = kwargs
        self.no_save_base_model = no_save_base_model
        print(f'self.rank: {self.rank}')

    def save_peft_model(self, model, output_dir):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if not isinstance(model, PeftModel):
            if isinstance(unwrap_model(model), PeftModel):
                unwrap_model(model).save_pretrained(output_dir)
            else:
                state_dict = model.state_dict()
                torch.save(state_dict, os.path.join(
                    output_dir, "pytorch_model.bin"))
        else:
            model.save_pretrained(output_dir)

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
        if self.train_peft:
            self.save_peft_model(model, output_dir)
        if not self.no_save_base_model:
            self.save_model(output_dir, _internal_call=True)

        # Save optimizer and scheduler
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            self.optimizer.consolidate_state_dict()

        if is_torch_tpu_available():
            xm.rendezvous("saving_optimizer_states")
            xm.save(self.optimizer.state_dict(),
                    os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                xm.save(self.lr_scheduler.state_dict(),
                        os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
        elif is_sagemaker_mp_enabled():
            opt_state_dict = self.optimizer.local_state_dict(
                gather_if_shard=False)
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
                    torch.save(self.lr_scheduler.state_dict(),
                               os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
                if self.do_grad_scaling:
                    torch.save(self.scaler.state_dict(),
                               os.path.join(output_dir, SCALER_NAME))
        elif self.args.should_save and not self.deepspeed:
            # deepspeed.save_checkpoint above saves model/optim/sched
            torch.save(self.optimizer.state_dict(),
                       os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(),
                           os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)
            if self.do_grad_scaling:
                torch.save(self.scaler.state_dict(),
                           os.path.join(output_dir, SCALER_NAME))

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
            self.state.save_to_json(os.path.join(
                output_dir, TRAINER_STATE_NAME))

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
            torch.save(rng_states, os.path.join(
                output_dir, f"rng_state_{self.args.process_index}.pth"))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

        if self.rank == 0:
            dir_to_remove = os.path.dirname(output_dir)
            print(f'Delete old deepspeed checkpoint in {dir_to_remove}')
            for file_or_dir in os.listdir(dir_to_remove):
                if file_or_dir.startswith(f'{PREFIX_CHECKPOINT_DIR}'):
                    if file_or_dir != f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}':
                        ab_file_or_dir = os.path.join(
                            os.path.join(dir_to_remove, file_or_dir))
                        for sub_file_or_dir in os.listdir(ab_file_or_dir):
                            if sub_file_or_dir.startswith('global_step'):
                                shutil.rmtree(os.path.join(
                                    ab_file_or_dir, sub_file_or_dir), ignore_errors=True)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                if state_dict is None:
                    state_dict = self.model.state_dict()
                unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, max_shard_size=self.max_shard_size)

            # Save the origin model parameters in peft method
            elif isinstance(unwrap_model(self.model), PeftModel):
                if isinstance(unwrap_model(self.model).base_model, GLMForConditionalGeneration):
                    if state_dict is None:
                        state_dict = unwrap_model(
                            self.model).base_model.state_dict()
                    unwrap_model(self.model).save_pretrained(
                        output_dir, state_dict=state_dict, max_shard_size=self.max_shard_size)
                else:
                    state_dict = unwrap_model(
                        self.model).base_model.model.state_dict()

                    # Filter the peft params ...
                    param_keys = list(state_dict.keys())
                    peft_param_prefix = "base_model.model."
                    for key in param_keys:
                        if "lora" in key:
                            state_dict.pop(key)
                        elif peft_param_prefix in key:
                            value = state_dict.pop(key)
                            new_key = key.replace(peft_param_prefix, "")
                            state_dict[new_key] = value

                    torch.save(state_dict, os.path.join(
                        output_dir, WEIGHTS_NAME))

            else:
                print(
                    "Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, max_shard_size=self.max_shard_size)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        pass

    def _get_train_sampler(self):
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

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
            model_input_name = self.tokenizer.model_input_names[
                0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=seed,
                )

        else:
            if self.args.world_size <= 1:
                return RandomSampler(self.train_dataset, generator=generator)
            elif (
                self.args.parallel_mode in [
                    ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                and not self.args.dataloader_drop_last
            ):
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )
            else:
                if (hasattr(self.train_dataset, "shard_data") and self.train_dataset.shard_data) or \
                        (hasattr(self.train_dataset, "scatter_num") and self.train_dataset.scatter_num > 1):
                    return ShardDistributedSampler(
                        self.train_dataset,
                        num_replicas=self.args.world_size,
                        rank=self.args.process_index,
                        seed=seed,
                    )
                else:
                    return DistributedSampler(
                        self.train_dataset,
                        num_replicas=self.args.world_size,
                        rank=self.args.process_index,
                        seed=seed,
                    )
