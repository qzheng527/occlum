#!/usr/bin/env python
# coding=utf-8
# @Author: tianxuan.jl
# @Date: Wed 31 May 2023 02:33:16 PM CST

import copy
import datetime
import json
import logging
import math
import os
import random
import re
import shutil
import sys  # noqa: F401
import time
import traceback
import warnings
from functools import partial
from pathlib import Path

import atorch
import numpy as np
import torch
from atorch.utils.fsdp_save_util import (ShardOptim, save_fsdp_flat_param,
                                         save_fsdp_optim_param)
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.multiprocessing import Process
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import get_scheduler as get_scheduler_trans
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.trainer import (SCHEDULER_NAME, TRAINER_STATE_NAME,
                                  TRAINING_ARGS_NAME)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import WEIGHTS_NAME

from solutions.antllm.antllm.data.dataset.shard_distributed_sampler import \
    ShardDistributedSampler
from solutions.antllm.antllm.models.peft.modeling_peft import (  # noqa
    AntPeftForCausalLM, PeftModel)

HYPER_PARAMETER_NAME = 'hyper_parameters.json'
NAN_CHECKPOINT_NAME = 'nan'
ATORCH_CHECKPOINT_NAME = 'atorch_checkpoint.bin'
EPOCH_CHECKPOINT_NAME = 'epoch'
PEFT_PARAM_PREFIX = "base_model.model."
LORA_KEY = 'lora'
LAST_EXIT_INFO_FILE = 'last_exit_info.json'
STREAMING_CKPT_DIR = "streaming_ckpt"  # FSDP流式save/load存储目录

logger = logging.getLogger(__name__)


def is_local_main_process():
    return atorch.local_rank() == 0


def is_global_main_process():
    return atorch.global_rank() == 0


def is_valid_checkpoint(path, save_load_by_streaming=False):
    if save_load_by_streaming:
        atorch_checkpoint_path = os.path.join(path, STREAMING_CKPT_DIR)
    else:
        atorch_checkpoint_path = os.path.join(path, ATORCH_CHECKPOINT_NAME)
    if not os.path.exists(atorch_checkpoint_path):
        logger.info(f'{atorch_checkpoint_path} not exist')
        return False
    return True


def count_model_params(model):
    trainable_params = 0
    all_params = 0
    for param in model.parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    return all_params, trainable_params


def has_inf_or_nan(x):
    try:
        # if x is half, the .float() incurs an additional deep copy, but it's necessary if
        # Pytorch's .sum() creates a one-element tensor of the same type as x
        # (which is true for some recent version of pytorch).
        cpu_sum = float(x.float().sum())
        # More efficient version that can be used if .sum() returns a Python scalar
        # cpu_sum = float(x.sum())
    except RuntimeError as instance:
        # We want to check if inst is actually an overflow exception.
        # RuntimeError could come from a different error.
        # If so, we still want the exception to propagate.
        if "value cannot be converted" not in instance.args[0]:
            raise
        return True
    else:
        if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
            return True
        return False


def local_token_level_cross_entropy(outputs, labels, **kwargs):
    # return outputs.loss / torch.distributed.get_world_size()
    # 在每个batch内部做token-level的平均,然后在所有batch间做平均
    return outputs.loss


def mini_batch_token_level_cross_entropy(outputs, labels, mini_batch=1, **kwargs):
    # 这个loss会先把batch分成小的mini_batch,在mini_batch内做个token-level的平均,然后做所有卡之间的平均
    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
    if labels.shape[0] % mini_batch != 0:
        # 如果batch % mini_batch != 0, 则不切分计算. 有的数据量一个epoch结束的时候可能会出现这个情况
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
    else:
        loss = loss_fct(outputs.logits.view(-1, outputs.logits.size(-1)),
                        labels.view(-1)).reshape(labels.shape[0] // mini_batch, -1)

        labels = labels.reshape(labels.shape[0] // mini_batch, -1)
        loss = loss.sum(-1) / (labels != -100).sum(-1)
        loss = loss.mean()
    return loss


def sample_level_cross_entropy(outputs, labels, **kwargs):
    # 先对所有样本字token-level的平均,然后计算所有sample的平均值
    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
    loss = loss_fct(outputs.logits.view(-1, outputs.logits.size(-1)),
                    labels.view(-1)).reshape(labels.shape[0], -1)
    loss = loss.sum(-1) / (labels != -100).sum(-1)
    loss = loss.mean()
    return loss


def global_token_level_cross_entropy(outputs, labels, **kwargs):
    # 对所有样本一起做token-level的平均
    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
    loss = loss_fct(outputs.logits.view(-1, outputs.logits.size(-1)),
                    labels.view(-1)).reshape(labels.shape[0], -1)
    num_tokens = (loss != 0).sum()
    loss = loss.sum()

    num_tokens_tensor = torch.zeros(
        [1], device=loss.device, dtype=loss.dtype)
    num_tokens_tensor[0] = num_tokens.item()

    torch.distributed.all_reduce(num_tokens_tensor)

    global_num_tokens = num_tokens_tensor.sum()

    torch.distributed.barrier()
    # global_num_tokens是全局的token数，因为在梯度更新的时候回自动对所有卡求mean
    # 所有这里要乘一个world_size
    loss = loss.sum() / global_num_tokens * torch.distributed.get_world_size()

    return loss


LOSS_MAP = {
    'local_token_level_cross_entropy': local_token_level_cross_entropy,
    'mini_batch_token_level_cross_entropy': mini_batch_token_level_cross_entropy,
    'sample_level_cross_entropy': sample_level_cross_entropy,
    'global_token_level_cross_entropy': global_token_level_cross_entropy,
}


class AtorchArguments:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def get_linear_schedule_with_log_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):

    def lr_lambda(current_step: int):
        inverse_log_warm_up = 1.0 / math.log(num_warmup_steps + 1e-6)
        if current_step == 0:
            return 0.0
        if current_step < num_warmup_steps:
            return inverse_log_warm_up * math.log(current_step + 1e-6)
        return max(
            0.0, float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_scheduler(name, optimizer, num_warmup_steps, num_training_steps):
    scheduler_map = {
        'log_warmup_linear_decay': get_linear_schedule_with_log_warmup}
    try:
        lr_scheduler = get_scheduler_trans(
            name, optimizer, num_warmup_steps, num_training_steps)
        return lr_scheduler
    except Exception:
        schedule_func = scheduler_map[name]
        return schedule_func(optimizer, num_warmup_steps, num_training_steps)


def recursively_to_cpu(data):
    if isinstance(data, dict):
        return type(data)(
            {k: recursively_to_cpu(v) for k, v in data.items()}
        )
    elif isinstance(data, (list, tuple)):
        return type(data)(
            [recursively_to_cpu(v) for v in data]
        )
    elif isinstance(data, torch.Tensor):
        if data.is_cuda:
            return data.cpu()
        else:
            return data
    else:
        return data


def get_last_checkpoint(folder, save_load_by_streaming=False):

    _re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
    content = sorted(os.listdir(folder))
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return None
    _sorted_checkpoints = sorted(checkpoints, key=lambda x: int(
        _re_checkpoint.search(x).groups()[0]), reverse=True)
    print(_sorted_checkpoints)
    for checkpoint in _sorted_checkpoints:
        path = os.path.join(folder, checkpoint)
        print(path)
        print(f'check correctness of {path}')
        if is_valid_checkpoint(path, save_load_by_streaming=save_load_by_streaming):
            print(f'{path} if a valid checkpoint')
            return path
    return None


class AtorchTrainer:
    def __init__(self,
                 model,
                 args,
                 train_dataset,
                 eval_dataset,
                 evaluator=None,
                 tokenizer=None,
                 callbacks=None,
                 no_save_atorch_checkpoint=False,
                 no_save_base_model=False,
                 save_pytorch_model_bin_checkpoint=True,
                 blocking_save=False,
                 train_peft=False,
                 rank=0,
                 max_shard_size='50GB',
                 files_to_save=None,
                 args_to_save=None,
                 dynamic_padding=False,
                 pad_id=None,
                 **kwargs,
                 ):
        self.args = args
        print(self.args)
        self.model = model
        self.no_save_atorch_checkpoint = no_save_atorch_checkpoint
        self.no_save_base_model = no_save_base_model
        self.save_pytorch_model_bin_checkpoint = save_pytorch_model_bin_checkpoint
        self.train_peft = train_peft
        self.blocking_save = blocking_save
        self.rank = rank
        self.kwargs = kwargs
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.max_shard_size = max_shard_size
        self.files_to_save = files_to_save
        self.args_to_save = args_to_save
        self.dynamic_padding = dynamic_padding
        self.evaluator = evaluator
        if self.dynamic_padding:
            logger.info('Using dynamic_padding')
        self.pad_id = pad_id
        self.eval_dataloader = None
        if eval_dataset is not None:
            self.eval_dataloader = DataLoader(
                eval_dataset,
                sampler=DistributedSampler(eval_dataset),
                batch_size=args.per_device_eval_batch_size,
                pin_memory=True,
            )
        self.total_train_batch_size = self.args.per_device_train_batch_size * \
            self.args.gradient_accumulation_steps * \
            atorch.world_size()

        self.train_dataloader_args = {
            "shuffle": True,
            "batch_size": self.total_train_batch_size,
            "pin_memory": True,
            "num_workers": self.args.dataloader_num_workers,
        }
        if self.args.loss_func == 'mini_batch_token_level_cross_entropy':
            assert self.args.per_device_train_batch_size % self.args.mini_batch == 0

        self.resume_checkpoint_dir = None
        if self.args.resume_from_checkpoint == 'true':
            self.resume_checkpoint_dir = get_last_checkpoint(
                self.args.output_dir, save_load_by_streaming=self.args.save_load_by_streaming)
        print(f'last_checkpoint: {self.resume_checkpoint_dir}')
        if self.resume_checkpoint_dir is None:
            print(
                f'Checkpoint not found in {self.args.output_dir}, train from scratch')
            self.args.resume_from_checkpoint = 'false'

        self.atorch_args = AtorchArguments(
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            adam_eps=args.adam_epsilon,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2)

        print('atorch initialization start')
        self.atorch_init()
        print('atorch initialization finished')

        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if self.args.max_steps == -1:
            self.args.max_steps = int(
                self.args.num_train_epochs * self.num_update_steps_per_epoch)
        else:
            self.args.num_train_epochs = math.ceil(
                self.args.max_steps / self.num_update_steps_per_epoch)

        self.args.warmup_steps = self.args.get_warmup_steps(
            self.args.max_steps)
        custom_lr_scheduler_type = self.kwargs.get(
            'custom_lr_scheduler_type', None)
        self.lr_scheduler = get_scheduler(
            name=custom_lr_scheduler_type if custom_lr_scheduler_type else self.args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.args.max_steps,
        )
        if self.args.resume_from_checkpoint == 'true':
            with warnings.catch_warnings(record=True):
                self.lr_scheduler.load_state_dict(torch.load(
                    os.path.join(self.resume_checkpoint_dir, SCHEDULER_NAME)))
            self._load_rng_state(self.resume_checkpoint_dir)
        torch.distributed.barrier()
        now_datetime = datetime.datetime.now()
        timestr = datetime.datetime.strftime(now_datetime, '%Y%m%d-%H%M%S')
        self.log_dir = os.path.join(self.args.output_dir, 'runs', timestr)
        self.summary_writer = None
        self.all_logs = {}
        self.last_log_procs = {}
        if torch.distributed.get_rank() == 0:
            try:
                self.summary_writer = SummaryWriter(log_dir=self.log_dir)
            except Exception:
                pass
        self.device = f"cuda:{atorch.local_rank()}"

    def _load_rng_state(self, resume_checkpoint_dir):
        # Load RNG states from `checkpoint`
        if resume_checkpoint_dir is None:
            return

        if self.args.world_size > 1:
            rng_file = os.path.join(
                resume_checkpoint_dir, f"rng_state_{self.rank}.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    f"Didn't find an RNG file for process {self.rank}, if you are resuming a training that "
                    "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
                )
                return
        else:
            rng_file = os.path.join(resume_checkpoint_dir, "rng_state.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
                    "fashion, reproducibility is not guaranteed."
                )
                return

        checkpoint_rng_state = torch.load(rng_file)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        if torch.cuda.is_available():
            if self.args.local_rank != -1:
                torch.cuda.random.set_rng_state(checkpoint_rng_state["cuda"])
            else:
                try:
                    torch.cuda.random.set_rng_state_all(
                        checkpoint_rng_state["cuda"])
                except Exception as e:
                    logger.info(
                        f"Didn't manage to set back the RNG states of the GPU because of the following error:\n {e}"
                        "\nThis won't yield the same results as if the training had not been interrupted."
                    )

    def load_atorch_model_state(self, model_state_dict, **kwargs):
        print('resume atorch model state')
        if self.is_rank0():
            self.model.load_state_dict(model_state_dict)
        # 在 rank 0 加载完毕后，再通过sync_module_states分发参数
        torch.distributed.barrier()
        # self.model = FSDP(self.model, sync_module_states=True, **kwargs)

    def load_atorch_optim_state(self, optim_state_dict):
        logger.info('resume optimizer state')
        if self.args.atorch_opt == "fsdp":
            optim_state_dict = FSDP.scatter_full_optim_state_dict(
                optim_state_dict, self.model)  # may be removed after PyTorch 2.2
        optim_state_dict = recursively_to_cpu(optim_state_dict)
        self.optimizer.load_state_dict(optim_state_dict)

    def atorch_init(self):

        from atorch.auto import auto_accelerate
        from atorch.utils.version import torch_version

        from solutions.antllm.antllm.models.glm.modeling_glm import GLMBlock

        assert torch_version() >= (2, 0, 0), "use pt2.0 for use orig param if fsdp"
        # fsdp_config = {
        #     "sync_module_states": True,
        #     "use_orig_params": True,
        #     "limit_all_gathers": True,
        #     "cpu_offload": self.args.cpu_offload,
        # }
        # fsdp_config["atorch_wrap_cls"] = {
        #     GLMBlock,
        # }
        p_mode = ([("data", torch.distributed.get_world_size())], None)
        strategy = [
            ("parallel_mode", p_mode),
            "module_replace",
        ]
        if self.args.atorch_opt == "fsdp":
            if self.args.peft_type is None:
                fsdp_config = {
                    "sync_module_states": True,
                    "use_orig_params": True,
                    "limit_all_gathers": True,
                    "cpu_offload": self.args.cpu_offload,
                    "atorch_wrap_cls": {GLMBlock, },
                }
                strategy.append(("fsdp", fsdp_config))
            else:
                num_all_params, num_trainable_params = count_model_params(
                    self.model)
                if num_all_params < 11e9 or self.args.peft_type == "qlora":  # For GLM-10B
                    logger.info(
                        f"Found using {self.args.peft_type} method. The peft model has {num_all_params} and only "
                        f"{num_trainable_params} params are trainable({100 * num_trainable_params / num_all_params}%)"
                        ". Set atorch opt to DistributedDataParallel."
                    )
                    self.args.atorch_opt = "ddp"

        if self.args.bf16 or self.args.fp16:
            if self.args.bf16:
                amp_config = {"dtype": torch.bfloat16,
                              "skip_if_nonfinite": True}
                if self.args.peft_type == "qlora":
                    # The dtype of grads is bf16 when using qlora.
                    # atorch scaler does not support bf16 grads.
                    amp_config["skip_if_nonfinite"] = False
            elif self.args.fp16:
                amp_config = {"dtype": torch.float16}
            strategy.append(("amp_native", amp_config))

        if self.args.gradient_checkpointing:
            strategy.append(("checkpoint", (GLMBlock,)))
        print(f"Manually loaded auto acc strategy: {strategy}")

        def prepare_input(batch, device):
            if self.dynamic_padding:
                if self.pad_id is not None:
                    input_ids = batch['input_ids']
                    max_length = -1

                    for row in range(input_ids.shape[0]):
                        for col in range(input_ids.shape[1] - 1, -1, -1):
                            if input_ids[row][col] != self.pad_id:
                                break
                        max_length = col if col > max_length else max_length
                    max_length += 1
                    labels = batch['labels'][:, :max_length]
                    num_effective_labels = (labels != -100).sum(-1)
                    if torch.any(num_effective_labels == 0):
                        pass
                    else:
                        batch['input_ids'] = batch['input_ids'][:, :max_length]
                        batch['labels'] = batch['labels'][:, :max_length]
                        batch['position_ids'] = batch['position_ids'][..., :max_length]
                        if len(batch['attention_mask'].shape) == 4:
                            batch['attention_mask'] = batch['attention_mask'][
                                :, :max_length, :max_length]
                        # support for chatglm2
                        elif len(batch['attention_mask'].shape) == 2:
                            batch['attention_mask'] = batch['attention_mask'][:, :max_length]  
                else:
                    print(
                        'Ignore dynamic_padding, while dynamic_padding, pad_id muast be set')
            batch = {k: v.to(device=device, non_blocking=True)
                     for k, v in batch.items()}
            return batch

        def optim_param_func(model, args):
            no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            return optimizer_grouped_parameters

        # load fsdp checkpoint参数
        if self.args.resume_from_checkpoint == 'true':
            logger.info(f'Resume training from {self.resume_checkpoint_dir}')
            if not self.args.save_load_by_streaming:
                if self.is_rank0():
                    sd = torch.load(os.path.join(
                        self.resume_checkpoint_dir, ATORCH_CHECKPOINT_NAME), map_location='cpu')
                    model_state_dict, optim_state_dict = sd['model_state_dict'], sd['optimizer_state_dict']
                else:
                    model_state_dict, optim_state_dict = None, None
                torch.distributed.barrier()  # other rank waiting
                ##########
                self.load_atorch_model_state(model_state_dict)
                ##########

        optim_func = torch.optim.AdamW
        ddp_find_unused_parameters = None
        if self.args.atorch_opt == "ddp" \
                and not (self.args.peft_type in ["lora", "qlora"] and self.args.gradient_checkpointing):
            ddp_find_unused_parameters = True
        status, result, best_strategy = auto_accelerate(
            self.model, optim_func, self.train_dataset, distributed_sampler_cls=ShardDistributedSampler
            if (hasattr(self.train_dataset, "shard_data") and self.train_dataset.shard_data)
            or (hasattr(self.train_dataset, "scatter_num") and self.train_dataset.scatter_num > 1) else None,
            dataloader_args=self.train_dataloader_args, loss_func=LOSS_MAP.get(
                self.args.loss_func, sample_level_cross_entropy),
            prepare_input=prepare_input,
            optim_args={"lr": self.atorch_args.lr, "weight_decay": self.atorch_args.weight_decay,
                        "eps": self.atorch_args.adam_eps,
                        "betas": (self.atorch_args.adam_beta1, self.atorch_args.adam_beta2), },
            optim_param_func=partial(optim_param_func, args=self.atorch_args),
            load_strategy=strategy, ignore_dryrun_on_load_strategy=True,
            find_unused_parameters=ddp_find_unused_parameters,)
        assert (
            status
        ), f"auto_accelerate failed. status: {status}, result: {result}, best_strategy: {best_strategy}"
        logger.info(f"Best strategy is: {best_strategy}")

        self.model = result.model
        self.optimizer = result.optim
        self.loss_func = result.loss_func
        self.train_dataloader = result.dataloader
        self.prepare_input = result.prepare_input

        if self.args.resume_from_checkpoint == 'true':
            if not self.args.save_load_by_streaming:
                self.load_atorch_optim_state(optim_state_dict)
            else:
                sm = ShardOptim(os.path.join(
                    self.resume_checkpoint_dir, STREAMING_CKPT_DIR))
                reshard_optim_state = sm.reshard_optim_state_dict(self.model)
                self.optimizer.load_state_dict(reshard_optim_state)
        logger.info(f"atorch use optimizer: {self.optimizer}")

    def evaluate(self):
        logger.info(f"Start evaluation")
        self.model.eval()
        losses = []
        for step, batch in enumerate(self.eval_dataloader):
            with torch.no_grad():
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)

                loss = self.loss_func(
                    outputs, batch['labels'], mini_batch=self.args.mini_batch)
                repeated_loss = loss.repeat(
                    self.args.per_device_eval_batch_size)
                if repeated_loss.ndim == 0:
                    repeated_loss = repeated_loss.clone()[None]
                output_tensors = [repeated_loss.clone()
                                  for _ in range(atorch.world_size())]
                torch.distributed.all_gather(output_tensors, repeated_loss)
                losses.append(torch.cat(output_tensors, dim=0))

        losses = torch.cat(losses)
        losses = losses[: len(self.eval_dataset)]
        mean_loss = torch.mean(losses).item()
        logs = {'eval_loss': mean_loss}
        if self.is_rank0():
            self.log(logs, step=self.global_steps, phase='Evaluation')

    def log(self, logs, step, phase='Train'):
        # 开一个进程写tb,避免写入失败的时候卡住主进程,适用于磁盘满的时候
        def write_tb(writer, phase, logs_to_write):
            for item in logs_to_write:
                step = item['step']
                logs = item['logs']
                for key, value in logs.items():
                    print(f'step: {step}, {key}: {value}')
                    writer.add_scalar(f'{phase}/{key}', value, step)

        logger.info(json.dumps(logs))
        logs_to_write = [{'step': step, 'logs': logs}]
        phase_logs = self.all_logs.get(phase, [])
        phase_logs.append({'step': step, 'logs': logs})
        self.all_logs[phase] = phase_logs
        if self.last_log_procs.get(phase, None) and self.last_log_procs[phase].is_alive():
            # 如果前一个写tb的进程还在,说明前一个进程应该是出错了. 重置tb writer
            self.last_log_procs[phase].kill()
            self.summary_writer = None
        if not self.summary_writer:
            try:
                self.summary_writer = SummaryWriter(log_dir=self.log_dir)
                logs_to_write.extend(self.all_logs[phase])
            except Exception:
                pass
        if not self.summary_writer:
            logger.info(
                'Write failed, stash logs , will save at the next time')
            return
        p = Process(target=write_tb, args=(
            self.summary_writer,
            phase, logs_to_write)
        )
        self.last_log_procs[phase] = p
        p.start()

    def _sorted_checkpoints(
        self,
        output_dir=None,
        checkpoint_prefix=PREFIX_CHECKPOINT_DIR,
        checkpoint_name_pattern='([0-9]+)',
        use_mtime=False
    ):
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(output_dir).glob(
            f"{checkpoint_prefix}-*") if os.path.isdir(x)]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append(
                    (os.path.getmtime(path), path))
            else:
                regex_match = re.search(
                    f".*{checkpoint_prefix}-({checkpoint_name_pattern})", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append(
                        (int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1]
                              for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def _rotate_checkpoints(
            self,
            use_mtime=False,
            output_dir=None,
            prefix=PREFIX_CHECKPOINT_DIR,
            checkpoint_name_pattern='.*') -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(
            use_mtime=use_mtime,
            output_dir=output_dir,
            checkpoint_prefix=prefix,
            checkpoint_name_pattern=checkpoint_name_pattern)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
        # we don't do to allow resuming.
        save_total_limit = self.args.save_total_limit
        checkpoints_to_be_deleted = []

        valid_checkpoints = []
        for checkpoint in checkpoints_sorted:
            if is_valid_checkpoint(checkpoint):
                valid_checkpoints.append(checkpoint)
            else:
                checkpoints_to_be_deleted.append(checkpoint)

        number_of_checkpoints_to_delete = max(
            0, len(valid_checkpoints) - save_total_limit)
        checkpoints_to_be_deleted.extend(
            valid_checkpoints[:number_of_checkpoints_to_delete])
        # checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info(
                f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
            shutil.rmtree(checkpoint, ignore_errors=True)

    def _clean_atorch_checkpoints(
            self,
            output_dir=None,
            prefix=PREFIX_CHECKPOINT_DIR,
            checkpoint_name_pattern='([0-9]+)',
            max_atorch_checkpoints=1):
        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(
            output_dir=output_dir,
            checkpoint_prefix=prefix,
            checkpoint_name_pattern=checkpoint_name_pattern)

        # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
        # we don't do to allow resuming.

        for checkpoint in checkpoints_sorted[:len(checkpoints_sorted) - max_atorch_checkpoints]:
            logger.info(
                f"Deleting older atorch checkpoint [{checkpoint}] due to self.args.save_total_limit is "
                f"{self.args.save_total_limit}"
            )
            atorch_ckpt_path = os.path.join(checkpoint, ATORCH_CHECKPOINT_NAME)
            if os.path.exists(atorch_ckpt_path):
                try:
                    os.remove(atorch_ckpt_path)
                except Exception as e:
                    logger.info(
                        f"Failed to delete {atorch_ckpt_path}. Skip it. Error: {e}")
                    continue

            if self.args.save_load_by_streaming:
                atorch_streaming_ckpt_path = os.path.join(
                    checkpoint, STREAMING_CKPT_DIR)
                if os.path.exists(atorch_streaming_ckpt_path):
                    shutil.rmtree(atorch_streaming_ckpt_path,
                                  ignore_errors=True)

    def _save_peft_model(self, output_dir, state_dict=None):
        logger.info(f"Start saving peft model to {output_dir}")
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        model = unwrap_model(self.model)
        if isinstance(model, PeftModel):
            if state_dict is None:
                state_dict = model.state_dict()
            model.save_pretrained(
                output_dir, state_dict=state_dict, is_main_process=self.is_rank0())
        else:
            if state_dict is None:
                state_dict = self.model.state_dict()
            if self.is_rank0():
                torch.save(state_dict, os.path.join(
                    output_dir, "pytorch_model.bin"))
        logger.info(f"Saving peft model done.")

    def _save_model(self, output_dir=None, state_dict=None, tokenizer=None, args=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        logger.info(f"Start saving model checkpoint to {output_dir}")
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                model = unwrap_model(self.model)
                if state_dict is None:
                    state_dict = model.state_dict()
                model.save_pretrained(
                    output_dir,
                    state_dict=state_dict,
                    max_shard_size=self.max_shard_size,
                    is_main_process=self.is_rank0())
                # unwrap_model(self.model).save_pretrained(
                #     output_dir, state_dict=state_dict, max_shard_size=self.max_shard_size)
            elif isinstance(unwrap_model(self.model), PeftModel):
                if state_dict is None:
                    state_dict = unwrap_model(
                        self.model).base_model.model.state_dict()
                # Filter the peft params ...
                param_keys = list(state_dict.keys())
                base_model_state_dict = {}
                for key in param_keys:
                    if LORA_KEY in key:
                        # state_dict.pop(key)
                        continue
                    elif PEFT_PARAM_PREFIX in key:
                        # value = state_dict.pop(key)
                        value = state_dict[key]
                        new_key = key.replace(PEFT_PARAM_PREFIX, "")
                        base_model_state_dict[new_key] = value
                    else:
                        base_model_state_dict[key] = value
                if self.is_rank0():
                    torch.save(base_model_state_dict,
                               os.path.join(output_dir, WEIGHTS_NAME))
                    # save config.json of model
                    if hasattr(unwrap_model(self.model), "config"):
                        unwrap_model(self.model).config.save_pretrained(output_dir)
            else:
                logger.info(
                    "Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, max_shard_size=self.max_shard_size)
        if tokenizer is None:
            tokenizer = self.tokenizer
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        if args is None:
            args = self.args
        torch.save(args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def is_rank0(self):
        return self.rank == 0

    def _save_atorch_checkpoint(self, output_dir, model_state_dict=None, optimizer_state_dict=None, global_steps=None):

        if optimizer_state_dict is None or model_state_dict is None:
            if isinstance(self.model, FSDP):
                # StateDictType.FULL_STATE_DICT得到完整的模型状态。
                # FullStateDictConfig指定保存到CPU，仅rank0保存
                save_policy = FullStateDictConfig(
                    offload_to_cpu=atorch.world_size() > 1, rank0_only=atorch.world_size() > 1)
                with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                    model_state_dict = self.model.state_dict()
                    optim_state_dict = FSDP.full_optim_state_dict(
                        self.model, self.optimizer)  # may be removed after PyTorch 2.2
            else:
                model_state_dict = self.model.state_dict()
                optim_state_dict = self.optimizer.state_dict()
        if global_steps is None:
            global_steps = self.global_steps

        if self.is_rank0():
            torch.save(
                {
                    "model_state_dict": model_state_dict,
                    "optimizer_state_dict": optim_state_dict,
                    "global_steps": global_steps,
                },
                os.path.join(output_dir, ATORCH_CHECKPOINT_NAME),
            )
        torch.distributed.barrier()  # other rank waiting

    def _save_checkpoint(
            self,
            output_dir,
            model_state_dict,
            optim_state_dict,
            tokenizer,
            args,
            global_steps,
            no_save_atorch_checkpoint):
        try:
            if not no_save_atorch_checkpoint:
                torch.save(
                    {
                        "model_state_dict": model_state_dict,
                        "optimizer_state_dict": optim_state_dict,
                        "global_steps": global_steps,
                    },
                    os.path.join(output_dir, ATORCH_CHECKPOINT_NAME),
                )
            model_state_dict = {key: value.bfloat16() if self.args.bf16 else value.half(
            ) for key, value in model_state_dict.items()}
            output_dir = output_dir if output_dir is not None else self.args.output_dir

            if self.train_peft:
                if not self.no_save_base_model:
                    self._save_model(output_dir=output_dir,
                                     state_dict=model_state_dict, tokenizer=tokenizer, args=args)
                self._save_peft_model(output_dir=output_dir,
                                      state_dict=model_state_dict)
            else:
                self._save_model(output_dir=output_dir,
                                 state_dict=model_state_dict, tokenizer=tokenizer, args=args)
        except Exception:
            traceback.print_exc()
            self.abort_save(output_dir)
            return False

        if self.args.extra_save_by_epoch:
            # 如果是每个epoch extra save的，那么每个epoch的checkpoin不会删除，不受save_total_limit的影响，
            # 而对按step存的，则会只保留save_total_limit个
            self._rotate_checkpoints(
                output_dir=self.args.output_dir, prefix=PREFIX_CHECKPOINT_DIR, checkpoint_name_pattern='([0-9]+)$')
        else:
            self._rotate_checkpoints(
                output_dir=self.args.output_dir, prefix=PREFIX_CHECKPOINT_DIR)
        # 只保留最新一个checkpoint的atorch checkpoint
        logger.info(
            "Keep only the latest atorch checkpoint and delete the previous checkpoints.")
        self._clean_atorch_checkpoints(
            output_dir=self.args.output_dir,
            prefix=PREFIX_CHECKPOINT_DIR,
            checkpoint_name_pattern='([0-9]+)$')
        return True

    def abort_save(self, output_dir):
        logger.info('Expcetion during saving, skip save')
        if self.is_rank0():
            if os.path.isdir(output_dir):
                logger.info(f'Delete {output_dir}')
                shutil.rmtree(output_dir, ignore_errors=True)

    def save(self, suffix=None, blocking=False, no_save_atorch_checkpoint=False):
        if suffix is None:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.global_steps}"
        else:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{suffix}"
        run_dir = self.args.output_dir
        output_dir = os.path.join(run_dir, checkpoint_folder)
        # if self.is_rank0():
        #     os.makedirs(output_dir, exist_ok=True)
        # torch.distributed.barrier()
        print(f'no_save_atorch_checkpoint: {no_save_atorch_checkpoint}')

        # abort_flag = False
        # save_flag_tensor = torch.zeros(
        #     [1], device=self.device, dtype=torch.int)
        logger.info(f'rank {self.rank}: Save start')
        try:
            os.makedirs(output_dir, exist_ok=True)
            if not self.save_pytorch_model_bin_checkpoint:
                return

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

            if torch.distributed.get_world_size() <= 1:
                torch.save(rng_states, os.path.join(
                    output_dir, "rng_state.pth"))
            else:
                torch.save(rng_states, os.path.join(
                    output_dir, f"rng_state_{self.args.process_index}.pth"))
            # save_flag_tensor[0] = 0
        except Exception:
            print(f'rank {self.rank}, 存随机状态失败')
            traceback.print_exc()
            # save_flag_tensor[0] = 1

        # torch.distributed.all_reduce(save_flag_tensor)
        # save_flag = save_flag_tensor.sum()

        # if save_flag > 0:
        #     if self.is_rank0():
        #         self.abort_save(output_dir)
        #     abort_flag = True
        #     return

        torch.distributed.barrier()
        abort_flag = False
        if self.is_rank0():
            # rank0检测上一步随机状态是否存成功
            if torch.distributed.get_world_size() <= 1:
                if not os.path.exists(os.path.join(output_dir, 'rng_state.pth')):
                    abort_flag = True
            else:
                for process_index in range(torch.distributed.get_world_size()):
                    if not os.path.exists(os.path.join(output_dir, f'rng_state_{process_index}.pth')):
                        print('rng_state_{process_index}.pth 不存在')
                        abort_flag = True
            if abort_flag:
                print('存随机状态失败')
                self.abort_save(output_dir)
            if not abort_flag:
                try:
                    if self.args_to_save:
                        json.dump(
                            self.args_to_save,
                            open(os.path.join(output_dir,
                                 HYPER_PARAMETER_NAME), 'w'),
                            ensure_ascii=False,
                            indent=2
                        )
                    # save state
                    state = {'global_steps': self.global_steps,
                             'loss': self.all_logs}
                    json.dump(state, open(os.path.join(
                        output_dir, TRAINER_STATE_NAME), 'w'), ensure_ascii=False, indent=2)
                    if self.files_to_save and not self.args.save_load_by_streaming:
                        for name in self.files_to_save:
                            if not os.path.exists(name):
                                continue
                            try:
                                if os.path.isfile(name):
                                    shutil.copy(name, output_dir)
                                elif os.path.isdir(name):
                                    shutil.copytree(name, os.path.join(
                                        output_dir, os.path.basename(name)))
                            except Exception:
                                continue
                    torch.save(
                        self.lr_scheduler.state_dict(),
                        os.path.join(output_dir, SCHEDULER_NAME)
                    )
                except Exception:
                    print(f'rank, {self.rank}存训练状态失败')
                    traceback.print_exc()
                    self.abort_save(output_dir)
                    abort_flag = True

        if not self.args.save_load_by_streaming:
            # 获取要存的state_dict, 每个rank都要调用
            if isinstance(self.model, FSDP):
                save_policy = FullStateDictConfig(
                    offload_to_cpu=atorch.world_size() > 1, rank0_only=atorch.world_size() > 1)
                with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                    model_state_dict = self.model.state_dict()
                    optim_state_dict = FSDP.full_optim_state_dict(
                        self.model, self.optimizer)  # may be removed after PyTorch 2.2
            else:
                model_state_dict = unwrap_model(self.model).state_dict()
                optim_state_dict = self.optimizer.state_dict()
        if self.is_rank0():
            # 获取需要存的tokenzer和args
            tokenizer_to_save = copy.deepcopy(self.tokenizer)
            args_to_save = copy.deepcopy(self.args)

        if blocking:  # 同步存ckpt
            if self.args.save_load_by_streaming:
                # 不需要生成单独创建 state_dict
                try:
                    # 所有rank都调用
                    streaming_ckpt_dir = os.path.join(
                        output_dir, STREAMING_CKPT_DIR)
                    os.makedirs(streaming_ckpt_dir, exist_ok=True)
                    save_fsdp_flat_param(self.model, streaming_ckpt_dir)
                    save_fsdp_optim_param(
                        self.model, self.optimizer, streaming_ckpt_dir)
                except Exception as e:
                    logger.warning(f"Save model and optim param failed. {e}")
                    self.abort_save(output_dir)
                if self.is_rank0():
                    try:
                        if tokenizer_to_save is not None:
                            tokenizer_to_save.save_pretrained()
                        if args_to_save is not None:
                            torch.save(args_to_save, os.path.join(
                                output_dir, TRAINING_ARGS_NAME))
                    except Exception as e:
                        logger.warning(f"Save tokenizer and args. {e}")
                        self.abort_save(output_dir)
                    logger.info(
                        f"Keep only {self.args.save_total_limit} atorch streaming checkpoint(s) and delete the"
                        " previous checkpoints."
                    )
                    self._clean_atorch_checkpoints(
                        output_dir=self.args.output_dir,
                        prefix=PREFIX_CHECKPOINT_DIR,
                        checkpoint_name_pattern='([0-9]+)$'
                    )
            else:
                if self.is_rank0():
                    if not abort_flag:
                        self._save_checkpoint(
                            output_dir,
                            model_state_dict,
                            optim_state_dict,
                            tokenizer_to_save,
                            args_to_save,
                            self.global_steps,
                            no_save_atorch_checkpoint)
        else:
            if self.is_rank0():
                if not abort_flag:
                    model_state_dict = recursively_to_cpu(model_state_dict)
                    optim_state_dict = recursively_to_cpu(optim_state_dict)
                    p = Process(target=self._save_checkpoint, args=(
                        output_dir,
                        model_state_dict,
                        optim_state_dict,
                        tokenizer_to_save,
                        args_to_save,
                        self.global_steps,
                        no_save_atorch_checkpoint)
                    )
                    p.start()
                # 启动一个进程来调用_save_model和save_atorch_checkpoint存这些数据

        torch.distributed.barrier()
        saving_log = f"rank {self.rank} at the end of `save` method. "
        if not self.blocking_save:
            async_saving_log = (f"Note that blocking_save is {self.blocking_save}. "
                                "Trainer will spawn a subprocess to save checkpoint asynchronously. "
                                "This log does not mean saving has finished.")
            saving_log += async_saving_log
        logger.info(saving_log)

    def train(self, **kwargs):
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_train_batch_size}")
        logger.info(
            f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.args.max_steps}")

        progress_bar = tqdm(range(self.args.max_steps),
                            disable=not is_local_main_process(), smoothing=0)
        training_time = 0

        self.global_steps = 0
        start_epoch = 0
        steps_trained_in_current_epoch = 0
        self.samples_trained = 0
        self.steps_to_skip = 0
        if self.args.resume_from_checkpoint == 'true':
            state = json.load(
                open(os.path.join(self.resume_checkpoint_dir, TRAINER_STATE_NAME), 'r'))
            self.global_steps = state.get('global_steps', 0)
            start_epoch = self.global_steps // self.num_update_steps_per_epoch
            steps_trained_in_current_epoch = self.global_steps % self.num_update_steps_per_epoch
            steps_trained_in_current_epoch *= self.args.gradient_accumulation_steps
            self.samples_trained = self.global_steps * self.total_train_batch_size
            if os.path.isfile(os.path.join(self.args.output_dir, LAST_EXIT_INFO_FILE)):
                last_exit_info = json.load(
                    open(os.path.join(self.args.output_dir, LAST_EXIT_INFO_FILE), 'r'))
                self.steps_to_skip = last_exit_info.get('steps_to_skip', 0)
                logger.info(f'steps_to_skip: {self.steps_to_skip}')
                if self.is_rank0():
                    os.remove(os.path.join(
                        self.args.output_dir, LAST_EXIT_INFO_FILE))
            self.global_steps += self.steps_to_skip
            progress_bar = tqdm(range(self.args.max_steps),
                                disable=not is_local_main_process(),
                                initial=self.global_steps, smoothing=0)
            print(
                f'Start training at step {self.global_steps}, trained {self.samples_trained} samples')
        self.last_step_logged = self.global_steps
        self.accumulated_loss = 0
        if 'interval' == self.args.save_policy:
            self.last_time_save_checkpoint = time.time()
        # loss_fout = open(os.path.join(self.args.output_dir, f'loss_{self.rank}'), 'w')
        self.last_step_saved = self.global_steps
        for epoch in range(start_epoch, int(self.args.num_train_epochs)):
            self.train_dataloader.set_epoch(epoch)
            self.model.train()
            start_time = time.time()

            for step, batch in enumerate(self.train_dataloader):
                self.model.train()

                # To avoid loss compute get NaN when use fp16 pretrained model, set model dtype to bfloat16
                if self.args.bf16 and self.model.dtype == torch.float16:
                    self.model.bfloat16()
                    
                batch = self.prepare_input(batch, self.device)
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                if self.steps_to_skip > 0:
                    # 如果发生了因为nan退出,下次重启的时候要跳过一些数据
                    self.steps_to_skip -= 1
                    continue
                outputs = self.model(**batch)
                loss = self.loss_func(
                    outputs, batch['labels'], mini_batch=self.args.mini_batch)
                if torch.isnan(loss) or torch.isinf(loss):
                    file_name = f"nan_info_rank{self.rank}_step{self.global_steps}.bin"
                    save_path = os.path.join(self.args.output_dir, file_name)
                    logger.info(
                        f"Rank {self.rank} computes overflow loss at step {self.global_steps}."
                        f"Save related file to {save_path}"
                    )
                    nan_info = {
                        "global_steps": self.global_steps,
                        "rank": self.rank,
                        "logits": outputs.logits,
                        "inputs": batch,
                    }
                    torch.save(nan_info, save_path)
                loss = loss / self.args.gradient_accumulation_steps
                loss_tensor = torch.zeros(
                    [1], device=loss.device, dtype=loss.dtype)
                loss_tensor[0] = loss.item()
                torch.distributed.all_reduce(loss_tensor)
                reduce_loss = loss_tensor.sum() / torch.distributed.get_world_size()
                if self.args.resume_and_skip_data_if_nan and has_inf_or_nan(reduce_loss):
                    if self.args.save_nan_checkpoint:
                        # 记录和上一次保存之间有多少个step
                        steps_to_skip = self.global_steps - self.last_step_saved + 2
                        exit_info = {'steps_to_skip': steps_to_skip}
                        fout = open(os.path.join(
                            self.args.output_dir, LAST_EXIT_INFO_FILE), 'w')
                        fout.write(json.dumps(
                            exit_info, ensure_ascii=False) + '\n')
                        fout.close()
                        self.args.output_dir = os.path.join(
                            self.args.output_dir, 'nan_ckpts')
                        self.save(
                            suffix=f'{self.global_steps}-{NAN_CHECKPOINT_NAME}',
                            blocking=True,
                            no_save_atorch_checkpoint=False)
                    self.accumulated_loss += reduce_loss.item()
                    if self.global_steps > self.last_step_logged:
                        log_interval = self.global_steps - self.last_step_logged
                    else:
                        log_interval = 1
                    train_loss = round(
                        self.accumulated_loss / log_interval, 4)
                    self.epoch = self.global_steps / self.num_update_steps_per_epoch
                    learning_rate = self.lr_scheduler.get_last_lr()[0]
                    if torch.is_tensor(learning_rate):
                        learning_rate = learning_rate.item()
                    # gather all number of samples
                    batch_len_tensor = torch.zeros([1], device=loss.device)
                    batch_len_tensor[0] = batch['input_ids'].shape[0]
                    torch.distributed.all_reduce(batch_len_tensor)
                    num_samples_in_current_batch = int(
                        torch.sum(batch_len_tensor).item())
                    self.samples_trained += num_samples_in_current_batch

                    logs = {'loss': train_loss,
                            'epoch': self.epoch, 'learning_rate': learning_rate}
                    if self.is_rank0():
                        self.log(logs, step=self.global_steps,
                                 phase='train')
                        self.log(logs, step=self.samples_trained,
                                 phase='train_wrt_samples')
                        logger.info('Failed due to NAN loss')
                    # sys.exit(-1)
                self.accumulated_loss += reduce_loss.item()
                loss.backward()
                self.global_steps += 1
                # gather all number of samples
                batch_len_tensor = torch.zeros([1], device=loss.device)
                batch_len_tensor[0] = batch['input_ids'].shape[0]
                torch.distributed.all_reduce(batch_len_tensor)
                num_samples_in_current_batch = int(
                    torch.sum(batch_len_tensor).item())
                self.samples_trained += num_samples_in_current_batch
                ######
                if step % self.args.gradient_accumulation_steps == 0 or step == len(self.train_dataloader) - 1:
                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                        # 如果是fp16，需要unscale。如果是bf16，self.optimizer里没有unscale这个方法
                        try:
                            self.optimizer.unscale_()
                        except Exception:
                            pass
                        if isinstance(self.model, FSDP):
                            self.model.clip_grad_norm_(self.args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    overflow = hasattr(
                        self.optimizer, "step_was_skipped") and self.optimizer.step_was_skipped
                    if not overflow:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    progress_bar.update(1)

                    if 'steps' == self.args.save_policy:
                        if self.global_steps % self.args.save_steps == 0:
                            self.save(
                                blocking=self.blocking_save,
                                no_save_atorch_checkpoint=self.no_save_atorch_checkpoint)
                            self.last_step_saved = self.global_steps
                    elif 'interval' == self.args.save_policy:
                        # 用rank0上的记录判断是否应该存模型
                        cur_time = time.time()
                        interval_tensor = torch.zeros(
                            [1], device=loss.device, dtype=loss.dtype)
                        if self.is_rank0():
                            cur_time = time.time()
                            interval_tensor[0] = cur_time - \
                                self.last_time_save_checkpoint
                        else:
                            interval_tensor[0] = 0

                        torch.distributed.all_reduce(interval_tensor)

                        if interval_tensor.item() >= self.args.save_interval:
                            self.save(
                                blocking=self.blocking_save,
                                no_save_atorch_checkpoint=self.no_save_atorch_checkpoint)
                            self.last_time_save_checkpoint = cur_time
                            self.last_step_saved = self.global_steps

                    if self.global_steps >= self.args.max_steps:
                        break
                    if self.global_steps % self.args.logging_steps == 0:
                        train_loss = round(
                            self.accumulated_loss / (self.global_steps - self.last_step_logged), 4)
                        self.accumulated_loss = 0
                        self.last_step_logged = self.global_steps
                        self.epoch = self.global_steps / self.num_update_steps_per_epoch
                        learning_rate = self.lr_scheduler.get_last_lr()[0]
                        if torch.is_tensor(learning_rate):
                            learning_rate = learning_rate.item()
                        logs = {'loss': train_loss,
                                'epoch': self.epoch, 'learning_rate': learning_rate}
                        if self.is_rank0():
                            self.log(logs, step=self.global_steps,
                                     phase='train')
                            self.log(logs, step=self.samples_trained,
                                     phase='train_wrt_samples')
                    if self.args.evaluation_strategy == 'steps' and self.global_steps % self.args.eval_steps == 0:
                        self.evaluate()

            logger.info(f"Training of epoch {epoch} finished")

            if self.steps_to_skip > 0:
                # 如果epoch结束还需要继续skip,,则不存模型以及evaluate
                continue
            training_time += time.time() - start_time
            if self.args.evaluation_strategy == 'epoch':
                self.evaluate()
            if self.args.save_policy == 'epoch':
                self.save(
                    blocking=self.blocking_save,
                    no_save_atorch_checkpoint=self.no_save_atorch_checkpoint)
                if self.args.save_policy == 'interval':
                    self.last_time_save_checkpoint = time.time()
            elif self.args.extra_save_by_epoch:
                no_save_atorch_checkpoint = True
                self.save(
                    suffix=f'{self.global_steps}-{EPOCH_CHECKPOINT_NAME}-{epoch + 1}',
                    blocking=self.blocking_save,
                    no_save_atorch_checkpoint=no_save_atorch_checkpoint)

        self._clean_atorch_checkpoints(output_dir=self.args.output_dir, prefix=PREFIX_CHECKPOINT_DIR,
                                       checkpoint_name_pattern="([0-9]+)$", max_atorch_checkpoints=0)
