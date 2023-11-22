import contextlib
import os
import sys

import colossalai
import colossalai.utils as colo_utils
import torch
import torch.nn as nn
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn import LinearWarmupLR
from colossalai.registry import HOOKS
from colossalai.trainer import Trainer as ColoTrainer
from colossalai.trainer import hooks
from colossalai.trainer.hooks import BaseHook, LRSchedulerHook
# from colossalai.utils import colo_set_process_memory_fraction
from colossalai.utils import is_using_pp
from colossalai.utils.checkpointing import gather_pipeline_parallel_state_dict
from colossalai.utils.timer import MultiTimer
from colossalai.zero.init_ctx import ZeroInitContext
from solutions.antllm.antllm.data.dataset.glm_instruction_dataset import (
    GLMInstructionDataset
)
from solutions.antllm.antllm.data.dataset.instruction_dataset import (
    InstructionDataset
)
from solutions.antllm.antllm.models.glm.configuration_glm import *  # noqa
from solutions.antllm.antllm.models.glm.modeling_glm import *  # noqa
from solutions.antllm.antllm.models.glm.tokenization_glm import *  # noqa
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
from transformers.models.gpt2.configuration_gpt2 import *  # noqa
from transformers.models.gpt2.modeling_gpt2 import *  # noqa
from transformers.models.gpt2.tokenization_gpt2 import *  # noqa


def save_checkpoint(file,
                    epoch_or_step: int,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer = None,
                    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                    **kwargs):
    """Stores the checkpoint to disk. Saves all the training components' parameters or buffers, such as model, optimizer,
    lr_scheduler etc. into a checkpoint dictionary.

    Args:
        file: a file-like object (has to implement write and flush) or a string or os.PathLike object containing a
            file name.
        epoch (int): Epoch number (indicates how many epochs have you trained this model).
        model (:class:`torch.nn.Module`): Model to be saved.
        optimizer (Union[:class:`torch.optim.Optimizer`, :class:`colossalai.nn.optimizer`]): Optimizer to be saved.
        lr_scheduler (Union[:class:`torch.optim.lr_scheduler`, :class:`colossalai.nn.lr_scheduler`], optional):
            lr_scheduler to be saved, defaults to None.
        pickle_module: module used for pickling metadata and objects
        pickle_protocol: can be specified to override the default protocol
    """
    # ckpt container
    checkpoint = {"epoch_or_step": epoch_or_step}

    model_state = model.state_dict()
    if is_using_pp() and gpc.get_local_rank(ParallelMode.TENSOR) == 0:
        model_state = gather_pipeline_parallel_state_dict(model_state)

    if gpc.get_global_rank() == 0:
        checkpoint["model"] = model_state

        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()

        if lr_scheduler is not None:
            checkpoint['lr_scheduler'] = lr_scheduler.state_dict()

        torch.save(checkpoint, file, **kwargs)


@HOOKS.register_module
class HuggingfaceFormatSaveCheckpointHook(BaseHook):
    """Saves the model by interval in training process.

    Args:
       interval (int, optional): Number of epochs between saving the checkpoint, defaults to 1.
            if save_by_iter is True, this arg refers to the number of iters between saving.
       checkpoint_dir (str, optional): File name to save the checkpoint, defaults to None.
       model (torch.nn.Module, Optional): The model to save, defaults to None. When not passing,
            'trainer.engine.model' will be used. We encourage you to pass the model in it to avoid some
            unexpected bugs, especially when using **DDP**.
       save_by_iter (bool, optional): Whether saving the checkpoint by iter, default to False.
       priority (int, optional): Priority in the printing, hooks with small priority will be printed in front
            defaults to 10. If different hooks share same priority, the order of printing would
            depend on the hooks order in the hook list.
    """

    def __init__(self,
                 interval: int = 1,
                 checkpoint_dir: str = None,
                 model: torch.nn.Module = None,
                 save_by_iter: bool = False,
                 priority: int = 10):
        super().__init__(priority=priority)
        self.interval = interval
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.model = model
        self.save_by_iter = save_by_iter
        self.logger = get_dist_logger()

        # get lr scheduler from the LRSchedulerHook before train
        self._lr_scheduler = None

    def after_hook_is_attached(self, trainer):
        # get lr scheduler if exists
        for hook in trainer.hooks:
            if isinstance(hook, LRSchedulerHook):
                self._lr_scheduler = hook.lr_scheduler
                break
        self.model = self.model if self.model is not None else trainer.engine.model

    def after_train_iter(self, trainer, output, label, loss):
        """Saves the model after a training iter.
        """
        # save by interval
        if self.save_by_iter and trainer.cur_step % self.interval == 0:
            filename = os.path.join(
                self.checkpoint_dir, 'checkpoint-step-' + str(trainer.cur_step))
            save_checkpoint(filename, trainer.cur_step, self.model, optimizer=trainer.engine.optimizer,
                            lr_scheduler=self._lr_scheduler)
            self.logger.info(f'checkpoint for iteration {trainer.cur_step} is saved to {filename}',
                             ranks=[0])
        else:
            pass

    def after_train_epoch(self, trainer):
        """Saves the model after a training epoch.
        """
        # save by interval
        if trainer.cur_epoch % self.interval == 0:
            filename = os.path.join(
                self.checkpoint_dir, 'checkpoint-epoch-' + str(trainer.cur_epoch))
            save_checkpoint(filename, trainer.cur_epoch, self.model, optimizer=trainer.engine.optimizer,
                            lr_scheduler=self._lr_scheduler)
            self.logger.info(
                f'checkpoint for epoch {trainer.cur_epoch} is saved to {filename}', ranks=[0])


def calc_local_model_size(model: torch.nn.Module):
    numel_per_device = 0
    for p in model.parameters():
        numel_per_device += p.numel()
    return numel_per_device


class LMLossWrapper(torch.nn.Module):

    def forward(self, loss, **kwargs):
        return loss


VOCAB_SIZE = 50257


def main():
    parser = colossalai.get_default_parser()
    parser.add_argument('--from_torch', default=False, action='store_true')
    parser.add_argument('--use_dummy_dataset',
                        default=False, action='store_true')
    parser.add_argument('--lm_type',
                        default='causal',
                        choices=['causal', 'seq2seq'],
                        action='store',
                        help='type of language model, determining \
                                which automodel class to use')
    parser.add_argument('--output_dir', default=None,
                        type=str, action='store', help='')
    args = parser.parse_args()
    disable_existing_loggers()
    if args.from_torch:
        colossalai.launch_from_torch(config=args.config)
    else:
        colossalai.launch_from_slurm(
            config=args.config, host=args.host, port=29500, seed=42)
    logger = get_dist_logger()
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    global_rank = int(os.environ.get('RANK', 0))
    logger.info(f'world_size: {world_size}, global_rank: {global_rank}')

    train_data_path = None if args.use_dummy_dataset else gpc.config.TRAIN_DATA
    test_data_path = None if args.use_dummy_dataset else gpc.config.TEST_DATA
    logger.info('gpc.config: ' + str(gpc.config))

    checkpoint_dir = args.output_dir
    save_checkpoint_interval = gpc.config.SAVE_CHECKPOINT_INTERVAL
    save_checkpoint_by_iter = gpc.config.SAVE_CHECKPOINT_BY_ITER

    if args.lm_type == 'seq2seq':
        auto_model_class = AutoModelForSeq2SeqLM  # noqa
        dataset_class = GLMInstructionDataset
        auto_model_class = GLMForConditionalGeneration  # noqa
        auto_tokenizer_class = GLMChineseTokenizer  # noqa
        sys.path.insert(0, gpc.config.PRETRAINED_MODEL_NAME_OR_PATH)
    else:
        auto_model_class = AutoModelForCausalLM  # noqa
        dataset_class = InstructionDataset
        auto_tokenizer_class = AutoTokenizer

    logger.info('Build model', ranks=[0])
    use_pipeline = is_using_pp()
    # use_interleaved = hasattr(gpc.config.model, 'num_chunks')
    use_interleaved = False
    use_zero3 = hasattr(gpc.config, 'zero')
    ctx = contextlib.nullcontext()
    if use_zero3:
        ctx = ZeroInitContext(target_device=torch.cuda.current_device(),
                              shard_strategy=gpc.config.zero.model_config.shard_strategy,
                              shard_param=True)
    with ctx:
        model = auto_model_class.from_pretrained(  # noqa
            gpc.config.PRETRAINED_MODEL_NAME_OR_PATH, trust_remote_code=True)
        tokenizer = auto_tokenizer_class.from_pretrained(  # noqa
            gpc.config.PRETRAINED_MODEL_NAME_OR_PATH, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.info(f'Build data loader from path {train_data_path}', ranks=[0])
    train_ds = dataset_class(data_path=train_data_path,
                             tokenizer=tokenizer,
                             max_length=gpc.config.SEQ_LEN,
                             max_input_length=gpc.config.MAX_INPUT_LENGTH,
                             max_output_length=gpc.config.MAX_OUTPUT_LENGTH,
                             world_size=world_size,
                             rank=global_rank,
                             return_colossal_format=True,
                             )
    test_ds = dataset_class(data_path=test_data_path,
                            tokenizer=tokenizer,
                            max_length=gpc.config.SEQ_LEN,
                            max_input_length=gpc.config.MAX_INPUT_LENGTH,
                            max_output_length=gpc.config.MAX_OUTPUT_LENGTH,
                            world_size=world_size,
                            rank=global_rank,
                            return_colossal_format=True,
                            )
    train_dataloader = colo_utils.get_dataloader(train_ds,
                                                 seed=42,
                                                 add_sampler=False,
                                                 batch_size=gpc.config.BATCH_SIZE,
                                                 pin_memory=True,
                                                 shuffle=True,
                                                 drop_last=True)
    test_dataloader = colo_utils.get_dataloader(test_ds,
                                                seed=42,
                                                add_sampler=False,
                                                batch_size=gpc.config.BATCH_SIZE,
                                                pin_memory=True,
                                                shuffle=False,
                                                drop_last=False)
    if use_pipeline and use_interleaved and not isinstance(model, nn.ModuleList):
        model = nn.ModuleList([model])

    if use_zero3:
        numel = ctx.model_numel_tensor.item()
    else:
        numel = calc_local_model_size(model)

    tflop = numel * gpc.config.BATCH_SIZE * gpc.config.SEQ_LEN \
        * gpc.get_world_size(ParallelMode.MODEL) * gpc.get_world_size(ParallelMode.DATA) * 8 / (1024 ** 4)

    criterion = getattr(gpc.config, 'loss_fn', None)
    if criterion is not None:
        criterion = criterion.type()
    else:
        criterion = LMLossWrapper()
    logger.info('Build optimizer', ranks=[0])
    optimizer = gpc.config.optimizer.pop('type')(
        model.parameters(), **gpc.config.optimizer)
    steps_per_epoch = len(train_ds) // gpc.config.BATCH_SIZE

    lr_scheduler = LinearWarmupLR(
        optimizer, total_steps=gpc.config.NUM_EPOCHS * steps_per_epoch, warmup_steps=5)
    engine, train_dataloader, test_dataloader, lr_scheduler = colossalai.initialize(model,
                                                                                    optimizer,
                                                                                    criterion,
                                                                                    train_dataloader=train_dataloader,
                                                                                    test_dataloader=test_dataloader,
                                                                                    lr_scheduler=lr_scheduler)
    global_batch_size = gpc.config.BATCH_SIZE * \
        gpc.get_world_size(ParallelMode.DATA) * \
        getattr(gpc.config, "gradient_accumulation", 1)
    logger.info(
        f'Init done, global batch size = {global_batch_size}', ranks=[0])
    timier = MultiTimer()
    trainer = ColoTrainer(engine=engine, logger=logger, timer=timier)
    hook_list = [
        hooks.LossHook(),
        hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=False),
        hooks.LogMetricByEpochHook(logger),
        hooks.ThroughputHook(ignored_steps=10, tflop_per_step=tflop),
        hooks.LogMetricByStepHook(),
        hooks.LogMemoryByEpochHook(logger),
        HuggingfaceFormatSaveCheckpointHook(model=engine.model,
                                            checkpoint_dir=checkpoint_dir,
                                            interval=save_checkpoint_interval,
                                            save_by_iter=save_checkpoint_by_iter)
    ]
    trainer.fit(train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                epochs=gpc.config.NUM_EPOCHS,
                test_interval=1,
                hooks=hook_list,
                display_progress=True,
                return_output_label=False)


if __name__ == '__main__':
    main()
