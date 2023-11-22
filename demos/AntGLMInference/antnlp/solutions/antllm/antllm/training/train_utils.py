from functools import partial

import deepspeed
import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from solutions.antllm.antllm.nn.parallel.distributed import \
    DistributedDataParallel as LocalDDP
from solutions.antllm.antllm.nn.parallel.distributed import \
    PyTorchDistributedDataParallel as TorchDDP
from solutions.antllm.antllm.training.learning_rates import AnnealingLR
from solutions.antllm.antllm.training.optimizer.fp16 import (DynamicLossScaler,
                                                             FP16_Module,
                                                             FP16_Optimizer)
from solutions.antllm.antllm.utils import mpu
from solutions.antllm.antllm.utils.glm_utils import (get_checkpoint_iteration,
                                                     get_checkpoint_name,
                                                     is_load_from_modelhub,
                                                     parse_modelhub_path)
from solutions.antllm.antllm.utils.logging.logger import log_dist


# return model_name for print logs
def get_model_name(args):
    return "AntGPT" if args.is_gpt else "AntGLM"


def load_pretrained(model, checkpoint_path, args, task_tokens=None):
    load_dir, tag, release, success = get_checkpoint_iteration(checkpoint_path)
    checkpoint_name = get_checkpoint_name(load_dir, tag, release)
    if mpu.get_data_parallel_rank() == 0:
        print(
            "global rank {} is loading pretrained model {}".format(
                torch.distributed.get_rank(), checkpoint_name
            )
        )
    # Load the checkpoint.
    sd = torch.load(checkpoint_name, map_location="cpu")
    if args.deepspeed:
        model = model.module
    if isinstance(model, TorchDDP):
        model = model.module
    if isinstance(model, FP16_Module):
        model = model.module
    if hasattr(model, "model"):
        model = model.model

    # Model.
    def extend_embedding_weights(state_weights, model_weights):
        original_length = state_weights.shape[0]
        assert original_length <= args.max_position_embeddings + 1
        new_weights = model_weights.clone()
        new_weights[:original_length] = state_weights
        return new_weights

    if args.block_lm:
        if "transformer.block_position_embeddings.weight" in sd["module"]:
            position_weights = sd["module"]["transformer.position_embeddings.weight"]
            if args.max_position_embeddings + 1 > position_weights.shape[0]:
                sd["module"][
                    "transformer.position_embeddings.weight"
                ] = extend_embedding_weights(
                    position_weights,
                    model.state_dict()["transformer.position_embeddings.weight"].data,
                )
                log_dist(
                    f"Extend position embedding to {args.max_position_embeddings + 1}"
                )
        if "transformer.block_position_embeddings.weight" in sd["module"]:
            block_position_weights = sd["module"][
                "transformer.block_position_embeddings.weight"
            ]
            if args.max_position_embeddings + 1 > block_position_weights.shape[0]:
                sd["module"][
                    "transformer.block_position_embeddings.weight"
                ] = extend_embedding_weights(
                    block_position_weights,
                    model.state_dict()[
                        "transformer.block_position_embeddings.weight"
                    ].data,
                )
                log_dist(
                    f"Extend block position embedding to {args.max_position_embeddings + 1}"
                )
    missing_keys, unexpected_keys = model.load_state_dict(sd["module"], strict=False)
    if missing_keys or unexpected_keys:
        log_dist(f"Missing keys {missing_keys}, unexpected keys {unexpected_keys}")
    if args.continuous_prompt and args.prompt_init:
        model.prompt_spell.init_embedding(
            model.word_embeddings.weight.data, task_tokens
        )


def get_model(
        args, model_type=None, multi_token=True, num_labels=None, spell_length=None
):
    """Build the model."""
    log_dist("building PreTrain model ...")
    output_predict, parallel_output = True, True
    if (
            model_type == "multiple_choice" or model_type == "classification"
    ) and not args.cloze_eval:
        output_predict = False
    if model_type is not None:
        parallel_output = False
    if spell_length is not None:
        log_dist(f"Continuous spell length {spell_length}")

    load_from_hub = (
        is_load_from_modelhub(args.load_pretrained)
        if args.load_pretrained
        else False
    )
    has_load = False
    if load_from_hub:
        from alps.pytorch.modelhub.hub_layer import TorchHubLayer

        name, version, dt = parse_modelhub_path(args.load_pretrained)
        model = TorchHubLayer.restore_from_modelhub(
            name=name, version=version, dt=dt
        )
        print(f"load model from modelhub: {args.load_pretrained}")
        has_load = True
    else:
        from antllm.models.glm.modeling_glm import GLMConfig, GLMModel

        config = GLMConfig(
            num_layers=args.num_layers,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_attention_heads,
            num_key_value_heads=args.num_key_value_heads,
            embedding_dropout_prob=args.hidden_dropout,
            attention_dropout_prob=args.attention_dropout,
            output_dropout_prob=args.hidden_dropout,
            max_sequence_length=args.max_position_embeddings,
            max_memory_length=args.mem_length,
            checkpoint_activations=args.checkpoint_activations,
            checkpoint_num_layers=args.checkpoint_num_layers,
            parallel_output=parallel_output,
            relative_encoding=args.transformer_xl,
            block_position_encoding=args.block_lm and not args.masked_lm,
            output_predict=output_predict,
            spell_length=spell_length,
            spell_func=args.prompt_func,
            attention_scale=args.attention_scale,
        )
        if args.atorch_accelerate and args.atorch_meta_init:
            from atorch.utils.meta_model_utils import \
                init_empty_weights_with_disk_offload

            context = init_empty_weights_with_disk_offload(ignore_tie_weights=False)
        else:
            from contextlib import nullcontext

            context = nullcontext()
        with context:
            model = GLMModel(config)
            # atorch load module sd on rank0 before auto accelerate
            if args.load and args.atorch_accelerate:
                if torch.distributed.get_rank() == 0:
                    load_pretrained(model, args.load, args)
                torch.distributed.barrier()

    if mpu.get_data_parallel_rank() == 0:
        print(
            " > number of parameters on model parallel rank {}: {}".format(
                mpu.get_model_parallel_rank(),
                sum([p.nelement() for p in model.parameters()]),
            ),
            flush=True,
        )

    if args.atorch_accelerate:
        # no need to handle device or wrap other things
        return model, has_load

    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if args.fp16:
        model.half()

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)

    # Wrap model for distributed training.
    if not args.deepspeed and (args.train_iters or args.epochs):
        if args.DDP_impl == "torch":
            i = torch.cuda.current_device()
            model = TorchDDP(
                model,
                device_ids=[i],
                output_device=i,
                process_group=mpu.get_data_parallel_group(),
            )
        elif args.DDP_impl == "local":
            model = LocalDDP(model)
        else:
            log_dist("Skip DDP model")
    return model, has_load


def glm_get_params_for_weight_decay_optimization(module):
    weight_decay_params = {"params": []}
    no_weight_decay_params = {"params": [], "weight_decay": 0.0}
    for module_ in module.modules():
        if isinstance(module_, (mpu.LayerNorm, torch.nn.LayerNorm)):
            no_weight_decay_params["params"].extend(
                [
                    p
                    for p in list(module_._parameters.values())
                    if p is not None and p.requires_grad
                ]
            )
        else:
            weight_decay_params["params"].extend(
                [
                    p
                    for n, p in list(module_._parameters.items())
                    if p is not None and p.requires_grad and n != "bias"
                ]
            )
            no_weight_decay_params["params"].extend(
                [
                    p
                    for n, p in list(module_._parameters.items())
                    if p is not None and p.requires_grad and n == "bias"
                ]
            )

    return weight_decay_params, no_weight_decay_params


def get_optimizer_param_groups(model):
    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, (LocalDDP, TorchDDP, FP16_Module)):
        model = model.module
    param_groups = glm_get_params_for_weight_decay_optimization(model)

    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        # print('## param_group', len(param_group['params']))
        for param in param_group["params"]:
            if not hasattr(param, "model_parallel"):
                param.model_parallel = False

    return param_groups


def get_optimizer(param_groups, args):
    """Set up the optimizer."""
    if args.cpu_optimizer:
        # Apex FusedAdam uses decoupled weight decay so use the same here
        if args.cpu_torch_adam:
            cpu_adam_optimizer = torch.optim.AdamW
        else:
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            cpu_adam_optimizer = DeepSpeedCPUAdam
        optimizer = cpu_adam_optimizer(
            param_groups, lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        # Use FusedAdam.
        from apex.optimizers import FusedAdam as Adam

        if args.optimizer == "adam":
            optimizer = Adam(
                param_groups,
                lr=args.lr,
                weight_decay=args.weight_decay,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_eps,
            )
        elif args.optimizer == "adafactor":
            from transformers import Adafactor

            optimizer = Adafactor(
                param_groups, lr=args.lr, relative_step=False, warmup_init=False
            )
        else:
            raise NotImplementedError

    log_dist(f"Optimizer = {optimizer.__class__.__name__}")
    if hasattr(args, "deepspeed") and args.deepspeed:
        raise NotImplementedError
        # fp16 wrapper is not required for DeepSpeed.
        # return optimizer

    # Wrap into fp16 optimizer.
    if args.fp16:
        optimizer = FP16_Optimizer(
            optimizer,
            static_loss_scale=args.loss_scale,
            dynamic_loss_scale=args.dynamic_loss_scale,
            dynamic_loss_args={
                "scale_window": args.loss_scale_window,
                "min_scale": args.min_scale,
                "delayed_shift": args.hysteresis,
            },
        )

    return optimizer


def get_learning_rate_scheduler(optimizer, args):
    """Build the learning rate scheduler."""

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters
    if args.finetune:
        num_iters = num_iters // args.gradient_accumulation_steps
    num_iters = max(1, num_iters)
    init_step = -1
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(
        optimizer,
        start_lr=args.lr,
        warmup_iter=warmup_iter,
        num_iters=num_iters - warmup_iter,
        decay_style=args.lr_decay_style,
        last_iter=init_step,
        decay_ratio=args.lr_decay_ratio,
    )

    return lr_scheduler


def atorch_accelerate(args, model):
    from antllm.models.glm.modeling_glm import GLMBlock
    from atorch.auto import auto_accelerate
    from atorch.utils.version import torch_version

    assert torch_version() >= (2, 0, 0), "use pt2.0 for use orig param if fsdp"
    fsdp_config = {
        "sync_module_states": True,
        "use_orig_params": True,
        "limit_all_gathers": True,
    }
    fsdp_config["atorch_wrap_cls"] = {
        GLMBlock,
    }
    p_mode = ([("data", torch.distributed.get_world_size())], None)
    strategy = [
        ("parallel_mode", p_mode),
        "module_replace",
        ("fsdp", fsdp_config),
        "amp_native",
        ("checkpoint", (GLMBlock,)),
    ]
    log_dist(f"Manually loaded auto acc strategy: {strategy}")

    def my_loss_func(logits, labels, loss_mask):
        losses = torch.nn.CrossEntropyLoss(reduction="none")(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )
        loss = torch.sum(losses * loss_mask.view(-1))
        if loss_mask.sum().item() > 0:
            loss = loss / loss_mask.sum()
        return loss

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

    status, result, best_strategy = auto_accelerate(
        model,
        torch.optim.AdamW,
        loss_func=my_loss_func,
        optim_args={
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "eps": args.adam_eps,
            "betas": (args.adam_beta1, args.adam_beta2),
        },
        optim_param_func=partial(optim_param_func, args=args),
        load_strategy=strategy,
        ignore_dryrun_on_load_strategy=True,
    )
    assert (
        status
    ), f"auto_accelerate failed. status: {status}, result: {result}, best_strategy: {best_strategy}"
    log_dist(f"Best strategy is: {best_strategy}")

    model = result.model
    optimizer = result.optim
    log_dist(f"atorch use optimizer: {optimizer}")
    model.atorch_loss_func = result.loss_func

    return model, optimizer


def setup_model_and_optimizer(
        args, model_type=None, multi_token=True, num_labels=None, spell_length=None
):
    """Setup model and optimizer."""

    model, has_load = get_model(
        args,
        model_type=model_type,
        multi_token=multi_token,
        num_labels=num_labels,
        spell_length=spell_length,
    )
    if not args.atorch_accelerate:
        param_groups = get_optimizer_param_groups(model)

    if (
            args.train_data is not None
            or args.data_dir is not None
            and (args.epochs > 0 or args.train_iters > 0)
    ):
        if args.atorch_accelerate:
            model, optimizer = atorch_accelerate(args, model)
        elif args.deepspeed:
            log_dist("DeepSpeed is enabled.")

            model, optimizer, _, _ = deepspeed.initialize(
                model=model,
                model_parameters=param_groups,
                args=args,
                mpu=mpu,
                dist_init_required=False,
            )
        else:
            optimizer = get_optimizer(param_groups, args)
        lr_scheduler = get_learning_rate_scheduler(optimizer, args)
    else:
        optimizer, lr_scheduler = None, None

    return model, optimizer, lr_scheduler, has_load


def backward_step(optimizer, model, lm_loss, args, timers):
    """Backward step."""

    # Total loss.
    loss = lm_loss

    # Backward pass.
    if args.deepspeed:
        model.backward(loss)
    else:
        # optimizer.zero_grad()
        if args.fp16:
            optimizer.backward(loss, update_master_grads=False)
        else:
            loss.backward()

    if args.deepspeed or args.DDP_impl == "torch" or args.atorch_accelerate:
        # DeepSpeed backward propagation already addressed all reduce communication.
        # Reset the timer to avoid breaking timer logs below.
        timers("allreduce").reset()
    else:
        timers("allreduce").start()
        model.allreduce_params(reduce_after=False, fp32_allreduce=args.fp32_allreduce)
        timers("allreduce").stop()

    # Update master gradients.
    if not args.deepspeed:
        if args.fp16:
            optimizer.update_master_grads()

        # Clipping gradients helps prevent the exploding gradient.
        if args.clip_grad > 0:
            if args.atorch_accelerate:
                optimizer.unscale_()
                if isinstance(model, FSDP):
                    model.clip_grad_norm_(args.clip_grad)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            elif not args.fp16:
                mpu.clip_grad_norm(model.parameters(), args.clip_grad)
            else:
                optimizer.clip_master_grads(args.clip_grad)

    return lm_loss


def see_memory_usage(message, force=False):
    if not force:
        return
    dist.barrier()
    if dist.get_rank() == 0:
        print(message)
        print(
            "Memory Allocated ",
            torch.cuda.memory_allocated() / (1024 * 1024 * 1024),
            "GigaBytes",
        )
        print(
            "Max Memory Allocated ",
            torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024),
            "GigaBytes",
        )
        print(
            "Cache Allocated ",
            torch.cuda.memory_cached() / (1024 * 1024 * 1024),
            "GigaBytes",
        )
        print(
            "Max cache Allocated ",
            torch.cuda.max_memory_cached() / (1024 * 1024 * 1024),
            "GigaBytes",
        )
        print(" ")
        # input("Press Any Key To Continue ..")


def train_step(
        data_iterator,
        model,
        optimizer,
        lr_scheduler,
        args,
        timers,
        forward_step_func,
        mems=None,
        single_step=False,
):
    """Single training step."""
    lm_loss_total, count, gpt_loss, bert_loss, sent_loss = 0.0, 0, 0.0, 0.0, 0.0
    gpt_cnt, bert_cnt, sent_cnt = 0, 0, 0
    mems = [] if mems is None else mems
    timers("train_step").start()
    if not args.deepspeed:
        optimizer.zero_grad()
    while True:
        skipped_iter, complete = 0, False
        # Forward model for one step.
        timers("forward").start()
        lm_loss, mems, mode = forward_step_func(
            data_iterator, model, args, timers, mems
        )
        timers("forward").stop()

        timers("middle1").start()
        # log_dist("Forward step")
        if not args.deepspeed:
            lm_loss /= args.gradient_accumulation_steps

        loss_tensors = torch.zeros([3], device=lm_loss.device, dtype=lm_loss.dtype)
        loss_cnt_tensors = torch.zeros([3], device=lm_loss.device, dtype=torch.long)
        idx = -1
        if mode == "gpt":
            idx = 2
        elif mode == "sentence":
            idx = 1
        elif mode == "bert":
            idx = 0
        loss_tensors[idx] = lm_loss.item()
        loss_cnt_tensors[idx] = 1

        torch.distributed.all_reduce(loss_tensors, group=mpu.get_data_parallel_group())
        torch.distributed.all_reduce(
            loss_cnt_tensors, group=mpu.get_data_parallel_group()
        )
        reduced_loss = loss_tensors.sum()

        reduced_loss = reduced_loss / (args.world_size / args.model_parallel_size)

        timers("middle1").stop()

        timers("middle2").start()

        if not DynamicLossScaler._has_inf_or_nan(reduced_loss):
            lm_loss_total += reduced_loss

            bert_loss += loss_tensors[0]
            sent_loss += loss_tensors[1]
            gpt_loss += loss_tensors[2]

            bert_cnt += loss_cnt_tensors[0]
            sent_cnt += loss_cnt_tensors[1]
            gpt_cnt += loss_cnt_tensors[2]

            count += 1

            # Calculate gradients, reduce across processes, and clip.
            timers("backward").start()
            backward_step(optimizer, model, lm_loss, args, timers)
            timers("backward").stop()
            # log_dist("Backward step")
            # Update parameters.
            timers("optimizer").start()
            if args.deepspeed:
                if model.is_gradient_accumulation_boundary():
                    model.step()
                    complete = True
                    if not (args.fp16 and optimizer.overflow):
                        lr_scheduler.step()
                    else:
                        skipped_iter = 1
                else:
                    model.step()
            else:
                if count == args.gradient_accumulation_steps:
                    optimizer.step()
                    complete = True
                    # Update learning rate.
                    if not (args.fp16 and optimizer.overflow) or (
                            args.atorch_accelerate and optimizer.step_was_skipped
                    ):
                        lr_scheduler.step()
                    else:
                        skipped_iter = 1
            # log_dist("Optimizer step")
            timers("optimizer").stop()
            timers("middle2").stop()
            if complete:
                break
        else:
            print("Found NaN loss, skip backward")
            del lm_loss, reduced_loss
            timers("middle2").stop()
            mems = []
        if single_step:
            break
    if args.deepspeed:
        lm_loss_total = lm_loss_total / count
    bert_loss = bert_loss / bert_cnt if bert_cnt > 0 else bert_loss
    sent_loss = sent_loss / sent_cnt if sent_cnt > 0 else sent_loss
    gpt_loss = gpt_loss / gpt_cnt if gpt_cnt > 0 else gpt_loss
    timers("train_step").stop()
    return lm_loss_total, skipped_iter, mems, (bert_loss, sent_loss, gpt_loss)
