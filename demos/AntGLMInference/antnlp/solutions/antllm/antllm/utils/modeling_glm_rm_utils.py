from typing import Optional, Union, Callable

import loralib as lora
import torch
import torch.nn as nn
import torch.nn.functional as F
from solutions.antllm.antllm.utils.modeling_glm_ppo_utils import hf_get_glm_embeddings


def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html
    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    log_ratio = log_probs - log_probs_base
    approx_kl = (log_ratio.exp() - 1) - log_ratio
    if action_mask is not None:
        approx_kl = masked_mean(approx_kl, action_mask, dim=1)
        return approx_kl
    approx_kl = approx_kl.mean(dim=1)
    return approx_kl


def compute_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if kl_coef <= 0.0:
        return r
    kl = compute_approx_kl(log_probs, log_probs_base, action_mask=action_mask)
    reward = r - kl_coef * kl
    return reward


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    tensor = tensor * mask
    tensor = tensor.sum(dim=dim)
    mask_sum = mask.sum(dim=dim)
    mean = tensor / (mask_sum + 1e-8)
    return mean


def masked_normalize(
    tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8
) -> torch.Tensor:
    tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim)
    mean_centered = tensor - mean
    var = masked_mean(mean_centered**2, mask, dim=dim)
    return mean_centered * var.clamp(min=eps).rsqrt()


def normalize(tensor: torch.Tensor, dim: int = 0, eps: float = 1e-8) -> torch.Tensor:
    mean = tensor.mean(dim)
    mean_centered = tensor - mean
    var = (mean_centered**2).mean(dim)
    norm = mean_centered * var.clamp(min=eps).rsqrt()
    return norm


def convert_to_lora(
    model: nn.Module,
    input_size: int,
    output_size: int,
    lora_rank: int = 16,
    lora_alpha: int = 1,
    lora_dropout: float = 0.0,
    fan_in_fan_out: bool = False,
    merge_weights: bool = True,
):
    if lora_rank > min(input_size, output_size):
        raise ValueError(
            f"LoRA rank {lora_rank} must be less or equal than {min(input_size, output_size)}"
        )

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._modules[name] = lora.Linear(
                input_size,
                output_size,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                fan_in_fan_out=fan_in_fan_out,
                merge_weights=merge_weights,
            )


def build_glm_inputs_from_sample(
    prompt: str,
    response: str,
    tokenizer: Callable,
    max_length: int,
    max_input_length: int,
    mask="[gMASK]",
    truncation_side="left",
    dynamic_padding=False,
    eos_token="<|endoftext|>",
    rotary_type="none"
):

    sop_id = tokenizer.sop_token_id
    # eop_id = tokenizer.eop_token_id
    eop_id = tokenizer.convert_tokens_to_ids(eos_token)
    cls_id = tokenizer.cls_token_id
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.convert_tokens_to_ids(mask)

    assert tokenizer.convert_tokens_to_ids("<|startofpiece|>") == sop_id
    # assert tokenizer.convert_tokens_to_ids("<|endofpiece|>") == eop_id

    tokenizer_outs = tokenizer(
        prompt,
        padding=False,
        add_special_tokens=False,
        return_attention_mask=True,
    )

    input_ids = tokenizer_outs["input_ids"]

    # 截断 prompt
    if truncation_side == "right":
        if len(input_ids) > max_input_length - 2:
            input_ids = input_ids[: max_input_length - 2]
    else:
        if len(input_ids) > max_input_length - 2:
            input_ids = input_ids[len(input_ids) - max_input_length + 2:]

    # 添加 cls 和 mask
    input_ids = [cls_id] + input_ids + [mask_id]

    sep = len(input_ids)
    mask_pos = input_ids.index(mask_id)
    position_ids = list(range(len(input_ids)))
    block_position_ids = [0] * len(input_ids)

    # sop & eop token
    max_output_length = max_length - max_input_length - 2

    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

    # response 默认右截断
    if len(response_ids) > max_output_length:
        response_ids = response_ids[:max_output_length]

    response_ids = input_ids + [sop_id] + response_ids + [eop_id]
    if len(response_ids) < max_length and not dynamic_padding:
        response_pad_length = max_length - len(response_ids)
        response_ids += [pad_id] * response_pad_length

    if "1d" in rotary_type:
        response_position_ids = list(range(len(response_ids)))
    else:
        # position_ids在mask之后全部补mask_pos
        response_position_ids = position_ids + \
            [mask_pos] * (len(response_ids) - len(position_ids))
    response_position_ids = torch.tensor(response_position_ids, dtype=torch.long)

    # block_position_ids在mask之后补1 2 3 4 5..
    response_block_position_ids = block_position_ids + \
        list(range(1, len(response_ids) - len(block_position_ids) + 1))
    response_block_position_ids = torch.tensor(response_block_position_ids, dtype=torch.long)

    # attention_mask = build_mask(max_length, sep)
    assert len(response_ids) == len(response_position_ids) == \
        len(response_block_position_ids)
    return {
        "input_ids": torch.Tensor(response_ids).long(),
        "attention_mask": torch.Tensor([sep]).long(),
        "position_ids": torch.stack((response_position_ids, response_block_position_ids), dim=0)
    }


def freeze_model(model, num_layers_unfrozen=2):
    # freeze layer
    hidden_layers = model.glm.transformer.layers
    if num_layers_unfrozen == 0:
        hidden_layers_to_freeze = list(hidden_layers)
    elif num_layers_unfrozen > 0:
        hidden_layers_to_freeze = list(hidden_layers)[:-num_layers_unfrozen]
    else:
        hidden_layers_to_freeze = []

    embeddings_to_freeze = hf_get_glm_embeddings(model)
    layers = hidden_layers_to_freeze + embeddings_to_freeze if hidden_layers_to_freeze else []
    for layer in layers:
        layer.requires_grad_(False)