import torch
from numpy import random
from dataclasses import dataclass


@dataclass
class LongGLMMemCache:
    """
    Class with LongLlama's memory cache

    Args:
        key (`torch.FloatTensor` of shape `(batch_size, mem_length, head_nums, embed_size_per_head)`)
        value (`torch.FloatTensor` of shape `(batch_size, mem_length, head_nums, embed_size_per_head)`)
        masks (`torch.FloatTensor` of shape `(batch_size, 1, mem_length, 1)`)
            For masking out parts of memory
    """

    key: torch.FloatTensor
    value: torch.FloatTensor
    masks: torch.FloatTensor


def mem_apply_update(
    prev_external_mem_cache: LongGLMMemCache, new_mem_content: LongGLMMemCache
):
    def update_one(prev, new, dim=1):
        if len(prev.shape) != len(new.shape):
            raise ValueError(f"Memory cache content should be consistent in shape got {prev.shape} {new.shape}")

        return torch.concat([prev, new], dim=dim)

    insert_size = new_mem_content.key.shape[1]

    assert new_mem_content.key.shape[1] == new_mem_content.value.shape[1]
    if new_mem_content.masks.shape[-2] != insert_size:
        raise ValueError(f"Inconsistent mem_length in new_mem_content")

    return LongGLMMemCache(
        key=update_one(prev_external_mem_cache.key, new_mem_content.key),
        value=update_one(prev_external_mem_cache.value, new_mem_content.value),
        masks=update_one(prev_external_mem_cache.masks, new_mem_content.masks, dim=-2),
    )


def generate_prompt_keypass(n_garbage: int, seed: int = None):
    """Generates a text file and inserts an execute line at a random position."""
    if seed is not None:
        rnd_state = random.get_state()
        random.seed(seed)
    n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "在下文的大量无关紧要的文字中隐藏着一个非常重要的信息，请找到并记住它们，后面将使用到这个信息。"
    garbage = "草是绿色的。天空是蓝色的。太阳是黄色的。我们走。我们离开又回来了。"
    garbage_inf = "".join([garbage] * 5000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = f"以下是本段文本的重要信息: “通行密码是'{pass_key}'，这是非常重要的信息，请记住'{pass_key}'是通行密码。”"
    information_line = "\n".join([information_line] * 3)
    final_question = "请问通行密码是多少？"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question,
    ]
    if seed is not None:
        random.set_state(rnd_state)
    return "\n".join(lines), str(pass_key)