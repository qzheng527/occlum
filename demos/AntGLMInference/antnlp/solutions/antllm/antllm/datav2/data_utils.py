#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"

import os
import random

import numpy as np
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from solutions.antllm.antllm.datav2.samplers import (DistributedBatchSampler,
                                                     RandomSampler)
from solutions.antllm.antllm.models.glm.tokenization_glm import GLMTokenizer
from solutions.antllm.antllm.utils.dist_utils import (
    get_data_parallel_rank, get_data_parallel_world_size)
from solutions.antllm.antllm.utils.utils import contains_any

TOKENIZER_MAP = {
    "glm": GLMTokenizer
}

SENTENCE_END = [".", "?", "!", ";", ":", "\n", "。", "？", "！", "；", "："]


def truncate_left(tokens, max_len, keep_sentence=True, strict=True, tolerance=1, tokenizer=None):
    """truncate a token list from left

    Args:
        tokens (list): the token list
        max_len (int): the maximum length of the truncated token list,
                       if strict is False, the maximum length could be max_len + tolerance
        keep_sentence (bool, optional): whether to truncate at sentence end. Defaults to True.
        strict (bool, optional): if strict the truncated token list must shorter than maximum length. Defaults to True.
        tolerance (int, optional): the length allowed to be larger or smaller than max_len. Defaults to 1.

    Returns:
        list: the truncated token list
    """
    if len(tokens) <= max_len:
        return tokens

    strip_num = len(tokens) - max_len

    if not keep_sentence:
        return tokens[strip_num:]

    right_sent_end_idx, left_sent_end_idx = strip_num - 1, strip_num - 1
    move_cnt = 0
    left_sent_end = False
    right_sent_end = False
    while left_sent_end_idx >= 0 and move_cnt < tolerance:
        if contains_sentence_end(tokens[left_sent_end_idx], tokenizer):
            left_sent_end = True
            break
        left_sent_end_idx -= 1
        move_cnt += 1

    move_cnt = 0
    while right_sent_end_idx < len(tokens) and move_cnt < tolerance:
        if contains_sentence_end(tokens[right_sent_end_idx], tokenizer):
            right_sent_end = True
            break
        right_sent_end_idx += 1
        move_cnt += 1

    if strict or (right_sent_end and not left_sent_end):
        return tokens[right_sent_end_idx + 1:]
    else:
        return tokens[left_sent_end_idx + 1:]


def truncate_right(tokens, max_len, keep_sentence=True, strict=True, tolerance=1, tokenizer=None):
    """truncate a token list from right

    Args:
        tokens (list): the token list
        max_len (int): the maximum length of the truncated token list,
                       if strict is False, the maximum length could be max_len + tolerance
        keep_sentence (bool, optional): whether to truncate at sentence end. Defaults to True.
        strict (bool, optional): if strict the truncated token list must shorter than maximum length. Defaults to True.
        tolerance (int, optional): the length allowed to be larger or smaller than max_len. Defaults to 1.

    Returns:
        list: the truncated token list
    """
    if len(tokens) <= max_len:
        return tokens

    strip_num = len(tokens) - max_len

    if not keep_sentence:
        return tokens[:-strip_num]

    right_sent_end_idx, left_sent_end_idx = len(
        tokens) - strip_num, len(tokens) - strip_num
    move_cnt = 0
    left_sent_end = False
    right_sent_end = False
    while left_sent_end_idx >= 0 and move_cnt < tolerance:
        if contains_sentence_end(tokens[left_sent_end_idx], tokenizer):
            left_sent_end = True
            break
        left_sent_end_idx -= 1
        move_cnt += 1

    move_cnt = 0
    while right_sent_end_idx < len(tokens) and move_cnt < tolerance:
        if contains_sentence_end(tokens[right_sent_end_idx], tokenizer):
            right_sent_end = True
            break
        right_sent_end_idx += 1
        move_cnt += 1

    if strict or (left_sent_end and not right_sent_end):
        return tokens[:left_sent_end_idx + 1]
    else:
        return tokens[:right_sent_end_idx + 1]


def truncate_doc(tokens, max_len, keep_sentence=True, tolerance=1, left_truncate_prob=0.0, rng=None, tokenizer=None):
    """truncate a token list

    Args:
        tokens (list): the token list
        max_len (int): the maximum length of the truncated token list,
                       if strict is False, the maximum length could be max_len + tolerance
        keep_sentence (bool, optional): whether to truncate at sentence end. Defaults to True.
        tolerance (int, optional): the length allowed to be larger or smaller than max_len. Defaults to 1.
        left_truncate_prob (float, optional): the probability to truncate from left. Defaults to 0.0.
        rng (random.Random, optional): the random object. Defaults to None.

    Returns:
        list: the truncated token list
    """
    if len(tokens) <= max_len:
        return tokens

    if rng is None:
        rng = random.Random()

    if rng.random() < left_truncate_prob:
        tokens = truncate_left(
            tokens, max_len, keep_sentence, False, tolerance, tokenizer)
    tokens = truncate_right(tokens, max_len, keep_sentence, True, tolerance, tokenizer)
    return tokens


class Span:
    def __init__(self, start, length, content, tokens=None) -> None:
        self.start = start
        self.length = length
        self.content = content
        self.tokens = tokens


def prepare_tokenizer(tokenizer=None, tokenizer_model=None):
    if isinstance(tokenizer, str):
        assert os.path.exists(
            tokenizer_model), f"cannot find tokenizer model {tokenizer_model}"
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        except ValueError:
            assert tokenizer in TOKENIZER_MAP, f"Cannot find tokenizer {tokenizer}"
            tokenizer = TOKENIZER_MAP[tokenizer].from_pretrained(tokenizer_model)
        return tokenizer
    elif isinstance(tokenizer, PreTrainedTokenizer):
        return tokenizer
    raise ValueError(
        "the tokenizer should be a PreTrainedTokenizer instance \
        or a tokenizer name which can be used to initialize a tokenizer.")


def resolve_path(path, base_dir):
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def contains_sentence_end(token, tokenizer=None):
    if isinstance(token, (int, np.integer)):
        assert tokenizer is not None
        token = tokenizer.convert_ids_to_tokens([token])
    return contains_any(token, SENTENCE_END)


def build_sampler(
        dataset,
        batch_size,
        num_iters,
        gradient_accumulation_steps=1,
        scatter_num=None,
        shuffle=False):
    # 将load同一个scatter的卡放到一个group里，统一采样然后分配，保证不同的卡拿到的是不同的数据
    if scatter_num:
        rank = get_data_parallel_rank() // scatter_num
        world_size = get_data_parallel_world_size() // scatter_num
        batch_size = batch_size // scatter_num

    distributed = world_size > 1

    if shuffle:
        sampler = RandomSampler(
            dataset,
            replacement=True,
            num_samples=batch_size * num_iters * gradient_accumulation_steps
        )
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    drop_last = distributed
    # the GPUs in the same model parallel group receive the same data
    if distributed:
        batch_sampler = DistributedBatchSampler(
            sampler,
            batch_size,
            drop_last,
            rank,
            world_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
    else:
        batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last)
    return batch_sampler


def build_data_loader(
    dataset,
    batch_size,
    num_iters,
    collate_fn,
    gradient_accumulation_steps=1,
    scatter_num=None,
    shuffle=False,
    num_workers=2,
    pin_memory=True
):
    batch_sampler = build_sampler(dataset, batch_size, num_iters, gradient_accumulation_steps, scatter_num, shuffle)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return data_loader
