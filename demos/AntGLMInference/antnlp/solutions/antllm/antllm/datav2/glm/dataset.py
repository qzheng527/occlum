#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"

import random
from bisect import bisect_right
from itertools import accumulate
from pathlib import Path
from typing import Union

import numpy as np
from transformers import PreTrainedTokenizer

from solutions.antllm.antllm.datav2.data_utils import (truncate_doc,
                                                       truncate_right)
from solutions.antllm.antllm.datav2.datasets import TextDataset
from solutions.antllm.antllm.datav2.glm.featurizer import (
    GLMSeq2SeqFeaturizer, ensure_glm_tokenizer)
from solutions.antllm.antllm.datav2.lazy_loader import LazyLoader
from solutions.antllm.antllm.datav2.lazy_loader_v2 import LazyLoaderV2
from solutions.antllm.antllm.utils.logging.logger import log_dist


class GLMSeq2SeqDataset(TextDataset):
    """The dataset used for GLM SFT.

    Args:
    """

    def __init__(self,
                 name: str,
                 data_path: Union[str, Path, LazyLoader, LazyLoaderV2],
                 tokenizer: PreTrainedTokenizer,
                 need_tokenize=True,
                 pre_tokenize=True,  # if true, will tokenize the data at the begining
                 input_key="input",
                 output_key="output",
                 lazy_loader_opt="naive",  # directly read the data
                 load_memory=True,
                 mem_map=False,
                 scatter_num=8,
                 mode="train",
                 max_length=1024,
                 max_input_length=512,
                 max_output_length=512,
                 left_truncate=True,
                 **kwargs):
        super().__init__(
            name=name,
            data_path=data_path,
            tokenize_keys=[input_key, output_key],
            need_tokenize=need_tokenize and pre_tokenize,
            tokenizer=tokenizer,
            inplace_tokenize=True,
            lazy_loader_opt=lazy_loader_opt,
            load_memory=load_memory,
            mem_map=mem_map,
            scatter_num=scatter_num,
            global_view=True,
            **kwargs)

        ensure_glm_tokenizer(tokenizer)

        self.featurizer = GLMSeq2SeqFeaturizer(
            name="s2s",
            need_tokenize=need_tokenize and not pre_tokenize,
            tokenizer=self.tokenizer,
            mode=mode,
            max_length=max_length,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            left_truncate=left_truncate,
            input_key=input_key,
            output_key=output_key)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        return self.featurizer.featurize(sample)


class GLMBlockDataset(TextDataset):
    """The dataset used for GLM pretraining

    Args:
    """

    def __init__(self,
                 name: str,
                 data_path: Union[str, Path, LazyLoader, LazyLoaderV2],
                 tokenizer: PreTrainedTokenizer,
                 need_tokenize=True,
                 format: str = "jsonl",
                 content_key: str = "content",
                 max_len: int = 2048,
                 left_truncate_prob: float = 0.0,
                 sample_across_doc: bool = False,
                 lazy_loader_opt="v1",
                 load_memory=False,
                 mem_map=True,
                 scatter_num=1,
                 no_lazy_loader=False,
                 half_lazy_loader=False,
                 load_old_format=False,
                 weighted_sampling=True,
                 **kwargs):
        # use to compatible with data format from legacy glm data format
        self.load_old_format = load_old_format
        if load_old_format:
            tokenize_keys = ["prompt", "text"]
            content_key = "tokens"
        else:
            tokenize_keys = [content_key]
        super().__init__(
            name=name,
            data_path=data_path,
            format=format,
            tokenize_keys=tokenize_keys,
            need_tokenize=need_tokenize,
            tokenizer=tokenizer,
            inplace_tokenize=True,
            lazy_loader_opt=lazy_loader_opt,
            load_memory=load_memory,
            mem_map=mem_map,
            scatter_num=scatter_num,
            no_lazy_loader=no_lazy_loader,
            half_lazy_loader=half_lazy_loader,
            **kwargs)

        ensure_glm_tokenizer(tokenizer)

        self.max_len = max_len
        self.left_truncate_prob = left_truncate_prob
        self.sample_across_doc = sample_across_doc

        self.content_key = content_key

        self.token_cnt = 0
        self.weighted_sampling = weighted_sampling
        if self.weighted_sampling:
            self._weights = []
            self._init_weighting()

        log_dist(
            f"Dataset {self.name}: {len(self)} documents, {self.token_cnt} tokens")

    def _init_weighting(self):
        """initialize the weights for each sample according to the length of the sample.
        """
        lens = []
        for idx in range(len(self)):
            sample = super().__getitem__(idx)
            if self.load_old_format:
                content = list(sample["prompt"]) + list(sample["text"])
            else:
                content = sample[self.content_key]
            lens.append(len(content))
        self.token_cnt = np.sum(lens)
        self._weights = list(accumulate(lens))

    def _get_weighted_sample(self, rng):
        idx = rng.randint(self.token_cnt)
        data_idx = bisect_right(self._weights, idx)
        sample = super().__getitem__(data_idx)

        return sample

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        if self.weighted_sampling:
            # init rng
            rng = random.Random(idx)
            rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])

            # get possibly weighted random index from dataset
            sample = self._get_weighted_sample(rng)
        else:
            sample = super().__getitem__(idx)

        if self.load_old_format:
            prompt = list(sample["prompt"])
            text = list(sample["text"])
            sample = {"tokens": prompt + text, "loss_masks": [0] * len(prompt) + [1] * len(text)}

        tokens = sample[self.content_key]
        tokens += [self.tokenizer.eos_token_id]

        # [cls] + tokens + [eos]
        if len(tokens) + 1 > self.max_len:
            # need to truncate
            tokens = truncate_doc(tokens, self.max_len - 1,
                                  True, (self.max_len - 1) // 2, self.left_truncate_prob, rng, self.tokenizer)
            tokens = [self.tokenizer.cls_token_id] + tokens
            loss_mask = [0] + [1] * (len(tokens) - 1)
        else:
            tokens = [self.tokenizer.cls_token_id] + tokens
            loss_mask = [0] + [1] * (len(tokens) - 1)
            # Sample multiple documents
            if self.sample_across_doc:
                tokens = [self.tokenizer.cls_token_id] + \
                    tokens + [self.tokenizer.eos_token_id]
                loss_mask = [0] + [1] * (len(tokens) - 1)
                while len(tokens) < self.max_len:
                    new_tokens = self._get_weighted_sample(rng)[self.content_key]
                    new_tokens = [self.tokenizer.cls_token_id] + \
                        new_tokens + [self.tokenizer.eos_token_id]

                    if len(tokens) + len(new_tokens) > self.max_len:
                        new_max_len = self.max_len - len(tokens)
                        new_tokens = truncate_right(
                            new_tokens, new_max_len, True, True, new_max_len // 2, self.tokenizer)
                        if not (
                            len(new_tokens) == 1 or (
                                len(new_tokens) == 2 and new_tokens[1] == self.tokenizer.eos_token_id)):
                            tokens += new_tokens
                            loss_mask += [0] + [1] * (len(new_tokens) - 1)
                        break
                    else:
                        tokens += new_tokens

        return {'text': np.array(tokens), "loss_mask": np.array(loss_mask)}
