#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"

from typing import Dict, Sequence

from transformers import DataCollatorForLanguageModeling

from solutions.antllm.antllm.datav2.batch_collators import BaseBatchCollator
from solutions.antllm.antllm.datav2.featurizers import INPUT_IDS


class LLaMAPretrainBatchCollator(BaseBatchCollator):
    """Batch collator used for LLaMA pretraining.
    Reuse DataCollatorForLanguageModeling, only change content token key to input_ids.

    Args:
        BaseBatchCollator (_type_): _description_
    """

    def __init__(self, name, tokenizer, input_ids_key=INPUT_IDS) -> None:
        super().__init__(name)
        self.input_ids_key = input_ids_key
        self.tokenizer = tokenizer
        self.lm_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    def collate(self, samples: Sequence[Dict]) -> Dict:
        new_samples = []
        for sample in samples:
            if INPUT_IDS in sample:
                new_samples.append(sample)
            else:
                assert self.input_ids_key in sample
                new_sample = {}
                for k, v in sample.items():
                    if k == self.input_ids_key:
                        new_sample[INPUT_IDS] = v
                    else:
                        new_sample[k] = v
                new_samples.append(new_sample)
        return self.lm_collator(new_samples)
