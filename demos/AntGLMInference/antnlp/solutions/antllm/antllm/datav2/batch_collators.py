#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"

from abc import ABC, abstractmethod
from typing import Dict, Sequence

import numpy as np
import torch

from .featurizers import IGNORE_INDEX

TASK = "task"
NAME = "name"
MODE = "mode"


class BaseBatchCollator(ABC):
    """The base class of a batch collator.

    Args:
        name: (str, Optional): the name of the batch collator. Default: empty str
    """

    def __init__(self, name="") -> None:
        super().__init__()
        self.name = name
        self.task = None  # the task of current batch
        self.source = None  # the data source of current batch

    def __call__(self, samples: Sequence[Dict]) -> Dict:
        return self.collate(samples)

    @abstractmethod
    def collate(self, samples: Sequence[Dict]) -> Dict:
        pass


class PadSequenceCollator(BaseBatchCollator):
    """Pad sequence in a batch

    Args:
    """

    def __init__(self, pad_token_id, ignore_index=IGNORE_INDEX, input_ids_key="input_ids", labels_key="labels") -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.input_ids_key = input_ids_key
        self.labels_key = labels_key

    def collate(self, samples: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = [sample[self.input_ids_key] for sample in samples], [
            sample[self.labels_key] for sample in samples]

        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.ignore_index)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.pad_token_id),
        )


def pad_batch(batch_seq, pad=0):
    seq_lengths = list(map(len, batch_seq))
    if seq_lengths.count(seq_lengths[0]) != len(seq_lengths):
        max_length = max(seq_lengths)
        batch_seq = [
            np.concatenate(
                (tokens, np.asarray([pad] * (max_length - len(tokens)), dtype=np.int_))
            )
            for tokens in batch_seq
        ]
    return batch_seq
