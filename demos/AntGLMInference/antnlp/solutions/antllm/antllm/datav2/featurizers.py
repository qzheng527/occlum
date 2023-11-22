#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"


import random
from abc import ABC
from typing import Dict

IGNORE_INDEX = -100

INPUT_IDS = "input_ids"
LABEL = "label"
LOSS_MASK = "loss_mask"


def sample_spans(span_lengths, total_length, rng: random.Random, offset=0):
    blank_length = total_length - sum(span_lengths)
    m = blank_length - len(span_lengths) + 1
    places = [rng.randrange(m + 1) for _ in range(len(span_lengths))]
    places.sort()
    spans = []
    for place, span_length in zip(places, span_lengths):
        start = offset + place
        end = offset + place + span_length
        spans.append((start, end))
        offset += span_length + 1
    return spans


class BaseFeaturizer(ABC):
    """The base class of featurizer. A featurizer is used to extract features from a single data instance.

    Args:
        name (str, Optional): the name of the featurizer. Default: ""
    """

    def __init__(self, name="") -> None:
        super().__init__()
        self.name = name

    def featurize(self, sample: Dict, **kwargs) -> Dict:
        """extract features from sample

        Args:
            sample (Dict): the input for feature extraction

        Returns:
            Dict: the extracted features
        """
        pass
