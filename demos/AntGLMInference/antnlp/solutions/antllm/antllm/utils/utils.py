#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"


import random
from pathlib import Path
from typing import Dict, Iterable, Union

import yaml


def get_first(dic, keys):
    for key in keys:
        if key in dic:
            return dic[key]
    return None


def contains_any(str, strs):
    for s in strs:
        if s in str:
            return True
    return False


def rindex(lst, val, start=None):
    if start is None:
        start = len(lst) - 1
    for i in range(start, -1, -1):
        if lst[i] == val:
            return i
    return -1


def index_in_list(lst, val, start=None):
    if start is None:
        start = 0
    for i in range(start, len(lst)):
        if lst[i] == val:
            return i
    return -1


def batch_iter(data: Iterable, size, shuffle=False):
    new_data = list(data)
    if shuffle:
        random.shuffle(new_data)

    for ndx in range(0, len(new_data), size):
        yield new_data[ndx:min(ndx + size, len(new_data))]


def load_yaml(yaml_path: Union[str, Path]) -> Dict:
    return yaml.safe_load(Path(yaml_path).read_text())


def is_yaml(path: Union[str, Path]) -> bool:
    suffix = Path(path).suffix
    return suffix.endswith("yaml") or suffix.endswith("yml")
