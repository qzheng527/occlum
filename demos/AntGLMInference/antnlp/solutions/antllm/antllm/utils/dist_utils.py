#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"

import os

import torch.distributed as dist

from solutions.antllm.antllm.utils import mpu


def get_rank(default=0):
    if dist.is_initialized():
        return dist.get_rank()
    return default


def get_local_rank(default=0):
    return int(os.environ.get("LOCAL_RANK", str(default)))


def get_world_size(default=1):
    if dist.is_initialized():
        return dist.get_world_size()
    return default


def get_data_parallel_rank(default=0):
    if dist.is_initialized():
        return mpu.get_data_parallel_rank()
    return default


def get_data_parallel_world_size(default=1):
    if dist.is_initialized():
        return mpu.get_data_parallel_world_size()
    return default


def get_model_parallel_rank(default=0):
    if dist.is_initialized():
        return mpu.get_model_parallel_rank()
    return default


def get_model_parallel_world_size(default=1):
    if dist.is_initialized():
        return mpu.get_model_parallel_world_size()
    return default
