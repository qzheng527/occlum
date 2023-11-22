#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"

import logging

import torch.distributed as dist


def log_dist(msg, logger=None, should_log_ranks=[0], level=logging.INFO):
    should_log = not dist.is_initialized()
    ranks = should_log_ranks or []
    my_rank = dist.get_rank() if dist.is_initialized() else -1
    if ranks and not should_log:
        should_log = ranks[0] == -1
        should_log = should_log or (my_rank in set(ranks))
        if should_log:
            final_msg = "[Rank {}]{}".format(my_rank, msg)
            if not logger:
                print(final_msg, flush=True)
            else:
                logger.log(level, final_msg)
