#!/usr/bin/env python
# coding=utf-8
# @Author: tianxuan.jl
# @Date: Wed 14 Jun 2023 05:11:33 PM CST
import math
import random
from typing import Iterator

import torch
from torch.utils.data.distributed import DistributedSampler, T_co


class ShardDistributedSampler(DistributedSampler):
    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            gen = torch.Generator()
            gen.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=gen).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size /
                            len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        gen.manual_seed(self.seed + self.epoch + self.rank)
        if self.shuffle:
            shard_indices = torch.randperm(
                self.dataset.local_num_samples, generator=gen).tolist()
        else:
            shard_indices = list(range(self.dataset.local_num_samples))
        if len(indices) <= len(shard_indices):
            shard_indices = shard_indices[:len(indices)]
        else:
            if self.shuffle:
                shard_indices = shard_indices + \
                    [random.randint(0, self.dataset.local_num_samples - 1)] * \
                    (len(indices) - len(shard_indices))
            else:
                shard_indices = shard_indices + \
                    [i for i in range((len(indices) - len(shard_indices)))]

        assert len(indices) == len(shard_indices) == self.num_samples

        return iter(shard_indices)
