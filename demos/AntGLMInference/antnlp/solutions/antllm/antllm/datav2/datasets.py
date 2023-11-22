#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"

import inspect
import json
import os
import random
import time
from bisect import bisect_right
from collections import OrderedDict
from itertools import accumulate
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from multiprocess import Queue
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from solutions.antllm.antllm.datav2.data_reader import (COMPLETE, STOP,
                                                        DataReader)
from solutions.antllm.antllm.datav2.data_utils import resolve_path
from solutions.antllm.antllm.datav2.lazy_loader import (LazyLoader, LazyWriter,
                                                        exists_lazy,
                                                        exists_scatter,
                                                        get_scatter_path)
from solutions.antllm.antllm.datav2.lazy_loader_v2 import (
    LazyLoaderV2, ScatterWriter, check_scatter_path, get_scatter_directory,
    get_scatter_offset_path)
from solutions.antllm.antllm.utils.dist_utils import (get_data_parallel_rank,
                                                      get_local_rank, get_rank)
from solutions.antllm.antllm.utils.logging import log_dist
from solutions.antllm.antllm.utils.utils import load_yaml

DEFAULT_TOKENIZE_KEYS = ["input", "output",
                         "prompt", "response", "content", "target"]


class TokenizeWorker:
    """A worker used to tokenize data. It could be used in DataReader.
    """

    def __init__(self,
                 tokenizer,
                 format="json",
                 tokenize_keys=DEFAULT_TOKENIZE_KEYS,
                 tokenize_suffix="tokens",
                 inplace_tokenize=False) -> None:
        self.format = format
        self.tokenizer = tokenizer
        self.tokenize_keys = tokenize_keys
        self.tokenize_suffix = tokenize_suffix
        self.inplace_tokenize = inplace_tokenize

    def tokenize(self, input_queue: Queue, output_queue: Queue):
        for row in iter(input_queue.get, STOP):
            if row:
                if self.format == "json":
                    json_obj = json.loads(row)
                    new_json_obj = tokenize_dict(
                        json_obj, self.tokenizer, self.tokenize_keys, self.tokenize_suffix, self.inplace_tokenize)
                    output_queue.put(json.dumps(
                        new_json_obj, ensure_ascii=False))

        output_queue.put(COMPLETE)


def tokenize_dict(dic,
                  tokenizer,
                  tokenize_keys=DEFAULT_TOKENIZE_KEYS,
                  tokenize_suffix="tokens",
                  inplace_tokenize=False):
    new_dict = OrderedDict()
    for k, v in dic.items():
        if k in tokenize_keys:
            tokenized = tokenizer.encode(v)
            if inplace_tokenize:
                new_dict[k] = tokenized
            else:
                new_dict[k] = v
                new_dict[k + "_" +
                         tokenize_suffix] = tokenized
        else:
            new_dict[k] = v

    return new_dict


def get_antllm_dataset_by_name(cls_name):
    textdataset_cls = None
    for cls in Dataset.__subclasses__():
        if "antllm" in cls.__module__ and cls.__name__ == 'TextDataset':
            textdataset_cls = cls
            break
    if textdataset_cls:
        for cls in textdataset_cls.__subclasses__():
            if cls.__name__ == cls_name:
                return cls

    for cls in Dataset.__subclasses__():
        if "antllm" in cls.__module__ and cls.__name__ == cls_name:
            return cls
    return None


class TextDataset(Dataset):
    """A generic dataset that can be used in most scenarios.

    Args:
        name (str): the name of the dataset
        data_path (str, Path, LazyLoader): the path of the dataloader.
        format (str, Optional): the format of the data file. It will be ignored if the data path is a LazyLoader.
        need_tokenize (bool, Optional): If true, will tokenize the data.
        tokenizer (PretrainedTokenizer, Optional): the pretrained tokenizer.
        tokenize_keys (List, Optional): the keys in the json which need to be tokenized.
        inplace_tokenize (bool, Optional): If true, will replace the original data with tokenized data.
        tokenize_suffix (str, Optional): If inplace_tokenize is false, will add the tokenized data with \
                                         tokenize_suffix as the suffix of the key.
        lazy_loader_opt (str, Optional): specify the way to read the data.\
                                         `naive` means directly read the data file.\
                                         `v1` means using the lazy loader from GLM.\
                                         `v1` means using the lazy loader v2 which is more flexible.
        load_memory (bool, Optional): whether to load the full data to memory.
        mem_map (bool, Optional): whether to use memory map.
        scatter_num (int, Optional): the number of scatter files.
    """

    def __init__(self,
                 name,
                 data_path: Union[str, Path, LazyLoaderV2, LazyLoader],
                 format: str = "jsonl",
                 need_tokenize=True,
                 tokenizer=None,
                 tokenize_keys=DEFAULT_TOKENIZE_KEYS,
                 inplace_tokenize=True,
                 tokenize_suffix="tokens",
                 lazy_loader_opt="v1",
                 load_memory=False,
                 mem_map=True,
                 scatter_num=8,
                 global_view=False,
                 no_lazy_loader=False,
                 half_lazy_loader=False,
                 **kwargs
                 ):
        self.name = name
        self.data_path = data_path
        self.format = format
        self.tokenize_keys = tokenize_keys
        self.need_tokenize = need_tokenize
        self.tokenizer = tokenizer
        self.tokenize_suffix = tokenize_suffix
        self.inplace_tokenize = inplace_tokenize

        self.load_memory = load_memory
        self.mem_map = mem_map
        self.scatter_num = scatter_num
        self.global_view = global_view

        assert self.scatter_num > 0

        self.lazy_loader_opt = lazy_loader_opt  # naive: simple line read; v1: GLM lazy loader, v2: lazy loader v2
        assert self.lazy_loader_opt in ("naive", "v1", "v2")

        self.no_lazy_loader = no_lazy_loader
        self.half_lazy_loader = half_lazy_loader

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.samples = []

        # only used in file read mode
        self.global_num_samples = 0
        self.local_num_samples = 0

        self._tokenized = True

        if isinstance(self.data_path, LazyLoaderV2):
            self.samples = self.data_path
        else:
            if self.lazy_loader_opt.strip().lower() == "naive":
                # directly load data to memory, compatible with old SFT logic
                log_dist(f"the data {self.name} will be loaded to memory directly.")
                self._build_dataset_from_line_read()
            elif self.lazy_loader_opt.strip().lower() == "v1":
                log_dist(f"the data {self.name} will be loaded using Lazy Loader v1.")
                self._build_dataset_from_lazyloader()
            else:
                log_dist(f"the data {self.name} will be loaded using Lazy Loader v2.")
                self._build_dataset_from_lazyloaderv2()

    def __getitem__(self, index):
        sample = self.samples[index]
        json_obj = json.loads(sample) if isinstance(sample, str) else sample
        if not self.need_tokenize or self._tokenized:
            return json_obj

        new_json_obj = tokenize_dict(
            json_obj, self.tokenizer, self.tokenize_keys, self.tokenize_suffix, self.inplace_tokenize)
        return new_json_obj

    def __len__(self):
        # only used to be compatible with current SFT sampler
        if self.global_view:
            return self.global_num_samples
        return len(self.samples)

    def _build_dataset_from_line_read(self):
        # only count local and global sample numbers in file line read mode
        self.global_num_samples = 0
        self.local_num_samples = 0
        self._tokenized = False
        self.samples = []
        scatter_idx = get_data_parallel_rank() % self.scatter_num
        paths = []
        if os.path.isdir(self.data_path):
            paths = [os.path.join(top, name) for top, _,
                     names in os.walk(self.data_path) for name in names]
        else:
            paths = [self.data_path]
        for path in paths:
            with open(path, "r", encoding="utf-8") as file:
                for row in file:
                    # only load part of data
                    self.global_num_samples += 1
                    if (self.global_num_samples - 1) % self.scatter_num != scatter_idx:
                        continue
                    self.local_num_samples += 1
                    json_obj = json.loads(row.strip())
                    if self.need_tokenize:
                        self._tokenized = True
                        new_obj = tokenize_dict(
                            json_obj, self.tokenizer, self.tokenize_keys,
                            self.tokenize_suffix, self.inplace_tokenize)
                        self.samples.append(new_obj)
                    else:
                        self.samples.append(json_obj)

    def _build_dataset_from_lazyloaderv2(self):
        """build dataset using lazy loader v2
        """
        scatter_dir = get_scatter_directory(self.data_path)
        if not os.path.exists(scatter_dir):
            if get_rank() == 0:
                # build scatter file in scatter folder for lazy loader v2
                os.makedirs(scatter_dir)
                tokenize_worker = TokenizeWorker(
                    self.tokenizer, tokenize_keys=self.tokenize_keys, tokenize_suffix=self.tokenize_suffix,
                    inplace_tokenize=self.inplace_tokenize)
                writer = ScatterWriter(scatter_dir, self.scatter_num)
                reader = DataReader(self.data_path, worker_ftn=tokenize_worker.tokenize, postprocess_ftn=writer.write)
                reader.process()
                writer.close()
            else:
                # for other GPUs, wait for scatter finished
                while not check_scatter_path(scatter_dir, self.scatter_num, not self.load_memory):
                    time.sleep(1)

        # load scatter file
        scatter_idx = get_data_parallel_rank() % self.scatter_num
        part_path, offset_path = get_scatter_offset_path(scatter_dir, scatter_idx)
        self.samples = LazyLoaderV2(part_path, self.load_memory, offset_path, self.mem_map)

    def _build_dataset_from_lazyloader(self):
        """build dataset using lazy loader v1 in order to be compatible with current GLM pretrain logic.
        """
        if not all([exists_scatter(self.data_path, self.scatter_num, key) for key in self.tokenize_keys]):
            if not all([exists_lazy(self.data_path, key) for key in self.tokenize_keys]):
                # the data hasn't been preprocessed, try to preprocess the data to lazy format in rank 0 GPU.
                if get_rank() == 0:
                    log_dist(f"Creating lazy loader for dataset {self.name}")
                    writers: Dict[str, LazyWriter] = {}
                    for key in self.tokenize_keys:
                        writer = LazyWriter(
                            self.data_path, data_type=key, is_array=True
                        )
                        writers[key] = writer
                    tokenize_worker = TokenizeWorker(
                        self.tokenizer, tokenize_keys=self.tokenize_keys, tokenize_suffix=self.tokenize_suffix,
                        inplace_tokenize=self.inplace_tokenize)

                    def write_ftn(data):
                        for k, v in json.loads(data).items():
                            writers[k].write(v)

                    reader = DataReader(
                        path=self.data_path,
                        worker_ftn=tokenize_worker.tokenize,
                        postprocess_ftn=write_ftn)
                    reader.process()
                    for w in writers.values():
                        w.close()
                else:
                    # for other GPU, wait util the data preprocessing finished.
                    while not os.path.exists(
                        LazyWriter.get_len_path(self.data_path, data_type="prompt")
                    ):
                        time.sleep(1)
        map_fn = (lambda x: x.tolist()) if not self.need_tokenize else None
        if self.scatter_num > 1:
            if not all([exists_scatter(self.data_path, self.scatter_num, key) for key in self.tokenize_keys]):
                # scatter the data
                if get_rank() == 0:
                    log_dist(f'Creating scatter loader for dataset {self.name}')
                    loaders = {}
                    data_len = 0
                    for key in self.tokenize_keys:
                        loader = LazyLoader(
                            self.data_path,
                            data_type=key,
                            map_fn=map_fn,
                            mem_map=True,
                            is_array=True,
                        )
                        loaders[key] = loader
                        data_len = len(loader)

                    indices = np.arange(data_len)
                    random.shuffle(indices)

                    log_dist(f"load indices: {indices.shape}.")
                    segment_length = (len(indices) - 1) // self.scatter_num + 1
                    for i in range(self.scatter_num):
                        log_dist(f"Start process scatter {i}")
                        scatter_path = get_scatter_path(self.data_path, scatter_rank=i)
                        writers = {}
                        for key in self.tokenize_keys:
                            writers[key] = LazyWriter(
                                scatter_path, data_type=key, is_array=True
                            )

                        for idx in indices[i * segment_length: (i + 1) * segment_length]:
                            for key in self.tokenize_keys:
                                writers[key].write(loaders[key][idx])
                        for w in writers.values():
                            w.close()
                else:
                    while not all([exists_scatter(
                        self.data_path, data_type=key, scatter_num=self.scatter_num
                    ) for key in self.tokenize_keys]):
                        time.sleep(1)
            scatter_path = get_scatter_path(
                self.data_path, scatter_rank=get_data_parallel_rank() % self.scatter_num
            )
            log_dist(f"Rank {get_rank()} is using scatter from {scatter_path}")
            loaders = {}
            for key in self.tokenize_keys:
                loader = LazyLoader(
                    scatter_path,
                    data_type=key,
                    map_fn=map_fn,
                    mem_map=True,
                    is_array=True,
                    load_memory=self.no_lazy_loader,
                    half_load=self.half_lazy_loader,
                )
                loaders[key] = loader
        else:
            loaders = {}
            if not all([exists_scatter(self.data_path, self.scatter_num, key) for key in self.tokenize_keys]):
                for key in self.tokenize_keys:
                    loader = LazyLoader(
                        self.data_path,
                        data_type=key,
                        map_fn=map_fn,
                        mem_map=True,
                        is_array=True,
                        load_memory=self.no_lazy_loader,
                        half_load=self.half_lazy_loader,
                    )
                    loaders[key] = loader
            else:
                scatter_path = get_scatter_path(
                    self.data_path, scatter_rank=get_data_parallel_rank() % self.scatter_num
                )
                log_dist(f"Rank {get_rank()} is using scatter from {scatter_path}")
                for key in self.tokenize_keys:
                    loader = LazyLoader(
                        scatter_path,
                        data_type=key,
                        map_fn=map_fn,
                        mem_map=True,
                        is_array=True,
                        load_memory=self.no_lazy_loader,
                        half_load=self.half_lazy_loader,
                    )
                    loaders[key] = loader

        self.samples = MultiLoaderDataset(
            loaders=loaders,
            tokenizer=self.tokenizer,
            to_tokenize=not self._tokenized
        )


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.

    Arguments:
        datasets (Sequence): List of datasets to be concatenated.
    """

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            x = len(e)
            r.append(x + s)
            s += x
        return r

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


class BlendableDataset(Dataset):
    """Merge multiple datasets into one and sample instance from the merged dataset according to given weights.
    from Megatron-LM
    """

    def __init__(self, datasets: List[Dataset],
                 weights: List[float] = None) -> None:
        super().__init__()
        self.datasets = datasets
        self.weights = weights

        if weights and len(weights) > 0:
            assert len(self.datasets) == len(self.weights)
            s = sum(weights)
            assert s > 0, 'the weights sum should be positive.'
            s = sum([len(d) * w for d, w in zip(self.datasets, self.weights)])
            norm_weights = [len(d) * w / s for d, w in zip(self.datasets, self.weights)]
            self.weights = norm_weights
            log_dist(
                f"the real weights of BlendableDataset is {[(d.name, w) for d, w in zip(self.datasets, self.weights)]}")

        self.total_num = sum([len(ds) for ds in self.datasets])
        self.dataset_index = np.zeros(self.total_num, dtype=np.uint8)
        self.dataset_sample_index = np.zeros(self.total_num, dtype=np.int64)

        # make helpers before run
        def _exists_helper_so():
            import glob
            so_files = glob.glob(os.path.join(os.path.dirname(__file__), "helpers.*.so"))
            return so_files and len(so_files) > 0

        if not _exists_helper_so:
            if get_local_rank() == 0:
                import subprocess
                p = subprocess.Popen(["make"], cwd=os.path.dirname(__file__))
                p.wait()
            else:
                while not _exists_helper_so():
                    time.sleep(1)
        try:
            from solutions.antllm.antllm.datav2 import helpers
            helpers.build_blending_indices(self.dataset_index,
                                           self.dataset_sample_index,
                                           self.weights, len(self.datasets), self.total_num,
                                           get_rank() == 0)
        except ImportError:
            log_dist(
                """fail to compile C++ helper, will fallback to python version \
                which is expected to be much slowly than C++ version.""")
            self._build_blending_indices(self.dataset_index, self.dataset_sample_index,
                                         self.weights, len(self.datasets), self.total_num, get_rank() == 0)

    def _build_blending_indices(self, dataset_index, dataset_sample_index, weights, num_datasets, size, verbose):
        """Given multiple datasets and a weighting array, build samples such that it follows those weights."""
        if verbose:
            print("> building indices for blendable datasets ...")

        # Initialize buffer for number of samples used for each dataset.
        current_samples = np.zeros(num_datasets, dtype=np.int64)

        # For each sample:
        for sample_idx in range(size):
            # Determine where the max error in sampling is happening.
            sample_idx_double = max(sample_idx, 1.0)
            max_error_index = 0
            max_error = weights[0] * sample_idx_double - float(current_samples[0])
            for dataset_idx in range(1, num_datasets):
                error = weights[dataset_idx] * sample_idx_double - float(current_samples[dataset_idx])
                if error > max_error:
                    max_error = error
                    max_error_index = dataset_idx

            # Populate the indices.
            dataset_index[sample_idx] = np.uint8(max_error_index)
            dataset_sample_index[sample_idx] = current_samples[max_error_index]

            # Update the total samples.
            current_samples[max_error_index] += 1

        # print info
        if verbose:
            print(" > sample ratios:")
            for dataset_idx in range(num_datasets):
                ratio = float(current_samples[dataset_idx]) / float(size)
                print("   dataset {}, input: {}, achieved: {}".format(dataset_idx, weights[dataset_idx], ratio))

    @classmethod
    def from_config(cls, config_path: Union[str, Path], tokenizer: PreTrainedTokenizer = None):
        log_dist(f"creating WeightedDataset from config {config_path}")
        dataset_configs = load_yaml(config_path)
        datasets = []
        weights = []
        for ds_config in dataset_configs:
            ds_type = ds_config["type"]
            args = ds_config.get("args", dict())
            for k, v in args.items():
                if k in ["data_path", "path", "data"]:
                    args[k] = resolve_path(v, os.path.dirname(config_path))
            weight = ds_config.get("weight", 1)
            ds_cls = get_antllm_dataset_by_name(ds_type)
            assert ds_cls, f"Cannot find dataset type {ds_type}"
            sig = inspect.signature(ds_cls.__init__)
            if "tokenizer" in set(sig.parameters.keys()):
                args["tokenizer"] = tokenizer
            dataset = ds_cls(**args)
            datasets.append(dataset)
            weights.append(weight)
        return BlendableDataset(datasets, weights)

    def __len__(self):
        return self.total_num

    def __getitem__(self, index):
        dataset_idx = self.dataset_index[index]
        sample_idx = self.dataset_sample_index[index]
        sample_idx %= len(self.datasets[dataset_idx])
        return self.datasets[dataset_idx][sample_idx]


class WeightedDataset(Dataset):
    """First sample dataset according to the given weights, then sample item from the dataset.
    """

    def __init__(self, datasets: List[Dataset],
                 weights: List[float] = None, total_num: int = 0) -> None:
        super().__init__()
        self.datasets = datasets
        self.weights = weights

        if weights and len(weights) > 0:
            assert len(self.datasets) == len(self.weights)
            s = sum(weights)
            assert s > 0, 'the weights sum should be positive.'
            norm_weights = [w / s for w in weights]
            self.weights = norm_weights
            self.weights = list(accumulate(self.weights))

        self.total_num = total_num if total_num > 0 else sum([len(ds) for ds in self.datasets])

    def __len__(self):
        return self.total_num

    def __getitem__(self, index):
        rng = random.Random(index)
        rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])

        dataset_idx = bisect_right(self.weights, rng.rand())
        ds = self.datasets[dataset_idx]
        return ds[index % len(ds)]


class SplitDataset(Dataset):
    """
    Dataset wrapper to access a subset of another dataset.
    Purpose: useful to index into existing datasets, possibly
    large-scale datasets as the subindexing operation is done in an
    on-the-fly manner.
    Arguments:
        ds (Dataset or array-like): List of datasets to be subindexed
        split_inds (1D array-like): List of indices part of subset
    """

    def __init__(self, ds, split_inds):
        self.split_inds = list(split_inds)
        self.wrapped_data = ds

    def __len__(self):
        return len(self.split_inds)

    def __getitem__(self, index):
        return self.wrapped_data[self.split_inds[index]]

    def __iter__(self):
        for idx in self.split_inds:
            yield self.wrapped_data[idx]


def split_dataset(ds, split=None, shuffle=True, save_splits=None, load_splits=None, rank=0):
    """
    Split a dataset into subsets given proportions of how
    much to allocate per split. If a split is 0% returns None for that split.
    Purpose: Useful for creating train/val/test splits
    Arguments:
        ds (Dataset or array-like): Data to be split.
        split (1D array-like): proportions to split `ds`. `sum(splits) != 0`
        shuffle (boolean): Randomly split dataset. Default: True
        save_splits: save split indices to file
        load_splits: load split indices from file
    """
    if split is None:
        split = [.8, .2, .0]
    split_sum = sum(split)
    if split_sum == 0:
        raise Exception('Split cannot sum to 0.')
    split = np.array(split)
    split /= split_sum
    ds_len = len(ds)
    inds = np.arange(ds_len)
    if shuffle:
        rng = np.random.RandomState(1234)
        rng.shuffle(inds)
    if load_splits is not None:
        inds = np.load(load_splits)
        assert len(inds) == ds_len
        log_dist(f"Load split indices from {load_splits}")
    elif save_splits is not None:
        if rank == 0:
            np.save(save_splits, inds)
            log_dist(f"Save split indices to {save_splits}")
    start_idx = 0
    residual_idx = 0
    rtn_ds = [None] * len(split)
    for i, f in enumerate(split):
        if f != 0:
            proportion = ds_len * split[i]
            residual_idx += proportion % 1
            split_ = int(int(proportion) + residual_idx)
            split_inds = inds[start_idx:start_idx + max(split_, 1)]
            rtn_ds[i] = SplitDataset(ds, split_inds)
            start_idx += split_
            residual_idx %= 1
    return rtn_ds


class MultiLoaderDataset(Dataset):
    """Compatible with old GLM dataset
    """

    def __init__(self, loaders, tokenizer=None, to_tokenize=False, **kwargs):
        self.loaders = loaders
        lens = set()
        for _, loader in self.loaders.items():
            lens.add(len(loader))
        assert len(lens) == 1
        self.data_len = list(lens)[0]
        self.tokenizer = tokenizer
        self.to_tokenize = to_tokenize

    def __getitem__(self, index):
        res = {}
        for key, loader in self.loaders.items():
            val = loader[index]
            if self.to_tokenize:
                val = self.tokenizer.encode(val)
            res[key] = val
        return res

    def __len__(self):
        return self.data_len


def init_dataset(
        type: str, args: Dict, tokenizer: PreTrainedTokenizer = None, root_dir: str = None):
    for k, v in args.items():
        if k in ["data_path", "path", "data"] and root_dir:
            args[k] = resolve_path(v, root_dir)
    ds_cls = get_antllm_dataset_by_name(type)
    assert ds_cls, f"Cannot find dataset type {type}"
    sig = inspect.signature(ds_cls.__init__)
    if "tokenizer" in set(sig.parameters.keys()):
        args["tokenizer"] = tokenizer

    log_dist(f"Creating dataset {type} with args as:\n{args}")
    dataset = ds_cls(**args)
    return dataset


def _resolve_sub_dataset(config, tokenizer=None, root_dir=None):
    if isinstance(config, Dict) and "type" in config:
        return AutoDataset.from_config(config, tokenizer, root_dir)
    elif isinstance(config, List):
        items = []
        for cfg in config:
            items.append(_resolve_sub_dataset(cfg, tokenizer, root_dir))
        return items
    return config


class AutoDataset:
    @classmethod
    def from_config(
            cls, config: Union[str, Path, Dict],
            tokenizer: PreTrainedTokenizer = None, root_dir: Union[str, Path] = None):
        log_dist(f"creating AutoDataset from config {config}")
        if isinstance(config, (str, Path)):
            dataset_config = load_yaml(config)
            if not root_dir:
                root_dir = os.path.dirname(config)
        else:
            dataset_config = config
        assert "type" in dataset_config

        ds_type = dataset_config["type"]
        args = dataset_config.get("args", dict())
        for k, v in args.items():
            args[k] = _resolve_sub_dataset(v, tokenizer, root_dir)
        dataset = init_dataset(ds_type, args, tokenizer, root_dir)
        return dataset
