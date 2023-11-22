# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Tasks data utility."""
import copy
import json
import os
import pickle
import random
import re
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from solutions.antllm.antllm.data import corpora
from solutions.antllm.antllm.data.datasets import (BertSentencepairDataset,
                                                   BlockDataset, ConcatDataset,
                                                   GPT2Dataset, XLDataset,
                                                   split_ds)
from solutions.antllm.antllm.data.lazy_loader import (LazyLoader, LazyWriter,
                                                      exists_lazy,
                                                      exists_scatter,
                                                      get_scatter_path)
from solutions.antllm.antllm.utils import mpu


def clean_text(text):
    """Remove new lines and multiple spaces and adjust end of sentence dot."""

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    for _ in range(3):
        text = text.replace(" . ", ". ")

    return text


class InputExample(object):
    """A raw input example consisting of one or two segments of text and a label"""

    def __init__(
        self,
        guid,
        text_a,
        text_b=None,
        label=None,
        logits=None,
        meta: Optional[Dict] = None,
        idx=-1,
        num_choices=1,
    ):
        """
        Create a new InputExample.

        :param guid: a unique textual identifier
        :param text_a: the sequence of text
        :param text_b: an optional, second sequence of text
        :param label: an optional label
        :param logits: an optional list of per-class logits
        :param meta: an optional dictionary to store arbitrary meta information
        :param idx: an optional numeric index
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.logits = logits
        self.idx = idx
        self.num_choices = num_choices
        self.meta = meta if meta else {}

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @staticmethod
    def load_examples(path: str) -> List["InputExample"]:
        """Load a set of input examples from a file"""
        with open(path, "rb") as fh:
            return pickle.load(fh)

    @staticmethod
    def save_examples(examples: List["InputExample"], path: str) -> None:
        """Save a set of input examples to a file"""
        with open(path, "wb") as fh:
            pickle.dump(examples, fh)


def num_special_tokens_to_add(
    text_a_ids, text_b_ids, answer_ids, add_cls, add_sep, add_piece, add_eos=True
):
    num_tokens = 0
    if add_cls:
        num_tokens += 1
    if text_b_ids and add_sep:
        num_tokens += 1
    if add_eos:
        num_tokens += 1
    if not answer_ids and add_piece:
        num_tokens += 1
    return num_tokens


def build_uni_input_from_ids(
    text_a_ids,
    answer_ids,
    max_seq_length,
    tokenizer,
    args=None,
    add_cls=True,
    add_sep=False,
    add_eos=True,
    mask_id=None,
):
    if mask_id is None:
        mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id
    cls_id = tokenizer.cls_token_id
    sop_id = tokenizer.sop_token_id
    ids = []
    if add_cls:
        ids = [cls_id]
    ids.append(mask_id)
    sep = len(ids)
    mask_position = sep - 1
    position_ids = list(range(len(ids)))
    block_position_ids = [0] * len(ids)
    ids.append(sop_id)
    # A
    ids.extend(text_a_ids)
    target_ids = [0] * (len(ids) - 1)
    loss_masks = [0] * (len(ids) - 1)
    # Piece
    ids.extend(answer_ids[:-1])
    target_ids.extend(answer_ids)
    loss_masks.extend([1] * len(answer_ids))
    position_ids.extend([mask_position] * (len(ids) - len(position_ids)))
    block_position_ids.extend(range(1, len(ids) - len(block_position_ids) + 1))
    # Padding.
    padding_length = max_seq_length - len(ids)
    if padding_length > 0:
        ids.extend([eos_id] * padding_length)
        position_ids.extend([0] * padding_length)
        block_position_ids.extend([0] * padding_length)
        target_ids.extend([0] * padding_length)
        loss_masks.extend([0] * padding_length)
    position_ids = [position_ids, block_position_ids]
    return ids, None, None, position_ids, sep, target_ids, loss_masks


def build_input_from_ids(
    text_a_ids,
    text_b_ids,
    answer_ids,
    max_seq_length,
    tokenizer,
    args=None,
    add_cls=True,
    add_sep=False,
    add_piece=False,
    add_eos=True,
    mask_id=None,
):
    if mask_id is None:
        mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    ids = []
    types = []
    paddings = []
    # CLS
    if add_cls:
        ids.append(cls_id)
        types.append(0)
        paddings.append(1)
    # A
    len_text_a = len(text_a_ids)
    ids.extend(text_a_ids)
    types.extend([0] * len_text_a)
    paddings.extend([1] * len_text_a)
    # B
    if text_b_ids is not None:
        # SEP
        if add_sep:
            ids.append(sep_id)
            types.append(0)
            paddings.append(1)
        len_text_b = len(text_b_ids)
        ids.extend(text_b_ids)
        types.extend([1] * len_text_b)
        paddings.extend([1] * len_text_b)
    eos_length = 1 if add_eos else 0
    # Cap the size.
    if len(ids) >= max_seq_length - eos_length:
        max_seq_length_m1 = max_seq_length - 1
        ids = ids[0:max_seq_length_m1]
        types = types[0:max_seq_length_m1]
        paddings = paddings[0:max_seq_length_m1]
    end_type = 0 if text_b_ids is None else 1
    if add_eos:
        ids.append(eos_id)
        types.append(end_type)
        paddings.append(1)
    sep = len(ids)
    target_ids = [0] * len(ids)
    loss_masks = [0] * len(ids)
    position_ids = list(range(len(ids)))
    block_position_ids = [0] * len(ids)
    # Piece
    if add_piece or answer_ids is not None:
        sop_id = tokenizer.sop_token_id
        assert mask_id in ids
        mask_position = (
            len(ids) - ids[-1::-1].index(mask_id) - 1
            if not args.sentinel_token
            else args.max_position_embeddings
        )
        ids.append(sop_id)
        types.append(end_type)
        paddings.append(1)
        position_ids.append(mask_position)
        block_position_ids.append(1)
        if answer_ids is not None:
            len_answer = len(answer_ids)
            ids.extend(answer_ids[:-1])
            types.extend([end_type] * (len_answer - 1))
            paddings.extend([1] * (len_answer - 1))
            position_ids.extend([mask_position] * (len_answer - 1))
            if not args.no_block_position:
                block_position_ids.extend(range(2, len(answer_ids) + 1))
            else:
                block_position_ids.extend([1] * (len(answer_ids) - 1))
            target_ids.extend(answer_ids)
            loss_masks.extend([1] * len(answer_ids))
        else:
            target_ids.append(0)
            loss_masks.append(1)
    # Padding.
    padding_length = max_seq_length - len(ids)
    if padding_length > 0:
        ids.extend([eos_id] * padding_length)
        types.extend([eos_id] * padding_length)
        paddings.extend([0] * padding_length)
        position_ids.extend([0] * padding_length)
        block_position_ids.extend([0] * padding_length)
        target_ids.extend([0] * padding_length)
        loss_masks.extend([0] * padding_length)
    if not args.masked_lm:
        position_ids = [position_ids, block_position_ids]
    return ids, types, paddings, position_ids, sep, target_ids, loss_masks


def build_decoder_input(
    enc_ids, answer_ids, max_seq_length, max_dec_seq_length, tokenizer
):
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.mask_eos_id
    sop_id = tokenizer.sop_token_id
    # enc_len = len(enc_ids)
    masks = []
    # TODO: it probably takes too much memory
    # for i in range(max_dec_seq_length):
    #     m = [1]*enc_len + [0]*(max_seq_length - enc_len) + [1]*(i+1) + [0]*(max_dec_seq_length-1-i)
    #     masks.append(m)
    mask_position = enc_ids.index(mask_id)
    len_answer = len(answer_ids)
    ids = [sop_id] + answer_ids[:-1]
    types = [0] * len_answer  # not used
    paddings = [1] * len_answer
    position_ids = [mask_position] * len_answer
    block_position_ids = list(range(1, len_answer + 1))
    target_ids = answer_ids
    loss_masks = [1] * len_answer
    # Padding.
    padding_length = max_dec_seq_length - len(ids)
    if padding_length > 0:
        ids.extend([eos_id] * padding_length)
        types.extend([0] * padding_length)
        paddings.extend([0] * padding_length)
        position_ids.extend([0] * padding_length)
        block_position_ids.extend([0] * padding_length)
        target_ids.extend([0] * padding_length)
        loss_masks.extend([0] * padding_length)
    position_ids = [position_ids, block_position_ids]
    return ids, types, paddings, position_ids, masks, target_ids, loss_masks


def build_sample(
    ids,
    types=None,
    paddings=None,
    positions=None,
    masks=None,
    label=None,
    unique_id=None,
    target=None,
    logit_mask=None,
    segment_ids=None,
    prompt_ids=None,
):
    """Convert to numpy and return a sample consumed by the batch producer."""

    ids_np = np.array(ids, dtype=np.int64)
    sample = {"text": ids_np, "label": int(label)}
    if types is not None:
        types_np = np.array(types, dtype=np.int64)
        sample["types"] = types_np
    if paddings is not None:
        paddings_np = np.array(paddings, dtype=np.int64)
        sample["padding_mask"] = paddings_np
    if positions is not None:
        positions_np = np.array(positions, dtype=np.int64)
        sample["position"] = positions_np
    if masks is not None:
        masks_np = np.array(masks, dtype=np.int64)
        sample["mask"] = masks_np
    if target is not None:
        target_np = np.array(target, dtype=np.int64)
        sample["target"] = target_np
    if logit_mask is not None:
        logit_mask_np = np.array(logit_mask, dtype=np.int64)
        sample["logit_mask"] = logit_mask_np
    if segment_ids is not None:
        segment_ids = np.array(segment_ids, dtype=np.int64)
        sample["segment_id"] = segment_ids
    if prompt_ids is not None:
        prompt_ids = np.array(prompt_ids, dtype=np.int64)
        sample["prompt_pos"] = prompt_ids
    if unique_id is not None:
        sample["uid"] = unique_id
    return sample


def build_decoder_sample(
    sample, dec_ids, dec_position, dec_masks, dec_target, dec_logit_mask
):
    sample["dec_text"] = np.array(dec_ids)
    sample["dec_position"] = np.array(dec_position)
    sample["dec_mask"] = np.array(dec_masks)
    sample["dec_target"] = np.array(dec_target)
    sample["dec_logit_mask"] = np.array(dec_logit_mask)
    return sample


def my_collate(batch):
    new_batch = [
        {key: value for key, value in sample.items() if key != "uid"}
        for sample in batch
    ]
    text_list = [sample["text"] for sample in batch]

    def pad_choice_dim(data, choice_num):
        if len(data) < choice_num:
            data = np.concatenate([data] + [data[0:1]] * (choice_num - len(data)))
        return data

    if len(text_list[0].shape) == 2:
        choice_nums = list(map(len, text_list))
        max_choice_num = max(choice_nums)
        for i, sample in enumerate(new_batch):
            for key, value in sample.items():
                if key != "label":
                    sample[key] = pad_choice_dim(value, max_choice_num)
                else:
                    sample[key] = value
            sample["loss_mask"] = np.array(
                [1] * choice_nums[i] + [0] * (max_choice_num - choice_nums[i]),
                dtype=np.int64,
            )

    if "dec_text" in new_batch[0]:
        choice_nums = [len(sample["dec_text"]) for sample in new_batch]
        if choice_nums.count(choice_nums[0]) != len(choice_nums):
            max_choice_num = max(choice_nums)
            for i, sample in enumerate(new_batch):
                for key, value in sample.items():
                    if key.startswith("dec_"):
                        sample[key] = pad_choice_dim(value, max_choice_num)
                sample["loss_mask"] = np.array(
                    [1] * choice_nums[i] + [0] * (max_choice_num - choice_nums[i]),
                    dtype=np.int64,
                )

    new_batch = default_collate(new_batch)
    if "uid" in batch[0]:
        uid_list = [sample["uid"] for sample in batch]
        new_batch["uid"] = uid_list
    return new_batch


class FakeDataloader:
    def __init__(self, num_iters):
        self.num_iters = num_iters

    def __iter__(self):
        if self.num_iters is not None:
            for _ in range(self.num_iters):
                yield None
        else:
            while True:
                yield None


def build_data_loader(
    dataset, batch_size, num_workers, drop_last, shuffle=True, only_rank0=False
):
    """Data loader. Note that batch-size is the local (per GPU) batch-size."""

    # Sampler.
    if only_rank0:
        rank, world_size = 0, 1
    else:
        world_size = mpu.get_data_parallel_world_size()
        rank = mpu.get_data_parallel_rank()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
    )

    # Data loader. Note that batch size is the per GPU batch size.
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,
        collate_fn=my_collate,
    )

    return data_loader


TRAIN_DATA = 0
VAL_DATA = 1
TEST_DATA = 2


def should_split(split):
    """
    given split proportions checks if should split
    Examples:
    >>> should_split([10,0,0])
    False
    >>> should_split([1,.1,.2])
    True
    """
    return max(split) / sum(split) != 1.0


def get_ext(path):
    """gets path extension"""
    return os.path.splitext(path)[1]


def get_dataset(
    name,
    tokenizer,
    pre_tokenize,
    data_parallel_rank,
    loader_scatter=None,
    no_lazy_loader=False,
    half_lazy_loader=False,
):
    """gets dataset object based on keyword args and file at `path`"""
    global_rank = torch.distributed.get_rank()
    if not supported_corpus(name):
        raise NotImplementedError("dataset %s is not supported" % name)
    dataset = corpora.NAMED_CORPORA[name]
    path = dataset.PATH
    if issubclass(dataset, corpora.PromptReader):
        if not (
            loader_scatter is not None
            and exists_scatter(path, data_type="prompt", scatter_num=loader_scatter)
            and exists_scatter(path, data_type="text", scatter_num=loader_scatter)
        ):
            if not (
                exists_lazy(path, data_type="prompt")
                and exists_lazy(path, data_type="text")
            ):
                # create cached version of dataset for lazy loading if it doesn't exist
                if global_rank == 0:
                    print(f"Creating lazy loader for dataset {name}")
                    prompt_writer = LazyWriter(
                        path, data_type="prompt", is_array=pre_tokenize
                    )
                    text_writer = LazyWriter(
                        path, data_type="text", is_array=pre_tokenize
                    )
                    writers = {"prompt": prompt_writer, "text": text_writer}
                    reader = dataset(
                        writers=writers, tokenizer=tokenizer, tokenize=pre_tokenize
                    )
                    reader.process()
                    prompt_writer.close()
                    text_writer.close()
                else:
                    while not os.path.exists(
                        LazyWriter.get_len_path(path, data_type="prompt")
                    ):
                        time.sleep(1)
        map_fn = (lambda x: x.tolist()) if pre_tokenize else None
        if loader_scatter is not None:
            if not (
                exists_scatter(path, data_type="prompt", scatter_num=loader_scatter)
                and exists_scatter(path, data_type="text", scatter_num=loader_scatter)
            ):
                if global_rank == 0:
                    print(
                        f'''Creating scatter loader for
                        dataset {name},
                        pre_tokenize: {pre_tokenize},
                        global_rank:{global_rank}'''
                    )
                    prompts = LazyLoader(
                        path,
                        data_type="prompt",
                        map_fn=map_fn,
                        mem_map=True,
                        is_array=pre_tokenize,
                    )
                    texts = LazyLoader(
                        path,
                        data_type="text",
                        map_fn=map_fn,
                        mem_map=True,
                        is_array=pre_tokenize,
                    )
                    t_len_path = os.path.join(path, f"len_{len(texts)}.npy")
                    done_len_path = t_len_path + ".done"
                    if not os.path.exists(done_len_path):
                        if global_rank == 0:
                            indices = np.arange(len(texts))
                            random.shuffle(indices)
                            np.save(t_len_path, indices)
                            wt = open(done_len_path, "w")
                            wt.close()
                        else:
                            while not os.path.exists(done_len_path):
                                time.sleep(5)

                    indices = np.load(t_len_path)

                    print(f"load indices: {indices.shape}.")
                    segment_length = (len(indices) - 1) // loader_scatter + 1
                    for i in range(loader_scatter):
                        print(f"Start process scatter {i}")
                        scatter_path = get_scatter_path(path, scatter_rank=i)
                        prompt_writer = LazyWriter(
                            scatter_path, data_type="prompt", is_array=pre_tokenize
                        )
                        text_writer = LazyWriter(
                            scatter_path, data_type="text", is_array=pre_tokenize
                        )
                        for idx in tqdm(
                            indices[i * segment_length: (i + 1) * segment_length]
                        ):
                            prompt_writer.write(prompts[idx])
                            text_writer.write(texts[idx])
                        prompt_writer.close()
                        text_writer.close()
                else:
                    while not (
                        exists_scatter(
                            path, data_type="prompt", scatter_num=loader_scatter
                        )
                        and exists_scatter(
                            path, data_type="text", scatter_num=loader_scatter
                        )
                    ):
                        time.sleep(1)
            scatter_path = get_scatter_path(
                path, scatter_rank=data_parallel_rank % loader_scatter
            )
            print(f"Rank {global_rank} is using scatter from {scatter_path}")
            prompts = LazyLoader(
                scatter_path,
                data_type="prompt",
                map_fn=map_fn,
                mem_map=True,
                is_array=pre_tokenize,
                load_memory=no_lazy_loader,
                half_load=half_lazy_loader,
            )
            texts = LazyLoader(
                scatter_path,
                data_type="text",
                map_fn=map_fn,
                mem_map=True,
                is_array=pre_tokenize,
                load_memory=no_lazy_loader,
                half_load=half_lazy_loader,
            )
        else:
            prompts = LazyLoader(
                path,
                data_type="prompt",
                map_fn=map_fn,
                mem_map=True,
                is_array=pre_tokenize,
                load_memory=no_lazy_loader,
                half_load=half_lazy_loader,
            )
            texts = LazyLoader(
                path,
                data_type="text",
                map_fn=map_fn,
                mem_map=True,
                is_array=pre_tokenize,
                load_memory=no_lazy_loader,
                half_load=half_lazy_loader,
            )
        text = corpora.PromptDataset(
            prompt_loader=prompts,
            text_loader=texts,
            tokenizer=tokenizer,
            to_tokenize=not pre_tokenize,
        )
        if loader_scatter is None:
            if global_rank == 0:
                print(f"Create dataset {name} with {len(text)} documents")
                for i in range(10):
                    rand_id = i if i < 5 else random.randrange(len(text))
                    sample_tokens = text[rand_id]["tokens"][:1024]
                    print(sample_tokens)
                    print(tokenizer.decode(sample_tokens).encode("utf-8"))
        else:
            for scatter_id in range(loader_scatter):
                print(
                    f'''data_parallel_rank: {data_parallel_rank},
                    loader_scatter: {loader_scatter},
                    scatter_id:{scatter_id},
                    {data_parallel_rank % loader_scatter == scatter_id and data_parallel_rank // loader_scatter == 0}'''
                )
                if (
                    data_parallel_rank % loader_scatter == scatter_id
                    and data_parallel_rank // loader_scatter == 0
                ):
                    print(
                        f"Create dataset {name} at scatter {scatter_id} with {len(text)} documents"
                    )
                    for i in range(10):
                        sample_tokens = text[i]["tokens"][:1024]
                        print(sample_tokens)
                        print(tokenizer.decode(sample_tokens))
                # torch.distributed.barrier()
        return text
    elif issubclass(dataset, corpora.KeyReader):
        if not (
            exists_lazy(path, data_type="text") and exists_lazy(path, data_type="mask")
        ):
            # create cached version of dataset for lazy loading if it doesn't exist
            if global_rank == 0:
                text_writer = LazyWriter(path, data_type="text", is_array=pre_tokenize)
                mask_writer = LazyWriter(path, data_type="mask", is_array=True)
                writers = {"mask": mask_writer, "text": text_writer}
                dataset(writers=writers, tokenizer=tokenizer, tokenize=pre_tokenize)
                mask_writer.close()
                text_writer.close()
            else:
                while not os.path.exists(
                    LazyWriter.get_len_path(path, data_type="mask")
                ):
                    time.sleep(1)
        map_fn = (lambda x: x.tolist()) if pre_tokenize else None
        masks = LazyLoader(
            path, data_type="mask", map_fn=map_fn, mem_map=True, is_array=True
        )
        texts = LazyLoader(
            path, data_type="text", map_fn=map_fn, mem_map=True, is_array=pre_tokenize
        )
        text = corpora.KeyDataset(
            mask_loader=masks,
            text_loader=texts,
            tokenizer=tokenizer,
            to_tokenize=not pre_tokenize,
        )
        return text


def supported_corpus(corpus_name):
    """checks if corpus name is defined in `corpora.py`"""
    return corpus_name in corpora.NAMED_CORPORA


def make_dataset(
    path,
    seq_length,
    mem_length,
    shuffle=True,
    split=None,
    tokenizer=None,
    sample_one_document=False,
    pre_tokenize=False,
    ds_type="",
    save_splits=None,
    load_splits=None,
    save_test_data=None,
    no_lazy_loader=False,
    loader_scatter=None,
    data_parallel_rank=None,
    filter_english=False,
    non_sentence_start=0.0,
    half_lazy_loader=False,
    **kwargs,
):
    """function to create datasets+tokenizers for common options"""
    if split is None:
        split = [1.0]

    # get one or multiple datasets and concatenate
    if isinstance(path, str):
        ds = get_dataset(
            path,
            tokenizer=tokenizer,
            pre_tokenize=pre_tokenize,
            no_lazy_loader=no_lazy_loader,
            loader_scatter=loader_scatter,
            data_parallel_rank=data_parallel_rank,
            half_lazy_loader=half_lazy_loader,
        )
    else:
        ds = [
            get_dataset(
                p,
                tokenizer=tokenizer,
                pre_tokenize=pre_tokenize,
                no_lazy_loader=no_lazy_loader,
                loader_scatter=loader_scatter,
                data_parallel_rank=data_parallel_rank,
                half_lazy_loader=half_lazy_loader,
            )
            for p in path
        ]
        ds = ConcatDataset(ds)

    # Split dataset into train/val/test (and wrap bert dataset)
    def wrap_dataset(dataset):
        if ds_type.lower() == "bert":
            presplit_sentences = (
                kwargs["presplit_sentences"]
                if "presplit_sentences" in kwargs
                else False
            )
            dataset = BertSentencepairDataset(
                dataset, max_seq_len=seq_length, presplit_sentences=presplit_sentences
            )
        elif ds_type.lower() == "gpt-xl":
            assert pre_tokenize
            dataset = XLDataset(
                dataset,
                tokenizer,
                max_seq_len=seq_length,
                mem_len=mem_length,
                sample_across_doc=not sample_one_document,
            )
        elif ds_type.lower() == "gpt2":
            dataset = GPT2Dataset(
                dataset,
                tokenizer,
                max_seq_len=seq_length,
                sample_across_doc=not sample_one_document,
            )
        elif ds_type.lower() == "block":
            dataset = BlockDataset(
                dataset,
                tokenizer,
                max_seq_len=seq_length,
                sample_across_doc=not sample_one_document,
                filter_english=filter_english,
                non_sentence_start=non_sentence_start,
            )
        return dataset

    if should_split(split):
        ds = split_ds(
            ds, split, shuffle=shuffle, save_splits=save_splits, load_splits=load_splits
        )
        if save_test_data is not None and torch.distributed.get_rank() == 0:
            test_ds = ds[-1]
            with open(save_test_data, "w", encoding="utf-8") as output:
                for data in test_ds:
                    text = data["tokens"]
                    text = tokenizer.decode(text)
                    output.write(text)
                    output.write("\n")
            print(f"Write test data to {save_test_data}")
        ds = [wrap_dataset(d) if d is not None else None for d in ds]
    else:
        ds = wrap_dataset(ds)
    return ds


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for line in tqdm(data):
            try:
                json_record = json.dumps(line, ensure_ascii=False)
            except Exception:
                print(f'dump error: line: {line}')
            f.write(json_record + '\n')
    print('Wrote {} records to {}'.format(len(data), output_path))


def load_jsonl(input_path, line_num: Optional[int] = None) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    try:
        input_path = str(input_path)
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            invalid_cnt = 0
            for idx, line in tqdm(enumerate(f)):
                if line_num and idx >= line_num:
                    break
                try:
                    data.append(json.loads(line.rstrip('\n|\r')))
                except Exception:
                    invalid_cnt += 1
        print('Loaded {} records from {}'.format(len(data), input_path))
        print('Invalid json num: {}'.format(invalid_cnt))
    except Exception:
        raise Exception(f'load file failed, {input_path}')
    return data
