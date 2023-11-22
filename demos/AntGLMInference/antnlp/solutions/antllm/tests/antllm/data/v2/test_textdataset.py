#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"

import os
from tempfile import TemporaryDirectory
from typing import Iterable, List
from unittest import TestCase, main

from solutions.antllm.antllm.datav2.data_reader import DataReader
from solutions.antllm.antllm.datav2.datasets import (AutoDataset, TextDataset,
                                                     TokenizeWorker)
from solutions.antllm.antllm.datav2.lazy_loader_v2 import (
    LazyLoaderV2, ScatterWriter, get_scatter_offset_path)
from solutions.antllm.antllm.models.glm.tokenization_glm import GLMTokenizer


class TestTextDataset(TestCase):
    def setUp(self):
        self.base_dir = os.path.dirname(__file__)
        self.tokenizer = GLMTokenizer.from_pretrained(
            os.path.join(self.base_dir, '../../../..', 'zhen_sp5/')
        )

    def test_datareader(self):
        worker = TokenizeWorker(self.tokenizer, tokenize_keys=["content"], inplace_tokenize=True)
        res = []
        reader = DataReader(os.path.join(self.base_dir, "toy_pretrain_data.jsonl"),
                            worker_ftn=worker.tokenize, postprocess_ftn=res.append, num_processes=10)
        reader.process()
        self.assertEquals(len(res), 30)

    def test_lazyloader(self):
        with TemporaryDirectory() as dirname:
            # using datareader to read the data and scatter writer to write the processed data
            scatter_path = os.path.join(dirname, "scatter")
            os.makedirs(scatter_path)
            worker = TokenizeWorker(self.tokenizer, tokenize_keys=["content"], inplace_tokenize=True)
            writer = ScatterWriter(scatter_path, 8)
            reader = DataReader(os.path.join(self.base_dir, "toy_pretrain_data.jsonl"),
                                worker_ftn=worker.tokenize, postprocess_ftn=writer.write, num_processes=10)
            reader.process()
            writer.close()

            part_path, offset_path = get_scatter_offset_path(scatter_path, 1)
            loader1 = LazyLoaderV2(part_path, True)  # load into memory
            loader2 = LazyLoaderV2(part_path, False, offset_path=offset_path)  # using offset
            self.assertTrue(len(loader1) > 0)
            self.assertEqual(len(loader1), len(loader2))

    def test_textdataset(self):
        txt_ds = TextDataset("text",
                             os.path.join(self.base_dir, "toy_pretrain_data.jsonl"),
                             tokenize_keys=["content"],
                             tokenizer=self.tokenizer,
                             lazy_loader_opt="v2")
        self.assertTrue(len(txt_ds) > 0)
        self.assertTrue(isinstance(txt_ds[0]["content"], List))

        txt_ds = TextDataset("text",
                             os.path.join(self.base_dir, "toy_pretrain_data.jsonl"),
                             tokenize_keys=["content"],
                             tokenizer=self.tokenizer,
                             lazy_loader_opt="v1")
        self.assertTrue(len(txt_ds) > 0)
        self.assertTrue(isinstance(txt_ds[0]["content"], Iterable))

    def test_weigthed_dataset(self):
        ds = AutoDataset.from_config(os.path.join(self.base_dir, "weighted_ds_config.yaml"), self.tokenizer)
        self.assertTrue(len(ds) > 0)


if __name__ == '__main__':
    main()
