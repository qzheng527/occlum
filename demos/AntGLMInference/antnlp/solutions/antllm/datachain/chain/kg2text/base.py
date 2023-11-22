#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"

from solutions.antllm.datachain.chain import DataChain


class KG2TextChain(DataChain):
    def __init__(self, output_key="result") -> None:
        super().__init__(output_key)
