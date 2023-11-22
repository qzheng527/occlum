#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"

from typing import Any, Dict, List

from solutions.antllm.datachain.chain import DataChain


class KG2InstructChain(DataChain):
    def __init__(self, output_key="result") -> None:
        super().__init__(output_key)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return super().run(inputs)

    def batch_run(self, input_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return super().batch_run(input_list)
