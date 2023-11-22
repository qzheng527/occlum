#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"

from typing import Dict, List

from antllm.data.llms.base import AntLLM


class MockLLM(AntLLM):
    def __init__(self) -> None:
        super().__init__()

    def generate(self, prompt: str) -> str:
        return "hello world"

    def batch_generate(self, prompts: List[str]) -> Dict[str, str]:
        res = {}
        for p in prompts:
            res[p] = "hello world"
        return res
