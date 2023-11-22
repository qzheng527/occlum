#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"

from typing import Dict, List

from .base import AntLLM


class AntGLM(AntLLM):
    """AntGLM client

    Args:
        ckpt_path (str): the checkpoint of AntGLM
    """

    def __init__(self, ckpt_path: str, **kwargs) -> None:
        self.ckpt_path = ckpt_path
        self.kwargs = kwargs

    def generate(self, prompt: str) -> str:
        return super().generate(prompt)

    def batch_generate(self, prompts: List[str], batch_size=10) -> Dict[str, str]:
        return super().batch_generate(prompts, batch_size)
