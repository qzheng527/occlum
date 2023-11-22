#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"

from abc import ABC, abstractmethod
from typing import List, Union

from solutions.antllm.datachain.chain.base import DataChain
from solutions.antllm.datachain.prompts.base import Instruct


class PromptCollationChain(DataChain, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def collate(self, instructs: Union[List[Instruct], List[str]]) -> List[Instruct]:
        pass
