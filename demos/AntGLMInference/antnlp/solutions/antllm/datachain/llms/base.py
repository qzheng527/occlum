#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"

from abc import ABC, abstractmethod
from typing import Dict, List


class AntLLM(ABC):
    """the base class of LLM in datachain
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Get the response of LLM

        Args:
            prompt (str): the prompt to the LLM

        Returns:
            str: the response from LLM
        """
        pass

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Get the chat response of LLM

        Args:
            messages: List[Dict[str, str]], multiturn chat dialog history to LLM

        Returns:
            str: the response from LLM
        """
        pass

    @abstractmethod
    def batch_generate(self, prompts: List[str]) -> Dict[str, str]:
        pass

    @abstractmethod
    def batch_chat(self, list_messages: List[List[Dict[str, str]]]) -> Dict[str, str]:
        pass
