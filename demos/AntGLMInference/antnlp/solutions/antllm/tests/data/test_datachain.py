#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"

import os
from unittest import TestCase, main

from antllm.data.chain import InstructGenerationChain
from antllm.data.chain.llm.base import AntLLMChain
from antllm.data.chain.prompt_collation import WizardLMChain
from antllm.data.chain.qa_generation.base import QAGenerationChain
from antllm.data.output_parsers.json import SafeJsonOutputParser
from antllm.tests.data.mock_llm import MockLLM


class TestDataChain(TestCase):
    def setUp(self):
        pass

    def test_wizardlm(self):
        llm = MockLLM()
        wizardlm = WizardLMChain(llm, verbose=True)
        res = wizardlm.collate(["中国的首都在哪里"])
        self.assertTrue(len(res) > 0)

    def test_antllmchain(self):
        llm = MockLLM()
        llm_chain = AntLLMChain(llm, "gov_qa_fluent")
        res = llm_chain.run(
            {"question": "问题",
             "context": "参考文档",
             "answer": "答案"})
        self.assertTrue(len(res) > 0)

    def test_instruction_generation_chain(self):
        llm = MockLLM()
        llm_chain1 = AntLLMChain(llm)
        llm_chain2 = AntLLMChain(llm)

        chain = InstructGenerationChain(
            {"llm_chain1": llm_chain1, "llm_chain2": llm_chain2})

        res = chain.run({"input": "中国首都是哪里"})
        self.assertTrue(len(res) > 0)

    def test_qa_generation(self):
        llm = MockLLM()
        llm_chain = AntLLMChain(
            llm, "multi_qa_zh", output_parser=SafeJsonOutputParser())
        qa_chain = QAGenerationChain(llm_chain)
        res = qa_chain.extract(
            [open(os.path.join(os.path.dirname(__file__), "data/zh_doc.txt"), "r", encoding="utf-8").read()])
        self.assertTrue(len(res) > 0)


if __name__ == "__main__":
    main()
