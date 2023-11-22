#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"


from pathlib import Path
from typing import Any, Dict, List, Union

from solutions.antllm.datachain.chain.base import DataChain
from solutions.antllm.datachain.chain.llm.base import FAILED, PROMPT, AntLLMChain
from solutions.antllm.datachain.chain.prompt_collation import PromptCollationChain
from solutions.antllm.datachain.prompts.base import Instruct
from solutions.antllm.datachain.utils import load_yaml


class InstructGenerationChain(DataChain):
    """Given seed prompts (instructs), apply llms to generate responses \
       and prompt collation techniques to expand prompts.
    """

    def __init__(self,
                 llm_chains: Dict[str, AntLLMChain],
                 prompt_collators: List[PromptCollationChain] = None,
                 rank_llm_chain: AntLLMChain = None
                 ) -> None:
        super().__init__()
        self._llm_chains = llm_chains
        self._prompt_collators = prompt_collators

        self._rank_llm_chain = rank_llm_chain

    @classmethod
    def from_config(cls, config: Union[Dict, str, Path]):
        if not isinstance(config, Dict):
            config = load_yaml(config)
        assert "llm_chains" in config, "Cannot find llm chains configuration"

        pass

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        res = dict()
        # run llm_chains in parallel
        instructs = []
        for name, llm_chain in self._llm_chains.items():
            llm_res = llm_chain.run(inputs)
            if llm_chain.output_key not in llm_res or llm_res[llm_chain.output_key] == FAILED:
                continue
            instruct = Instruct(
                input=llm_res[PROMPT], response=llm_res[llm_chain.output_key], tags=[name])
            instructs.append(instruct)

        res["instructs"] = list(instructs)
        # run prompt collator to expand or select prompt
        if self._prompt_collators is not None and len(self._prompt_collators) > 0:
            for collator in self._prompt_collators:
                instructs = collator.collate(instructs)
            res["collated_instructs"] = instructs

        if self._rank_llm_chain is not None:
            qa_pairs = dict()
            for inst in instructs:
                if inst.input not in qa_pairs:
                    qa_pairs[inst.input] = set()
                qa_pairs[inst.input].add(inst.response)

            rank_llm_inputs = []
            for q, answers in qa_pairs.items():
                rank_llm_inputs.append({"question": q, "answers": answers})

            rank_llm_res = self._rank_llm_chain.batch_run(rank_llm_inputs)
            res["rank_llm_result"] = rank_llm_res
        return res
