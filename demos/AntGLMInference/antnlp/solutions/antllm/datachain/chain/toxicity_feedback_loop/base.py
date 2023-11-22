#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.
__author__ = "kexi"

import json
from typing import Any, Dict, List

from solutions.antllm.datachain.chain.base import DataChain
from solutions.antllm.datachain.chain.llm import AntLLMChain
from solutions.antllm.datachain.chain.llm.base import FAILED, OUTPUT, PROMPT
from solutions.antllm.datachain.io.odps_reader import ODPSReader


class ToxicityFeedbackLoop(DataChain):
    """

    Args:
        DataChain (_type_): _description_
    """

    def __init__(self,
                 llm: AntLLMChain,
                 prompt_column: str = "prompt",
                 candidates_column: str = "candidates",
                 gold_column: str = "gold",
                 verbose=False) -> None:
        super().__init__(verbose=verbose)
        self.llm = llm
        self.prompt_column = prompt_column
        self.candidates_column = candidates_column
        self.gold_column = gold_column

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        prompt = inputs[self.prompt_column]
        candidates = inputs[self.candidates_column]
        candidates = list(json.loads(candidates).values())
        gold = inputs[self.gold_column]

        rewrite_prompt = self.llm.run({PROMPT: prompt})
        prompts = [prompt]
        if OUTPUT in rewrite_prompt and rewrite_prompt[OUTPUT] != FAILED:
            if isinstance(rewrite_prompt[OUTPUT], List):
                prompts.extend(rewrite_prompt[OUTPUT])
            else:
                prompts.append(rewrite_prompt[OUTPUT])

        return {self.prompt_column: prompts, self.candidates_column: candidates, self.gold_column: gold}

    def load(self, input_path=None, **kwargs) -> List[Dict[str, Any]]:
        odps_project_table = kwargs.get("odps_project_table", input_path)
        columns = kwargs.get("columns", None)
        access_id = kwargs.get("access_id", None)
        access_key = kwargs.get("access_key", None)
        project = kwargs.get("project", None)
        endpoint = kwargs.get("endpoint", None)
        partition = kwargs.get("partition", [])
        odps = ODPSReader(odps_project_table, columns, access_id, access_key, project, endpoint, partition)
        self._inputs = odps.read()

    def save(self, sft_data_save_path=None, rm_data_save_path=None, data_source='knowledge_base', **kwargs):
        with open(sft_data_save_path, "w", encoding="utf-8") as fo_sft:
            with open(rm_data_save_path, "w", encoding="utf-8") as fo_rm:
                for item in self._outputs:
                    prompts = set(item[self.prompt_column])
                    candidates = set(item[self.candidates_column])
                    golds = item[self.gold_column]
                    if golds in candidates:
                        candidates.remove(golds)
                    for p in prompts:
                        fo_sft.write(json.dumps({'input': p, 'output': golds}, ensure_ascii=False) + "\n")
                        for c in candidates:
                            fo_rm.write(json.dumps({'prompt': p, 'chosen': golds, 'rejected': c, 
                                        'source': data_source, 'checked': False}, ensure_ascii=False) + "\n")
