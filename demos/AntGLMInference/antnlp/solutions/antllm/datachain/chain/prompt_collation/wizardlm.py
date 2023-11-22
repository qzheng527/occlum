#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kuangzhi"

import random
from typing import Any, Dict, List, Union

from solutions.antllm.datachain.llms.ant_openai import STATUS_FAILED
from solutions.antllm.datachain.llms.base import AntLLM
from solutions.antllm.datachain.prompts import get_prompt_by_name
from solutions.antllm.datachain.prompts.base import Instruct

from .base import PromptCollationChain


class WizardLMChain(PromptCollationChain):
    """Apply WizardLM to expand prompt set

    https://arxiv.org/abs/2304.12244

    Args:
    """

    def __init__(self,
                 llm: AntLLM,
                 prompts: List[str] = [
                     "indepth_prompt1",
                     "indepth_prompt2",
                     "indepth_prompt3",
                     "indepth_prompt4",
                     "indepth_prompt1",
                     "inbreadth_prompt1"],
                 evol_steps=4,
                 max_workers=20,
                 verbose=False) -> None:
        super().__init__()
        self.llm = llm
        self.prompts = []
        for p in prompts:
            prompt_content = get_prompt_by_name(p)
            if not prompt_content or len(prompt_content) == 0:
                self.prompts.append(p)
            else:
                self.prompts.append(prompt_content)

        self._evol_steps = evol_steps
        self._max_workers = max_workers
        self._verbose = verbose

    def _instruction_evol(self, instruction: Instruct) -> List[Instruct]:
        evol_res = []
        last_inst = instruction.instruct
        for _ in range(self._evol_steps):
            prompt = random.sample(self.prompts, 1)[0]
            try:
                llm_res = self.llm.generate(
                    prompt.format(last_inst))
                if llm_res != STATUS_FAILED:
                    evol_inst = Instruct(
                        llm_res, origin_inst=last_inst, prompt=prompt)
                    evol_res.append(evol_inst)
                    last_inst = llm_res
            except Exception:
                continue
        return evol_res

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inst = inputs["instruct"]
        return {self.output_key: self._instruction_evol(inst)}

    def collate(self, instructs: Union[List[Instruct], List[str]]) -> List[Instruct]:
        if isinstance(instructs[0], str):
            batch_input = [{"instruct": Instruct(inst)} for inst in instructs]
        else:
            batch_input = [{"instruct": inst} for inst in instructs]
        res = self.batch_run(batch_input)
        all_insts = []
        for item in res:
            if self.output_key in item and len(item[self.output_key]) > 0:
                all_insts.extend(item[self.output_key])
        return all_insts
