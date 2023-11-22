#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"

import os
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

from solutions.antllm.datachain.chain.base import DataChain
from solutions.antllm.datachain.llms.base import AntLLM
from solutions.antllm.datachain.prompts import get_prompt_by_name
from solutions.antllm.datachain.utils import load_yaml
from langchain import PromptTemplate
from langchain.prompts.base import StringPromptValue
from langchain.schema import BaseOutputParser, BasePromptTemplate, PromptValue

RAW_GENERATION = "raw"
PROMPT = "prompt"
OUTPUT = "output"
INPUT = "input"
FAILED = "failed"


class AntLLMChain(DataChain):
    def __init__(self, llm: AntLLM, prompt: Union[BasePromptTemplate, str, Path] = None,
                 output_parser: BaseOutputParser = None) -> None:
        super().__init__(output_key=OUTPUT)
        self.llm = llm
        if not isinstance(prompt, BasePromptTemplate):
            if os.path.exists(prompt):
                self.prompt = PromptTemplate.from_file(prompt)
            else:
                prompt_templ = get_prompt_by_name(prompt)
                assert prompt_templ is not None and prompt_templ != "", f"Cannot load prompt {prompt}"
                self.prompt = PromptTemplate.from_template(prompt_templ)
        else:
            self.prompt = prompt
        self.output_parser = output_parser

    @classmethod
    def from_config(cls, config: Union[Dict, str, Path]):
        if not isinstance(config, Dict):
            config = load_yaml(config)

        raise NotImplementedError

    def prep_prompts(
        self,
        input_list: List[Dict[str, Any]]
    ) -> List[PromptValue]:
        """Prepare prompts from inputs."""
        prompts = []
        for inputs in input_list:
            if self.prompt:
                selected_inputs = {}

                for k in self.prompt.input_variables:
                    v = inputs[k]
                    if isinstance(v, (List, Tuple, Set)):
                        v_str = "\n"
                        for idx, item in enumerate(v):
                            v_str += f"{idx+1}. {item}\n"
                    else:
                        v_str = str(v)
                    selected_inputs[k] = v_str

                prompt = self.prompt.format_prompt(**selected_inputs)
            else:
                assert INPUT in inputs, "since the prompt is None, make sure key `input` \
                    exists which will be directly used as prompt"
                prompt = StringPromptValue(text=inputs[INPUT])

            prompts.append(prompt)
        return prompts

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        prompts = self.prep_prompts([inputs])
        resp = self.llm.generate(prompts[0].to_string())
        res = {PROMPT: prompts[0].to_string(
        ), RAW_GENERATION: resp, self.output_key: resp}
        if self.output_parser:
            parsed_res = self.output_parser.parse(resp)
            res[self.output_key] = parsed_res
        return res
