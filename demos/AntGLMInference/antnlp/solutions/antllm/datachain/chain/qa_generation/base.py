#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"


from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from solutions.antllm.datachain.chain.base import DataChain
from solutions.antllm.datachain.chain.llm.base import AntLLMChain
from langchain.text_splitter import (RecursiveCharacterTextSplitter,
                                     TextSplitter)


class QAGenerationChain(DataChain):
    """Given documents, extract qa pairs from the documents.

    Note: the output parser of llm_chain should parse the raw LLM result to List of QA Tuple
    """

    def __init__(self,
                 llm_chain: AntLLMChain,
                 text_splitter: TextSplitter = RecursiveCharacterTextSplitter(
                     chunk_overlap=500)
                 ) -> None:
        super().__init__()
        self._llm_chain = llm_chain
        self._text_splitter = text_splitter

    @classmethod
    def from_config(cls, config: Union[Dict, str, Path]):
        pass

    def extract(self, docs: List[str]) -> List[List[Tuple[str, str]]]:
        docs = self._text_splitter.create_documents(docs)
        results = self._llm_chain.batch_run(
            [{"text": d.page_content} for d in docs]
        )
        res = []
        for qa_a_doc in results:
            if isinstance(qa_a_doc[self._llm_chain.output_key], List):
                res.append(qa_a_doc[self._llm_chain.output_key])
            elif isinstance(qa_a_doc[self._llm_chain.output_key], Dict):
                res.append([qa_a_doc[self._llm_chain.output_key]])
            else:
                res.append([])
        return res

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        doc = inputs["text"]
        res = self.extract([doc])
        return {self.output_key: res[0]}
