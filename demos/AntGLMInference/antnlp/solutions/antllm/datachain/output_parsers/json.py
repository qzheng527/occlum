#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"

import json
import re
from json import JSONDecodeError
from typing import Any

from langchain.schema import BaseOutputParser, OutputParserException


class SafeJsonOutputParser(BaseOutputParser[Any]):
    def parse(self, text: str) -> Any:
        text = text.strip()
        text = re.sub(r"\n", r" ", text)
        try:
            return json.loads(text)
        except JSONDecodeError as e:
            raise OutputParserException(f"Invalid json output: {text}") from e

    @property
    def _type(self) -> str:
        return "safe_json_output_parser"


class StrOutput2ListParser(BaseOutputParser[Any]):
    def parse(self, text: str) -> Any:
        parse_result = []
        try:
            text = text.strip().split('\\n')
            for t in text:
                try:
                    t = re.sub('^(改写|答案|结果|回答|)\d+(.|:|：)', '', t)  # chatgpt的返回结果可能会在改写的结果前面出现标号，需要替换掉
                    t = t.replace(' ', '')
                    if len(t) > 2 and t != 'failed':  # 替换后的结果长度不能太短并且解析后的结果不能是failed，如果是直接丢弃不用
                        parse_result.append(t)
                except (TypeError, IndexError):
                    continue
        except (TypeError, IndexError):  # chatgpt请求失败或者请求回来的结果格式无法解析，则直接丢弃
            return []
        return parse_result

    @property
    def _type(self) -> str:
        return "string_to_list_parser"
