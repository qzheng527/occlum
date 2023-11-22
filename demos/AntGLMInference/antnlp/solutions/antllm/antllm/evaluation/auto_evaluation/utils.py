#!/usr/bin/env python
# coding=utf-8
# @Author: xinyu.kxy
# @Date: Fri 17 July 2023 09:22:56 PM CST

import copy
import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Optional, Union
import yaml

from solutions.antllm.antllm.evaluation.auto_evaluation import annotators

DUMMY_EXAMPLE = dict(instruction="1+1=", output_1="2", input="", output_2="3")


def read_or_return(to_read: str, **kwargs):
    """
    读取文件
    """
    try:
        with open(Path(to_read), **kwargs) as f:
            out = f.read()
    except FileNotFoundError as e:
        if Path(to_read).is_absolute():
            # The path is not absolute, so it's not just a string
            raise e

        logging.warning(f"Returning input because file not found. Error: {e}")
        out = to_read

    return out


def get_annotator_by_name(name: str) -> annotators.BaseAnnotator:
    """获取annotator"""
    return getattr(annotators, name)


def shuffle_pairwise_preferences(data: list, output_a, output_b) -> list:
    """
    调换pairwise答案
    """
    for item in data:
        if item["is_switched_outputs"]:
            tmp = item[output_a]
            item[output_a] = item[output_b]
            item[output_b] = tmp
            if "label" in item and item["label"] in [0, 1]:
                item["label"] = 1 - item["label"]
                del item["is_switched_outputs"]
    return data


def find_first_match(text: str, outputs_to_match: dict) -> tuple:
    """根据输入文本和compiled regex进行匹配，返回第一个匹配项"""
    first_match = None
    first_key = None

    for key, compiled_regex in outputs_to_match.items():
        match = compiled_regex.search(text)
        if match and (not first_match or match.start() < first_match.start()):
            first_match = match
            first_key = key

    return first_match, first_key


def make_prompts(
    data: list, template: str, batch_size: int = 1, padding_example=DUMMY_EXAMPLE
) -> list:
    """
    Example
    -------
    >>> data = list({"instruction": ["solve", "write backwards", "other 1"],
    ...                    "input": ["1+1", "'abc'", ""]})
    >>> make_prompts(data, template="first: {instruction} {input}, second: {instruction} {input}",
    ...              batch_size=2, padding_example=dict(instruction="pad", input="pad_in"))[0]
    ["first: solve 1+1, second: write backwards 'abc'",
     'first: other 1 , second: pad pad_in']
    """

    if len(data) == 0:
        return []

    text_to_format = re.findall("{([^ \s]+?)}", template)
    n_occurrences = Counter(text_to_format)

    if not all([n == batch_size for n in n_occurrences.values()]):
        raise ValueError(
            f"All placeholders should be repeated batch_size={batch_size} times but {n_occurrences}.")

    # TODO
    # padding if you don't have enough examples
    # n_to_pad = (batch_size - len(data)) % batch_size
    # padding = list([padding_example] * n_to_pad)
    # padding["is_padding"] = True
    # data_out = pd.concat([data, padding], axis=0, ignore_index=True)
    # data_out["is_padding"] = data_out["is_padding"].fillna(False)

    prompts = []
    # ugly for loops, not trivial to vectorize because of the batching
    for i in range(0, len(data), batch_size):
        current_prompt = copy.deepcopy(template)
        for j in range(batch_size):
            for to_format in n_occurrences.keys():
                # replace only first occurrence (that's why we don't use .format)ƒ
                current_prompt = current_prompt.replace(
                    "{" + to_format + "}", str(data[i + j][to_format]), 1)
        prompts.append(current_prompt)

    return prompts


class Timer:
    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start = time.time()
        return self

    def __exit__(self, *args):
        """Stop the context manager timer"""
        self.end = time.time()
        self.duration = self.end - self.start

    def __str__(self):
        return f"{self.duration:.1f} seconds"


def load_configs(configs: Union[str, dict], relative_to: Optional[str] = None):
    """加载config yaml文件"""
    if not isinstance(configs, dict):
        if relative_to is not None:
            configs = Path(relative_to) / configs
        configs = Path(configs)
        if configs.is_dir():
            configs = configs / "configs.yaml"
        with open(configs, "r") as stream:
            try:
                configs = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logging.exception(exc)

    return configs


if __name__ == "__main__":
    config = load_configs(
        "solutions/antllm/antllm/evaluation/auto_evaluation/dataset_configs/alpacaEval")
    print(config)
