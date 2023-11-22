#!/usr/bin/env python
# coding=utf-8
# @Author: xinyu.kxy
# @Date: Fri 17 July 2023 09:22:56 PM CST

import ast
import copy
import json
import logging
import re

import numpy as np

from solutions.antllm.antllm.evaluation.auto_evaluation import utils


def regex_parser(completion: str, outputs_to_match: dict) -> list:
    """
    Examples
    --------
    >>> completion = ('\n(b)\n\n### Best output for example 8:\n(a)\n\n### Best output for example 9:\n(b)\n\n### Best'\
    ...               ' output for example 10:\n(a)\n\n### Best output for example 11:\n(a)')
    >>> regex_parser(completion, {1: r"\n\(a\)", 2: r"\n\(b\)"})
    [2, 1, 2, 1, 1]
    >>> regex_parser(' (a)', {1: r" \(a\)", 2: r" \(b\)"})
    [1]
    """
    for k, v in outputs_to_match.items():
        if not isinstance(v, re.Pattern):
            # inplace modification, which is bad practice but useful to speedup
            outputs_to_match[k] = re.compile(v)

    completion = copy.deepcopy(completion)
    responses = []
    while True:
        match, key = utils.find_first_match(completion, outputs_to_match)
        if not match:
            break
        responses.append(key)
        # avoid matching the same output twice
        completion = completion[match.end():]
    if len(responses) == 0:
        responses = [-1]
    else:
        responses = responses[:1]
    return responses


def ranking_parser(completion: str) -> list:
    """
    Examples
    --------
    >>> ranking_parser("[{'model': 'model_1', 'rank': 1}, {'model': 'model_2', 'rank': 2}]")
    [1]
    >>> ranking_parser("[{'model': 'model_1', 'rank': 2}, {'model': 'model_2', 'rank': 1}]")
    [2]
    >>> ranking_parser("[{'model': 'model_1', 'rank': 3}, {'model': 'model_2', 'rank': 1}]")
    [nan]
    """
    try:
        if isinstance(completion, str):
            ordered_completions = ast.literal_eval(completion)
        else:
            ordered_completions = completion

        rank = [c for c in ordered_completions if c["model"]
                == "model_1"][0]["rank"]
        assert rank in [1, 2]

        return [rank]
    except Exception as e:
        logging.error(
            f"{e}\nContent: {completion}\n" "You must manually fix the score pair.")
        return [np.nan]


def json_parser(completion: str, annotation_key: str) -> list:
    """
    Examples
    --------
    >>> completion = '{"short_explanation": "that is why", "is_incorporated": true}'
    >>> json_parser(completion, "is_incorporated")
    [True]
    >>> completion = '[{"short_explanation": "that is why", "is_incorporated": true}, {"is_incorporated": false}]'
    >>> json_parser(completion, "is_incorporated")
    [True, False]
    >>> completion = 'blah ```json\n{"short_explanation": "that is why", "integer": 1}```'
    >>> json_parser(completion, "integer")
    [1]
    """
    # search for a pattern "```json{...}```" and take what is inside the curly brackets
    if "```json" in completion:
        completion = re.search(
            r"```json(.*?)```", completion, re.DOTALL).group(1)

    json_loaded = json.loads(completion)
    if isinstance(json_loaded, dict):
        return [json_loaded[annotation_key]]
    return [d[annotation_key] for d in json.loads(completion)]


def eval_parser(completion: str) -> list:
    """
    Examples
    --------
    >>> eval_parser("True")
    [True]
    >>> eval_parser("(True,1,'False')")
    [(True, 1, 'False')]
    >>> eval_parser("[True,1,'False']")
    [True, 1, 'False']
    """
    evaluated_completion = ast.literal_eval(completion)
    if not isinstance(evaluated_completion, list):
        evaluated_completion = [evaluated_completion]
    return evaluated_completion


if __name__ == "__main__":
    print(regex_parser("Output (a) as f", {
          1: r"(?:^|\n) ?Output \(a\)", 2: r" \(b\)"}))
