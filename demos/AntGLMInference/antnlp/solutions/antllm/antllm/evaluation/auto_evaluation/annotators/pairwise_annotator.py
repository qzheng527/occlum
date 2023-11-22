#!/usr/bin/env python
# coding=utf-8
# @Author: xinyu.kxy
# @Date: Fri 17 July 2023 09:22:56 PM CST

# import logging
import random
from typing import Optional

from solutions.antllm.antllm.evaluation.auto_evaluation import utils, metrics
from solutions.antllm.antllm.evaluation.auto_evaluation.annotators import BaseAnnotator


random.seed(2023)


class PairwiseAnnotator(BaseAnnotator):
    """
    双模型答案对比评估标注器

    is_randomize_output_order : 是否随机调换两个模型答案的位置
    """

    def __init__(
        self,
        *args,
        is_randomize_output_order: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.is_randomize_output_order = is_randomize_output_order
        assert len(self.model_output_keys) == 2

    def _preprocess(self, data_to_annotate: list) -> list:
        if self.is_randomize_output_order:
            # 随机调换模型答案位置
            for item in data_to_annotate:
                item["is_switched_outputs"] = random.choice([False, True])

            data_to_annotate = utils.shuffle_pairwise_preferences(
                data_to_annotate, self.model_output_keys[0], self.model_output_keys[1])

        # 处理相等答案
        for item in data_to_annotate:
            if item[self.model_output_keys[0]] == item[self.model_output_keys[1]]:
                item[self.annotation_key] = 0

        data_to_annotate = super()._preprocess(data_to_annotate)

        return data_to_annotate

    def _get_metric(self):
        return metrics.pairwise_to_winrate

    def _make_prompts(
        self, data_to_annotate: list, prompt_template: Optional[str] = None
    ) -> list:
        if prompt_template is None:
            prompt_template = self.prompt_template
        prompt_template = self.prompt_template.replace(
            "output_1", self.model_output_keys[0]).replace("output_2", self.model_output_keys[1])
        return utils.make_prompts(data=data_to_annotate, template=prompt_template, batch_size=self.batch_size)

    def _postprocess(self, data_annotated: list) -> list:
        data_annotated = super()._postprocess(data_annotated)

        all_values = [item[self.annotation_key] for item in data_annotated]
        assert set(all_values) <= {-1, 0, 1, 2}

        if self.is_randomize_output_order:
            # unshuffles output 1 and output 2. For binary preference, unshuffling is equivalent to reshuffling
            data_annotated = utils.shuffle_pairwise_preferences(
                data_annotated, self.model_output_keys[0], self.model_output_keys[1])
        return data_annotated
