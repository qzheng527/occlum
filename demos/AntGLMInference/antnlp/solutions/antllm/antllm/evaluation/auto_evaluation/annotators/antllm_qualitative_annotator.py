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


class QualitativeAnnotator(BaseAnnotator):
    """
    蚂蚁大模型定性数据集标注器
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert len(self.model_output_keys) == 1

    def _make_prompts(
        self, data_to_annotate: list, prompt_template: Optional[str] = None
    ) -> list:
        if prompt_template is None:
            prompt_template = self.prompt_template
        prompt_template = prompt_template.replace("output", self.model_output_keys[0])
        return utils.make_prompts(data=data_to_annotate, template=prompt_template, batch_size=self.batch_size)
    
    def _get_metric(self):
        return metrics.single_to_acc

    def _postprocess(self, data_annotated: list) -> list:
        data_annotated = super()._postprocess(data_annotated)

        all_values = [item[self.annotation_key] for item in data_annotated]
        assert set(all_values) <= {-1, 0, 1, 2}

        return data_annotated
