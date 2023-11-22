#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_model_distill.py
# @Author: daniel.ljh
# @Date  : 2023/9/1

from unittest import TestCase, main
import pytest  # noqa
import os
import json
from solutions.antllm.antllm.api.distill import HardTargetDistill
from solutions.antllm.antllm.api.distill import SoftTargetDistillStudent


class MyTestCase(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_dir = os.path.dirname(__file__)

    def test_hard_distill(self):
        distiller = HardTargetDistill(
            teacher_model="solutions/antllm/glm-super-mini-model",
            student_model="solutions/antllm/super_mini_bart"
        )
        flag = distiller.train_local(
            distill_data=os.path.join(self.base_dir, "../data/dataset/model_distill/hard")
        )
        assert flag

    def test_cot_distill(self):
        cot_config_path = os.path.join(self.base_dir, "../../../antllm/api/configs/distill/hard_target_cot.json")
        cot_config = json.load(open(cot_config_path))
        distiller = HardTargetDistill(
            teacher_model="solutions/antllm/glm-super-mini-model",
            student_model="solutions/antllm/super_mini_bart",
            reasoning_model="solutions/antllm/glm-super-mini-model",
            distill_config=cot_config
        )
        flag = distiller.train_local(
            distill_data=os.path.join(self.base_dir, "../data/dataset/model_distill/hard")
        )
        assert flag

    def test_soft_distill(self):
        output_dir = "experiments/fine-tune-local/"
        data_folder = os.path.join(
            self.base_dir, "../data/dataset/model_distill/soft"
        )
        tuner = SoftTargetDistillStudent("solutions/antllm/glm-super-mini-model",
                                         "solutions/antllm/glm-super-mini-model")

        flag = tuner.train_local(
            data_folder=data_folder,
            output_dir=output_dir,
            epoch=1
        )
        self.assertTrue(flag)
        self.assertTrue(os.path.exists(output_dir))


if __name__ == '__main__':
    main()
