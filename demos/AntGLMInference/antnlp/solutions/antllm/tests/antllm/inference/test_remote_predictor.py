# coding=utf-8
# @Author: xinyu.kxy
# @Date: 2023-06-08
import pytest
from unittest import TestCase, main

from solutions.antllm.antllm.inference.remote_predictor import RemoteInference


class TestGLMInference(TestCase):
    """
    依赖于线上服务，暂不进行此单测
    """

    def setUp(self):
        scene_name = "lx_platform"
        chain_name = "antglm_5b"
        self.remote_predictor = RemoteInference(scene_name, chain_name)

    @pytest.mark.skip(reason="depends on remote service")
    def test_remote_anwers(self):
        query = "今天天气不错"
        adapter_path = "linxi_test_0612_7"
        answers = self.remote_predictor.remote_answer(
            query=query, adapter_path=adapter_path)

        self.assertTrue(answers.texts)


if __name__ == "__main__":
    main()
