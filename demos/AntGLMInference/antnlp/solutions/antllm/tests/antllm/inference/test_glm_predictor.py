# coding=utf-8
# @Author: xinyu.kxy
# @Date: 2023-06-08
import os
import pytest
from unittest import TestCase, main

from solutions.antllm.antllm.inference.glm_predictor import GLMForInference


class TestGLMInference(TestCase):
    """
    aci时间过长,暂不进行此单测
    """

    def setUp(self):
        self.base_dir = os.path.dirname(__file__)
        self.evaluator = GLMForInference(os.path.join(
            self.base_dir, "../../..", "glm-super-mini-model"))

    def test_multi_answers(self):
        answers = self.evaluator.generate('北京是中国首都吗？', num_return_sequences=2, num_beams=2,
                                          max_output_tokens=5)
        self.assertEqual(len(answers.texts), 2)
        self.assertEqual(len(answers.finish_reasons), 2)

    def test_expected_assertion_path_check(self):
        with pytest.raises(AssertionError):
            GLMForInference("no existing path")

    @pytest.mark.skip(reason="aci timeout")
    def test_answer(self):
        """
        模型预测
        """
        answer = self.evaluator.answer("北京在哪里")
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)

        answer = self.evaluator.answer("北京在哪里" * 500)  # 测试超长
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)

    @pytest.mark.skip(reason="aci timeout")
    def test_batch_answer(self):
        """
        模型批量预测
        """
        answers = self.evaluator.batch_answer(
            [
                {"input": "你在哪里"},
                {"input": "月亮在哪里" * 500},  # 测试超长
            ]
        )
        self.assertEqual(len(answers), 2)
        self.assertIsInstance(answers[0]["predictions"], list)

    @pytest.mark.skip(reason="aci timeout")
    def test_answer_with_options(self):
        """
        模型批量分类预测
        """
        answers = self.evaluator.batch_answer_with_options(
            [
                {"input": "你在哪里", "options": ["床前明月光", "在北京"]},
                {"input": "月亮在哪里", "options": ["地球里面", "地球旁边"]},
                {"input": "月亮在哪里" * 500,
                    "options": ["地球里面" * 500, "地球旁边" * 500]},  # 测试超长
            ],
            likelihood=True,  # 测试选项分数
        )
        self.assertEqual(len(answers), 3)
        self.assertIsInstance(answers[0]["likelihood"], dict)

    @pytest.mark.skip(reason="aci timeout")
    def test_stream_answer(self):
        """
        模型流式预测
        """
        generator = self.evaluator.generate_stream("北京在哪里")
        for token in generator:
            self.assertIsInstance(token, str)
            self.assertGreater(len(token), 0)


if __name__ == "__main__":
    main()
