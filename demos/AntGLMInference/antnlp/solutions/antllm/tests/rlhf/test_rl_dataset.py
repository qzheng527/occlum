import os
import unittest
from unittest import TestCase
from pathlib import Path

bazel_dir = str(Path(__file__).resolve().parent)
os.environ["HF_MODULES_CACHE"] = bazel_dir


@unittest.skip("trlx needs python3.8")
class PPODatasetTest(TestCase):
    def setUp(self):
        from solutions.antllm.antllm.data.dataset.rl_dataset.offline_pipeline import GlmPipeline  # noqa
        from transformers import AutoTokenizer  # noqa
        basepath = "solutions"
        self.model_path = os.path.join(basepath, "antllm/tests/data/glm-test-model")
        self.prompts = [
            "你能提供“小”的5个反义词吗？",
            "西班牙是否有权力分立？",
            "我希望您充当虚拟助理，我会给您指令，您将自动为我执行它们。"
        ]
        self.max_len = 512
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.dataset = GlmPipeline(
            self.prompts,
            max_prompt_length=self.max_len,
            tokenizer=self.tokenizer
        )

    # 测试函数，一般以test开头
    def test_dataset(self):
        self.assertEqual(len(self.prompts), len(self.dataset))
        for idx, sample in enumerate(self.dataset.prompts):
            input_ids = sample["input_ids"]
            self.assertLess(len(input_ids), self.max_len)


if __name__ == '__main__':
    unittest.main()