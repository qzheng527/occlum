import os
import unittest
from unittest import TestCase
from pathlib import Path

bazel_dir = str(Path(__file__).resolve().parent)
os.environ["HF_MODULES_CACHE"] = bazel_dir


@unittest.skip("trlx need python3.8")
class RewardDatasetTest(TestCase):
    def setUp(self):
        from transformers import AutoTokenizer  # noqa
        from solutions.antllm.antllm.data.dataset.rm_dataset.chatglm2_reward_dataset import ChatGLM2RewardDataset  # noqa

        self.base_dir = os.path.dirname(__file__)
        self.raw_dataset = [
            {
                "prompt": "意译以下句子：“客户不能用花呗收款了”。答案：",
                "chosen": "\"客户不能用花呗收款了\" 的意译是 \"Clients are no longer able to pay with芝麻信用.\"",
                "rejected": "\"Customer cannot use Mastercard or PayPal to pay.\""
            },
            {
                "prompt": "生成一个与这个意思相同的句子:“为什么我还了花呗还存在负面”。答案：",
                "chosen": "为什么我已经偿还了花呗,但仍有负面信用记录?",
                "rejected": "我生成的句子是:“为什么我还了花呗还存在负面?”。"
            },
            {
                "prompt": "意译以下句子：“花呗支付每天有额度限制么”。答案：",
                "chosen": "“没有额度限制。",
                "rejected": "没有。"
            }
        ]
        self.model_path = os.path.join(self.base_dir, '../../', 'chatglm2-6b/')
        self.max_input_len = 512
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.mask_type = "[gMASK]"
        self.mask_id = self.tokenizer.convert_tokens_to_ids(self.mask_type)
        self.dataset = ChatGLM2RewardDataset(
            self.raw_dataset,
            tokenizer=self.tokenizer,
            max_length=self.max_input_len * 2,
            max_input_length=self.max_input_len,
            return_dict=True
        )

    def check_glm_feature(self, input_ids, attention_mask, position_ids):
        mask_idx = input_ids.index(self.mask_id)
        total_len = len(input_ids)

        # check special token
        self.assertIn(self.mask_id, input_ids)

        # check attention mask
        self.assertEqual(1, attention_mask[mask_idx])

        # check position ids
        self.assertSetEqual(set([0]), set(position_ids[:mask_idx]))
        self.assertEqual(list(range(total_len - mask_idx)), position_ids[mask_idx:])

    # 测试函数，一般以test开头
    def test_dataset(self):
        self.assertEqual(len(self.raw_dataset), len(self.dataset))
        for sample in self.dataset:
            chosen_ids = sample["input_ids_chosen"].tolist()
            chosen_attention_mask = sample["attention_mask_chosen"].tolist()
            chosen_position_ids = sample["position_ids_chosen"].tolist()

            rejected_ids = sample["input_ids_rejected"].tolist()
            rejected_attention_mask = sample["attention_mask_rejected"].tolist()
            rejected_position_ids = sample["position_ids_rejected"].tolist()

            self.check_glm_feature(chosen_ids, chosen_attention_mask, chosen_position_ids)
            self.check_glm_feature(rejected_ids, rejected_attention_mask, rejected_position_ids)


if __name__ == '__main__':
    unittest.main()