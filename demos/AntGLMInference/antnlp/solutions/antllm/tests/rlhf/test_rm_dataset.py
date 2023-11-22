import os
import unittest
from unittest import TestCase
from pathlib import Path

bazel_dir = str(Path(__file__).resolve().parent)
os.environ["HF_MODULES_CACHE"] = bazel_dir


@unittest.skip("trlx need python3.8")
class RewardDatasetTest(TestCase):
    def setUp(self):
        from solutions.antllm.antllm.models.glm.tokenization_glm import GLMTokenizer  # noqa
        from solutions.antllm.antllm.data.dataset.rm_dataset.reward_dataset import GLMRewardDataset  # noqa

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
        self.model_path = os.path.join(self.base_dir, '../../', 'zhen_sp5/')
        self.max_input_len = 512
        self.tokenizer = GLMTokenizer.from_pretrained(self.model_path)
        self.mask_type = "[gMASK]"
        self.mask_id = self.tokenizer.convert_tokens_to_ids(self.mask_type)
        self.dataset = GLMRewardDataset(
            self.raw_dataset,
            tokenizer=self.tokenizer,
            max_length=self.max_input_len * 2,
            max_input_length=self.max_input_len,
            mask=self.mask_type,
            return_dict=True
        )

    def check_glm_feature(self, input_ids, attention_mask, position_ids):
        mask_idx = input_ids.index(self.mask_id)
        total_len = len(input_ids)

        # check special token
        self.assertIn(self.mask_id, input_ids)
        self.assertEqual(self.tokenizer.cls_token_id, input_ids[0])
        self.assertIn(self.tokenizer.sop_token_id, input_ids)
        self.assertIn(self.tokenizer.eop_token_id, input_ids)

        # check attention mask
        self.assertEqual(mask_idx + 1, attention_mask[0])

        # check position ids
        self.assertEqual(list(range(mask_idx)), position_ids[0][:mask_idx])
        self.assertSetEqual(set([mask_idx]), set(position_ids[0][mask_idx:]))
        self.assertSetEqual(set([0]), set(position_ids[1][:mask_idx]))
        self.assertEqual(list(range(total_len - mask_idx)), position_ids[1][mask_idx:])

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