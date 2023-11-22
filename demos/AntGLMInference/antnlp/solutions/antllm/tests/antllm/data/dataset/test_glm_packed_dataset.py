# !/usr/bin/env python
# coding=utf-8
# @Author: jiangpeijie.jpj
# @Date: 2023.10.17
import os
from unittest import TestCase, main

import torch
from solutions.antllm.antllm.data.dataset.glm_packed_dataset import (
    GLMPackedDataset
)
from solutions.antllm.antllm.models.glm.tokenization_glm import GLMTokenizer


class TestGLMPackedDataset(TestCase):
    def setUp(self):
        self.base_dir = os.path.dirname(__file__)
        self.tokenizer = GLMTokenizer.from_pretrained(
            os.path.join(self.base_dir, '../../../..', 'zhen_sp5/')
        )

    def test_glm_packed_dataset(self):
        data_file = os.path.join(
            self.base_dir, 'glm_packed_dataset_test_data.jsonl')
        self.dataset = GLMPackedDataset(data_file,
                                        self.tokenizer,
                                        max_input_length=10,
                                        max_output_length=10,
                                        max_length=30,
                                        no_append_glm_mask=False,
                                        gpt_data=False,
                                        world_size=1,
                                        global_rank=0,
                                        left_turncate=False,
                                        shard_data=False)
        self.assertEqual(len(self.dataset), 2)
        gt = {'input_ids': torch.Tensor([[50002, 11, 3475, 50, 50007, 50006, 11, 3475, 50, 1912,
                                          13497, 12706, 43362, 85, 50005, 132, 3475, 85, 50007, 50006,
                                          132, 3475, 85, 1912, 13497, 12706, 43362, 85, 50005, 50000]]),
              'position_ids': torch.Tensor([[[0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                              5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0],
                                             [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                              0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0]]]),
              'labels': torch.Tensor([[-100, -100, -100, -100, -100, 11, 3475, 50, 1912, 13497, 12706,
                                       43362, 85, 50005, -100, -100, -100, -100, -100, 132, 3475, 85,
                                       1912, 13497, 12706, 43362, 85, 50005, -100, -100]])}
        encoded_data = self.dataset[0]
        self.assertIn('input_ids', encoded_data)
        self.assertTrue(
            torch.all(gt['input_ids'] == encoded_data['input_ids']).item())
        self.assertIn('position_ids', encoded_data)
        self.assertTrue(
            torch.all(gt['position_ids'] == encoded_data['position_ids']).item())
        self.assertIn('labels', encoded_data)
        self.assertTrue(
            torch.all(gt['labels'] == encoded_data['labels']).item())

    def test_build_feature_from_sample(self):
        data = {"input": ["1 + 2", "1 - 3", "1 + 2"], "output": ["1 + 2 = 3, 答案是 3", "1 - 3 = -2, 答案是 -2"]}
        inputs = GLMPackedDataset.build_feature_from_sample(data, 
                                                            self.tokenizer,
                                                            max_input_length=20,
                                                            max_output_length=10,
                                                            mask_id=self.tokenizer.convert_tokens_to_ids('[gMASK]'),
                                                            for_generation=True)
        gt_inputs = {
            'input_ids': torch.Tensor([[85, 50007, 50006, 11, 449, 85, 1912, 449, 1715, 12706,
                                        43362, 449, 43372, 50005, 50002, 11, 3475, 50, 50007, 50006]]),
            'position_ids': torch.Tensor(
                [[[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 11, 12,
                   13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
                  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0,
                   0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]),
            'generation_attention_mask': torch.Tensor(
                [[[[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]]])
        }

        return_max_output_length = inputs[0]
        self.assertEqual(return_max_output_length, 10)
        inputs = dict(inputs[1])
        print('input_ids')
        print(inputs['input_ids'])
        print('position_ids')
        print(inputs['position_ids'])
        print('generation_attention_mask')
        print(inputs['generation_attention_mask'])
        self.assertIn('input_ids', inputs)
        self.assertTrue(
            torch.all(gt_inputs['input_ids'] == inputs['input_ids']).item())
        self.assertIn('position_ids', inputs)
        self.assertTrue(
            torch.all(gt_inputs['position_ids'] == inputs['position_ids']).item())
        self.assertIn('generation_attention_mask', inputs)
        self.assertTrue(
            torch.all(gt_inputs['generation_attention_mask'] == inputs['generation_attention_mask']).item())


if __name__ == '__main__':
    main()
