# !/usr/bin/env python
# coding=utf-8
# @Author: tianxuan.jl
# @Date: Thu 27 Apr 2023 02:10:23 PM CST
import os
from unittest import TestCase, main

import torch
from solutions.antllm.antllm.data.dataset.glm_instruction_dataset import (
    GLMInstructionDataset
)
from solutions.antllm.antllm.models.glm.tokenization_glm import GLMTokenizer


class TestGLMInstructionDataset(TestCase):
    def setUp(self):
        self.base_dir = os.path.dirname(__file__)
        self.tokenizer = GLMTokenizer.from_pretrained(
            os.path.join(self.base_dir, '../../../..', 'zhen_sp5/')
        )

    def test_glm_instrunction_dataset(self):
        data_file = os.path.join(
            self.base_dir, 'glm_instruction_dataset_test_data.jsonl')
        self.dataset = GLMInstructionDataset(data_file,
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
        self.assertEqual(len(self.dataset), 1)
        gt = {'input_ids': torch.Tensor([[50002, 7272, 21409, 1247, 20743, 43383, 43659, 39025, 15458, 43384,
                                          10683, 21396, 6990, 23100, 50007, 50006, 43358, 43358, 3, 160,
                                          6793, 24537, 15458, 4401, 44506, 16213, 18335, 43359, 525, 50005]]),
              'position_ids': torch.Tensor([[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 14,
                                              14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2,
                                              3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]]),
              'attention_mask': torch.Tensor([15]),
              'labels': torch.Tensor([[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                                       -100, -100, -100, -100, -100, 43358, 43358, 3, 160, 6793,
                                       24537, 15458, 4401, 44506, 16213, 18335, 43359, 525, 50005, -100]])}
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
        data = {'input': '北京在哪里', 'output': '在中国北部'}
        inputs = GLMInstructionDataset.build_feature_from_sample(data, self.tokenizer,
                                                                 max_input_length=10,
                                                                 max_output_length=10,
                                                                 mask_id=self.tokenizer.convert_tokens_to_ids(
                                                                     '[gMASK]'),
                                                                 for_generation=True)
        gt_inputs = {'input_ids': torch.Tensor([[50002, 2531, 3551, 50007, 50006]]),
                     'position_ids': torch.Tensor([[[0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                                                    [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]]),
                     'generation_attention_mask': torch.Tensor([[[[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                  [1, 1, 1, 1, 0, 0, 0, 0,
                                                                      0, 0, 0, 0, 0, 0, 0],
                                                                  [1, 1, 1, 1, 0, 0, 0, 0,
                                                                      0, 0, 0, 0, 0, 0, 0],
                                                                  [1, 1, 1, 1, 0, 0, 0, 0,
                                                                      0, 0, 0, 0, 0, 0, 0],
                                                                  [1, 1, 1, 1, 1, 0, 0, 0,
                                                                      0, 0, 0, 0, 0, 0, 0],
                                                                  [1, 1, 1, 1, 1, 1, 0, 0,
                                                                      0, 0, 0, 0, 0, 0, 0],
                                                                  [1, 1, 1, 1, 1, 1, 1, 0,
                                                                      0, 0, 0, 0, 0, 0, 0],
                                                                  [1, 1, 1, 1, 1, 1, 1, 1,
                                                                      0, 0, 0, 0, 0, 0, 0],
                                                                  [1, 1, 1, 1, 1, 1, 1, 1,
                                                                      1, 0, 0, 0, 0, 0, 0],
                                                                  [1, 1, 1, 1, 1, 1, 1, 1,
                                                                      1, 1, 0, 0, 0, 0, 0],
                                                                  [1, 1, 1, 1, 1, 1, 1, 1,
                                                                      1, 1, 1, 0, 0, 0, 0],
                                                                  [1, 1, 1, 1, 1, 1, 1, 1,
                                                                      1, 1, 1, 1, 0, 0, 0],
                                                                  [1, 1, 1, 1, 1, 1, 1, 1,
                                                                      1, 1, 1, 1, 1, 0, 0],
                                                                  [1, 1, 1, 1, 1, 1, 1, 1,
                                                                      1, 1, 1, 1, 1, 1, 0],
                                                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]]])}
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

        gt_inputs = {'input_ids': [50002, 2531, 3551, 50007, 50006, 14041, 6759, 50005, 50000, 50000, 50000, 50000, 50000, 50000, 50000, 50000, 50000, 50000, 50000, 50000],  # noqa
                     'position_ids': [[0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                                      [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]],
                     'attention_mask': 4, 'labels': [-100, -100, -100, -100, 14041, 6759, 50005, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]}  # noqa
        inputs = GLMInstructionDataset.build_feature_from_sample(data, self.tokenizer,
                                                                 max_length=20,
                                                                 mask_id=self.tokenizer.convert_tokens_to_ids(
                                                                     '[gMASK]'))
        print(type(inputs))
        self.assertIn('input_ids', inputs)
        self.assertTrue(
            gt_inputs['input_ids'] == inputs['input_ids'])
        self.assertIn('position_ids', inputs)
        self.assertTrue(
            gt_inputs['position_ids'] == inputs['position_ids'])
        self.assertIn('attention_mask', inputs)
        self.assertTrue(
            gt_inputs['attention_mask'] == inputs['attention_mask'])
        self.assertIn('labels', inputs)
        self.assertTrue(
            gt_inputs['labels'] == inputs['labels'])


if __name__ == '__main__':
    main()
