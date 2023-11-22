#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"

from unittest import TestCase, main

import torch

from solutions.antllm.antllm.models.glm.configuration_glm import GLMConfig
from solutions.antllm.antllm.models.glm.modeling_glm import GLMModel


class TestGLMModel(TestCase):
    def setUp(self):
        pass

    def test_gqa(self):
        config = GLMConfig(
            num_layers=6,
            vocab_size=1000,
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=2
        )
        model = GLMModel(config)
        input_ids = torch.randint(0, 999, (2, 512))
        output = model(input_ids)
        self.assertTrue(isinstance(output.last_hidden_states, torch.Tensor))


if __name__ == "__main__":
    main()
