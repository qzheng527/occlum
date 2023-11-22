#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : glm_evaluator.py
# @Author: xinyu.kxy
# @Date  : 2023/7/3

import os
import sys
import torch
import logging

from typing import List
from solutions.antllm.antllm.evaluation import utils
from solutions.antllm.antllm.evaluation.metrics import METRIC_CLASS_DICT  # noqa
from solutions.antllm.antllm.evaluation.evaluators.base_evaluator import BaseEvaluator
from solutions.antllm.antllm.data.dataset import *  # noqa
from torch.nn import CrossEntropyLoss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

require_cot_datasets = {"BIG-Bench-Hard", "BBH"}
GLM_DEFALF_DATASEAT = "GLMEvalGenDataset"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class GLMEvaluator(BaseEvaluator):
    """
    GLM模型评估器,加载多个评估数据集进行评估

    参数:
        -- model:评估模型
        -- tokenizer:编码器
        -- datasets_folder:多数据集共同父目录
        -- output_dir:输出路径
        -- training_args:模型训练参数
        -- dataset_name:指定需要跑的数据集名称
        -- test_file:默认的数据集评估文件，如果dataset_config中没有指定，则指向此文件
        -- batch_size:默认batch大小，如果dataset_config中没有指定，则指向此数值
        -- dataset_config:数据集配置文件，需要在其中对需要评估的数据集进行配置
        -- device_ids:gpu设备id
    """

    def __init__(self,
                 model_path,
                 datasets_folder,
                 output_dir,
                 model_size: str = "10b",
                 training_args: dict = {},
                 dataset_name=None,
                 test_file="test_prompts.1k.json",
                 batch_size: int = 4,
                 dataset_config: dict = None,
                 unidirectional: bool = False,
                 rotary_1d: bool = False,
                 device_ids=None):
        super().__init__(
            model_path=model_path,
            datasets_folder=datasets_folder,
            output_dir=output_dir,
            model_size=model_size,
            training_args=training_args,
            dataset_name=dataset_name,
            test_file=test_file,
            batch_size=batch_size,
            dataset_config=dataset_config,
            unidirectional=unidirectional,
            rotary_1d=rotary_1d,
            device_ids=device_ids
        )
        self.default_dataset = GLM_DEFALF_DATASEAT
        
    def load_model(self, model_path: str, model_size: str, device_id: int):
        model, tokenizer, max_length = utils.load_glm(model_path, model_size, device_id)
        if device_id == self.device_ids[0]:
            logging.info("max length : {}".format(max_length))
        self.special_tokens = [
            tokenizer.eop_token,
            tokenizer.sop_token,
            tokenizer.eos_token,
        ]
        return model, tokenizer, max_length

    # TODO 补充完善的分类预测逻辑
    def batch_answer(
        self,
        model,
        tokenizer,
        batch_data,
        max_new_tokens,
    ) -> List:
        outputs = model.generate(
            **batch_data,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eop_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        outputs = outputs.tolist()
        for i in range(len(outputs)):
            sop_index = outputs[i].index(tokenizer.sop_token_id)
            outputs[i] = outputs[i][sop_index + 1:]
        output = [tokenizer.decode(o) for o in outputs]
        answers = [[self._post_process(
            o, special_tokens=self.special_tokens)] for o in output]
        return answers

    # TODO 补充完善的分类预测逻辑
    def batch_answer_with_options(
        self,
        model,
        tokenizer,
        batch_data,
        batch_options,
        batch_size=4,
        extra_info=None,
        option_rank="logit",
        max_input_length=-1,
        max_output_length=-1,
        left_truncate=True,
    ) -> List:
        answers = []
        if option_rank == "logit":
            option_ids = [tokenizer.encode(
                options, add_special_tokens=False) for options in batch_options]
            output = model.generate(
                **batch_data,
                max_new_tokens=max_output_length,
                return_dict_in_generate=True,
                output_scores=True,
                eos_token_id=tokenizer.eop_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            for i in range(len(batch_options)):
                answers.append([batch_options[i][torch.argmax(
                    output.scores[0][i, [option_ids[i]]], -1).item()]])
        elif option_rank == "loss":
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
            option_normalized_loss = []
            for i in range(0, len(batch_data["input_ids"]), batch_size):
                mini_batch_data = {key: value[i: i + batch_size]
                                   for key, value in batch_data.items()}
                with torch.no_grad():
                    output = model(**mini_batch_data)
                loss = loss_fct(
                    output["logits"].view(-1, output["logits"].size(-1)),
                    mini_batch_data["labels"].view(-1),
                )
                loss = torch.sum(
                    loss.view(len(mini_batch_data["input_ids"]), -1), -1)
                option_length = torch.count_nonzero(
                    mini_batch_data["labels"] != -100, dim=-1)
                option_normalized_loss.append((loss / option_length))
            option_normalized_loss = torch.cat(option_normalized_loss)
            pre_index = 0
            for i in range(len(batch_options)):
                answers.append([batch_options[i][torch.argmin(
                    option_normalized_loss[pre_index: pre_index + len(batch_options[i])])]])
                pre_index += len(batch_options[i])
        return answers


def main():
    return
    # args = parse_args()


if __name__ == "__main__":
    main()
