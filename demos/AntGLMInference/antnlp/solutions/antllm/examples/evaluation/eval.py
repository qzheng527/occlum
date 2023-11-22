#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : evaluate.py
# @Author: xinyu.kxy
# @Date  : 2023/3/16

import argparse
import json
import os
import sys
import time
import logging
import torch.multiprocessing as mp
from solutions.antllm.antllm.evaluation.evaluators.base_evaluator import BaseEvaluator
from solutions.antllm.antllm.evaluation.evaluators.glm_evaluator import GLMEvaluator
# from solutions.antllm.antllm.evaluation import utils
from solutions.antllm.antllm.data.dataset import *  # noqa

DEFAULT_TEST_FILE = "test_prompts.1k.json"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description="config for prediction")
    parser.add_argument(
        "--model_folder",
        type=str,
        help="模型路径",
        required=True,
    )
    parser.add_argument(
        "--datasets_folder",
        type=str,
        help="数据集路径,路径下包括以各数据集名称命名的各文件夹",
        required=True,
        default="eval_data",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=False,
        help="默认跑目录下的全部数据集,也可以指定某些数据集名称,用逗号分隔即可, e.g. CEval,MMLU,AGIEval",
        default="",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=False,
        help="数据集下面需要跑的数据文件,默认为核心数据集中的1000个采样数据",
        default=DEFAULT_TEST_FILE,
    )
    parser.add_argument(
        "--gpu",
        type=str,
        required=False,
        help="指定可用的gpu机器资源, e.g. 0,1,2,3,4,5,6,7",
        default="0",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        help="批量预测的批次数量",
        default=16,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        choices=["glm", "llama2"],
        default="glm",
        help="指定评测模型的类型"
    )
    parser.add_argument(
        "--model_size",
        type=str,
        required=False,
        choices=["10b", "65b"],
        default="10b",
        help="指定评测模型的大小"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="预测结果文件夹输出路径",
        default="./"
    )
    parser.add_argument(
        "--unidirectional",
        action="store_true",
        help="编码模式是否使用单向"
    )
    parser.add_argument(
        "--rotary_1d",
        action="store_true",
        help="是否使用1d rotary"
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=False,
        default='solutions/antllm/antllm/evaluation/configs/datasets_des.json',
        help="可选参数：dataset的配置文件。默认将使用solutions/antllm/antllm/evaluation/configs/datasets_des.json"
    )
    args = parser.parse_args()
    return args


def evaluate(args):
    mp.set_start_method('spawn', force=True)
    model_path = args.model_folder
    datasets_path = args.datasets_folder
    dataset_name = args.dataset_name.split(
        ",") if args.dataset_name != "" else []
    output_path = args.output_folder
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    test_file = args.test_file
    batch_size = args.batch_size
    model_name = args.model_name
    unidirectional = args.unidirectional
    rotary_1d = args.rotary_1d
    dataset_config_path = args.dataset_config
    device_ids = list(map(int, args.gpu.split(",")))
    logging.info("*** gpu device -- cuda:{} ***".format(args.gpu))
    dataset_config = json.load(open(dataset_config_path))
    start = time.time()

    if model_name == 'glm':
        Evaluator = GLMEvaluator
    else:
        Evaluator = BaseEvaluator

    evaluator = Evaluator(
        model_path=model_path,
        datasets_folder=datasets_path,
        output_dir=output_path,
        model_size=args.model_size,
        dataset_name=dataset_name,
        test_file=test_file,
        batch_size=batch_size,
        dataset_config=dataset_config,
        unidirectional=unidirectional,
        rotary_1d=rotary_1d,
        device_ids=device_ids
    )
    eval_result = evaluator.evaluate()
    cost = time.time() - start
    logging.info(f'gpu predict cost {cost}')
    return eval_result


def main():
    args = parse_args()
    # dist.init_process_group(backend='nccl',rank=0, world_size=1)
    eval_result = evaluate(args)
    print(eval_result)


if __name__ == "__main__":
    main()