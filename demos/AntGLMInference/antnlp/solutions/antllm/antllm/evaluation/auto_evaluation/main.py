#!/usr/bin/env python
# coding=utf-8
# @Author: xinyu.kxy
# @Date: Fri 17 July 2023 09:22:56 PM CST

import json
import argparse
from solutions.antllm.antllm.evaluation.auto_evaluation import utils
# from solutions.antllm.antllm.evaluation.auto_evaluation.annotators import pairwise_evaluator,single_evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="config for auto evaluation")
    parser.add_argument(
        "--data_folder",
        type=str,
        help="评估数据目录,目录中需要包含config.yaml,数据,prompt三种文件",
        default="solutions/antllm/antllm/evaluation/auto_evaluation/dataset_configs/antllm_qualitative",
        required=False,
    )
    parser.add_argument(
        "--result_folder",
        type=str,
        default="solutions/antllm/antllm/evaluation/auto_evaluation/dataset_configs/antllm_qualitative_outs",
        help="结果数据目录",
        required=False,
    )

    args = parser.parse_args()

    return args


def ana_json_config(config_file):
    config = json.load(open(config_file))
    return config


def main():
    args = parse_args()
    data_folder = args.data_folder
    result_folder = args.result_folder

    annotator_config = utils.load_configs(data_folder)
    Annotator = utils.get_annotator_by_name(annotator_config["annotator"])
    annotator = Annotator(data_folder, **annotator_config)
    annotator.annotate(result_folder)


if __name__ == "__main__":
    main()
