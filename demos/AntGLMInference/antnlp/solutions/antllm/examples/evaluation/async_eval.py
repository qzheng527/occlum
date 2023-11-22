#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : evaluate.py
# @Author: xinyu.kxy
# @Date  : 2023/3/16

import argparse
import os
import sys
import time
import logging
import torch
import shutil
import json
from torch.utils.tensorboard import SummaryWriter
# from solutions.antllm.antllm.evaluation.evaluators.base_evaluator import BaseEvaluator
# from solutions.antllm.antllm.evaluation.evaluators.glm_evaluator import GLMEvaluator
# from solutions.antllm.antllm.evaluation import utils
from solutions.antllm.examples.evaluation.eval import evaluate
from solutions.antllm.antllm.data.dataset import *  # noqa

DEFAULT_TEST_FILE = "test_prompts.ppl5.json"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description="config for prediction")
    parser.add_argument(
        "--train_save_dir",
        type=str,
        help="训练时的模型保存路径",
        required=True,
    )
    parser.add_argument(
        "--datasets_folder",
        type=str,
        help="数据集路径,路径下包括以各数据集名称命名的各文件夹",
        required=True,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=False,
        help="默认跑目录下的全部数据集,也可以指定某些数据集名称,用逗号分隔即可, e.g. CEval,MMLU,AGIEval",
        default="MMLU",
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
        "--unidirectional",
        action='store_true',
        help="编码模式是否使用单向"
    )
    parser.add_argument(
        "--rotary_1d",
        action='store_true',
        help="是否使用1d rotary"
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
        "--dataset_config",
        type=str,
        required=False,
        default='solutions/antllm/antllm/evaluation/configs/datasets_des.json',
        help="可选参数：dataset的配置文件。默认将使用solutions/antllm/antllm/evaluation/configs/datasets_des.json"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="预测结果文件夹输出路径",
        default="./"
    )
    parser.add_argument(
        "--re_run",
        action="store_true",
        help="默认从当前时间点开始进行模型文件监控，评估后续生成的模型，如果设置此项参数，则重新评估全部生成模型文件并继续进行监控"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=False,
        help="数据集下面需要跑的数据文件,默认为核心数据集中的1000个采样数据",
        default=DEFAULT_TEST_FILE,
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="预训练模型结果文件中不包含tokenizer等文件，需要从另外文件中复制过来",
        default="/ossfs/workspace/nas_new/chatgpt/models_0602/glm/AntGLM-10B-20230602"
    )
    args = parser.parse_args()
    return args


def convert2hf(path_in, path_out):
    if not os.path.exists(path_out):
        os.mkdir(path_out)
    state = torch.load(os.path.join(path_in, "mp_rank_00_model_states.pt"))
    torch.save(state['module'], os.path.join(path_out, "pytorch_model.bin"))


def is_convertible_to_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def numerical_sort(string):
    # 提取字符串中的数字部分并转换为整数
    try:
        string = int(string)
    except Exception:
        string = -1
    return string


def occupy_gpu(device_ids, occupy_time):
    # 为了确保 PyTorch 使用 GPU
    print('occupying gpu:', device_ids)
    tensors = []
    for device in device_ids:
        device = torch.device('cuda:{}'.format(device))
        # 创建一个大型的随机张量，占用 GPU 存储
        tensors.append(torch.randn(60000, 60000).to(device))

    start_time = time.time()  # 获取循环开始的时间
    while time.time() - start_time < occupy_time:
        # 执行一些计算任务，例如矩阵乘法
        for tensor in tensors:
            result = tensor.matmul(tensor)  # noqa
        # 在任务执行完成后，等待一段时间，以便 GPU 仍然保持利用率
        time.sleep(0.001)


def log2tensorboard(eval_result: dict, writer, step):
    for dataset, metrics in eval_result.items():
        if isinstance(metrics, dict):
            for metric, res in metrics.items():
                if isinstance(res, dict):
                    for i, j in res.items():
                        if j is not None:
                            writer.add_scalar(
                                'dataset/{}/{}'.format(dataset, i), j, step)
                elif res is not None:
                    writer.add_scalar(
                        'dataset/{}'.format(dataset), res, step)
        elif isinstance(metrics, float):
            writer.add_scalar(
                '{}'.format(dataset), metrics, step)


def copy_folder(source_folder, destination_folder):
    try:
        # 检查目标文件夹是否存在，如果不存在则创建
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # 遍历源文件夹下的所有文件和子文件夹
        for item in os.listdir(source_folder):
            source_item = os.path.join(source_folder, item)
            if source_item.startswith("pytorch_model") or source_item.startswith("mp_rank"):
                continue
            destination_item = os.path.join(destination_folder, item)

            # 判断是文件还是文件夹
            if os.path.isfile(source_item):
                # 复制文件
                shutil.copy2(source_item, destination_item)
            elif os.path.isdir(source_item):
                # 递归调用自身，复制子文件夹
                copy_folder(source_item, destination_item)

        print("文件夹复制完成！")
    except Exception as e:
        print(f"复制文件夹失败：{e}")


def main():
    args = parse_args()
    save_dir = args.train_save_dir
    # 初始文件夹列表
    initial_folders = [] if args.re_run else os.listdir(save_dir)
    output_folder = args.output_folder
    # dataset_config = json.load(open(args.dataset_config))
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    writer = SummaryWriter(output_folder)

    while True:
        # 获取当前目录下的所有文件夹
        current_folders = sorted(os.listdir(save_dir), key=numerical_sort)

        # 检查是否有新的文件夹生成
        new_folders = [
            folder for folder in current_folders if folder not in initial_folders]

        # 如果有新的文件夹生成，打印出它们的名字
        if new_folders:
            print("New folders detected:{}".format(new_folders))
            for folder in new_folders:
                if not is_convertible_to_int(folder):
                    continue
                print("evaluating...", folder)
                convert_folder = os.path.join(output_folder, folder)
                if folder in []:
                    eval_res = json.load(
                        open(os.path.join(convert_folder, "eval_result.json")))
                    log2tensorboard(eval_res, writer, folder)
                    continue
                if not os.path.exists(convert_folder):
                    print("converting...")
                    copy_folder(args.base_model_path, convert_folder)
                    try:
                        convert2hf(os.path.join(save_dir, folder), convert_folder)
                    except Exception:
                        time.sleep(600)
                        convert2hf(os.path.join(save_dir, folder), convert_folder)
                else:
                    continue
                args.model_folder = convert_folder
                args.output_folder = convert_folder
                result = evaluate(args)
                log2tensorboard(result, writer, folder)
                print("*" * 10 + " {} ".format(folder) + "*" * 10)
                print(result)
        # 更新初始文件夹列表
        initial_folders = current_folders

        # 休眠10分钟后再次检查
        torch.cuda.empty_cache()
        print("sleeping...")
        time.sleep(600)
        # occupy_gpu(list(map(int, args.gpu.split(","))), 600)


if __name__ == "__main__":
    main()
