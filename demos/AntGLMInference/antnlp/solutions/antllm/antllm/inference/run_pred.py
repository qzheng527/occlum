#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : run_pred.py
# @Author: xinyu.kxy
# @Date  : 2023/3/16

import argparse
import json
import os
import shutil
import sys
import time

import torch
# from pathos.multiprocessing import ProcessingPool as Pool
from torch.multiprocessing import Process

# import multiprocessing as mp
from solutions.antllm.antllm.inference.glm_predictor import GLMForInference
DEFAULT_TEST_FILE = "test_prompts.1k.json"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


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
        help="数据集路径,路径下包括以各数据集名称命名的各文件夹,代码会默认在同级目录生成tmp文件夹保存切割数,tmp文件夹最终会删除",
        required=True,
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
        "--method",
        type=str,
        choices=["k-shot", "finetune", ""],
        default="k-shot",
        help="指定评估数据集的类型,其中k-shot包含zero shot和few shot"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="预测结果文件夹输出路径",
        required=True,
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default='solutions/antllm/antllm/evaluation/configs/datasets_des.json',
        help="可选参数：dataset的配置文件。默认将使用solutions/antllm/antllm/evaluation/configs/datasets_des.json"
    )
    args = parser.parse_args()
    return args


def data_split(datasets_path, output_path, dataset_name, method, test_file, dataset_config, gpu_available):
    """
    数据集切割到tmp文件夹
    """
    output_path_tmp = os.path.join(output_path, "tmp")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if os.path.exists(output_path_tmp):
        shutil.rmtree(output_path_tmp, ignore_errors=True)
    os.mkdir(output_path_tmp)
    if method == "k-shot":
        method = "shot"
    for dataset in os.listdir(datasets_path):
        if not os.path.isdir(os.path.join(datasets_path, dataset)) or  \
                dataset not in dataset_config or \
                (not dataset_name and dataset_config[dataset]["is_key_dataset"] is False):
            continue
        if dataset_name:
            if dataset not in dataset_name:
                continue
        if method:
            if method not in dataset_config[dataset]["method"]:
                continue
        for data in os.listdir(os.path.join(datasets_path, dataset)):
            if test_file == DEFAULT_TEST_FILE:
                dataset_test_file = dataset_config[dataset].get(
                    'test_file', DEFAULT_TEST_FILE)
            else:
                dataset_test_file = test_file
            if data == dataset_test_file:
                with open(os.path.join(os.path.join(datasets_path, dataset), data), "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    gpu_num = len(gpu_available)
                    for i, gpu_idx in enumerate(gpu_available):
                        tmp_file = os.path.join(
                            output_path_tmp, "gpu_{}".format(gpu_idx))
                        output_file = os.path.join(tmp_file, dataset)
                        if not os.path.exists(tmp_file):
                            os.mkdir(tmp_file)
                        with open(output_file, "w", encoding="utf-8") as fout:
                            for line in lines[i::gpu_num]:
                                fout.write(line)
    return output_path_tmp


def data_merge(output_path):
    """
    多卡预测的数据集结果合并
    """
    output_path_tmp = os.path.join(output_path, "tmp")
    for dataset in os.listdir(os.path.join(output_path_tmp, os.listdir(output_path_tmp)[0])):
        if dataset.endswith("_out"):
            data = []
            for tmp_file in os.listdir(output_path_tmp):
                tmp_file = os.path.join(output_path_tmp, tmp_file)
                with open(os.path.join(tmp_file, dataset), "r", encoding="utf-8") as f:
                    data.extend(f.readlines())
            with open(os.path.join(output_path, dataset.replace("_out", "")), "w", encoding="utf-8") as f:
                for line in data:
                    f.write(line)


def predict(
    gpu,
    model_path,
    dataset_name,
    output_path_tmp,
    dataset_config,
    max_batch_size=16
):
    require_cot_datasets = {"BIG-Bench-Hard", "BBH"}
    require_loss_datasets = {"Gov_Report", "GR"}

    print("***  loading model on GPU {}...  ***".format(gpu))
    model = GLMForInference(model_path, gpu_index=gpu)
    tmp_folder = os.path.join(output_path_tmp, "gpu_{}".format(gpu))
    num_datasets = len(os.listdir(tmp_folder))
    start = time.time()
    for dataset_index, dataset in enumerate(os.listdir(tmp_folder)):
        if dataset_name:
            if dataset not in dataset_name:
                continue
        assert dataset in dataset_config, "{} not in expected datasets!".format(
            dataset)
        likelihood = False
        require_cot = dataset in require_cot_datasets
        require_loss = dataset in require_loss_datasets
        if "CalibrationError" in dataset_config[dataset].get("metric", {}):
            likelihood = True
        print("***  predicting {} on GPU {}  ***".format(dataset, gpu))
        fin = open(os.path.join(tmp_folder, dataset), "r", encoding="utf-8")
        lines = fin.readlines()
        grouped_lines = {'options_data': [], 'others_data': []}
        max_input_length = 0
        max_output_length = 0
        for line in lines:
            data = json.loads(line.rstrip('\n\r'))
            input_str = data['input']
            input_tokens = model.tokenizer(
                input_str, add_special_tokens=False)['input_ids']
            max_input_length = max_input_length if len(
                input_tokens) < max_input_length else len(input_tokens)

            if 'options' in data:
                for option in data['options']:
                    option_tokens = model.tokenizer(
                        option, add_special_tokens=False)['input_ids']
                    max_output_length = max_output_length if len(
                        option_tokens) < max_output_length else len(option_tokens)
                grouped_lines['options_data'].append(line)
            elif 'references' in data:
                for ref in data['references']:
                    output_tokens = model.tokenizer(
                        ref, add_special_tokens=False)['input_ids']
                    max_output_length = max_output_length if len(
                        output_tokens) < max_output_length else len(output_tokens)
                grouped_lines['others_data'].append(line)
            else:
                max_output_length = len(model.tokenizer(
                    data['output'], add_special_tokens=False)['input_ids'])
                grouped_lines['others_data'].append(line)                
            batch_size = max_batch_size
        print(
            f'gpu: {gpu}, dataset: {dataset}, max_input_length: {max_input_length}')
        print(
            f'gpu: {gpu}, dataset: {dataset}, max_output_length: {max_output_length}')
        print(f'gpu: {gpu}, dataset: {dataset}, batch_size: {batch_size}')

        start = time.time()
        ori_start = start
        with open(os.path.join(tmp_folder, dataset), "r", encoding="utf-8") as fin, open(
                os.path.join(tmp_folder, dataset + "_out"), "w", encoding="utf-8") as fout:
            for _, lines in grouped_lines.items():
                if max_batch_size < len(lines) // 4:
                    batch_size = max_batch_size
                else:
                    batch_size = len(lines) // 4
                print(
                    f'gpu: {gpu}, dataset: {dataset}, batch_size: {batch_size}')
                datas = []
                index = 0
                total_lines = len(lines)
                start = time.time()
                while index < total_lines + 1:
                    if (len(datas) >= batch_size or index == len(lines)) and len(datas) > 0:
                        batch_size = len(datas)
                        while batch_size >= 1:
                            try:
                                if not require_cot and "options" in datas[-1]:
                                    data_outs = model.batch_answer_with_options(
                                        datas,
                                        batch_size,
                                        max_input_length=max_input_length,
                                        max_output_length=max_output_length,
                                        likelihood=likelihood,
                                        option_rank=dataset_config[dataset].get("option_rank", "loss"),
                                        left_truncate=True)
                                elif require_loss:
                                    data_outs = model.batch_answer_with_loss(
                                        datas,
                                        max_input_length=max_input_length,
                                        max_output_length=max_output_length,         
                                        left_truncate=True                               
                                    )
                                    cost = time.time() - start
                                else:
                                    data_outs = model.batch_answer(
                                        datas,
                                        max_output_length=max(
                                            max_output_length * 2, 1000 if require_cot else 100
                                        ),
                                        left_truncate=True,
                                    )
                                    cost = time.time() - start
                                for data in data_outs:
                                    fout.write(json.dumps(
                                        data, ensure_ascii=False) + '\n')
                                    fout.flush()
                                if index % (batch_size * 5) == 0:
                                    print(
                                        f'gpu: {gpu}, dataset: {dataset},'
                                        f' processed {index}/{total_lines}, cost {cost}')
                                break
                            except Exception as e:
                                if 'out of memory' in str(e).lower():
                                    old_batch_size = batch_size
                                    batch_size = batch_size // 2
                                    print(
                                        f'gpu: {gpu}, dataset: {dataset}, \
                                                decrease batch size from {old_batch_size} to {batch_size}')
                                    datas = datas[:batch_size]
                                    index = index - \
                                        (old_batch_size - batch_size)
                                else:
                                    print(f'gpu: {gpu} ', e)
                                    raise Exception
                        if batch_size == 0:
                            break
                        datas = []
                    if index == len(lines):
                        break
                    line = lines[index]
                    data = json.loads(line.rstrip('\n\r'))
                    datas.append(data)
                    cost = time.time() - start
                    index += 1
                    cost = time.time() - start
        cost = time.time() - ori_start
        print(
            f'gpu: {gpu}, processed {dataset_index} / {num_datasets}, cost {cost}')


def multiGPU_predict(
        model_path,
        dataset_name,
        output_path_tmp,
        dataset_config,
        gpu_available,
        max_batch_size=16):

    if len(gpu_available) == 1:
        predict(gpu_available[0], model_path, dataset_name, output_path_tmp, dataset_config, max_batch_size)
    else:
        # pool = Pool(len(gpu_available))
        # pool.map(predict, gpu_available)
        processes = []
        for rank in gpu_available:
            p = Process(target=predict, 
                        args=(rank, model_path, dataset_name, output_path_tmp, dataset_config, max_batch_size))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


def main():
    args = parse_args()
    model_path = args.model_folder
    datasets_path = args.datasets_folder
    dataset_name = args.dataset_name.split(
        ",") if args.dataset_name != "" else []
    output_path = args.output_folder
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    shutil.copytree('solutions/antllm',
                    os.path.join(output_path, 'antllm'), dirs_exist_ok=True)
    test_file = args.test_file
    batch_size = args.batch_size
    method = args.method
    dataset_config_path = args.dataset_config
    gpu_available = list(map(int, args.gpu.split(",")))
    print("*** use gpu on cuda:{} ***".format(args.gpu))
    dataset_config = json.load(open(dataset_config_path))
    output_path_tmp = data_split(datasets_path, output_path, dataset_name, method, test_file, dataset_config,
                                 gpu_available)  # 按照可用gpu数量切割数据集
    start = time.time()
    multiGPU_predict(model_path, dataset_name, output_path_tmp,
                     dataset_config, gpu_available, max_batch_size=batch_size)  # 多卡跑结果
    cost = time.time() - start
    print(f'gpu predict cost {cost}')
    data_merge(output_path)  # 数据结果合并


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
