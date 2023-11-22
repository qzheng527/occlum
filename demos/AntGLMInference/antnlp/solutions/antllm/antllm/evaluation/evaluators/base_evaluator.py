#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : glm_evaluator.py
# @Author: xinyu.kxy
# @Date  : 2023/7/3

import json
import os
import sys
import torch
import logging
import time
# import concurrent.futures
from multiprocessing import Manager
from torch.multiprocessing import Process
from typing import List
from tqdm import tqdm
from solutions.antllm.antllm.evaluation import utils
from solutions.antllm.antllm.evaluation.metrics import METRIC_CLASS_DICT  # noqa
from solutions.antllm.antllm.data.dataset import *  # noqa
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

require_cot_datasets = {"BIG-Bench-Hard", "BBH"}

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class BaseEvaluator:
    """
    通用模型评估器,加载多个评估数据集进行评估

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
                 model_path: str,
                 datasets_folder: str,
                 output_dir: str,
                 model_size: str = "10b",
                 training_args: dict = {},
                 dataset_name: str = None,
                 test_file: str = "test_prompts.1k.json",
                 batch_size: int = 4,
                 dataset_config: dict = None,
                 unidirectional: bool = False,
                 rotary_1d: bool = False,
                 device_ids: list = None):
        self.model_path = model_path
        self.datasets_folder = datasets_folder
        self.model_size = model_size
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.device_ids = device_ids
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.test_file = test_file
        self.unidirectional = unidirectional
        self.training_args = training_args
        self.rotary_1d = rotary_1d
        self.special_tokens = []
        self.default_dataset = "LlamaEvalDataset"
        self.init_res()

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    def load_model(self, model_path: str, model_size: str, device_id: int):
        model, tokenizer, max_length = utils.load_llama(
            model_path, device_id)
        if device_id == self.device_ids[0]:
            logging.info("max length : {}".format(max_length))
        return model, tokenizer, max_length

    def load_dataloader(
            self,
            datasets_folder: str,
            dataset: str,
            batch_size: int,
            device_id: int,
            tokenizer,
            max_length: int):
        device = torch.device("cuda:{}".format(device_id))
        data_path = os.path.join(datasets_folder, dataset, self.dataset_config[dataset].get(
            "test_file", self.test_file))
        # 获取数据集最大input长度和最大output长度
        max_input_length, max_output_length = self.get_dataset_length(
            data_path, dataset, tokenizer, max_length)
        if device_id == self.device_ids[0]:
            logging.info(
                f'dataset: {dataset}, max_input_length: {max_input_length}, \
                max_output_length: {max_output_length}')
        Dataset = utils.get_dataset_by_name(
            self.dataset_config[dataset].get("dataset", self.default_dataset))
        eval_dataset = Dataset(
            data_path=data_path,
            tokenizer=tokenizer,
            name=dataset,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            max_length=max_length,
            batch_size=batch_size,
            no_append_glm_mask=False,
            gpt_data=False,
            world_size=len(self.device_ids),
            unidirectional=self.unidirectional,
            rotary_1d=self.rotary_1d,
            global_rank=self.device_ids.index(device_id),
            left_truncate=True,
            shard_data=True,
            device=device,
        )
        dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            collate_fn=eval_dataset.collate_batch
        )
        return dataloader

    def evaluate(self) -> dict:
        result = {}
        normalized_scores = []
        processes = []
        for id in self.device_ids:
            p = Process(target=self.evaluate_singleGPU, args=(id,))
            p.start()
            processes.append(p)
        while len(self.res) > 0:
            to_remove = []
            for dataset in self.res:
                predictions = []
                extras_info = []
                if len(self.res[dataset]) == len(self.device_ids):
                    # 当前数据集已经在所有卡上评估完成
                    for device_id in self.device_ids:
                        predictions.extend(
                            self.res[dataset][device_id]["predictions"])
                        extras_info.extend(
                            self.res[dataset][device_id]["extras_info"])
                    to_remove.append(dataset)
                    metric_dict = {}
                    metric_names = self.dataset_config[dataset]["metric"]
                    references = [extra_data["references"] for extra_data in extras_info]
                    for name in metric_names:
                        metric_class = METRIC_CLASS_DICT.get(name, None)
                        if not metric_class:
                            continue
                        metric = metric_class()
                        eval_result = metric.compute(
                            references=references, predictions=predictions, extras=extras_info)
                        metric_dict[name] = eval_result
                    result[dataset] = metric_dict
                    logging.info("{} result: {}".format(
                        dataset, metric_dict))
                    # calculate each dataset normalized score and extend to normalized_scores list
                    self.get_normalize_score(
                        extras_info, metric_dict, normalized_scores)
                    for i in range(len(predictions)):
                        extras_info[i]["predictions"] = predictions[i]
                    self.save_json(extras_info, dataset)
            for dataset in to_remove:
                del self.res[dataset]
            time.sleep(5)
        for p in processes:
            p.join()
        self.save_json(self.dataset_config, "dataset_configs.json")
        result["normalized_score"] = sum(
            normalized_scores) / len(normalized_scores)
        self.save_json(result, "eval_result.json")
        return result

    def evaluate_singleGPU(self, device_id):
        model, tokenizer, max_length = self.load_model(
            self.model_path, self.model_size, device_id)
        for dataset_index, dataset in enumerate(os.listdir(self.datasets_folder)):
            if self.is_valid_dataset(dataset, device_id) is False:
                continue
            if device_id == self.device_ids[0]:
                logging.info("{} evaluating...".format(dataset))
            batch_size = self.dataset_config[dataset].get(
                "batch_size", self.batch_size)
            while batch_size > 0:
                try:
                    predictions = []
                    extras_info = []
                    dataloader = self.load_dataloader(
                        self.datasets_folder, dataset, batch_size, device_id, tokenizer, max_length)
                    with tqdm(total=len(dataloader), desc=dataset + "-gpu:{}".format(device_id), ncols=80) as pbar:
                        for batch in dataloader:
                            extra_info = batch.pop("extra")
                            # import pdb; pdb.set_trace()
                            if "options" in extra_info[0] and dataset != "BIG-Bench-Hard":
                                batch_options = [extra["options"]
                                                 for extra in extra_info]
                                answers = self.batch_answer_with_options(
                                    model=model,
                                    tokenizer=tokenizer,
                                    batch_data=batch,
                                    batch_options=batch_options,
                                    batch_size=dataloader.batch_size,
                                    option_rank=self.dataset_config[dataset].get(
                                        "option_rank", "loss"),
                                    extra_info=extra_info,
                                    max_output_length=dataloader.dataset.max_output_length
                                )
                            else:
                                answers = self.batch_answer(
                                    model=model,
                                    tokenizer=tokenizer,
                                    batch_data=batch,
                                    max_new_tokens=dataloader.dataset.max_output_length,
                                )
                            pbar.update(1)
                            predictions.extend(answers)
                            extras_info.extend(extra_info)
                    break
                except Exception as e:
                    if 'out of memory' in str(e).lower():
                        logging.info(f'gpu: {device_id}, dataset: {dataset}, \
                                    decrease batch size from {batch_size} to {batch_size//2}')

                        batch_size = batch_size // 2
                    else:
                        logging.error(e)
                        raise e
            if batch_size == 0:
                raise Exception(f"dataset: {dataset} stay oom, please decrease max length")
            self.res[dataset][device_id] = {
                "predictions": predictions, "extras_info": extras_info}

            # if dataset == "AGIEval_classification" or dataset == "AGIEval_generation":
            #     if len(agi_tmp_predictions) == 0:
            #         agi_tmp_predictions.extend(predictions)
            #         agi_tmp_extras_info.extend(extras_info)
            #         continue
            #     else:
            #         predictions.extend(agi_tmp_predictions)
            #         extras_info.extend(agi_tmp_extras_info)
            #         dataset = "AGIEval"

    def get_normalize_score(
        self,
        extras_info,
        metric_dict,
        normalized_scores,
    ) -> float:
        """
        计算normalized score
        """
        random_acc_score = 0.
        # if "options" in extras_info[0]:
        #     for i in range(len(extras_info)):
        #         random_acc_score += 100. / (len(extras_info[i]["options"]) * len(extras_info))

        if "Accuracy" in metric_dict:
            normalized_acc_score = 100. * \
                (metric_dict["Accuracy"] - random_acc_score) / \
                (100 - random_acc_score)
            normalized_scores.append(normalized_acc_score)
        elif "AccuracyMacro" in metric_dict:
            normalized_acc_score = 100. * \
                (metric_dict["AccuracyMacro"] - random_acc_score) / \
                (100 - random_acc_score)
            normalized_scores.append(normalized_acc_score)
        else:
            dataset_scores = []
            for metric, result in metric_dict.items():
                if metric == "CalibrationError":
                    continue
                if isinstance(result, dict):
                    for score in result.values():
                        if score is not None:
                            assert isinstance(score, float)
                            dataset_scores.append(score)
                else:
                    assert isinstance(result, float)
                    dataset_scores.append(result)
            if dataset_scores:
                normalized_scores.append(
                    sum(dataset_scores) / len(dataset_scores))

    def batch_answer(
        self,
        model,
        tokenizer,
        batch_data: dict,
        max_new_tokens: int,
    ) -> List:
        """
        通用批量generate方法,目前仅支持batch_size为1
        """
        outputs = model.generate(
            **batch_data,
            max_new_tokens=max_new_tokens,
        )
        outputs = outputs.tolist()
        outputs = [o[batch_data["input_ids"].shape[1]:] for o in outputs]
        output = [tokenizer.decode(o) for o in outputs]
        answers = [[self._post_process(
            o, special_tokens=self.special_tokens)] for o in output]
        return answers

    # TODO 补充完善的分类预测逻辑
    def batch_answer_with_options(
        self,
        model,
        tokenizer,
        batch_data: dict,
        batch_options: List,
        batch_size: int,
        option_rank: str = "logit",
        extra_info: list = None,
        max_input_length: int = -1,
        max_output_length: int = -1,
        left_truncate: bool = True,
    ) -> List:
        """
        根据分类候选项进行分类预测

        参数:
        -- option_rank:类别结果分类依据,选择"logit"/"loss"
        -- extra_info:原始数据样本内容
        """
        answers = []
        if option_rank == "logit":
            option_ids = [tokenizer.encode(
                options, add_special_tokens=False) for options in batch_options]
            output = model.generate(
                **batch_data,
                max_new_tokens=max_output_length,
                return_dict_in_generate=True,
                output_scores=True,
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

    def get_dataset_length(self, data_path: str, dataset: str, tokenizer, max_length: int):
        """
        获取数据集最大input长度和最大output长度
        """
        with open(data_path, "r", encoding="utf-8") as fin:
            # grouped_lines = {'options_data': [], 'others_data': []}
            max_input_length = 0
            max_output_length = 0
            is_generation = False
            for line in fin:
                data = json.loads(line.rstrip('\n\r'))
                input_str = data['input']
                input_tokens = tokenizer(
                    input_str, add_special_tokens=False)['input_ids']
                max_input_length = max_input_length if len(
                    input_tokens) < max_input_length else len(input_tokens)
                if 'options' in data:
                    for option in data['options']:
                        option_tokens = tokenizer(
                            option, add_special_tokens=False)['input_ids']
                        max_output_length = max_output_length if len(
                            option_tokens) < max_output_length else len(option_tokens)
                    # grouped_lines['options_data'].append(line)
                else:
                    is_generation = True
                    for ref in data['references']:
                        output_tokens = tokenizer(
                            ref, add_special_tokens=False)['input_ids']
                        max_output_length = max_output_length if len(
                            output_tokens) < max_output_length else len(output_tokens)
                    # grouped_lines['others_data'].append(line)
        if is_generation:
            max_output_length = max_output_length * 2 + 1
        
        if max_input_length > max_length - 2:
            max_input_length = max_length - 2
        if "max_output_length" in self.dataset_config[dataset]:
            max_output_length = self.dataset_config[dataset]["max_output_length"]
        if max_output_length > max_length - 2:
            max_output_length = max_length - 2

        return max_input_length, max_output_length

    def save_json(self, file, save_name: str):
        with open(os.path.join(self.output_dir, save_name), "w", encoding="utf-8") as f:
            json.dump(file, f, ensure_ascii=False, indent=4)

    def _post_process(
        self,
        output: str,
        special_tokens: list = []
    ):
        for token in special_tokens:
            output = output.replace(token, "")
        output = output.replace("<n>", "\n")
        return output

    def init_res(self):
        manager = Manager()
        self.res = {}
        if self.dataset_name:
            for dataset in self.dataset_name:
                assert dataset in os.listdir(self.datasets_folder)
                self.res[dataset] = manager.dict()
        else:
            for dataset in os.listdir(self.datasets_folder):
                if self.is_valid_dataset(dataset):
                    self.res[dataset] = manager.dict()

    def is_valid_dataset(self, dataset, device_id=None):
        if not os.path.isdir(os.path.join(self.datasets_folder, dataset)):
            return False
        if dataset not in self.dataset_config:
            if device_id is not None:
                if device_id == self.device_ids[0]:
                    logging.warning(
                        "{} not in expected datasets!".format(dataset))
            return False
        if self.dataset_name:
            if dataset not in self.dataset_name:
                return False
        else:
            if self.dataset_config[dataset]["is_key_dataset"] is False:
                return False
            if "-shot" not in self.dataset_config[dataset]["method"]:
                return False
        return True


def main():
    return
    # args = parse_args()


if __name__ == "__main__":
    main()
