#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : run_eval.py
# @Author: daniel.ljh
# @Date  : 2023/3/16

import argparse
import json
import os
import sys

from solutions.antllm.antllm.evaluation.metrics import \
    METRIC_CLASS_DICT  # noqa

NORMALIZE_SKIP_DATASEATS = ["BIG-Bench-Hard",
                            "BBH", "AGIEval", "CEval", "HumanEval"]
NORMALIZE_SKIP_METRICS = ["CalibrationError", "GenderRepresentation"]


def parse_args():
    parser = argparse.ArgumentParser(description="config for evaluation")
    parser.add_argument(
        "--result_folder",
        type=str,
        help="必须参数：预测结果目录",
        required=True,
    )
    parser.add_argument(
        "--eval_policy",
        default="all",
        type=str,
        choices=("dataset", "task", "ability", "language", "all"),
        required=True,
        help="必须参数：要采取的评估策略: \
                dataset: 只评估一个数据集上的结果, \
                task: 评估一种NLP任务下所有数据集， \
                ability: 评估一种能力下的所有数据集, \
                language: 评估一种语言下的所有数据集, \
                all: 评估resultfolder里的所有数据集文件",
    )
    parser.add_argument(
        "--target_name",
        type=str,
        required=False,
        help="必须参数：评价的目标，和eval_policy结合使用, \
              当eval_policy=dataset时，target_name应该为一个数据集名称, e.g. OCNLI \
              当eval_policy=task时, target_name应该为一个任务名称，e.g. 分类任务 \
              当eval_policy=ability时，target_name应该为一个能力名称, e.g. 泛化能力,\
              当eval_policy=language时，target_name应该为一个语言名称, e.g. 中文、英文,\
              当eval_policy=all时，target_name无需提供",
        default="",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default='solutions/antllm/antllm/evaluation/configs/datasets_des.json',
        help="可选参数：dataset的配置文件。默认将使用solutions/antllm/antllm/evaluation/configs/datasets_des.json"
    )
    parser.add_argument(
        "--multi_choice_baseline",
        type=str,
        choices=("0", "random"),
        default="0",
        help="可选参数:计算数据集归一化分数时的分类任务baseline: \
                0: 以0值作为baseline, \
                random: 以随机分类器作为baselin(PaLM-BIG-Bench)",
    )
    parser.add_argument('--result_file', default=None,
                        type=str, action='store', help='File to write result')

    args = parser.parse_args()

    return args


def ana_json_config(config_file):
    config = json.load(open(config_file))
    return config


class DatasetEval:
    def __init__(self, dataset_config, result_folder, multi_choice_baseline="0"):
        self.dataset_config = dataset_config
        self.result_folder = result_folder
        self.multi_choice_baseline = multi_choice_baseline
        self.normalized_scores = []

    def normalize(self, data):
        chinese_punctuations = " ！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏."
        english_punctuations = " !?.\"#$%&\'()*+,-\:;<>=@[]\\^_`{}|~"
        data = data.strip(chinese_punctuations)
        data = data.strip(english_punctuations)
        data = data.lower()
        return data

    def get_eval_data(self, result_file, do_normalize=False):
        try:
            with open(result_file) as f:
                lines = json.load(f)
        except Exception:
            with open(result_file) as f:
                lines = f.readlines()
        refs = []
        preds = []
        extras = []
        random_acc_score = 0.
        for line in lines:
            info = line
            if isinstance(line, str):
                info = json.loads(line)
            if "options" in info:
                random_acc_score += 100. / (len(info["options"]) * len(lines))
            reference = info["references"]
            prediction = info["predictions"]
            if do_normalize:
                reference = [self.normalize(ref) for ref in reference]
                prediction = [self.normalize(pred) for pred in prediction]
            refs.append(reference)
            preds.append(prediction)
            info.pop("references")
            info.pop("predictions")
            for option in list(info.get("likelihood", {}).keys()):
                likelihood = info["likelihood"].pop(option)
                info["likelihood"][self.normalize(option)] = likelihood
            extras.append(info)
        return refs, preds, extras, random_acc_score

    def check_data_format(self, result_file):
        fin = open(result_file, 'r')
        flag = True
        for i, line in enumerate(fin):
            try:
                data = json.loads(line.rstrip('\n\r'))
            except Exception:
                print(f'{result_file}: 第{i}行，预测结果文件的每一行应该为一个json，表示一个数据\
                        每个数据的格式为：\{"predictions": ["pred1", "pred2"], "references": ["ref1", "ref2"]\}')
                flag = False
                continue
            if 'predictions' not in data:
                print(
                    f'{result_file}: 第{i}行，预测结果文件的每一行应该包含predictions, \
                            格式为："predictions": ["pred1", "pred2"]')
                flag = False
            else:
                if not isinstance(data['predictions'], list):
                    print(
                        f'{result_file}: 第{i}行，预测结果文件的predictions应该是一个list, \
                                格式为："predictions": ["pred1", "pred2"]')
                    flag = False
                if isinstance(data['predictions'][0], list):
                    print(
                        f'{result_file}: 第{i}行，预测结果文件的predictions应该是一个list, \
                                格式为："predictions": ["pred1", "pred2"], 不应该是多层list')
                    flag = False
            if 'references' not in data:
                print(
                    f'{result_file}: 第{i}行，预测结果文件的每一行应该包含references, \
                            格式为："references": ["ref1", "ref2"]')
                flag = False
            else:
                if not isinstance(data['references'], list):
                    print(
                        f'{result_file}: 第{i}行，预测结果文件的references应该是一个list, \
                                格式为："references": ["ref1", "ref2"]')
                    flag = False
                else:
                    if len(data['references']) > 0 and isinstance(data['references'][0], list):
                        print(
                            f'{result_file}: 第{i}行，预测结果文件的references应该是一个list, \
                                    格式为："references": ["ref1", "ref2"], 不应该是多层list')
                        flag = False
        fin.close()
        return flag

    def eval_dataset(self, dataset_name):
        # 获取模型的评测指标
        target_dataset_config = self.dataset_config[dataset_name]

        # 得到ground truth和模型预测结果
        result_file = os.path.join(self.result_folder, dataset_name)
        if os.path.exists(result_file):
            # is_format_valid = self.check_data_format(result_file)
            # if not is_format_valid:
            #     print(f'{result_file}格式错误，跳过')
            #     return {}
            do_normalize = True
            if target_dataset_config.get('task', None) == '代码生成':
                do_normalize = False
            if dataset_name in {"BIG-Bench-Hard", "BBH"}:
                do_normalize = False
            refs, preds, extras, random_acc_score = self.get_eval_data(
                result_file, do_normalize=do_normalize)

            metric_names = target_dataset_config["metric"]
            metric_dict = {}
            for name in metric_names:
                metric_class = METRIC_CLASS_DICT.get(name, None)
                if not metric_class:
                    continue
                metric = metric_class()
                eval_result = metric.compute(
                    references=refs, predictions=preds, extras=extras)
                metric_dict[name] = eval_result
            # 计算归一化指标
            if dataset_name in NORMALIZE_SKIP_DATASEATS:
                pass
            elif "Accuracy" in metric_names and random_acc_score != 0:
                if self.multi_choice_baseline == "random":
                    normalized_acc_score = 100. * \
                        (metric_dict["Accuracy"] - random_acc_score) / \
                        (100 - random_acc_score)
                else:
                    assert self.multi_choice_baseline == "0"
                    normalized_acc_score = metric_dict["Accuracy"]
                self.normalized_scores.append(normalized_acc_score)
            else:
                normalized_scores = []
                for metric, result in metric_dict.items():
                    if metric in NORMALIZE_SKIP_METRICS:
                        continue
                    if isinstance(result, dict):
                        for score in result.values():
                            if score is not None:
                                assert isinstance(score, float)
                                normalized_scores.append(score)
                    else:
                        assert isinstance(result, float)
                        normalized_scores.append(result)
                if normalized_scores:
                    self.normalized_scores.append(
                        sum(normalized_scores) / len(normalized_scores))
        else:
            metric_dict = {}
        return metric_dict


def main():
    dataset_results = {}
    args = parse_args()
    eval_policy = args.eval_policy
    target_name = args.target_name
    result_folder = args.result_folder
    dataset_config_path = args.dataset_config
    multi_choice_baseline = args.multi_choice_baseline
    dataset_config = ana_json_config(dataset_config_path)
    metric_calculator = DatasetEval(
        dataset_config, result_folder, multi_choice_baseline)

    if eval_policy == "dataset":
        dataset_name = target_name
        metric_dict = metric_calculator.eval_dataset(dataset_name)
        if metric_dict:
            dataset_results[dataset_name] = metric_dict

    elif eval_policy == "task":
        # 获取模型的评测指标
        task_name = target_name
        for dataset_name, target_dataset_config in dataset_config.items():
            if target_dataset_config["task"] != task_name:
                continue
            metric_dict = metric_calculator.eval_dataset(dataset_name)
            if metric_dict:
                dataset_results[dataset_name] = metric_dict
    elif eval_policy == 'ability':
        # eval_policy == "ability"
        ability_name = target_name
        for dataset_name, target_dataset_config in dataset_config.items():
            if target_dataset_config["ability"] != ability_name:
                continue
            metric_dict = metric_calculator.eval_dataset(dataset_name)
            if metric_dict:
                dataset_results[dataset_name] = metric_dict
    elif eval_policy == 'language':
        # eval_policy == "ability"
        language = target_name
        for dataset_name, target_dataset_config in dataset_config.items():
            if target_dataset_config["language"] != language:
                continue
            metric_dict = metric_calculator.eval_dataset(dataset_name)
            if metric_dict:
                dataset_results[dataset_name] = metric_dict
    elif eval_policy == 'all':
        for dataset_name, target_dataset_config in dataset_config.items():
            metric_dict = metric_calculator.eval_dataset(dataset_name)
            if metric_dict:
                dataset_results[dataset_name] = metric_dict
    else:
        print('非法的eval_policy')
        sys.exit()
    if metric_calculator.normalized_scores:
        dataset_results["normalized_score"] = sum(
            metric_calculator.normalized_scores) / len(metric_calculator.normalized_scores)
    if args.result_file:
        json.dump(dataset_results, open(args.result_file, 'w'),
                  ensure_ascii=False, indent=2)
    else:
        print(json.dumps(dataset_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
