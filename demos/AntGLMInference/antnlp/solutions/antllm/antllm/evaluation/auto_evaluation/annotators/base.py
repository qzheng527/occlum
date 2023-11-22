#!/usr/bin/env python
# coding=utf-8
# @Author: xinyu.kxy
# @Date: Fri 17 July 2023 09:22:56 PM CST

import abc
import logging
import os
import json
from functools import partial
from solutions.antllm.antllm.evaluation.auto_evaluation import openai_api
from typing import Callable, Optional, Union
from solutions.antllm.antllm.evaluation.auto_evaluation import completion_parsers, metrics, utils


class BaseAnnotator(abc.ABC):
    """自动评估标注器父类

    Parameters
    ----------
    prompt_template : 模版路径
    fn_completion_parser : 接口返回解析器,选择在"completion_parsers.py"中的函数
    completion_parser_kwargs : 解析器参数
    fn_completions : 指定请求接口，默认openai
    api_param : 接口参数
    is_shuffle : 在进行batch请求的时候是否进行shuffle
    seed : 随机种子
    batch_size : batch数量
    annotation_key : 标注dict中的key值
    """

    def __init__(
        self,
        data_folder: str,
        prompt_template: str,
        dataset_path: str,
        completion_parser_kwargs: Optional[dict] = None,
        fn_completion_parser: Optional[Union[Callable, str]] = "regex_parser",
        api_param: Optional[dict] = None,
        model_output_keys: list = ["chatgpt"],
        is_shuffle: bool = False,
        seed: Optional[int] = 123,
        batch_size: int = 1,
        annotation_key: str = "label",
        golden_label_key: str = "golden_label",
        completion_key: str = "completion",
        **kwargs
    ):
        self.prompt_template = self._get_prompt_template(
            os.path.join(data_folder, prompt_template))

        fn_completion_parser = self._search_fn_completion_parser(
            fn_completion_parser)
        completion_parser_kwargs = completion_parser_kwargs or {}
        self.fn_completion_parser = partial(
            fn_completion_parser, **completion_parser_kwargs)

        # self.fn_completions = get_fn_completions(fn_completions)
        self.api_param = api_param or {}
        self.dataset_path = os.path.join(data_folder, dataset_path)
        self.model_output_keys = model_output_keys
        self.seed = seed
        self.is_shuffle = is_shuffle
        self.batch_size = batch_size
        self.annotation_key = annotation_key
        self.golden_label_key = golden_label_key
        self.completion_key = completion_key
        self.kwargs = kwargs

    def annotate(self, result_folder) -> list:
        """
        主标注函数
        """
        data_to_annotate = self._load_data()  # avoid in place modifications

        if len(data_to_annotate) == 0:
            data_to_annotate[self.annotation_key] = []
            return data_to_annotate

        data_to_annotate = self._preprocess(data_to_annotate)

        # prompts and completions here will not be the same length as the dataframe due to batching
        prompts = self._make_prompts(data_to_annotate)
        completions = openai_api.open_api_completions(
            prompts, self.api_param)

        annotations_to_save, completions_to_save = self._parse_completions(
            completions=completions["completions"])

        for item, annotation, completion in zip(data_to_annotate, annotations_to_save, completions_to_save):
            item[self.annotation_key] = annotation
            item[self.completion_key] = completion

        # TODO
        # for k, v in completions.items():
        #     if k != "completions":
        #         if len(data_to_annotate[self.annotation_key]) == len(v) * self.batch_size:
        #             v = [el for el in v for _ in range(self.batch_size)]
        #         data_to_annotate[k] = v
        #         if "per_example" in k:
        #             data_to_annotate[k] = data_to_annotate[k] / self.batch_size

        data_annotated = self._postprocess(data_to_annotate)
        metric = self._get_metric()
        result = metric(data_annotated=data_annotated,
                        annotation_key=self.annotation_key, golden_label_key=self.golden_label_key)
        logging.info("result is {}".format(result))
        self._save(result_folder, data_annotated, result)

    ######################

    def _search_fn_completion_parser(self, name: str) -> Callable:
        """获取解析器函数"""
        return getattr(completion_parsers, name)

    def _load_data(self) -> list:
        with open(self.dataset_path, "r") as f:
            data = json.load(f)
        # import random
        # data = random.sample(data,10)
        return data

    def _get_metric(self):
        return metrics.pairwise_to_winrate

    def _save(self, folder_path, data_annotated, result):
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        with open(os.path.join(folder_path, "data_annotated.json"), "w") as f:
            json.dump(data_annotated, f, ensure_ascii=False, indent=4)
        with open(os.path.join(folder_path, "result.json"), "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

    def _get_prompt_template(self, prompt_template: str):
        return utils.read_or_return(prompt_template)

    def _make_prompts(
        self, data_to_annotate: list, prompt_template: Optional[str] = None
    ) -> list:
        """
        合并prompt
        """
        if prompt_template is None:
            prompt_template = self.prompt_template
        return utils.make_prompts(data=data_to_annotate, template=prompt_template, batch_size=self.batch_size)

    def _preprocess(self, data_to_annotate: list) -> list:
        """
        预处理
        """
        if self.is_shuffle:
            data_to_annotate = data_to_annotate.sample(
                frac=1, random_state=self.seed)

        return data_to_annotate

    def _parse_completions(self, completions: list) -> tuple:
        """
        解析大模型返回内容
        """
        all_annotations = []
        all_completions = []
        for completion in completions:
            try:
                batch_annotations = self.fn_completion_parser(completion)

                if len(batch_annotations) != self.batch_size:
                    logging.warning(
                        f"Found {len(batch_annotations)} annotations in:'''\n{completion}\n''' but expected"
                        f" {self.batch_size}. We are setting all annotations to None."
                    )
                    batch_annotations = [None] * self.batch_size

            except Exception as e:
                logging.exception(
                    f"Error while parsing completion: '''\n{completion}\n'''")
                logging.warning("{}".format(e))
                batch_annotations = [None] * self.batch_size

            all_annotations += batch_annotations
            all_completions += [completion] * self.batch_size
        return all_annotations, all_completions

    def _postprocess(self, data_annotated: list) -> list:
        """
        后处理
        """
        # TODO remove padding examples when using batch_size > 1
        return data_annotated
