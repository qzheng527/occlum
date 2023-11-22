#!/usr/bin/env python
# coding=utf-8
# @Author: xinyu.kxy
# @Date: Fri 17 July 2023 09:22:56 PM CST

# import logging
# import random
from typing import Optional

from solutions.antllm.antllm.evaluation.auto_evaluation import utils, metrics
from solutions.antllm.antllm.evaluation.auto_evaluation.annotators import SingleAnnotator, PairwiseAnnotator


class MultiturnSingleAnnotator(SingleAnnotator):
    """
    蚂蚁大模型多轮数据集single标注器
    """

    def __init__(
        self,
        *args,
        dialog_key: str = "dialogs",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dialog_key = dialog_key
        assert len(self.model_output_keys) == 1

    def _preprocess(self, data_to_annotate: list) -> list:
        """
        预处理,多轮转化成单轮
        """
        data_to_annotate_singleTurn = []
        session_id = 0
        for data in data_to_annotate:
            session_id += 1
            # [{"role":"用户","content":"你是叫什么名字"},{"role":"贞仪","content":"我叫贞仪"},...]
            dialogs = data[self.dialog_key]
            assert len(dialogs) % 2 == 0
            for i in range(0, len(dialogs), 2):
                data_to_annotate_singleTurn.append({
                    "input": dialogs[i]["content"],
                    "output": dialogs[i + 1]["content"],
                    "session_id": session_id,
                    "ori_data": data
                })

        return data_to_annotate_singleTurn

    def merge_labels(self, labels: list) -> int:
        """
        合并所有的单轮标签
        """
        if -1 in labels:
            return -1
        elif 2 in labels:
            return 2
        else:
            return 1

    def _postprocess(self, data_annotated: list) -> list:
        data_annotated = super()._postprocess(data_annotated)

        all_values = [item[self.annotation_key] for item in data_annotated]
        assert set(all_values) <= {-1, 0, 1, 2}
        data_annotated_ori = []
        session_id = -1
        session_labels = []
        for i in range(len(data_annotated)):
            data = data_annotated[i]
            if session_id != data["session_id"]:
                # 合并所有单轮label:
                session_label = self.merge_labels(session_labels)
                if len(data_annotated_ori) != 0:
                    data_annotated_ori[-1]["label"] = session_label
                    data_annotated_ori[-1]["session_labels"] = session_labels
                data_annotated_ori.append(data["ori_data"])
                session_labels = [data[self.annotation_key]]
                session_id = data["session_id"]
            else:
                session_labels.append(data[self.annotation_key])
            if i == len(data_annotated) - 1:
                session_label = self.merge_labels(session_labels)
                if len(data_annotated_ori) != 0:
                    data_annotated_ori[-1]["label"] = session_label
                    data_annotated_ori[-1]["session_labels"] = session_labels
        return data_annotated_ori


class MultiturnPairwiseAnnotator(PairwiseAnnotator):
    """
    蚂蚁大模型多轮数据集Pairwise标注器
    """

    def __init__(
        self,
        *args,
        method: str = "single",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert len(self.model_output_keys) == 1

    def _make_prompts(
        self, data_to_annotate: list, prompt_template: Optional[str] = None
    ) -> list:
        if prompt_template is None:
            prompt_template = self.prompt_template
        prompt_template = prompt_template.replace(
            "output", self.model_output_keys[0])
        return utils.make_prompts(data=data_to_annotate, template=prompt_template, batch_size=self.batch_size)

    def _get_metric(self):
        return metrics.single_to_acc

    def _postprocess(self, data_annotated: list) -> list:
        data_annotated = super()._postprocess(data_annotated)

        all_values = [item[self.annotation_key] for item in data_annotated]
        assert set(all_values) <= {-1, 0, 1, 2}

        return data_annotated
