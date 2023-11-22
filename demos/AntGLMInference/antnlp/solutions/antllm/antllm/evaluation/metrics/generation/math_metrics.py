#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : math_metrics.py
# @Author: daniel.ljh
# @Date  : 2023/3/21

import os
import evaluate
import re
from typing import List, Dict


class GSM8kMetric():
    def __init__(self):
        self.ANS_RE = re.compile(r"(####|answer is) (\s*\-?[0-9\.\,]+)")
        self.acc_metric = evaluate.load(os.path.join(
            os.path.dirname(__file__), '../evaluate_factory/accuracy.py'))

    def compute(self, predictions: List[List], references: List[List], extras: List[Dict]):
        predictions = [self.pred_data_process(p) for p in predictions]  # select the majority answer from predictions 
        references = [self.extract_answer(r[0]) for r in references]

        values_list = list(set(references).union(set(predictions)))
        id_dict = {value: idx for idx, value in enumerate(values_list)}
        predictions = [id_dict[value] for value in predictions]
        references = [id_dict[value] for value in references]

        results = self.acc_metric.compute(predictions=predictions, references=references)
        return round(results["accuracy"] * 100, 2)

    def extract_answer(self, completion):
        match = self.ANS_RE.search(completion)
        try:
            if match:
                match_str = match.group(2).strip()
                match_str = match_str.replace(",", "") 
                if match_str[-1] == ".":
                    match_str = match_str[:-1]
                return match_str
            else:
                # find the last number in answer
                # import pdb; pdb.set_trace()
                last_number = re.findall(r"\d+", completion)[-1] 
                return last_number
        except Exception as ex:
            print("Cannot parse number form answer {0}, exception: {1}".format(completion, ex))
            return ""

    def pred_data_process(self, pred):
        ans_list = [self.extract_answer(ans) for ans in pred]
        return max(set(ans_list), key=ans_list.count)


if __name__ == "__main__":
    metric = GSM8kMetric()
    # gold = [[
    #     "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the \
    #     farmer\u2019s market.\n#### 18"]]
    # pred = [["market.#### 18", "market.#### 18"]]
    # print(metric.compute(pred, gold, None))
    test = "janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a daay at the farmer’s market.\n#### 18.1 19.1"
    test2 = "'each girl got 1/6 x 24 = <<1/6*24=4>>>10 liters of water were left.\nthe 10.1'"
    res = metric.extract_answer(test)
    res2 = metric.extract_answer(test2)
    print(res, res2)
