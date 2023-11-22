import os
import evaluate
# import nltk
# from functools import partial
# from rouge_score import rouge_scorer
from typing import List, Dict
from ..utils.tokenizer import tokenize
from rouge_chinese import Rouge

'''
    说明：
    HuggingfaceROUGE 用在标准英文数据集上
    ChineseRouge 用在中文业务数据上

    rouge 计算实现了两个类，区别是调用的计算库不同
    HuggingfaceROUGE可以使用在英文的数据集上，由于HuggingfaceROUGE中使用的evaluate库与其他公开的工作（使用英文数据集做测评）
    使用的测评方案一致，可以比较方便做到评测逻辑对齐。
    ChineseRouge适用与在中文业务数据集上做评测，主要解决了evaluate库背后调用的rouge_score库会在tokennize阶段会对非
    r"^[a-z0-9]+$"范围内的字符做过滤的逻辑，导致中文字符被过滤，最终导致在中文上指标有问题的bug。
'''


class HuggingfaceRouge():
    def __init__(self, rouge_name):
        """
        rouge_name is in (rouge1, rouge2, rougeL)
        """
        self.metric = evaluate.load(os.path.join(
            os.path.dirname(__file__), '../evaluate_factory/rouge.py'))
        self.rouge_name = rouge_name

    def compute(self, predictions: List[List], references: List[List], extras: List[Dict]):
        predictions = [tokenize(p[0]) for p in predictions]
        references = [tokenize(r[0]) if len(r) > 0 else "" for r in references]
        results = self.metric.compute(
            references=references, predictions=predictions)
        results = {key: round(100 * results[key], 2) for key in results}
        return results[self.rouge_name]


class HuggingfaceRouge1():
    def __init__(self):
        self.rouge_name = "rouge1"

    def compute(self, predictions: List[List], references: List[List], extras: List[Dict]):
        rouge = HuggingfaceRouge(self.rouge_name)
        return rouge.compute(predictions=predictions, references=references, extras=extras)


class HuggingfaceRouge2():
    def __init__(self):
        self.rouge_name = "rouge2"
        
    def compute(self, predictions: List[List], references: List[List], extras: List[Dict]):
        rouge = HuggingfaceRouge(self.rouge_name)
        return rouge.compute(predictions=predictions, references=references, extras=extras)


class HuggingfaceRougeL():
    def __init__(self):
        self.rouge_name = "rougeL"
        
    def compute(self, predictions: List[List], references: List[List], extras: List[Dict]):
        rouge = HuggingfaceRouge(self.rouge_name)
        return rouge.compute(predictions=predictions, references=references, extras=extras)


class ChineseRouge():
    '''
    return
    rouge-1 f rounge-2 f and rouge-l f
    {
        "rouge-1": 100.0,
        "rouge-2": 100.0,
        "rouge-l": 100.0
    }
    '''

    def __init__(self, rouge_name) -> None:  # noqa
        self.rouge = Rouge()
        self.rouge_name = rouge_name

    def compute(self, predictions: List[List], references: List[List], extras: List[Dict]) -> dict:
        pred_list = []
        ref_list = []
        for item in zip(predictions, references):
            p, r = item
            # 如果pred和ref均为 ""，对指标计算无意义，为防止后续的空字符替换干扰指标结果，做跳过处理
            if (len(p) > 0 and not p[0]) and (len(r) > 0 and not r[0]):
                continue
            pred_list.append(tokenize(p[0]) if len(p) > 0 and p[0] else "没有")
            ref_list.append(tokenize(r[0]) if len(r) > 0 and r[0] else "没有")
        result = self.rouge.get_scores(pred_list, ref_list, avg=True)
        metric_result = {
            "rouge1": round(100 * result["rouge-1"]["f"], 2),
            "rouge2": round(100 * result["rouge-2"]["f"], 2),
            "rougeL": round(100 * result["rouge-l"]["f"], 2)
        }
        return metric_result[self.rouge_name]


class ChineseRouge1():
    def __init__(self):
        self.rouge_name = "rouge1"

    def compute(self, predictions: List[List], references: List[List], extras: List[Dict]):
        rouge = ChineseRouge(self.rouge_name)
        return rouge.compute(predictions, references, extras)


class ChineseRouge2():
    def __init__(self):
        self.rouge_name = "rouge2"

    def compute(self, predictions: List[List], references: List[List], extras: List[Dict]):
        rouge = ChineseRouge(self.rouge_name)
        return rouge.compute(predictions, references, extras)
    

class ChineseRougeL():
    def __init__(self):
        self.rouge_name = "rougeL"

    def compute(self, predictions: List[List], references: List[List], extras: List[Dict]):
        rouge = ChineseRouge(self.rouge_name)
        return rouge.compute(predictions, references, extras)
    

# try:
#     nltk.data.find("tokenizers/punkt")
# except LookupError:
#     nltk.download("punkt")  # Required for rouge


# class HelmROUGE():
#     # TODO: 输入是str，和我们的定义不符
#     def compute(self, gold: str, pred: str):
#         scorer = rouge_scorer.RougeScorer(
#             ["rouge1", "rouge2", "rougeL"], use_stemmer=True)
#         results = scorer.score(pred, gold)
#         new_results = {'rouge1': {}, 'rouge2': {}, 'rougeL': {}}
#         new_results['rouge1']['precision'] = round(results['rouge1'].precision * 100, 2)
#         new_results['rouge1']['recall'] = round(results['rouge1'].recall * 100, 2)
#         new_results['rouge1']['fmeasure'] = round(results['rouge1'].fmeasure * 100, 2)

#         new_results['rouge2']['precision'] = round(results['rouge2'].precision * 100, 2)
#         new_results['rouge2']['recall'] = round(results['rouge2'].recall * 100, 2)
#         new_results['rouge2']['fmeasure'] = round(results['rouge2'].fmeasure * 100, 2)

#         new_results['rougeL']['precision'] = round(results['rougeL'].precision * 100, 2)
#         new_results['rougeL']['recall'] = round(results['rougeL'].recall * 100, 2)
#         new_results['rougeL']['fmeasure'] = round(results['rougeL'].fmeasure * 100, 2)

#         return new_results

#     def rouge_score(self, gold: str, pred: str, rouge_type: str, scorer: rouge_scorer.RougeScorer) -> float:
#         scores = scorer.score(gold, pred)
#         return scores[rouge_type].fmeasure


if __name__ == "__main__":
    test = HuggingfaceRouge("rougeL")
    print(test.compute([["hello there", "general kenobi"], ],
                       [["hello there2", "general kenobi"], ], None))

    test = ChineseRouge("rougeL")
    print(test.compute([["你好呀", ], ["你好呀", ]],
                       [["你好呀2", ], ["你好呀", ]], None))

    # test = HelmROUGE()
    # print(test.compute("this is", "that is"))
