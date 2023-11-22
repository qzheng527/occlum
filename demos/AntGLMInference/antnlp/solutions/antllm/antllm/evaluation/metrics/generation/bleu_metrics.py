import os
import evaluate
from typing import List, Dict
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.translate.bleu_score import sentence_bleu

from ..utils.tokenizer import tokenize


"""
Huggingface BLEU
predictions: [['矛盾'], ['矛盾']]
references: [['矛盾'], ['中立']]
"""


class HuggingfaceBLEU():
    def __init__(self):
        self.metric = evaluate.load(os.path.join(
            os.path.dirname(__file__), '../evaluate_factory/bleu.py')) 

    def compute(self, predictions: List[List], references: List[List], extras: List[Dict]):
        predictions = [tokenize(p[0]) for p in predictions]
        references = [[tokenize(rr) for rr in r] for r in references]  # BLEU allows multiple references
        results = self.metric.compute(
            references=references, predictions=predictions)
        results = round(results['bleu'] * 100, 2)
        return results


# try:
#     nltk.data.find("tokenizers/punkt")
# except LookupError:
#     nltk.download("punkt")  # Required for bleu


# class HelmBLEU1():
#     def compute(self, references: List[List], predictions: List[List]):
#         references = [r[0] for r in references]
#         predictions = [p[0] for p in predictions]
#         results = []
#         for r, p in zip(references, predictions):
#             # 这里能兼容中文吗
#             score = sentence_bleu(word_tokenize(
#                 r), word_tokenize(p), weights=(1, 0, 0, 0))
#             results.append(score)
#         result = sum(results) / len(results)
#         return round(100 * result, 2)


# class HelmBLEU2():
#     def compute(self, references: List[List], predictions: List[List]):
#         references = [r[0] for r in references]
#         predictions = [p[0] for p in predictions]
#         results = []
#         for r, p in zip(references, predictions):
#             score = sentence_bleu(word_tokenize(
#                 r), word_tokenize(p), weights=(0, 1, 0, 0))
#             results.append(score)
#         result = sum(results) / len(results)
#         return round(100 * result, 2)


if __name__ == "__main__":
    test = HuggingfaceBLEU()
    print(test.compute([["明确告知"], ["遇到其他人的干扰"]],
                       [["明确告知"], ["遇到其他人的干扰"]], None))

    # test = HelmBLEU1()
    # print(test.compute([["明 确 告 知"], ["遇 到 其 他 人 的 干 扰"]],
    #       [["明 确 告"], ["遇 到 其 他 人 的 干"]]))

    # test = HelmBLEU2()
    # print(test.compute([["明 确 告 知"], ["遇 到 其 他 人 的 干 扰"]],
    #                    [["明 确 告"], ["遇 到 其 他 人 的 干"]]))
