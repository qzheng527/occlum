# import os
# import re
# from nltk.metrics.scores import f_measure
# import string
# import evaluate
from typing import List, Dict
from ..utils.tokenizer import tokenize

# """HELM take one string"""
# def normalize_text(text: str):
#     """Lower text and remove punctuation, articles and extra whitespace.
#     Copied from the [QuAC](http://quac.ai/) evaluation script found at
#     https://s3.amazonaws.com/my89public/quac/scorer.py"""

#     def remove_articles(text: str):
#         return re.sub(r"\b(a|an|the)\b", " ", text)

#     def white_space_fix(text: str):
#         return " ".join(text.split())

#     def remove_punc(text: str):
#         exclude = set(string.punctuation)
#         return "".join(ch for ch in text if ch not in exclude)

#     def lower(text: str):
#         return text.lower()

#     return white_space_fix(remove_articles(remove_punc(lower(text))))


# """HELM"""
# class HelmEM():
#     def compute(self, predictions: list, references: list):
#         # TODO
#         predictions = [prediction[0] for prediction in predictions]
#         references = [reference[0] for reference in references]
#         equal = [predictions[i].strip() == references[i].strip()
#                  for i in range(len(predictions))]
#         result = sum(equal) / len(equal)
#         return result


# """HELM"""
# class HelmQuasiEM():
#     def compute(self, predictions: list, references: str):
#         predictions = [prediction[0] for prediction in predictions]
#         references = [reference[0] for reference in references]
#         equal = [normalize_text(predictions[i]) == normalize_text(
#             references[i]) for i in range(len(predictions))]
#         result = sum(equal) / len(equal)
#         result = round(result * 100, 2)
#         return result


# class HuggingfaceEM():
#     def __init__(self):
#         self.metric = evaluate.load(os.path.join(
#             os.path.dirname(__file__), '../evaluate_factory/exact_match.py'))

#     def compute(self, predictions: List[List], references: List[List]):
#         predictions = [tokenize(p[0]) for p in predictions]
#         references = [tokenize(r[0]) if len(r) > 0 else "" for r in references]
#         results = self.metric.compute(
#             predictions=predictions, references=references)
#         return round(results['exact_match'] * 100, 2)


"""
Top1 prediction exact matches one of references, count as positive
"""


class MultiEM():
    def compute(self, predictions: List[List], references: List[List], extras: List[Dict]):
        result = 0
        predictions = [tokenize(p[0]) for p in predictions]
        references = [[tokenize(rr) for rr in r] if len(r) > 0 else [""] for r in references]
        for pre, ref in zip(predictions, references):
            if pre in ref:
                result += 1
        return round(result / len(predictions) * 100, 2)


if __name__ == "__main__":
    # test = HelmEM()
    # print(test.compute("This is", "this is"))

    # test = HelmQuasiEM()
    # print(test.compute("This is", "this is"))

    test = MultiEM()
    print(test.compute([["the cat"], ["theater"], ["YELLING"], ["agent007"]], [
          [], ["theater"], ["yelling"], ["agent"]], None))
