import os
import evaluate
from typing import List, Dict


"""
Huggingface accuracy, prediction & references are labels. i.e.
predictions: [['矛盾'], ['矛盾']]
references: [['矛盾'], ['中立']]
"""


class HuggingfaceAccuracy():
    def __init__(self):
        self.metric = evaluate.load(os.path.join(
            os.path.dirname(__file__), '../evaluate_factory/accuracy.py'))

    def compute(self, predictions: List[List], references: List[List], extras: List[Dict]):
        predictions = [p[0] for p in predictions]
        references = [r[0] for r in references]
        values_list = list(set(references).union(set(predictions)))
        id_dict = {value: idx for idx, value in enumerate(values_list)}
        references = [id_dict[value] for value in references]
        predictions = [id_dict[value] for value in predictions]
        results = self.metric.compute(references=references, predictions=predictions)
        results = round(results["accuracy"] * 100, 2)
        return results


if __name__ == "__main__":
    test = HuggingfaceAccuracy()
    print(test.compute([['矛盾'], ['矛盾']], [['矛盾'], ['中立']], None))
