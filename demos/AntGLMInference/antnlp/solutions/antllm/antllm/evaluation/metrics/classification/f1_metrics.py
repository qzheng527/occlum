import os
import evaluate
from typing import List, Dict


"""
Huggingface F1, prediction & references are labels. i.e.
predictions: [['矛盾'], ['矛盾']]
references: [['矛盾'], ['中立']]
"""


class HuggingfaceF1():
    def __init__(self):
        self.metric = evaluate.load(os.path.join(
            os.path.dirname(__file__), '../evaluate_factory/f1.py'))

    def compute(self, predictions: List[List], references: List[List], extras: List[Dict]):
        predictions = [p[0] for p in predictions]
        references = [r[0] for r in references]
        values_list = list(set(references).union(set(predictions)))
        id_dict = {value: idx for idx, value in enumerate(values_list)}
        references = [id_dict[value] for value in references]
        predictions = [id_dict[value] for value in predictions]
        results = self.metric.compute(references=references, predictions=predictions, average='micro')
        results = round(results['f1'] * 100, 2)
        return results


if __name__ == "__main__":
    test = HuggingfaceF1()
    print(test.compute([[1], [2]], [[1], [3]], None))
