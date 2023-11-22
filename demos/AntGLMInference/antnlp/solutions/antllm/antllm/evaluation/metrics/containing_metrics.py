import os
import evaluate
from typing import List, Dict

"""
Containing accuracy, prediction & references are labels. i.e.
predictions: [['8'], ['YES'],['(A)'],['valid'],['False']]
references: [['...the answer is 8'], ['YES'],['(A)'],['valid'],['False']]
"""


class ContainingAccuracy():
    def __init__(self):
        self.metric = evaluate.load(os.path.join(
            os.path.dirname(__file__), './evaluate_factory/accuracy.py'))

    def compute(self, predictions: List[List], references: List[List], extras: List[Dict]):
        num_correct = 0
        for i in range(len(predictions)):
            if references[i][0] in predictions[i][0]:
                num_correct += 1

        accuracy = num_correct / len(predictions)
        results = round(accuracy * 100, 2)
        return results


if __name__ == "__main__":
    test = ContainingAccuracy()
    print(test.compute([['3'], ['YES'], ['(A)'], ['valid'], ['False']],
                       [['...the answer is 8'], ['No'], ['(A)'], ['valid is True'], ['False']], None))