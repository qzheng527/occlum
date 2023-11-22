import os
import evaluate
from typing import List, Dict
import re

"""
BBH accuracy, prediction & references are labels. i.e.
predictions: [['8'], ['YES'],['(A)'],['valid'],['False']]
references: [['...the answer is 8'], ['YES'],['(A)'],['valid'],['False']]
"""


class BBHAccuracy():
    def __init__(self):
        self.metric = evaluate.load(os.path.join(
            os.path.dirname(__file__), './evaluate_factory/accuracy.py'))

    def compute(self, predictions: List[List], references: List[List], extras: List[Dict]):
        predictions = [self.post_processing(p[0].strip()) for p in predictions]
        references = [r[0] for r in references]
        types = [extra["type"] for extra in extras]

        values_list = list(set(references).union(set(predictions)))
        id_dict = {value: idx for idx, value in enumerate(values_list)}
        references = [id_dict[value] for value in references]
        predictions = [id_dict[value] for value in predictions]

        type_dict = dict()

        for p, r, t in zip(predictions, references, types):
            if t not in type_dict:
                type_dict[t] = ([], [])
            type_dict[t][0].append(p)
            type_dict[t][1].append(r)

        type_acc_l = []
        for type, res in type_dict.items():
            result = self.metric.compute(references=res[1], predictions=res[0])
            type_acc_l.append(result["accuracy"])
        results = sum(type_acc_l) / len(type_acc_l)
        results = round(results * 100, 2)
        return results
    
    def post_processing(self, content):
        p1 = re.compile(r'answer is ([^\.]+)\.')
        res = p1.search(content)
        if res:
            content = res[1]
        return content


if __name__ == "__main__":
    test = BBHAccuracy()
    print(test.compute([['8'], ['YES'], ['(A)'], ['valid'], ['False']], 
                       [['...the answer is 8'], ['YES'], ['(A)'], ['valid'], ['False']]
                       , None))
