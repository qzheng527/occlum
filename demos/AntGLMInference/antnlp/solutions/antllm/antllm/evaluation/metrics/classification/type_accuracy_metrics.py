import os
import evaluate
from typing import List, Dict


"""
Huggingface accuracy, prediction & references are labels. i.e.
predictions: [['矛盾'], ['矛盾']]
references: [['矛盾'], ['中立']]
"""


class HuggingfaceTypeAccuracyMacro():
    def __init__(self):
        self.metric = evaluate.load(os.path.join(
            os.path.dirname(__file__), '../evaluate_factory/accuracy.py'))

    def compute(self, predictions: List[List], references: List[List], extras: List[Dict]):

        predictions = [p[0] for p in predictions]
        references = [r[0] for r in references]
        if "type" in extras[0]:  # for AGIEval and CEval
            types = [extra["type"] for extra in extras]
        elif "task_name" in extras[0]:  # for MMLU
            types = [extra["task_name"] for extra in extras]
        else:
            raise Exception("Sorry, you need to have type or task_name in your input fields.")

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

        results = dict()
        for type, res in type_dict.items():
            result = self.metric.compute(references=res[1], predictions=res[0])
            results[type] = round(result["accuracy"] * 100, 2)
        return results


if __name__ == "__main__":
    test = HuggingfaceTypeAccuracyMacro()
    # print(test.compute([['矛盾'], ['矛盾']], [['矛盾'], ['中立']], ))
