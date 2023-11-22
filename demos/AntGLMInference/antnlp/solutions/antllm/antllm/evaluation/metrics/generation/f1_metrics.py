from typing import List, Dict
from nltk.metrics.scores import f_measure
from ..utils.tokenizer import tokenize

"""Borrow from HELM"""


class HelmF1():
    def compute(self, predictions: List[List], references: List[List], extras: List[Dict]):
        predictions = [tokenize(p[0]) for p in predictions]
        # references = [tokenize(r[0]) if len(r) > 0 else "" for r in references]
        references = [[tokenize(rr) for rr in r] if len(r) > 0 else [""] for r in references]
        results = []
        for i in range(len(predictions)):
            max_result = 0
            if references[i] == [""]:  # handle cases where reference is empty
                results.append(max_result)
                continue

            for ref in references[i]:
                result = f_measure(set(ref.lower().split()), 
                                   set(predictions[i].lower().split()))
                if result is None:
                    result = 0
                if result >= max_result:
                    max_result = result
            results.append(max_result)
        result = sum(results) / len(results)
        result = round(result * 100, 2)
        return result


if __name__ == "__main__":
    test = HelmF1()
    print(test.compute([["This is我是中国人"]], [["this is我是中国"]], None))
