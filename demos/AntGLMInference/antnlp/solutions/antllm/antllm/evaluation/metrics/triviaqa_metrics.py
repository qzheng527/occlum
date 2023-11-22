from typing import List, Dict
import string
import re
from collections import Counter
"""
TriviaQA evaluation code, borrowed from 
https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py
This code is only used for TriviaQA evaluation
"""


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


class TriviaQAEval():
    def compute(self, predictions: List[List], references: List[List], extras: List[Dict]): 
        predictions = [p[0] for p in predictions]
        f1 = exact_match = 0
        for pred, ref in zip(predictions, references): 
            if len(ref) == 0:  # if reference is empty
                continue
            em_for_this_question = self.metric_max_over_references(self.exact_match_score, pred, ref)
            exact_match += em_for_this_question
            f1_for_this_question = self.metric_max_over_references(self.f1_score, pred, ref)
            f1 += f1_for_this_question

        exact_match = 100.0 * exact_match / len(predictions)
        f1 = 100.0 * f1 / len(predictions)
        return {"exact_match": exact_match, "f1": f1}

    def f1_score(self, prediction, reference):
        prediction_tokens = normalize_answer(prediction).split()
        reference_tokens = normalize_answer(reference).split()
        common = Counter(prediction_tokens) & Counter(reference_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(reference_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def exact_match_score(self, prediction, reference):
        return normalize_answer(prediction) == normalize_answer(reference)

    def metric_max_over_references(self, metric_fn, prediction, references):
        scores_for_references = []
    
        if len(references) == 0:  # reference is empty
            return 0.0
                           
        for reference in references:
            score = metric_fn(prediction, reference)
            scores_for_references.append(score)
        return max(scores_for_references)


if __name__ == "__main__":
    test = TriviaQAEval()
    predictions = [["list of bond girls in octopussy"]]
    references = [["list of bond girls in octopussy", "bond 13", "list of james bond allies in octopussy"]]
    print(test.compute(predictions, references, None))
