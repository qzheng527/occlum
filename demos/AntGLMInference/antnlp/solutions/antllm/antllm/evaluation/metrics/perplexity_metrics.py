import math
from typing import List, Dict

"""
calculate perplexity for language models
"""


class Perplexity():
    def compute(self, predictions: List[List], references: List[List], extras: List[Dict]): 
        """
        extras[0]["probability"]: probability of generating each token in the predicted sentences
        should match with the top1 predicted sentence length. e.g.
            extras[0]["probability"]: [0.2, 0.1, 0.3]
            extras[1]["probability"]: [0.4, 0.5, 0.1, 0.01]
        predictions: the predicted sentences as the final output. e.g. 
            [["this is dog", "this is"], ["this is a dog", "this is"]]
        """
        use_logits = "probability" in extras[0].keys()
        use_loss = "loss" in extras[0].keys()
        if not use_logits and not use_loss:
            raise Exception("Probability or Loss field is required for perplexity calculation!")
        logprobs = []
        if use_logits:
            for extra in extras:
                logprob = [math.log(ex) for ex in extra["probability"]]
                logprobs.append(logprob)
            results = [math.e ** (-sum(logp) / len(logp)) for logp in logprobs]
        elif use_loss:
            results = [math.exp(sum(extra["loss"]) / len(extra["loss"])) for extra in extras]
        return sum(results) / len(results)


if __name__ == "__main__":
    test = Perplexity()
    print(test.compute([["this is dog", "this is"], ["this is a dog", "this is"]], 
                       [["this is dog", "this is"], ["this is a dog", "this is"]], 
                       [{"probability": [0.2, 0.1, 0.3]}, {"probability": [0.4, 0.5, 0.1, 0.01]}]))

