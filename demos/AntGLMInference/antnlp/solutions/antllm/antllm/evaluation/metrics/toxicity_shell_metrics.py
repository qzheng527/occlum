import os
import evaluate
from typing import List, Dict
from .utils.tokenizer import tokenize

"""
Huggingface toxicity
The toxicity measurement aims to quantify toxicity
using a pretrained hate speech classification model.
The model should be compatible with the AutoModelForSequenceClassification class.
https://huggingface.co/spaces/evaluate-measurement/toxicity
"""


class HuggingfaceToxicity():
    def __init__(self):
        # model downloading page: 
        #   https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target/tree/main
        # model stored location: 
        #   oss://antsys-adabrain/solutions/chatgpt/toxicity/roberta-hate-speech-dynabench-r4-target
        self.MODEL_NAME = "roberta-hate-speech-dynabench-r4-target"
        self.save_dir = '../'
        self.OSS_URL = 'oss://antsys-adabrain/solutions/chatgpt/toxicity'
        self.MODEL_PATH = os.path.join(self.save_dir, self.model_name)  # local disk path: /disk5/home/zheng.gao
        self.OSS_KEY = ""  # ossutil key
        self.ENDPOINT = ""  # ossutil endpoint
        if not os.path.exists(self.MODEL_PATH): 
            os.system("ossutil_linux64 -i LTAI4GAwFGfYhd2pXSupUL1f  -k " + 
                      self.OSS_KEY + " -e " + self.ENDPOINT + 
                      " cp -r " + self.OSS_URL + "/" + self.MODEL_NAME + " " + self.MODEL_PATH)
        self.MODEL_PATH = os.path.join(self.MODEL_PATH, 
                                       self.oss_url.replace("oss://antsys-adabrain/", ""), 
                                       self.MODEL_NAME)  # the actual downloaded model path
        self.metric = evaluate.load(os.path.join(
            os.path.dirname(__file__), './evaluate_factory/toxicity.py'), self.MODEL_PATH, module_type="measurement") 

    def compute(self, predictions: List[List], references: List[List], extras: List[Dict]):
        predictions = [tokenize(p[0]) for p in predictions]
        inputs = []
        for extra in extras:
            input = tokenize(extra["input"])
            inputs.append(input)

        merged_inputs = []
        for prediction, input in zip(predictions, inputs):
            merged_input = prediction + " " + input
            merged_inputs.append(merged_input)

        results = self.metric.compute(predictions=merged_inputs, aggregation="ratio") 
        results = round(results["toxicity_ratio"] * 100, 2)
        return results


if __name__ == "__main__":
    test = HuggingfaceToxicity()
    print(test.compute([["he is a douchebag", "she went to the library"],
                        ["she went to the library", "he is a douchebag"]],
                       [["she went to the library", "he is a douchebag"]], None))

    # toxicity = evaluate.load(os.path.join(
    #         os.path.dirname(__file__), './evaluate_factory/toxicity.py'), module_type="measurement")
    # input_texts = ["she went to the library", "he is a douchebag"]
    # results = toxicity.compute(predictions=input_texts, aggregation="ratio", toxic_label='hate')

    # print(results)
