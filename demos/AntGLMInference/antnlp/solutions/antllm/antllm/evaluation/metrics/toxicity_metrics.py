import os
import evaluate
from typing import List, Dict
import sys
import solutions.antllm.antllm.evaluation.scripts.fetch_benchmark_datasets as fbd
from tools.util.kmitool_util import kmi_cryptor
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
        self.model_name = "roberta-hate-speech-dynabench-r4-target"
        self.save_dir = './'
        self.oss_url = 'oss://antsys-adabrain/solutions/chatgpt/toxicity'
        self.MODEL_PATH = os.path.join(self.save_dir, self.model_name)  # local disk path: /disk5/home/zheng.gao
        if not os.path.exists(self.MODEL_PATH):
            FetchBenchmarkDataset(self.model_name, 
                                  self.save_dir, 
                                  self.oss_url, 
                                  ossutil="/usr/bin/ossutil")

        self.MODEL_PATH = os.path.join(self.MODEL_PATH, 
                                       self.oss_url.replace("oss://antsys-adabrain/", ""), 
                                       self.model_name)  # the actual downloaded model path
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


def FetchBenchmarkDataset(
        dataset_name="roberta-hate-speech-dynabench-r4-target",
        save_dir='~/',
        oss_url='oss://antsys-adabrain/solutions/chatgpt/toxicity',
        ossutil="/usr/bin/ossutil"
):
    oss_id = kmi_cryptor.get_value("adabrain_oss_id")
    oss_key = kmi_cryptor.get_value("adabrain_oss_key")
    oss_endpoint = kmi_cryptor.get_value("adabrain_oss_host")

    if ossutil is not None:
        ossutil = ossutil
    else:
        result = fbd.subprocess_popen('which ossutil64') 
        if isinstance(result, bool):
            print('Not found ossutil command! Please install it first or put its path under PATH variable!')
            sys.exit(0)
        else:
            ossutil = result[0]

    dataset_url = '{}/{}'.format(oss_url, dataset_name) 
    download_util = fbd.DownloadUtilNew() 
    try:
        download_util.download(
            dataset_url,
            output_dir=save_dir,
            oss_access_id=oss_id,
            oss_access_key=oss_key,
            oss_endpoint=oss_endpoint,
            keep_archive=True,
            ossutil=ossutil
        )
    except BaseException:
        raise BaseException('Not reached dataset {}.'.format(oss_url))


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
