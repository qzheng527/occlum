import time
import os
import argparse

from bigdl.llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


class BigDLBaichuan2Evaluator:
    def __init__(self, model_path="/models/Baichuan2-13B-Chat"):
        self.model_path = model_path

        # Load model in 4 bit,
        # which convert the relevant layers in the model into INT4 format
        # if your selected model is capable of utilizing previous key/value attentions
        # to enhance decoding speed, but has `"use_cache": false` in its model config,
        # it is important to set `use_cache=True` explicitly in the `generate` function
        # to obtain optimal performance with BigDL-LLM INT4 optimizations
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                    load_in_4bit=True,
                                                    trust_remote_code=True,
                                                    use_cache=True)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                trust_remote_code=True)

    def answer(self, query):
        messages = []
        messages.append({"role": "user", "content": query})
        response = self.model.chat(self.tokenizer, messages)
        return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict demo for Baichuan2')
    parser.add_argument('--model-path', type=str, default="/models/Baichuan2-13B-Chat",
                        help='Model absolute path')
    parser.add_argument('--prompt', type=str, default="股票里的金叉代表什么意思？"
                , help='Prompt to infer')
    # parser.add_argument('--prompt', type=str, default="给我一份去北京玩三天的计划"
    #         , help='Prompt to infer')
    args = parser.parse_args()
    model_path = args.model_path
    evaluator = BigDLBaichuan2Evaluator(model_path)
    print("问：")
    print(args.prompt)
    st = time.time()
    output = evaluator.answer(args.prompt)
    et = time.time()
    print("答:")
    print(output)
    print(f'cost time: {et-st} s')
