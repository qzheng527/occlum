import time
import os
import argparse

from bigdl.llm.transformers import AutoModel
from transformers import AutoTokenizer


class BigDLChatGLM2Evaluator:
    def __init__(self, model_path="/models/chatglm2-6b"):
        self.model_path = model_path
        # Load model in 4 bit,
        # which convert the relevant layers in the model into INT4 format
        self.model = AutoModel.from_pretrained(model_path,
                                        load_in_4bit=True,
                                        trust_remote_code=True)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                trust_remote_code=True)

    def answer(self, query):
        response, history = self.model.chat(self.tokenizer, query, history=[])
        return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict demo for ChatGLM2')
    parser.add_argument('--model-path', type=str, default="/models/chatglm2-6b",
                        help='Model absolute path')
    parser.add_argument('--prompt', type=str, default="股票里的金叉代表什么意思？"
                , help='Prompt to infer')
    args = parser.parse_args()
    model_path = args.model_path
    evaluator = BigDLChatGLM2Evaluator(model_path)
    print("问：")
    print(args.prompt)
    st = time.time()
    output = evaluator.answer(args.prompt)
    et = time.time()
    print("答:")
    print(output)
    print(f'cost time: {et-st} s')
