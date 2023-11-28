import time
import os
import argparse

from transformers import AutoModel, AutoTokenizer


class ChatGLM2Evaluator:
    def __init__(self, model_path="/models/chatglm2-6b"):
        self.model_path = model_path
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).float()
        self.model = self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

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
    evaluator = ChatGLM2Evaluator(model_path)
    print("问：")
    print(args.prompt)
    st = time.time()
    output = evaluator.answer(args.prompt)
    et = time.time()
    print("答:")
    print(output)
    print(f'cost time: {et-st} s')
