import time
import os
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


class Baichuan2Evaluator:
    def __init__(self, model_path="/models/Baichuan2-13B-Chat"):
        self.model_path = model_path
        set_seed(42)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", trust_remote_code=True).eval()

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
    evaluator = Baichuan2Evaluator(model_path)
    print("问：")
    print(args.prompt)
    st = time.time()
    output = evaluator.answer(args.prompt)
    et = time.time()
    print("答:")
    print(output)
    print(f'cost time: {et-st} s')
