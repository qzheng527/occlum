import time
import os
import argparse

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

B_INST, E_INST = "[INST]", "[/INST]"

class LLaMA2Evaluator:
    def __init__(self, model_path="/models/llama-2-7b-hf"):
        self.model_path = model_path
        # Load model
        self.model = LlamaForCausalLM.from_pretrained(
            model_path, device_map="auto").eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def answer(self, query, max_new_tokens=512):
        query = f"{B_INST} {query.strip()} {E_INST}"
        inputs = self.tokenizer.encode(query, return_tensors="pt")
        output = self.model.generate(inputs, max_length=max_new_tokens)
        output_str = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return output_str[len(query):].strip("!@#$%^&*()_-+=[]{}|\\;':\",.<>/?~` \t\n？。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict demo for llama2')
    parser.add_argument('--model-path', type=str, default="/models/llama-2-7b-hf",
                        help='Model absolute path')
    parser.add_argument('--prompt', type=str, default="股票里的金叉代表什么意思？"
                , help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                help='Max tokens to predict')
    args = parser.parse_args()
    model_path = args.model_path

    evaluator = LLaMA2Evaluator(model_path)
    print("问：")
    print(args.prompt)
    st = time.time()
    output = evaluator.answer(args.prompt, args.n_predict)
    et = time.time()
    print("答:")
    print(output)
    print(f'cost time: {et-st} s')
    