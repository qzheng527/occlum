import time
import argparse

from bigdl.llm.transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer

B_INST, E_INST = "[INST]", "[/INST]"

class BigDLLLaMA2Evaluator:
    def __init__(self, model_path="/models/llama-2-7b-hf"):
        self.model_path = model_path
        # Load model in 4 bit,
        # which convert the relevant layers in the model into INT4 format
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                    load_in_4bit=True,
                                                    trust_remote_code=True)

        # Load tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def answer(self, query, max_new_tokens=512):
        query = f"{B_INST} {query.strip()} {E_INST}"
        inputs = self.tokenizer.encode(query, return_tensors="pt")
        output = self.model.generate(inputs, max_length=max_new_tokens)
        output_str = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return output_str[len(query):].strip("!@#$%^&*()_-+=[]{}|\\;':\",.<>/?~` \t\n？。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict demo for ChatGLM2')
    parser.add_argument('--model-path', type=str, default="/models/llama-2-7b-hf",
                        help='Model absolute path')
    parser.add_argument('--prompt', type=str, default="股票里的金叉代表什么意思？"
                , help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                    help='Max tokens to predict')
    args = parser.parse_args()
    model_path = args.model_path

    evaluator = BigDLLLaMA2Evaluator(model_path)
    print("问：")
    print(args.prompt)
    st = time.time()
    output = evaluator.answer(args.prompt, args.n_predict)
    et = time.time()
    print("答:")
    print(output)
    print(f'cost time: {et-st} s')
    