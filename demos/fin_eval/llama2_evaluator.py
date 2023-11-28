import time
import os
import argparse

import torch
from bigdl.llm.transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer, AutoTokenizer

# you could tune the prompt based on your own model,
# here the prompt tuning refers to https://huggingface.co/georgesung/llama2_7b_chat_uncensored#prompt-style
LLAMA2_PROMPT_FORMAT = """### HUMAN:
{prompt}

### RESPONSE:
"""


class LLaMA2Evaluator:
    def __init__(self, model_path="/models/llama-2-7b-hf"):
        self.model_path = model_path
        # Load model in 4 bit,
        # which convert the relevant layers in the model into INT4 format
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                    load_in_4bit=True,
                                                    trust_remote_code=True)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def answer(self, query):
        # prompt = LLAMA2_PROMPT_FORMAT.format(prompt=query)
        inputs = self.tokenizer.encode(query, return_tensors="pt")
        generate_ids = self.model.generate(inputs, max_length=512)
        output = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # output = self.model.generate(input_ids,
        #                         max_new_tokens=1024)
        # output_str = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # return output_str
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict demo for ChatGLM2')
    parser.add_argument('--model-path', type=str, default="/models/llama-2-7b-hf",
                        help='Model absolute path')
    parser.add_argument('--prompt', type=str, default="你是一名专业的金融专家，你现在进行关于国际经济学的问答\n题目是：在直接标价法情况下，如果远期汇率高于即期汇率，则____。\n以下哪个选项可以该题目最恰当的答案？\nA.远期外汇升水\nB.远期外汇贴水\nC.远期平价\nD.远期本币升水\n你的输出只能是A、B、C、D四个字母中的一个"
                , help='Prompt to infer')
    args = parser.parse_args()
    model_path = args.model_path

    evaluator = LLaMA2Evaluator(model_path)
    print("问：")
    print(args.prompt)
    st = time.time()
    output = evaluator.answer(args.prompt)
    et = time.time()
    print("答:")
    print(output)
    print(f'cost time: {et-st} s')
    