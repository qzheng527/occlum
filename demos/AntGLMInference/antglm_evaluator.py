import sys
import time
import os
import argparse

# 将antnlp所在的文件夹路径放入sys.path中
curpath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curpath, "./antnlp"))
from solutions.antllm.antllm.inference.glm_predictor import GLMForInference


class AntGLMEvaluator:
    def __init__(self, model_path):
        self.model = GLMForInference(model_path)

    def answer(self, query):
        print("问：")
        print(query)
        st = time.time()
        output = self.model.answer(query).strip()
        et = time.time()
        print("答:")
        print(output)
        print('cost time:', round((et - st) * 1000, 4), 'ms')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict demo for AntGLM')
    parser.add_argument('--model-path', type=str, default="/models/AntGLM-10B-RLHF-20230930",
                        help='Model absolute path')
    parser.add_argument('--prompt', type=str, default="你是谁？",
                        help='Prompt to infer')

    args = parser.parse_args()
    model_path = args.model_path
    evaluator = AntGLMEvaluator(model_path)
    evaluator.answer(args.prompt)
