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
        return self.model.answer(query).strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict demo for AntGLM')
    parser.add_argument('--model-path', type=str, default="/models/AntGLM-10B-RLHF-20230930",
                        help='Model absolute path')
    parser.add_argument('--prompt', type=str, default="你是一名专业的金融专家，你现在进行关于中国精算师的问答。题目是：已知$\alpha(12)=1.000281$, $\beta(12)=0.46811951$, $\alpha_{65}=9.89693$,假设死亡均匀分布。计算(65)退休每月期初1000的终生年金精算现值为____。以下哪个选项可以该题目最恰当的答案？\n\nA.113179\nB.113189\nC.113199\nD.113209\n\n你的输出只能是A、B、C、D四个字母中的一个"
                , help='Prompt to infer')

    args = parser.parse_args()
    model_path = args.model_path
    evaluator = AntGLMEvaluator(model_path)
    print("问：")
    print(args.prompt)
    st = time.time()
    output = evaluator.answer(args.prompt)
    et = time.time()
    print("答:")
    print(output)
    print('cost time:', round((et - st) * 1000, 4), 'ms')
