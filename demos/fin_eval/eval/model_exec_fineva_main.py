import argparse
import os
import time

from tqdm import tqdm

from chatglm2_evaluator import ChatGLM2Evaluator
from llama2_evaluator import LLaMA2Evaluator
from qwen_evaluator import QwenEvaluator
from baichuan2_evaluator import Baichuan2Evaluator

# bigdl prefix means that it is optimized by BigDL LLM
from bigdl_chatglm2_evaluator import BigDLChatGLM2Evaluator
from bigdl_llama2_evaluator import BigDLLLaMA2Evaluator
from bigdl_qwen_evaluator import BigDLQwenEvaluator
from bigdl_baichuan2_evaluator import BigDLBaichuan2Evaluator

from utils.file_utils import load_json, save_json

def get_evaluator(eval, model_path):
    """
    获取对应的 Evaluator
    """
    if eval == 'bigdl-chatglm2':
        evaluator = BigDLChatGLM2Evaluator(model_path)
    elif eval == 'chatglm2':
        evaluator = ChatGLM2Evaluator(model_path)
    elif eval == 'llama2':
        evaluator = LLaMA2Evaluator(model_path)
    elif eval == 'bigdl-llama2':
        evaluator = BigDLLLaMA2Evaluator(model_path)
    elif eval == 'qwen':
        evaluator = QwenEvaluator(model_path)
    elif eval == 'bigdl-qwen':
        evaluator = BigDLQwenEvaluator(model_path)
    elif eval == 'baichuan2':
        evaluator = Baichuan2Evaluator(model_path)
    elif eval == 'bigdl-baichuan2':
        evaluator = BigDLBaichuan2Evaluator(model_path)
    else:
        print(f"invalid evaluator: {eval}")
        raise ValueError("invalid evaluator")
    
    return evaluator

def model_fineva_main(args):
    model_name = args.model_name
    model_path = args.model_path
    datasets_path = args.datasets_path
    save_path = args.save_path
    eval = args.eval

    # 导入模型
    evaluator = get_evaluator(eval, model_path)
    print(f'模型 {model_name} 加载成功')

    # 导入数据集
    dataset_list = load_json(datasets_path)
    print(f'数据集 {datasets_path} 加载成功')

    st = time.time()
    for data_dict in tqdm(dataset_list):
        prompt_query = data_dict['query']
        print(f'正在处理问题：{prompt_query}')
        # 大模型输出
        st1 = time.time()
        model_answer = evaluator.answer(prompt_query)
        et1 = time.time()
        print(f' 回答：{model_answer}')
        print(f' case 耗时：{round((et1-st1)*1000, 4)}ms')
        data_dict[f'{model_name}_answer'] = model_answer
    
    et = time.time()
    print(f'cost time: {et-st} s')
    
    # 保存为 json
    try:
        save_json(dataset_list, os.path.join(save_path, f'{model_name}_result.json'))
    except Exception as e:
        print(f"Error occurred while saving as json: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default="llm")
    parser.add_argument('--model-path', required=False, type=str)
    parser.add_argument('--datasets-path', required=True, type=str)
    parser.add_argument('--save-path', required=True, type=str)
    parser.add_argument('--eval', required=True, type=str)
    args = parser.parse_args()
    model_fineva_main(args)
