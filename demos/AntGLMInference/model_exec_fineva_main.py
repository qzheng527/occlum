import argparse
import os
import time

from tqdm import tqdm

from antglm_evaluator import AntGLMEvaluator
from utils.file_utils import load_json, save_json


def model_fineva_main(args):
    model_name = args.model_name
    model_path = args.model_path
    datasets_path = args.datasets_path
    save_path = args.save_path

    # 导入模型
    evaluator = AntGLMEvaluator(model_path)
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
        print(f' case 耗时：{round((et1-st1)*1000, 4)}ms')
        data_dict[f'{model_name}_answer'] = model_answer
    
    et = time.time()
    print('cost time:', round((et - st) * 1000, 4), 'ms')
    
    # 保存为 json
    try:
        save_json(dataset_list, os.path.join(save_path, f'{model_name}_result.json'))
    except Exception as e:
        print(f"Error occurred while saving as json: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True, type=str)
    parser.add_argument('--model_path', required=False, type=str)
    parser.add_argument('--datasets_path', required=True, type=str)
    parser.add_argument('--save_path', required=True, type=str)
    args = parser.parse_args()
    model_fineva_main(args)
