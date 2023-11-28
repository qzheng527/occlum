import argparse
import os
import re

import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from utils.file_utils import load_json, save_json


def extract_choice(response: str) -> str:
    '''
        Always return a choice, even cannot match by regex,
        to ensure fair comparison to other models.
    '''
    if response == '':
        return ""
    choices = ["A", "B", "C", "D", "E"]
    if response == '':
        return ""
    # 1. Single match
    patterns = [
        (r'答案(选项)?(是|为)：? ?([ABCDE])', 3),
        (r'答案(是|为)选项 ?([ABCDE])', 2),
        (r'故?选择?：? ?([ABCDE])',1),
        (r'([ABCDE]) ?选?项(是|为)?正确',1),
        (r'正确的?选项(是|为) ?([ABCDE])',2),
        (r'答案(应该)?(是|为)([ABCDE])',3),
        (r'选项 ?([ABCDE]) ?(是|为)?正确',1),
        (r'选择答案 ?([ABCDE])',1),
        (r'答案?：?([ABCDE])',1),
        (r'([ABCDE])(选?项)?是?符合题意',1),
        (r'答案选项：? ?([ABCDE])', 1), # chatglm
        (r'答案(选项)?为(.*?)([ABCDE])', 3), # chatgpt
        (r'选项([ABCDE])是最恰当的', 1),
        (r'选项([ABCDE]).*最恰当', 1),
        (r'选项([ABCDE]).*最能恰当', 1),
        (r'选项([ABCDE]).*最能', 1),
        (r'最恰当.*是选项([ABCDE])', 1),
        (r'correct answer is.*([ABCDE])', 1),
    ]
    for pattern, idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            answer = m.group(idx)
            assert answer in choices
            return answer

    # 2. Recursive match
    patterns = [
        (r'([ABCDE])(.*?)当选', 1),
        (r'([ABCDE])(.*?)正确', 1),
    ]
    for pattern, idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            while m:
                answer = m.group(idx)
                m = re.search(pattern, m.group(0)[1:], re.M)
            assert answer in choices
            return answer

    # 3. Weak single match
    patterns = [
        (r'[^不]是：? ?([ABCDE])', 1),
    ]
    for pattern,idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            answer = m.group(idx)
            assert answer in choices
            return answer

    # 4. Check the only mentioned choices
    pattern = r'^[^ABCDE]*([ABCDE])[^ABCDE]*$'
    m = re.match(pattern, response)
    if m:
        answer = m.group(1)
        assert answer in choices
        return answer

    # 5. Check the only mentioned choices in the start of the sentence
    m = re.match(pattern, response[:4])
    if m:
        answer = m.group(1)
        assert answer in choices
        return answer

    m = re.match(pattern, response[:2])
    if m:
        answer = m.group(1)
        assert answer in choices
        return answer

    return ""


def get_score(args):
    model_name = args.model_name
    result_path = args.result_path
    score_save_path = args.score_save_path

    score_map = dict()

    result_path = os.path.join(result_path, f'{model_name}_result.json')
    result_list = load_json(result_path)

    df = pd.DataFrame(result_list)

    for index, row in df.iterrows():
        df.at[index, f'{model_name}_extract'] = extract_choice(row[f'{model_name}_answer'])

    grouped = df.groupby('task')
    for task, group in grouped:
        y_true_task = group['answer'].values
        y_pred_task = group[f'{model_name}_extract'].values
        accuracy_task = accuracy_score(y_true_task, y_pred_task)
        print("Accuracy for Task", task, ":", accuracy_task)
        score_map[task] = accuracy_task
    save_json(score_map, os.path.join(score_save_path, 'score.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True, type=str)
    parser.add_argument('--result_path', required=True, type=str)
    parser.add_argument('--score_save_path', required=True, type=str)
    args = parser.parse_args()
    get_score(args)
