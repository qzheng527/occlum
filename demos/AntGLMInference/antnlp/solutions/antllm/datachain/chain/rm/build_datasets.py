import json
import os
import random
import pandas as pd
from collections import defaultdict
from typing import List
from solutions.antllm.datachain.utils import load_jsonl, dump_jsonl


def check_reason_via_keywords(item, kws=[]):
    for reason in item.get("sortStamp", []):
        for kw in kws:
            if kw in reason:
                return True
    return False


def get_samples(prompt, list1, list2, prompt_type="", source=""):
    samples = []
    for item1 in list1:
        for item2 in list2:
            if check_reason_via_keywords(item2, ["判断困难"]):
                continue
            samples.append({
                "prompt": prompt,
                "prompt_type": prompt_type,
                "chosen": item1["value"],
                "rejected": item2["value"],
                "source": source,
            })
    return samples


def parse_rm_task_result(inpath: str, 
                         outpath: str,
                         rank_keys: List[str] = ["Rank 1", "Rank 2", "Rank 3", "Rank 4", "Rank 5"],
                         omit_bad: bool = False,
                         omit_hard: bool = False,
                         only_choose_best: bool = False,
                         only_choose_rank1: bool = False,
                         source: str = "itag"):        
    df = pd.read_csv(inpath, encoding="utf-8", error_bad_lines=False)
    mark_cnt = 0
    bad_prompt_cnt = 0
    hard_prompt_cnt = 0
    tie_cnt = 0
    samples = []
    for idx, row in df.iterrows():
        if row["子任务包状态"] in ["标注中"] or pd.isna(row["对话排序"]):
            continue
        mark_cnt += 1
            
        mark_result = json.loads(row["对话排序"])
        # mark_result = json.loads(mark_result[0]["MarkResult"])
        prompt_label = mark_result["annotations"][0]["labels"].get("对话标签", "")
        prompt_type = mark_result["annotations"][0]["labels"].get("问题分类", "")   
        mark_result = json.loads(mark_result["annotations"][0]["exif"]["data"])
        
        if omit_bad and prompt_label == "题目质量差":
            bad_prompt_cnt += 1
            continue
        if omit_hard and prompt_label == "题目困难":
            hard_prompt_cnt += 1
            continue
        flag = False
        for i in range(len(rank_keys)):
            key1 = rank_keys[i]
            for key2 in rank_keys[i + 1:]:
                values1 = mark_result[key1]
                values2 = mark_result[key2]
                if values1 and values2:
                    flag = True
                    samples += get_samples(row["post"], values1, values2, prompt_type, source)
            if only_choose_rank1:
                break
            if only_choose_best and flag:
                break
        if not flag:
            tie_cnt += 1

    print(f'df len: {len(df)}\tmark cnt: {mark_cnt}\tbad cnt: {bad_prompt_cnt}\t'
          f'hard cnt: {hard_prompt_cnt}\ttie cnt: {tie_cnt}\tcomparision cnt: {len(samples)}')
    if outpath:
        dump_jsonl(samples, outpath)
    return samples


def shuffle_data(samples,
                 inpath,
                 train_path,
                 dev_path,
                 key="prompt",
                 train_ratio=0.8,
                 k=100):
    # prompt, chosen, rejected
    if inpath:
        samples = load_jsonl(inpath)
    values = set()
    v2s = defaultdict(list)
    for sample in samples:
        values.add(sample[key])
        v2s[sample[key]].append(sample)
    values = list(values)
    random.shuffle(values)
    train_size = int(train_ratio * len(values))
    train_values = values[:train_size]
    dev_values = values[train_size:]
    train = []
    dev = []
    
    for v in train_values:
        if k and len(v2s[v]) > k:
            random.shuffle(v2s[v])
        train += v2s[v][:k]
    for v in dev_values:
        if k and len(v2s[v]) > k:
            random.shuffle(v2s[v])
        dev += v2s[v][:k]

    if train_path:
        dump_jsonl(train, train_path)
    if dev_path:
        dump_jsonl(dev, dev_path)
    print(f"train size: {len(train)}")
    print(f"dev size: {len(dev)}")


def overall_process_med_rm_data():
    train_path = "chatgpt/RM/rm_med_v3.train.jsonl"
    dev_path = "chatgpt/RM/rm_med_v3.dev.jsonl"
    samples = []

    dirnames = [
        "chatgpt/itag/itag-medsftv1",
        "chatgpt/itag/itag-medsft-wumav2"
    ]
    for dirname in dirnames:
        itag_inpaths = [os.path.join(dirname, filename) for filename in os.listdir(dirname)]
        for inpath in itag_inpaths:
            print(inpath)
            samples += parse_rm_task_result(inpath=inpath,
                                            outpath="",
                                            source=os.path.basename(dirname),
                                            omit_bad=True,
                                            omit_hard=True,
                                            only_choose_rank1=True
                                            )

    shuffle_data(samples=samples,
                 inpath="",
                 train_path=train_path,
                 dev_path=dev_path,
                 train_ratio=0.9,
                 k=7)
