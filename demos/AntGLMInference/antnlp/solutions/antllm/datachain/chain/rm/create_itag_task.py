import json
import uuid
import pandas as pd
from datetime import datetime
from typing import List
from collections import defaultdict
from solutions.antllm.datachain.utils import load_jsonl
from solutions.antllm.datachain.itag import Itag


def create_task(file_path,
                task_name,
                user_id="1508785651817058304",
                user_no="272483",
                biz="antllm_med",
                biz_code="antllm_med_biz_code",
                template_id="1669324871732826112"):
    # CTO大模型租户
    tenant = "72f60808"
    prk = '''MIICdgIBADANBgkqhkiG9w0BAQEFAASCAmAwggJcAgEAAoGBAMmyERQ7kMujtPDXhy+vnMXnczstfsOOaTH34bdNO5tZybXmP18frBfXjGrGZpzMqUjyk8JN4ZkenszIWy3xa5J037L1vsRzolkKl0radwE37wlbidnQXIKjrN8i2GC+I6Xb3iYU2kUvvPGM+kYItZYe8zF/bjBTknwODaS4zcRlAgMBAAECgYAs5uWCeZhMnY9kprbD2Pav4Ez4+bHk20l0BFlNs3X3qc+MHUwyYxyu2h+6jZy+f5mYUdivQyNcMULtGBWkbsChnw+bXXCRhL9K50Vjrufgxslju9gVl9/r3BE8cq4ZjLQdY1PbsqcTEqfNfXiSd4Fv66I9/ERl0eHShw8e43H8wQJBAOagUo88/z9eElANOPj/EUcjKUBhDU89KSNhDfLoIPkYzeiymqjLSYYhd/MUAQxYKz0U5J8dwzXGbt3omLij8fkCQQDf4uZ4ebeip1BDkg3gAJfYwU2trILPkBJMcf0f5D5RGj9gmhFud2NdndvWgNFsu52tb/DjyBcZhaYSHJcq7QDNAkByZXwOSPdje0oiIyzrdbogSzSfFoT/lRrezbmZj8MrTD52+oD00UF7IwbYsEeE1Ac+mSp+Mskt12wO7t0yWUAhAkEAux8Sj4jzsY9zpzYQQLNeNnzBprFzl3WLxbbT3+7NIs30QJIklZZVR25jyjFaWC2rCMVxqX+Xxu4MMkERG4CA1QJAS+LY57JHJegsWyIhnDu4yNPhM46v0ZrkKLbRadJRiPuZ3x/P7XGTaSbxGyE5Iad2uZOI9GXFgLrvc5Wt/IV7uA=='''  # noqa
    itag = Itag(tenant, prk, user_id, user_no)
    
    task_config = {
        'sharedUsers': user_id,
        'sharedUserGroups': '',
        'markUsers': user_id,
        'markUserGroups': '',
        'checkUsers': user_id,
        'checkUserGroups': '',
        'samplingUsers': user_id,
        'samplingUserGroups': '',
        'voteNum': 2,
        'assignCount': 1
    }

    biz_no = f"{biz}#{datetime.now().strftime('%Y-%m-%d')}#{uuid.uuid4()}" 
    task_id = itag.create_task(task_name, file_path, template_id, biz_code, biz_no, **task_config)
    print(task_id)


def gen_rm_task_data(inpaths: List[str], 
                     outpath: str, 
                     st: int = 0, 
                     ed: int = 10000,
                     inputkey: str = "input",
                     goldkey: str = "output",
                     candkey: str = "llm_cands",
                     k: int = 3,
                     max_cand: int = 7,
                     min_cand_num: int = 2,
                     dedup: bool = True,
                     eval: bool = False):
    def format_cand_content(cand: str):
        # 暂时不做后处理
        return cand

    def is_bad_prompt(prompt):
        # 过滤demo日志中批量调用的无用数据
        bad_parts = ["Generate query sql based on table information and questions",
                     "你需要提取一个问句的api名字和输入参数，输入参数从以下input_description中选取，输出格式为xxx(yyy=zzz),其中xxx为api名字，yyy为输入参数名字",
                     "Generate query expression based on entity and relation\nYou have access to the following tools"]
        for part in bad_parts:
            if part in prompt:
                return True
        return False
    
    task_data = []
    prompt2cands = defaultdict(list)
    for inpath in inpaths:
        if inpath.endswith(".csv"):
            df = pd.read_csv(inpath, encoding="utf-8")
            df.to_json(inpath + ".jsonl", orient="records", lines=True, force_ascii=False)
            inpath += ".jsonl"
        samples = load_jsonl(inpath)[st: ed]
        for sample in samples:
            post = sample[inputkey]
            if is_bad_prompt(post):
                continue
            if goldkey:
                cand = sample[goldkey]
                if eval:
                    post = post + f'\n\n参考答案：{cand}'
                elif not dedup or cand not in prompt2cands[post]:
                    prompt2cands[post].append(cand)
            if isinstance(sample[candkey], str):
                sample[candkey] = json.loads(sample[candkey])
            cands = sample[candkey][:k]
            cands = [cand for cand in cands if cand and cand != "FAIL"]
            for cand in cands:
                cand = format_cand_content(cand)
                if not dedup or cand not in prompt2cands[post]:
                    prompt2cands[post].append(cand)

    max_cand_num = 0
    for post, cands in prompt2cands.items():
        cands = cands[:max_cand]  # 最多保留max_cand个候选
        if min_cand_num and len(cands) >= min_cand_num:
            task_data.append([post] + cands)
        if len(cands) > max_cand_num:
            max_cand_num = len(cands)
    
    columns = ["post"]
    for i in range(max_cand_num):
        columns.append(f"item{i+1}")
    
    df = pd.DataFrame(task_data, columns=columns)
    df.to_csv(outpath, encoding="utf-8-sig", index=False)


if __name__ == "__main__":
    delta = 4
    for st in range(8, 20, delta):
        outpath = f"datasets/huatuo_sft_train_{st}k-{st+delta}k.csv"
        gen_rm_task_data(inpaths=["/ossfs/workspace/data/huatuo_sft_train_medsft0731.jsonl",
                                  "/ossfs/workspace/data/huatuo_sft_train_medsftv8.jsonl"],
                         outpath=outpath,
                         k=3,
                         st=st * 1000,
                         ed=(st + delta) * 1000)
        create_task(outpath, f"RM标注任务-医疗-huatuo_sft_{st}k-{st+delta}k") 
