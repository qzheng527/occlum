# coding=utf-8
# @Author: ytt360131
# @Date: 2023-09-08
import os
import csv
import json
import fire
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.generation import GenerationConfig
from solutions.antllm.datachain.utils import download_oss
from solutions.antllm.antllm.inference.glm_predictor import GLMForInference
from solutions.antllm.datachain.chain.multiturn_dialogs.convert import DialogDataConvertChain


def write_csv(save_dir, all_info):
    with open(save_dir, 'w', encoding='utf-8-sig') as f_w:
        writer = csv.writer(f_w)
        writer.writerows(all_info)


def get_oss_file(oss_dir, temp_dir='./temp_dir'):
    """
    从oss上面下载需要测试的文件
    """
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    local_path = Path(temp_dir).joinpath(Path(oss_dir).name)
    if not local_path.exists():
        download_oss(oss_dir, local_path)
    return local_path


def get_query_list(file_dir):
    """
    提取所有多轮对话的
    file_dir：文件路径

    Return：
    all_multurns_list: 所有待测试的多轮query
    [
        [xxxx, ["您好", "请给我推荐", ..., "谢谢"], "多轮重复"],
        [xxxx, ["我想了解手机", "怎么学数学", ..., "好的"], "多轮重复"]
    ]
    """
    all_multurns_list = []
    with open(file_dir, 'r', encoding='utf-8') as f_r:
        all_lines = csv.reader(f_r)
        for idx, sgl_line in enumerate(all_lines):
            if idx == 0 :
                continue
            contextId, query_concate, cate = sgl_line[:3]
            query_list_o = query_concate.split("\n")
            query_list = []
            for query in query_list_o:
                if len(query.strip()) == 0:
                    continue
                query_list.append(query.lstrip())
            all_multurns_list.append([contextId, query_list, cate])
    return all_multurns_list


def get_llm_multurn_res(
    model_path,
    oss_dir,
    save_dir="./repition_test_result.csv",
    llm_role="机器人",
    multi_gpu=True,
    num_beams=1,
    temperature=0.4,
    top_k=50,
    top_p=1,
    do_sample=True,
    max_length=2048,
):
    """
    通过llm对每一轮输入都进行模型输出
    Input：
        model_path:  llm模型路径
        file_dir: 测试数据路径
        save_dir: 结果保存路径
        llm_role: 模型的角色信息
    """
    file_dir = get_oss_file(oss_dir)
    all_multurns_list = get_query_list(file_dir)
    all_multurn_dialog_list = []

    prompt2turn_chain = DialogDataConvertChain(is_prompt_to_turns=True)

    if 'chatglm2' in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        llm_bot = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda().eval()
    elif 'chatglm' in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        llm_bot = AutoModel.from_pretrained(
            model_path, trust_remote_code=True).half().cuda().eval()
    elif 'qwen' in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        llm_bot = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", trust_remote_code=True).eval()
        llm_bot.generation_config = GenerationConfig.from_pretrained(model_path)
        llm_bot.generation_config.chat_format = 'chatml'
        llm_bot.generation_config.max_window_size = max_length
    elif 'baichuan' in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        llm_bot = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        llm_bot.generation_config = GenerationConfig.from_pretrained(model_path)
        llm_bot.generation_config.user_token_id = 195
        llm_bot.generation_config.assistant_token_id = 196
    else:
        llm_bot = GLMForInference(path=model_path, multi_gpu=multi_gpu)

    for multurns in all_multurns_list:
        dialog = []
        contextId, query_list, cate = multurns
        chat_history = ""
        for p_idx, query in enumerate(query_list):
            query_prompt = query if p_idx == 0 else chat_history + f"第{p_idx + 1}轮\n用户: {query}\n{llm_role}: "

            if any([
                'chatglm' in model_path.lower(),
                'qwen' in model_path.lower(),
                'baichuan' in model_path.lower(),
            ]):
                try:
                    to_dialog = prompt2turn_chain.run({'input': chat_history, 'output': ''}) or {'turns': []}
                    glm_history = [
                        (item['user'], item['assistant'])
                        for item in to_dialog['turns']
                    ]
                    # 截断
                    glm_history = glm_history[::-1]
                    count = 0
                    max_len = max_length - len(query)
                    history_idx = 0
                    for idx, (a, b) in enumerate(glm_history):
                        count = len(a) + len(b)
                        if count > max_len:
                            history_idx = idx
                            break
                    glm_history = glm_history[:history_idx]
                    glm_history = glm_history[::-1]

                    if 'chatglm' in model_path.lower():
                        out = llm_bot.chat(
                            tokenizer,
                            query,
                            glm_history,
                            num_beams=num_beams,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            do_sample=do_sample,
                        )[0]
                    elif 'qwen' in model_path.lower():
                        llm_bot.generation_config.num_beams = num_beams
                        llm_bot.generation_config.temperature = temperature
                        llm_bot.generation_config.top_k = top_k
                        llm_bot.generation_config.top_p = top_p
                        llm_bot.generation_config.do_sample = do_sample
                        llm_bot.generation_config.repetition_penalty = None

                        out = llm_bot.chat(
                            tokenizer,
                            query,
                            history=glm_history,
                        )[0]
                    elif 'baichuan' in model_path.lower():
                        messages = []
                        llm_bot.generation_config.num_beams = num_beams
                        llm_bot.generation_config.temperature = temperature
                        llm_bot.generation_config.top_k = top_k
                        llm_bot.generation_config.top_p = top_p
                        llm_bot.generation_config.do_sample = do_sample
                        llm_bot.generation_config.repetition_penalty = None

                        for turn in to_dialog['turns']:
                            messages.append({'role': 'user', 'content': turn['user']})
                            messages.append({'role': 'assistant', 'content': turn['assistant']})
                        messages.append({'role': 'user', 'content': query})
                        out = llm_bot.chat(tokenizer, messages)
                except Exception as e:
                    print(f'other model inference failed, query_prompt: {query_prompt}, error: {e}')
            else:
                out = llm_bot.answer(
                    query_prompt,
                    num_beams=num_beams,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                )

            chat_history += f"第{p_idx + 1}轮\n用户: {query}\n{llm_role}: {out}\n"
            dialog.append({
                "paragraphId": p_idx + 1,
                "sentenceId": p_idx * 2 + 1,
                "speaker": "用户",
                "speech": query,
                "flag": "1",
                "extra": {"paragraphRemark": cate}
            })
            dialog.append({
                "paragraphId": p_idx + 1,
                "sentenceId": p_idx * 2 + 2,
                "speaker": "机器人",
                "speech": out,
                "flag": "1",
                "extra": ""
            })
        all_multurn_dialog_list.append([contextId, query_list, cate, json.dumps(dialog, ensure_ascii=False)])
    write_csv(save_dir, [["contextId", "query", "cate", "dialogue"]] + all_multurn_dialog_list)
    return all_multurn_dialog_list


if __name__ == "__main__":
    fire.Fire()
