# coding=utf-8
# @Author: ytt360131
# @Date: 2023-07-27

import json
import requests
# from Crypto.Cipher import AES
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from binascii import b2a_hex
from html import unescape
from concurrent.futures import ThreadPoolExecutor, as_completed


def aes_encrypt(data, key):
    """aes加密函数，如果data不是16的倍数【加密文本data必须为16的倍数！】，那就补足为16的倍数
    :param key:
    :param data:
    """
    iv = b"1234567890123456"
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), None)
    block_size = algorithms.AES.block_size
    # 判断data是不是16的倍数，如果不是用b'\0'补足
    if len(data) % block_size != 0:
        add = block_size - (len(data) % block_size)
    else:
        add = 0
    data += b'\0' * add
    encryptor = cipher.encryptor()
    encrypted = encryptor.update(data) + encryptor.finalize()  # aes加密
    result = b2a_hex(encrypted)  # b2a_hex encode  将二进制转换成16进制
    return result


def get_ans_from_gpt35_turbo(
    api_key: str,
    key: str,
    messages: dict,
    temperature: float = 1,
    max_tokens: int = 2048,
    model: str = "gpt-3.5-turbo",
    retry_times: int = 3,
):
    """
    history:[{"role":"user","content":"xxx"},{"role":"assistant","content":"xxx"}...]
    """
    try:
        if isinstance(key, str):
            key = bytes(key, encoding='utf-8')
        param = {
            "serviceName": "chatgpt_prompts_completions_query_dataview",
            # "visitDomain": "BU_cto",
            # "visitBiz": "BU_cto_llm",
            # "visitBizLine": "BU_cto_llm_line",
            "visitBiz": "BU_cto_dialogue",
            "visitBizLine": "BU_cto_dialogue_line",
            "visitDomain": "BU_cto",
            "cacheInterval": 0,
            "queryConditions": {
                "url": "%s",
                "model": model,
                "max_tokens": str(max_tokens),
                "n": "1",
                "temperature": "{}".format(temperature),
                "api_key": api_key,
                "messages": messages,
            },
        }

        headers = {
            'Content-Type': 'application/json'
        }
        url = 'https://zdfmng.alipay.com/commonQuery/queryData'
        # url = "https://zdfmng-pre.alipay.com/commonQuery/queryData"
        data = json.dumps(param) % url
        data = data.encode('utf-8')

        post_data = {
            "encryptedParam": aes_encrypt(data, key).decode()
        }

        for _ in range(retry_times):
            response = requests.post(url, data=json.dumps(post_data), headers=headers)
            # result = json.loads(unescape(response.json()["data"]["values"]["data"]).strip("\\n"))
            if response.json()["success"]:
                break
        if not response.json()["success"]:
            return None
        result = unescape(response.json()["data"]["values"]["data"]).strip("\\n")
        replace_token = "\\n           "
        for _ in range(0, 11):
            replace_token = replace_token[:-1]
            result = result.replace(replace_token, "")
        result = json.loads(result.replace("\\", ""))
        content = result["choices"][0]["message"]["content"]
    except Exception:
        content = ""
    return content


def mp_requests_gpt(
    api_key: str,
    key: str,
    query_list: list,
    max_workers: int = 30,
    temperature: float = 1.0,
    max_tokens: int = 2048,
    model: str = 'gpt-3.5-turbo',
):
    """
    多线程请求openai的服务

    query_list: 请求列表, ChatGPT chat message 数据结构
        示例: [[{"role":"user","content":"xxx"},{"role":"assistant","content":"xxx"}...]]

    max_workers: 进程数

    model: 模型名称, `gpt-3.5-turbo`, `gpt-3.5-turbo-16k`
    """
    executor = ThreadPoolExecutor(max_workers=max_workers)
    all_task = [
        executor.submit(get_ans_from_gpt35_turbo, api_key, key, message, temperature, max_tokens, model)
        for message in query_list
    ]
    all_result = [None] * len(all_task)
    for future in as_completed(all_task):
        index = all_task.index(future)
        all_result[index] = future.result()
    return all_result


if __name__ == '__main__':
    query_1 = [{"role": "user", "content": "你好"}]
    query_2 = [{"role": "user", "content": "你是哪家公司开发的"}]
    print(mp_requests_gpt([query_1, query_2]))
