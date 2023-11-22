#!/usr/bin/env python
# coding=utf-8
# @Author: xinyu.kxy
# @Date: Fri 17 July 2023 09:22:56 PM CST

import json
import logging
import time
import requests
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from binascii import b2a_hex
from html import unescape
from tqdm import tqdm


def aes_encrypt(data, key, iv):
    """aes加密函数，如果data不是16的倍数【加密文本data必须为16的倍数！】，那就补足为16的倍数
    :param key:
    :param data:
    """
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


def get_ans_from_api(history: list, params: dict, waiting_time: float = 0.2):
    """
    history:[{"role":"user","content":"..."},{"role":"assistant","content":"..."},...]
    """

    headers = {
        'Content-Type': 'application/json'
    }
    params["queryConditions"]["messages"] = history
    data = json.dumps(params)
    data = data.encode('utf-8')

    post_data = {
        "encryptedParam": aes_encrypt(data, params["aes_key"].encode("utf-8"), 
                                      params["aes_iv"].encode("utf-8")).decode()
    }
    try:
        for _ in range(params["retry_times"]):
            response = requests.post(
                params["queryConditions"]["url"], data=json.dumps(post_data), headers=headers)
            res = response.json()
            if not res["success"]:
                error = res["data"]["errorMessage"]
                if "Rate limit reached" in error:
                    logging.warning(
                        f"Rate limit reached... Trying again in {waiting_time} seconds. Full error: {error}")
                    time.sleep(waiting_time)
                elif error == "":
                    error = res["msg"]
                    logging.warning(
                        f"Rate limit reached... Trying again in {waiting_time} seconds. Full error: {error}")
                    time.sleep(waiting_time)
                elif "Input validation error" in error and "max_new_tokens" in error:
                    # params["max_new_tokens"] = int(params["max_new_tokens"] * 0.8)
                    logging.warning(
                        f"`max_new_tokens` too large. Reducing target length to {params['max_tokens']}, " f"Retrying..."
                    )
                else:
                    raise ValueError(
                        f"Error in inference. Full error: {error}")
        if not res["success"]:
            raise ValueError("Error in inference. We tried {} times and failed.".format(
                params["retry_times"]))
        result = unescape(
            response.json()["data"]["values"]["data"]).strip("\\n")
        replace_token = "\\n           "
        for _ in range(0, 11):
            replace_token = replace_token[:-1]
            result = result.replace(replace_token, "")
        result = json.loads(result.replace("\\", ""))

        content = result["choices"][0]["message"]["content"]
    except Exception as e:
        logging.warning("Error in openai api. {}".format(e))
        return "api failed"
    return content


def open_api_completions(
    prompts: list,
    param: dict
) -> dict:
    completions = []
    for prompt in tqdm(prompts):
        history = _prompt_to_chatml(prompt)
        completion = get_ans_from_api(history, param)
        completions.append(completion)
    return dict(completions=completions)


def _prompt_to_chatml(prompt: str, start_token: str = "<|im_start|>", end_token: str = "<|im_end|>"):
    r"""转换prompt到对话格式

    Examples
    --------
    >>> prompt = (
    ... "<|im_start|>system\n"
    ... "You are a helpful assistant.\n<|im_end|>\n"
    ... "<|im_start|>system name=example_user\nKnock knock.\n<|im_end|>\n<|im_start|>system name=example_assistant\n"
    ... "Who's there?\n<|im_end|>\n<|im_start|>user\nOrange.\n<|im_end|>"
    ... )
    >>> print(prompt)
    <|im_start|>system
    You are a helpful assistant.
    <|im_end|>
    <|im_start|>system name=example_user
    Knock knock.
    <|im_end|>
    <|im_start|>system name=example_assistant
    Who's there?
    <|im_end|>
    <|im_start|>user
    Orange.
    <|im_end|>
    >>> _prompt_to_chatml(prompt)
    [{'content': 'You are a helpful assistant.', 'role': 'system'},
      {'content': 'Knock knock.', 'role': 'system', 'name': 'example_user'},
      {'content': "Who's there?", 'role': 'system', 'name': 'example_assistant'},
      {'content': 'Orange.', 'role': 'user'}]
    """
    prompt = prompt.strip()
    assert prompt.startswith(start_token)
    assert prompt.endswith(end_token)

    message = []
    for p in prompt.split("<|im_start|>")[1:]:
        newline_splitted = p.split("\n", 1)
        role = newline_splitted[0].strip()
        content = newline_splitted[1].split(end_token, 1)[0].strip()
        other_params = dict()

        message.append(dict(content=content, role=role, **other_params))

    return message


if __name__ == "__main__":
    query = "北京在哪里"
    history = [{"role": "user", "content": query}]
    print(get_ans_from_api(history))
