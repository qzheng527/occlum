#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from binascii import b2a_hex
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from html import unescape
from typing import Dict, List, Tuple, Optional
from solutions.antllm.datachain.utils import batch_iter, post_with_retry, safe_get
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from tqdm import tqdm

from .base import AntLLM

STATUS_FAILED = "failed"
CONTENT = "content"
MSG_KEY = "msg_key"
RESULT = "result"
QUERY_URL = "https://zdfmng.alipay.com/commonQuery/queryData"


ANT_OPENAI_API_KEY = "ANT_OPENAI_API_KEY"
ANT_OPENAI_AES_KEY = "ANT_OPENAI_AES_KEY"

logger = logging.getLogger(__file__)


class AntOpenAI(AntLLM):
    """openai gateway in ant

    Args:
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: str = ANT_OPENAI_API_KEY,
        aes_key: str = ANT_OPENAI_AES_KEY,
        openai_params: dict = None,
        use_async_api: bool = False,
        max_workers: int = 20,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        work_dir: str = None,
        receive_after: int = 0,
        retried_times: int = 3,
        verbose: bool = False,
        visitDomain: Optional[str] = None,
        visitBiz: Optional[str] = None,
        visitBizLine: Optional[str] = None,
        serviceName: Optional[str] = None,
    ):
        self._model = model
        self._api_key = os.environ.get(ANT_OPENAI_API_KEY, api_key)
        self._aes_key = os.environ.get(ANT_OPENAI_AES_KEY, aes_key)
        self._use_async_api = use_async_api
        self._max_workers = max_workers
        self._openai_params = openai_params
        self._work_dir = work_dir
        self._receive_after = receive_after
        self._retried_times = retried_times
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._verbose = verbose
        if visitDomain:
            self._visitDomain = visitDomain
        else:
            self._visitDomain = 'BU_cto'
        if visitBiz:
            self._visitBiz = visitBiz
        else:
            self._visitBiz = 'BU_cto_llm'
        if visitBizLine:
            self._visitBizLine = visitBizLine
        else:
            self._visitBizLine = 'BU_cto_llm_line'
        if serviceName:
            self._serviceName = serviceName
        else:
            self._serviceName = 'chatgpt_prompts_completions_query_dataview'

    def _aes_encrypt(self, data, key):
        """aes加密函数，如果data不是16的倍数【加密文本data必须为16的倍数！】，那就补足为16的倍数
        :param key:
        :param data:
        """
        iv = b"1234567890123456"
        cipher = Cipher(algorithms.AES(bytes(key, 'utf-8')), modes.CBC(iv))
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

    async def _single_send(self, data_item):
        res = STATUS_FAILED
        m = hashlib.md5()
        m.update(data_item[CONTENT].encode('utf-8'))
        md5_value = m.hexdigest()
        res = await self._async_send_invoke(data_item[CONTENT], md5_value)
        if res != STATUS_FAILED:
            data_item[MSG_KEY] = res

    async def _async_send_invoke(self, query: str, md5_value: str) -> str:
        # query = query.replace("%", " percent")
        service = "asyn_chatgpt_prompts_completions_query_dataview"
        param = self._default_openai_param(service)
        param["queryConditions"]["messageKey"] = f"{md5_value}"
        param["queryConditions"]["outputType"] = "PULL"

        # Check if query is chat message json string
        try:
            messages = json.loads(query)
            param["queryConditions"]["messages"] = messages
        except ValueError:
            param["queryConditions"]["messages"] = [
                {
                    "role": "user",
                    "content": query
                }
            ]

        if self._openai_params:
            for k, v in self._openai_params.items():
                param["queryConditions"][k] = v

        headers = {
            'Content-Type': 'application/json'
        }
        data = json.dumps(param)
        data = data.encode('utf-8')

        post_data = {
            "encryptedParam": self._aes_encrypt(data, self._aes_key).decode()
        }
        content = STATUS_FAILED

        response = self._retry_util_success(
            QUERY_URL,
            data=json.dumps(post_data),
            headers=headers,
        )
        if response:
            res = unescape(response.json())
            if safe_get(res, ["data", "values", "messageKey"])[0]:
                # return the message key for collecting the results
                content = res['data']['values']['messageKey']
            else:
                logger.info("error in res, res is %s" % res)
        else:
            logger.info(f"fail to call gpt for prompt {query}")
        return content

    def _retry_util_success(self, url, data, headers):
        for _ in range(self._retried_times):
            response = post_with_retry(
                self._retried_times, url, data=data, headers=headers)
            if response:
                res = response.json()
                if "success" in res and res["success"]:
                    return response
        return None

    async def _single_receive(self, data_item):
        if MSG_KEY in data_item:
            msg_key = data_item[MSG_KEY]
        else:
            content = data_item[CONTENT]
            res = STATUS_FAILED
            m = hashlib.md5()
            m.update(content.encode('utf-8'))
            msg_key = m.hexdigest()
        res = await self._async_receive_invoke(msg_key)
        if res != STATUS_FAILED:
            data_item[RESULT] = res

    async def _async_receive_invoke(self, msgKey) -> str:
        param = {
            "serviceName": self._serviceName,
            "visitDomain": self._visitDomain,
            "visitBiz": self._visitBiz,
            "visitBizLine": self._visitBizLine,
            "cacheInterval": 0,
            "queryConditions": {
                "url": QUERY_URL,
                "messageKey": msgKey
            }
        }

        headers = {
            'Content-Type': 'application/json'
        }

        data = json.dumps(param).encode('utf-8')

        post_data = {
            "encryptedParam": self._aes_encrypt(data, self._aes_key).decode()
        }
        content = STATUS_FAILED
        response = post_with_retry(
            self._retried_times, QUERY_URL, data=json.dumps(post_data), headers=headers)
        if not response:
            logger.info(f"fail to receive message {msgKey}")
        else:
            res = response.json()
            if safe_get(res, ["data", "values", "response"])[0]:
                resp_json = unescape(res["data"]["values"]["response"])
                try:
                    result = json.loads(resp_json)
                except json.JSONDecodeError:
                    try:
                        resp_json = re.sub("\\+", r"\\", resp_json)
                        result = json.loads(resp_json)
                    except json.JSONDecodeError:
                        logger.info("fail to parse response as json, response is %s" %
                                    resp_json)
                if "choices" in result and len(result["choices"]) > 0 and \
                        safe_get(result["choices"][0], ["message", "content"]):
                    content = result["choices"][0]["message"]["content"]
        return content

    async def _batch_task(self, task_ftn, data, *args, **kwargs):
        task_list = []
        for input in data:
            task_list.append(asyncio.create_task(
                task_ftn(input, *args, **kwargs)))
        for task in task_list:
            await task

    def _default_openai_param(self, service_name: str) -> dict:
        return {
            "serviceName": self._serviceName,
            "visitDomain": self._visitDomain,
            "visitBiz": self._visitBiz,
            "visitBizLine": self._visitBizLine,
            "cacheInterval": 0,
            "queryConditions": {
                "url": QUERY_URL,
                "model": self._model,
                "max_tokens": str(self._max_tokens),
                "n": "1",
                "temperature": str(self._temperature),
                "api_key": self._api_key,
                "messages": None,
            },
        }

    def generate(self, prompt: str) -> str:
        """Instruction generate."""
        service = "chatgpt_prompts_completions_query_dataview"
        param = self._default_openai_param(service)
        param["queryConditions"]["messages"] = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        return self._generate(param)

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Chat completion."""
        service = "chatgpt_prompts_completions_query_dataview"
        param = self._default_openai_param(service)
        param["queryConditions"]["messages"] = messages
        return self._generate(param)

    def _generate(self, param: dict) -> str:
        """OpenAI generate."""
        if self._openai_params:
            for k, v in self._openai_params.items():
                param["queryConditions"][k] = v

        headers = {
            'Content-Type': 'application/json'
        }
        data = json.dumps(param).encode('utf-8')

        post_data = {
            "encryptedParam": self._aes_encrypt(data, self._aes_key).decode()
        }

        response = self._retry_util_success(
            QUERY_URL,
            data=json.dumps(post_data),
            headers=headers,
        )

        content = STATUS_FAILED

        if not response:
            logger.info(f"fail to call chatgpt for param {param}")
        else:
            res = response.json()
            if safe_get(res, ["data", "values", "data"])[0]:
                abnormals_map_in_res = {
                    ",\\n": ",",
                    "[\\n": "[",
                    "\\n]": "]",
                    "\"\\n": "\"",
                    "{\\n": "{",
                    "\\n}": "}",
                    "\\n  ": "",
                }
                resp_json = unescape(
                    res["data"]["values"]["data"]).strip("\\n")
                for k, v in abnormals_map_in_res.items():
                    resp_json = resp_json.replace(k, v)
                try:
                    result = json.loads(resp_json)
                except json.JSONDecodeError:
                    try:
                        resp_json = re.sub(r"\\+", r"\\", resp_json)
                        result = json.loads(resp_json)
                    except json.JSONDecodeError:
                        logger.info("fail to parse response as json, response is %s" %
                                    resp_json)

                if "choices" in result and len(result["choices"]) > 0 \
                        and safe_get(result["choices"][0], ["message", "content"])[0]:
                    content = result["choices"][0]["message"]["content"]
        return content

    def batch_generate(self, prompts: List[str]) -> Dict[str, str]:
        if self._use_async_api:
            return self._batch_generate_async(prompts, batch_size=10)
        else:
            return self._batch_generate(prompts)

    def batch_chat(self, list_messages: List[List[Dict[str, str]]]) -> List[str]:
        prompts = [json.dumps(message) for message in list_messages]
        if self._use_async_api:
            results = self._batch_generate_async(prompts, batch_size=10)
        else:
            results = self._batch_generate(prompts)

        return [
            results[item] for item in prompts
        ]

    def _generate_with_input(self, prompt: str) -> Tuple[str, str]:
        # Check if prompt is chat message json string
        try:
            messages = json.loads(prompt)
            return (prompt, self.chat(messages))
        except ValueError:
            return (prompt, self.generate(prompt))

    def _batch_generate(self, prompts: List[str]) -> Dict[str, str]:
        all_res = dict()
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            for result in list(tqdm(executor.map(self._generate_with_input, prompts),
                                    total=len(prompts), disable=not self._verbose)):
                all_res[result[0]] = result[1]
        return all_res

    def _batch_generate_async(self, prompts: List[str], batch_size=10) -> Dict[str, str]:
        all_data = []
        for p in prompts:
            all_data.append({CONTENT: p})

        for batch in tqdm(list(batch_iter(all_data, batch_size)), disable=not self._verbose):
            asyncio.run(self._batch_task(self._single_send, batch))

        if self._work_dir is not None:
            output_name = f"sent_{datetime.now().strftime(r'%m%d_%H%M%S')}.jsonl"
            output_path = os.path.join(self._work_dir, output_name)
            with open(output_path, "w", encoding="utf-8") as fo:
                for item in all_data:
                    fo.write(json.dumps(item, ensure_ascii=False) + "\n")

            output_name = f"received_{datetime.now().strftime(r'%m%d_%H%M%S')}.jsonl"
            fo = open(os.path.join(self._work_dir, output_name),
                      "w", encoding="utf-8")

        if self._receive_after > 0:
            time.sleep(self._receive_after * 60)

        left_data = all_data
        times = 0
        ret_dict = dict()
        while len(left_data) > 0:
            if times >= self._retried_times:
                break
            for batch in tqdm(list(batch_iter(left_data, batch_size)), disable=not self._verbose):
                asyncio.run(self._batch_task(self._single_receive, batch))

            new_left_data = []
            for item in left_data:
                if RESULT not in item:
                    new_left_data.append(item)
                else:
                    ret_dict[item[CONTENT]] = item[RESULT]
                    if self._work_dir is not None:
                        fo.write(json.dumps(item, ensure_ascii=False) + "\n")

            if len(new_left_data) == len(left_data):
                times += 1
            else:
                times = 0
            left_data = new_left_data
            if self._work_dir is not None:
                fo.flush()
            time.sleep(3)

        if self._work_dir is not None:
            fo.close()

        return ret_dict

    def _is_json_string(self, input: str) -> bool:
        """Check if a string is json string."""
        try:
            json.loads(input)
        except ValueError:
            return False

        return True
