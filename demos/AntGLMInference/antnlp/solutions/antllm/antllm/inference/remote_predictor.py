"""
Author: yuzhi.wzx
Date: 2023-06-19 10:52:45
Description: RemotePredictor for Maya service
"""
import json
from typing import List
import requests
from requests.adapters import HTTPAdapter


# 添加当前文件的上级目录到 sys.path
# import sys
# sys.path.append(
# "/Users/brickea/Documents/developments/algorithms/max_models/antnlp")
from .glm_predictor import CompletionOutput


class RemoteInference(object):
    def __init__(self, scene_name: str, chain_name: str) -> None:
        """
        参数参考自「向zark_deploy发起部署章节」：https://yuque.antfin.com/zark/iudsay/agy3pzyg7o1rm09x#t4HQr
        Args:
            scene_name (str): 场景名称
            chain_name (str): 链名称

        """
        # 检查模型是否已经部署
        self.scene_name = scene_name
        self.chain_name = chain_name

    def remote_generate(self,
                        query: str,
                        adapter_name: str = None,

                        timeout: int = 10,
                        num_beams: int = 1,
                        temperature: float = 1,
                        top_k: int = 50,
                        top_p: float = 1,

                        do_sample: bool = False,
                        left_truncate: bool = False,
                        max_output_tokens: int = -1,
                        num_return_sequences: int = 1,

                        **gen_kwargs,
                        ) -> CompletionOutput:
        """
        Args:
            参数参考适用于 antllm 的 maya userHandler 实现：https://yuque.antfin.com/zark/iudsay/agy3pzyg7o1rm09x#bv7hq
            query (string):
                AntGLM大模型的输入文本
            adapter_path (str): 
                参考自现有服务 c7992997f76af6b3_lx_platform/antglm_5b
            timeout (int):
                请求超时时间，单位为秒

            num_beams (int):
                解码过程所使用的集束搜索(beamsearch)的集束大小，通常情况下`num_beams`越大，
                生成结果越接近全局最优，但是会带来额外`num_beams`倍的计算开销。
                默认`num_beams=1`，即不使用集束搜索策略。
            temperature (int):
                采样生成策略中使用的温度参数，
                `temperature`越高采样的分布会越平滑，越低分布越尖锐。
            top_k (int):
                采样生成策略中使用的`top_k`采样参数，每一步解码将从top K个候选输出中选择生成。
            top_p (float):
                采样生成策略中使用的`top_p`采样参数，每一步解码将从累计概率为`P`的多个候选输出中选择生成。

            以下参数参考 glm_predictor.py 实现
            do_sample (boolean):
                是否使用采样生成策略，采样生成策略会给输出带来多样性，
                同时也会带来不确定行，请根据实际任务配置，默认不使用。
            left_truncate (boolean):
                是否进行左侧进行截断
            max_output_tokens (int):
                最大输出长度
            num_return_sequences (int):
                返回的文本条数，需要确保 num_return_sequences <= num_beams


        Returns:
            CompletionOutput
        """

        if query is None:
            raise Exception("query is None!")

        # 1. 构造请求数据
        data = {
            "query": query,
            "adapter_name": adapter_name,
            "num_beams": num_beams,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "do_sample": do_sample,
            "left_truncate": left_truncate,
            "max_output_tokens": max_output_tokens,
            "num_return_sequences": num_return_sequences
        }
        # 2. 发送请求
        remote_call_result = self._maya_infer_client(
            scene_name=self.scene_name,
            chain_name=self.chain_name,
            features={
                "data": json.dumps(data)

            },
            timeout=timeout
        )  # 3. 解析响应

        # 4. 构造 单条 CompletionOutput
        completion_output = CompletionOutput(
            texts=remote_call_result[0].get("texts", []),
            finish_reasons=remote_call_result[0].get("finish_reasons", [])
        )
        return completion_output

    def remote_batch_generate(self,
                              query: List[str],
                              adapter_name: str = None,

                              timeout: int = 10,
                              num_beams: int = 1,
                              temperature: float = 1,
                              top_k: int = 50,
                              top_p: float = 1,

                              do_sample: bool = False,
                              left_truncate: bool = False,
                              max_output_tokens: int = -1,
                              num_return_sequences: int = 1,

                              **gen_kwargs,
                              ) -> List[CompletionOutput]:
        """
        Args:
            参数参考适用于 antllm 的 maya userHandler 实现：https://yuque.antfin.com/zark/iudsay/agy3pzyg7o1rm09x#bv7hq
            query (string):
                AntGLM大模型的输入文本
            adapter_path (str): 
                参考自现有服务 c7992997f76af6b3_lx_platform/antglm_5b
            timeout (int):
                请求超时时间，单位为秒

            num_beams (int):
                解码过程所使用的集束搜索(beamsearch)的集束大小，通常情况下`num_beams`越大，
                生成结果越接近全局最优，但是会带来额外`num_beams`倍的计算开销。
                默认`num_beams=1`，即不使用集束搜索策略。
            temperature (int):
                采样生成策略中使用的温度参数，
                `temperature`越高采样的分布会越平滑，越低分布越尖锐。
            top_k (int):
                采样生成策略中使用的`top_k`采样参数，每一步解码将从top K个候选输出中选择生成。
            top_p (float):
                采样生成策略中使用的`top_p`采样参数，每一步解码将从累计概率为`P`的多个候选输出中选择生成。

            以下参数参考 glm_predictor.py 实现
            do_sample (boolean):
                是否使用采样生成策略，采样生成策略会给输出带来多样性，
                同时也会带来不确定行，请根据实际任务配置，默认不使用。
            left_truncate (boolean):
                是否进行左侧进行截断
            max_output_tokens (int):
                最大输出长度
            num_return_sequences (int):
                返回的文本条数，需要确保 num_return_sequences <= num_beams


        Returns:
            List[CompletionOutput]
        """

        # 1. 构造请求数据
        data = {
            "query": query,
            "adapter_name": adapter_name,
            "num_beams": num_beams,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "do_sample": do_sample,
            "left_truncate": left_truncate,
            "max_output_tokens": max_output_tokens,
            "num_return_sequences": num_return_sequences
        }
        # 2. 发送请求
        remote_call_result = self._maya_infer_client(
            scene_name=self.scene_name,
            chain_name=self.chain_name,
            features={
                "data": json.dumps(data)

            },
            timeout=timeout
        )
        # 4. 构造 batch CompletionOutput
        batch_completion_output = []
        for item in remote_call_result:
            completion_output = CompletionOutput(
                texts=item.get("texts", []),
                finish_reasons=item.get("finish_reasons", [])
            )
            batch_completion_output.append(completion_output)
        return batch_completion_output

    def _maya_infer_client(self, scene_name, chain_name, features, timeout=20):
        """
        Args:
            scene_name (str): 场景名称
            chain_name (str): 链名称
            params (dict): 请求参数
            timeout (int): 超时时间 (秒)
        """
        # 初始化请求session
        with requests.Session() as request_session:
            adapter = HTTPAdapter(
                pool_connections=10, pool_maxsize=50, max_retries=3)
            request_session.mount('http://', adapter)
            request_session.mount('https://', adapter)

            api_url = f"https://gateway-cv.alipay.com/ua/invoke"
            response = request_session.post(
                url=api_url,
                headers={"Content-Type": "application/json"},
                json={
                    "serviceCode": scene_name,
                    "uri": chain_name,
                    "params": {
                        "features": features
                    },
                    "appId": "antllm",
                    "attributes": {
                        "_ROUTE_": "MAYA",
                        "_TIMEOUT_": timeout * 1000
                    }
                }
            )

            if response.status_code != 200:
                raise Exception(
                    "completion-remote {} {} 请求失败，响应码：{} 详细返回：{}".format(
                        self.scene_name,
                        self.chain_name,
                        response.status_code,
                        response))

            response_data = response.json()
            if not response_data.get("success", False):
                raise Exception(
                    "completion-remote {} {} 请求失败，服务器 inference error! 响应码：{} 详细返回：{}".format(
                        self.scene_name,
                        self.chain_name,
                        response.status_code,
                        response_data))

            remote_call_result_str = response_data.get(
                "resultMap").get("attributes").get("result", None)

            if remote_call_result_str is None:
                raise Exception(
                    "completion-remote {} {} 请求失败，服务器返回结果为空! 响应码：{} 详细返回：{}".format(
                        self.scene_name,
                        self.chain_name,
                        response.status_code,
                        response_data))

            # remote_call_result 内容定义参考 https://code.alipay.com/ai-dls/antnlp/pull_requests/3823
            # [{
            #     "texts": ans.texts,
            #     "finish_reasons": ans.finish_reasons
            # }]
            remote_call_result = None
            try:
                remote_call_result = json.loads(remote_call_result_str)
            except Exception:
                raise Exception(
                    "completion-remote {} {} 请求失败，服务器返回结果 json 反序列化失败! 响应码：{} 详细返回：{}".format(
                        self.scene_name,
                        self.chain_name,
                        response.status_code,
                        response_data))
            if remote_call_result is None or len(remote_call_result) == 0:
                raise Exception(
                    "completion-remote {} {} 请求失败，服务器返回不符合 CompletionOutput 定义! 响应码：{} 详细返回：{}".format(
                        self.scene_name,
                        self.chain_name,
                        response.status_code,
                        response_data))

            return remote_call_result
