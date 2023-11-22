# coding=utf-8
# @Date: 2023-06-14
import os
import json
import tempfile
import shutil
from typing import List

from .error import InvalidParamError
from ..inference.glm_predictor import GLMForInference, CompletionOutput
from ..inference.remote_predictor import RemoteInference


class Completion:
    """
    调用本地模型服务的 API 接口
    """

    def __init__(self, model, gpu_index=None, use_long_glm: bool = False):
        """
        初始化模型
        :param model, 本地模型路径
        :param gpu_index, 指定使用的GPU的index，如0，1，2
        :return:
        """
        self.predictor = self.init_predictor(model, gpu_index, use_long_glm)

    def init_predictor(self, path, gpu_index=None, use_long_glm: bool = False):
        model_files = os.listdir(path)
        # cog-pretrain.model for compatible with old version
        if not ('merge.model' in model_files or 'cog-pretrain.model' in model_files) \
                or 'tokenizer_config.json' not in model_files:
            raise FileNotFoundError(
                'tokenizer model file [merge.model/cog-pretrain.model]' 
                ' or config file [tokenizer_config.json] not exist')
        if 'adapter_model.bin' in model_files and 'adapter_config.json' in model_files:
            # peft模型
            with open(os.path.join(path, 'adapter_config.json')) as fi:
                adapter_config = json.load(fi)
            base_model_path = adapter_config['base_model_name_or_path']
            with tempfile.TemporaryDirectory() as temp_dir:
                adapter_path = os.path.join(temp_dir, 'adapter')
                shutil.copytree(path, adapter_path, dirs_exist_ok=True)
                predictor = GLMForInference(
                    path=base_model_path, adapter_path=temp_dir, gpu_index=gpu_index, use_long_glm=use_long_glm)
        else:
            # 基座模型
            predictor = GLMForInference(path=path, gpu_index=gpu_index, use_long_glm=use_long_glm)
        return predictor

    def generate(self,
                 prompt: str,
                 max_tokens=32,
                 num_beams=1,
                 temperature=1,
                 top_k=10,
                 top_p=0.9,
                 do_sample=False,
                 num_return_sequences=1,
                 left_truncate=False,
                 **gen_kwargs) -> CompletionOutput:
        """
        让蚂蚁大模型根据给定的输入进行回答，可能返回多条答案
        参数和使用说明，参考文档 https://huggingface.co/blog/how-to-generate

        Args:
            prompt (string):
                AntLLM大模型的输入文本
            max_tokens (int):
                最大输出token数，平均1个token约 1.6 个字
                -1 表示不限制输出的 token 数
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
            do_sample (boolean):
                是否使用采样生成策略，采样生成策略会给输出带来多样性，
                同时也会带来不确定行，请根据实际任务配置，默认不使用。
            left_truncate (boolean):
                是否进行左侧进行截断，默认超长输入则进行右侧截断
            num_return_sequences (int):
                返回的文本条数，需要确保 num_return_sequences <= num_beams
            gen_kwargs:
                更多参数见 https://huggingface.co/docs/transformers/main_classes/text_generation


        Example::
        >>> model_path = "path-to-antllm"
        >>> completer = Completion(model_path)
        >>> text = "请问北京在哪里？"
        >>> outs = completer.generate(text)
        >>> print(outs.texts[0])
        """
        if not prompt:
            raise InvalidParamError("prompt 输入不能为空")
        outs = self.predictor.generate(prompt=prompt,
                                       num_beams=num_beams,
                                       temperature=temperature,
                                       top_k=top_k,
                                       top_p=top_p,
                                       do_sample=do_sample,
                                       left_truncate=left_truncate,
                                       max_output_tokens=max_tokens,
                                       num_return_sequences=num_return_sequences,
                                       **gen_kwargs)
        return outs

    def generate_batch(self,
                       prompts: List[str],
                       max_tokens=32,
                       num_beams=1,
                       temperature=1,
                       top_k=10,
                       top_p=0.9,
                       do_sample=False,
                       num_return_sequences=1,
                       left_truncate=False,
                       **gen_kwargs) -> List[CompletionOutput]:
        """
        让蚂蚁大模型根据给定的输入进行回答，可能返回多条答案
        参数和使用说明，参考文档 https://huggingface.co/blog/how-to-generate

        Args:
            prompts (string):
                AntLLM大模型的输入文本列表，作为一个`batch`输入，请调用方控制`batch`的大小
            max_tokens (int):
                最大输出token数，平均1个token约 1.6 个字
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
            do_sample (boolean):
                是否使用采样生成策略，采样生成策略会给输出带来多样性，
                同时也会带来不确定行，请根据实际任务配置，默认不使用。
            left_truncate (boolean):
                是否进行左侧进行截断，默认超长输入则进行右侧截断
            num_return_sequences (int):
                返回的文本条数，需要确保 num_return_sequences <= num_beams
            gen_kwargs:
                更多参数见 https://huggingface.co/docs/transformers/main_classes/text_generation


        Example::
        >>> model_path = "path-to-antllm"
        >>> completer = Completion(model_path)
        >>> texts = ["请问北京在哪里？", "中国的首都在哪里？"]
        >>> outs = completer.generate_batch(texts)
        >>> print(outs[0].texts[0])
        """
        if not prompts:
            raise InvalidParamError("prompts 输入不能为空")
        outs = self.predictor.generate_batch(prompts,
                                             num_beams=num_beams,
                                             temperature=temperature,
                                             top_k=top_k,
                                             top_p=top_p,
                                             do_sample=do_sample,
                                             left_truncate=left_truncate,
                                             max_output_tokens=max_tokens,
                                             num_return_sequences=num_return_sequences,
                                             **gen_kwargs)
        return outs

    def generate_stream(self,
                        prompt: str,
                        max_tokens=32,
                        left_truncate=False):
        """
        实时流的生成方式，暂时只支持 greedy search
        和非实时流方式的输出，有少量差异，如中英文符号、字符串前后的空格
        Args:
            prompt (string):
                AntLLM大模型的输入文本
            max_tokens (int):
                最大输出token数，平均1个token约 1.6 个字
                -1 表示不限制输出的 token 数
            left_truncate (boolean):
                是否进行左侧进行截断，默认超长输入则进行右侧截断

        Example::
        >>> model_path = "path-to-antllm"
        >>> completer = Completion(model_path)
        >>> text = "请问北京在哪里？"
        >>> generator = completer.generate_stream(text)
        >>> for token in generator:
        >>>     print(token)
        """
        if max_tokens > 0:
            # 从测试看GLMForInference.generate_stream的调用要少返回一个token，所以调用时+1
            max_tokens = max_tokens + 1
        if not prompt:
            raise InvalidParamError("prompt 输入不能为空")
        return self.predictor.generate_stream(prompt=prompt,
                                              num_beams=1,
                                              max_output_tokens=max_tokens,
                                              left_truncate=left_truncate)


class RemoteCompletion:
    '''
    调用远端模型服务的 API 接口
    @渔知
    '''

    def __init__(self, scene_name, chain_name) -> None:
        """
        初始化远程调用服务
        :param model:
        :return:
        """
        self.predictor = RemoteInference(scene_name, chain_name)

    def generate(self,
                 prompt: str,
                 adapter_name=None,
                 timeout=10,
                 max_tokens=32,
                 num_beams=1,
                 temperature=1,
                 top_k=10,
                 top_p=0.9,
                 do_sample=False,
                 num_return_sequences=1,
                 left_truncate=False,
                 **gen_kwargs) -> CompletionOutput:
        """
        让蚂蚁大模型根据给定的输入进行回答，可能返回多条答案

        Args:
            prompt (string):
                AntLLM大模型的输入文本
            adapter_name (string):
                适配器名称
            timeout (int):
                远程请求超时时间，单位为秒
            max_tokens (int):
                最大输出token数，平均1个token约 1.6 个字
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
            do_sample (boolean):
                是否使用采样生成策略，采样生成策略会给输出带来多样性，
                同时也会带来不确定行，请根据实际任务配置，默认不使用。
            left_truncate (boolean):
                是否进行左侧进行截断，默认超长输入则进行右侧截断
            num_return_sequences (int):
                返回的文本条数，需要确保 num_return_sequences <= num_beams
            gen_kwargs:
                更多参数和使用说明，参考文档 https://huggingface.co/blog/how-to-generate

        Example::
        >>> scene_name = "lx_platform"
        >>> chain_name = "antglm_5b"
        >>> remote_completion = RemoteCompletion(scene_name, chain_name)
        >>> # 2. 发送请求
        >>> query = "今天天气不错"
        >>> adapter_name = "test"
        >>> completion_output = remote_completion.generate(
        >>>     query, adapter_name)
        """
        if not prompt:
            raise InvalidParamError("prompt 输入不能为空")
        answers = self.predictor.remote_generate(query=prompt,
                                                 adapter_name=adapter_name,
                                                 timeout=timeout,
                                                 num_beams=num_beams,
                                                 temperature=temperature,
                                                 top_k=top_k,
                                                 top_p=top_p,
                                                 do_sample=do_sample,
                                                 left_truncate=left_truncate,
                                                 max_output_tokens=max_tokens,
                                                 num_return_sequences=num_return_sequences,
                                                 **gen_kwargs)
        return answers

    def generate_batch(self, prompts: List[str],
                       adapter_name=None,
                       timeout=10,
                       max_tokens=32,
                       num_beams=1,
                       temperature=1,
                       top_k=10,
                       top_p=0.9,
                       do_sample=False,
                       num_return_sequences=1,
                       left_truncate=False,
                       **gen_kwargs) -> List[CompletionOutput]:
        """
        让蚂蚁大模型根据给定的输入进行回答，可能返回多条答案

        Args:
            prompts (list[string]):
                AntLLM大模型的输入文本，批量输入
            adapter_name (string):
                适配器名称
            timeout (int):
                远程请求超时时间，单位为秒
            max_tokens (int):
                最大输出token数，平均1个token约 1.6 个字
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
            do_sample (boolean):
                是否使用采样生成策略，采样生成策略会给输出带来多样性，
                同时也会带来不确定行，请根据实际任务配置，默认不使用。
            left_truncate (boolean):
                是否进行左侧进行截断，默认超长输入则进行右侧截断
            num_return_sequences (int):
                返回的文本条数，需要确保 num_return_sequences <= num_beams
            gen_kwargs:
                更多参数和使用说明，参考文档 https://huggingface.co/blog/how-to-generate

        Example::
        >>> scene_name = "lx_platform"
        >>> chain_name = "antglm_5b"
        >>> remote_completion = RemoteCompletion(scene_name, chain_name)
        >>> # 2. 发送请求
        >>> query = ["今天天气不错", "中国在哪里？"]
        >>> adapter_name = "test"
        >>> completion_output = remote_completion.remote_batch_generate(
        >>>     query, adapter_name)
        """
        # todo batch infer，增加 feature list 作为输入
        if not prompts:
            raise InvalidParamError("prompts 输入不能为空")
        answers = self.predictor.remote_batch_generate(query=prompts,
                                                       adapter_name=adapter_name,
                                                       timeout=timeout,
                                                       num_beams=num_beams,
                                                       temperature=temperature,
                                                       top_k=top_k,
                                                       top_p=top_p,
                                                       do_sample=do_sample,
                                                       left_truncate=left_truncate,
                                                       max_output_tokens=max_tokens,
                                                       num_return_sequences=num_return_sequences,
                                                       **gen_kwargs)
        return answers

    def generate_stream(self, prompt: str, max_tokens=32, temperature=1, top_k=10, top_p=0.9):
        # todo 调研 maya 流式推理
        pass
