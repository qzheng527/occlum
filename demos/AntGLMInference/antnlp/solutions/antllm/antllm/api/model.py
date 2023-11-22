# coding=utf-8
# @Date: 2023-06-14
import os
import json
import torch
import logging

from ..models.glm.tokenization_glm import GLMTokenizer
from ..models.glm.modeling_glm import GLMForConditionalGeneration
from solutions.antllm.antllm.utils.version_utils import is_oldest_version


logger = logging.getLogger(__name__)


class LLMModel:
    '''
    提供大模型一些基础的能力
    '''

    def __init__(self, model_name_or_path):
        '''
        加载模型
        :param model_name_or_path: 官方训练的大模型名，或者一个可以访问加载的路径
        :return:
        '''
        if not os.path.isdir(model_name_or_path):
            # 模型下载暂未实现
            logger.error('unimplemented model download ' + model_name_or_path)
            raise RuntimeError('local model not exist ' + model_name_or_path)
        self.model_dir = model_dir = model_name_or_path
        ret = is_oldest_version(model_dir)
        if ret is True:
            self.is_old_version = True
            self.mask = "[sMASK]"
            from solutions.antllm.antllm.models.glm.tokenization_glm_deprecated import (
                GLMChineseTokenizer,
            )

            self.tokenizer = GLMChineseTokenizer.from_pretrained(model_dir)
        elif ret is False:
            self.is_old_version = False
            self.mask = "[gMASK]"
            self.tokenizer = GLMTokenizer.from_pretrained(model_dir)
        else:
            logger.error("模型目录中词典文件缺失")
            raise RuntimeError('check model: ' + model_dir)
        # 读取配置配置
        self.config = json.load(
            open(os.path.join(model_dir, "config.json"), "r")
        )
        self._load_model()
        self._post_load_model()

    def _load_model(self):
        '''
        加载模型
        :return:
        '''
        self.model = GLMForConditionalGeneration.from_pretrained(self.model_dir)

    def _post_load_model(self):
        '''
        加载模型之后的后处理
        :return:
        '''
        self.model.eval()
        if not torch.cuda.is_available():
            logger.warning('unfound gpu, use cpu')
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")
            self.model = self.model.half()
        self.model.to(self.device)
