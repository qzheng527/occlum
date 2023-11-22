# coding=utf-8
# @Date: 2023-06-14
from typing import List, Union
import os
import json
import torch
import logging
import numpy as np
from transformers.tokenization_utils_base import BatchEncoding

from .error import InvalidParamError
from .model import LLMModel
from ..data.dataset.glm_embedding_dataset import GLMEmbeddingDataset
from ..models.embeddings.modeling_embedding import GLMForEmbedding

logger = logging.getLogger(__name__)


class Embedding(LLMModel):
    def __init__(self, model: str):
        '''
        加载 emebedding 模型
        :param model:
        :return:
        '''
        super().__init__(model)
        self.max_sequence_length = self.config.get("max_sequence_length")
        self.embedding_dim = self.config.get("hidden_size", 3072)
        logger.info(f'max_sequence_length: {self.max_sequence_length}')

    def _load_model(self):
        model_files = os.listdir(self.model_dir)
        if 'adapter_model.bin' in model_files and 'adapter_config.json' in model_files:
            # 加载peft模型
            with open(os.path.join(self.model_dir, 'adapter_config.json')) as fi:
                adapter_config = json.load(fi)
            base_model_path = adapter_config['base_model_name_or_path']
            self.model = GLMForEmbedding.from_pretrained(base_model_path)

            from solutions.antllm.antllm.models.peft.modeling_peft import AntPeftForEmbedding
            self.model = AntPeftForEmbedding.from_pretrained(self.model, self.model_dir)
        else:
            # 基加载座模型
            self.model = GLMForEmbedding.from_pretrained(self.model_dir)

    def _get_batch_query_inputs(
            self,
            querys: List[str],
            left_truncate: bool = False,
            max_input_length: int = 1022
    ):
        batch_size = len(querys)
        input_ids, position_ids, attention_mask, embedding_mask = [], [], [], []

        max_output_length = 0
        for query in querys:
            data = {"input": query}
            max_output_length, instance_input = GLMEmbeddingDataset.build_feature_from_sample(
                data, self.tokenizer,
                max_input_length=max_input_length,
                max_output_length=max_output_length,
                mask_id=self.tokenizer.convert_tokens_to_ids(self.mask),
                for_generation=True,
                left_truncate=left_truncate,
                gpt_data=self.config.get("gpt_model", False),
                old_version_tokenizer=self.is_old_version,
            )
            input_ids.append(instance_input["query_ids"])
            position_ids.append(instance_input["query_position_ids"])
            attention_mask.append(
                instance_input["query_attention_mask"]
            )

        max_ids_length = max([input.size(1) for input in input_ids])

        for i in range(batch_size):
            cur_ids_length = input_ids[i].size(1)

            _embedding_mask = torch.LongTensor(
                [0] * (max_ids_length - cur_ids_length) + [1] * cur_ids_length
            ).unsqueeze(0)
            embedding_mask.append(_embedding_mask)

            if cur_ids_length < max_ids_length:
                # pad input ids
                pad_input_ids = input_ids[i].new_zeros(
                    (1, max_ids_length - cur_ids_length)
                )
                input_ids[i] = torch.cat([pad_input_ids, input_ids[i]], dim=-1)

                # pad postition ids with left pad
                # 0, 1, 2, 3, 4 ... -> 0, ..., 0, 1, 2, 3, 4, ...
                pad_position_ids = input_ids[i].new_zeros(
                    (1, 2, max_ids_length - cur_ids_length)
                )
                position_ids[i] = torch.cat([pad_position_ids, position_ids[i]], dim=-1)

                # pad generation attention mask with left and bottom pad
                new_attention_mask = input_ids[i].new_zeros(
                    1,
                    1,
                    max_ids_length + max_output_length,
                    max_ids_length + max_output_length,
                )
                new_attention_mask[
                    :,
                    :,
                    max_ids_length - cur_ids_length:,
                    max_ids_length - cur_ids_length:,
                ] = attention_mask[i]
                attention_mask[i] = new_attention_mask.contiguous()

        input_ids = torch.cat(input_ids, dim=0)
        position_ids = torch.cat(position_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        embedding_mask = torch.cat(embedding_mask, dim=0)

        inputs = {
            "query_ids": input_ids.to(self.device),
            "query_position_ids": position_ids.to(self.device),
            "query_attention_mask": attention_mask.to(self.device),
            "query_mask": embedding_mask.to(self.device)
        }

        inputs = BatchEncoding(inputs)
        return inputs

    def _get_single_query_inputs(self, query: str):
        max_output_length = 0
        data = {"input": query}

        max_output_length, inputs = GLMEmbeddingDataset.build_feature_from_sample(
            data,
            self.tokenizer,
            max_length=self.max_sequence_length,
            max_input_length=self.max_sequence_length - 2,
            max_output_length=max_output_length,
            mask_id=self.tokenizer.convert_tokens_to_ids(self.mask),
            for_generation=True,
            left_truncate=False,
            gpt_data=False,
            old_version_tokenizer=True,
        )
        input_ids = inputs.query_ids
        position_ids = inputs.query_position_ids
        attention_mask = inputs.query_attention_mask

        input_ids_length = input_ids.size(-1)
        embedding_mask = torch.LongTensor([1] * input_ids_length).view(1, -1)

        inputs = {
            "query_ids": input_ids.to(self.device),
            "query_position_ids": position_ids.to(self.device),
            "query_attention_mask": attention_mask.to(self.device),
            "query_mask": embedding_mask.to(self.device)
        }
        return BatchEncoding(inputs)

    @torch.no_grad()
    def get_embedding(
            self,
            text: Union[str, List[str]],
            reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Compute the text embedding with the LLM model.

        Args:
            text (`str` or `list[str]`): the text used to be encoded into embedding,
                which can be a single string of a list of text.
            reduction (string, optional): Specifies the reduction to apply to the output:
                ``'last'`` | ``'mean'`` | ``'sum'``. ``'last'``: use the hidden feature of
                last token as the final sentence embedding,
                ``'mean'``: the sum of the output will be divided by the number of
                elements in the output, ``'sum'``: the output will be summed.

        Examples::
            >>> texts = [
            >>>     "我觉得这只猫长得可爱",
            >>>     "这条狗好丑",
            >>>     "这只猫长得真可爱",
            >>>     "请问我的支付宝账号如何注销",
            >>>     "请问我如何注销我的支付宝，我不想用了",
            >>>     "支付宝应该如何体现？"
            >>> ]
            >>> embedder = Embedding('model_path')
            >>> embeddings = embedder.get_embedding(texts)
            >>> print(embeddings.size())
            >>> similarity = torch.cosine_similarity(embeddings[None, ...], embeddings[:, None], dim=-1)
            >>> print(similarity)
        """

        if not text:
            raise InvalidParamError("获取Embedding的输入文本不能为空")
        if isinstance(text, str):
            inputs = self._get_single_query_inputs(text)
        elif isinstance(text, list):
            inputs = self._get_batch_query_inputs(text)
        else:
            raise TypeError(f"The input text must be string or list of string.")

        embeddings = self.model(**inputs, reduction=reduction).query_embeddings

        return embeddings

    @property
    def get_embedding_dim(self) -> int:
        '''
        :return: 获取本模型的 ebmedding 维度
        '''
        return self.embedding_dim

    @property
    def get_max_token_num(self) -> int:
        '''
        :return: 支持的最大token数
        '''
        return self.max_sequence_length

    def encode(
        self,
        sentences: List[str],
        convert_to_tensor: bool = False,
        batch_size: int = 32,
        reduction: str = "mean"
    ) -> Union[torch.Tensor, List[np.ndarray]]:
        """
        Embedding模型encode API，用于支持对大文本数组进行embedding编码，
        同时兼容目前主流embedding模型和榜单那个所使用的接口。

        Args:
            - sentences (`list[str]`): 待编码的文本数组。
            - convert_to_tensor (bool): 是否将结果转换为torch格式`torch.Tensor`，默认为`False`，
                会讲结果转换为numpy格式`np.Array`。
            - batch_size (int): 每次进行批量embedding编码的批次大小。
            - reduction (string, optional): embedding的计算方法，同`get_embedding`函数。            
        """
        res = []

        for i in range(0, len(sentences), batch_size):
            sents = sentences[i: i + batch_size]
            embeding = self.get_embedding(sents, reduction)
            res.extend(embeding)

        if convert_to_tensor:
            res = [torch.tensor(s) for s in res]
            res = torch.stack(res)
        else:
            res = [np.array(s.cpu()) for s in res]

        return res
