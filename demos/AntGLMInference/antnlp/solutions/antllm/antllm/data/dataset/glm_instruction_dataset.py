import json
import os
import time

import numpy as np

import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import BatchEncoding

from . import mdatasets as mds


class GLMInstructionDataset(Dataset):
    '''
    GLM结构模型所使用的Dataset
    数据格式:
    {"input": "清华大学在哪里", "output": "北京"}
    '''

    def __init__(self,
                 data_path,
                 tokenizer,
                 max_input_length=550,
                 max_output_length=550,
                 max_length=1024,
                 no_append_glm_mask=False,
                 gpt_data=False,
                 world_size=1,
                 global_rank=0,
                 left_truncate=False,
                 shard_data=False,
                 undirectional_attention=False,
                 eos_token='<|endofpiece|>',
                 **kwargs):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.max_length = max_length
        self.undirectional_attention = undirectional_attention
        self.left_truncate = left_truncate
        self.sop_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.sop_token)
        self.eos_token = eos_token
        self.eop_id = self.tokenizer.convert_tokens_to_ids(
            self.eos_token)
        self.cls_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
        self.mask = kwargs.get('glm_mask', '[gMASK]')
        self.mask_id = self.tokenizer.convert_tokens_to_ids(self.mask)
        self.no_append_glm_mask = no_append_glm_mask
        self.gpt_data = gpt_data
        self.kwargs = kwargs
        self.old_version_tokenizer = self.kwargs.get(
            'old_version_tokenizer', False)
        self.add_cls = self.kwargs.get('add_cls', False)
        self.rotary_type = self.kwargs.get("rotary_type", 'none')
        self.isolation_position_ids = self.kwargs.get('isolation_position_ids', False)
        self.shard_data = shard_data
        self.world_size = world_size
        self.global_rank = global_rank
        if os.environ.get("MDS_ENABLE", "false").upper() == "TRUE":
            print("Loading a dataset through mds.load_dataset")
            self._load_dataset_from_mds()
        else:
            print("Loading a dataset through jsonl")
            self._load_dataset_from_jsonl()

    def _check(self, tokens, position_ids, block_position_ids, labels):
        assert len(tokens) == len(labels) == len(
            position_ids) == len(block_position_ids)
        mask_id = self.tokenizer.convert_tokens_to_ids(self.mask)
        mask_pos = tokens.index(mask_id)
        assert position_ids[:self.max_input_length] == list(
            range(self.max_input_length))
        assert set(position_ids[self.max_input_length:]) == set([mask_pos])
        for i in range(self.max_input_length, len(labels)):
            if labels[i] == -100:
                break
            assert labels[i] == tokens[i + 1]
        assert set(block_position_ids[:self.max_input_length]) == set([0])
        assert block_position_ids[self.max_input_length:] == list(
            range(1, len(tokens) - self.max_input_length + 1))

    @staticmethod
    def build_feature_from_sample(data,
                                  tokenizer,
                                  max_length=1024,
                                  max_input_length=512,  # 仅在生成预测样本时生效,最大不超过1024
                                  max_output_length=512,  # 仅在生成预测样本是生效,最大不超过1024
                                  no_append_glm_mask=False,
                                  left_truncate=False,
                                  sop_id=None,
                                  eop_id=None,
                                  mask_id=None,
                                  cls_id=None,
                                  for_generation=False,
                                  for_eval_classification=False,
                                  pad_id=None,
                                  gpt_data=False,
                                  old_version_tokenizer=False,
                                  undirectional_attention=False,
                                  eos_token='<|endofpiece|>',
                                  add_cls=True,
                                  isolation_position_ids=False,
                                  rotary_type: str = 'none'
                                  ):
        sop_id = sop_id if sop_id else tokenizer.convert_tokens_to_ids(
            tokenizer.sop_token)
        eop_id = eop_id if eop_id else tokenizer.convert_tokens_to_ids(eos_token)
        mask_id = mask_id if mask_id else tokenizer.convert_tokens_to_ids(
            '[gMASK]')
        cls_id = cls_id if cls_id else tokenizer.convert_tokens_to_ids(
            tokenizer.cls_token)
        pad_id = pad_id if pad_id else tokenizer.convert_tokens_to_ids(
            tokenizer.pad_token)
        data['input'] = str(data['input']).replace('\\n', '\n')
        if old_version_tokenizer:
            data['input'] = str(data['input']).replace('\n', '<n>')
        if not gpt_data and for_eval_classification:
            first_token = tokenizer.decode(
                tokenizer(data['input'])['input_ids'][1])
            new_data = {}
            new_data['output'] = data['input'][len(
                first_token):] + data['output']
            new_data['input'] = first_token
            data = new_data

        input_ids = tokenizer(data['input'])['input_ids'][1:-1]
        if for_generation:
            # 预留特殊字符的长度
            if len(input_ids) > max_input_length:
                if left_truncate:
                    input_ids = input_ids[-max_input_length:]
                else:
                    input_ids = input_ids[:max_input_length]
        else:
            if gpt_data:
                num_special_tokens = 3
            else:
                num_special_tokens = 4
            data['output'] = data['output'].replace('\\n', '\n')
            if old_version_tokenizer:
                data['output'] = data['output'].replace('\n', '<n>')
            output_ids = tokenizer(data['output'])['input_ids'][1:-1]
            if len(input_ids) + len(output_ids) > max_length - num_special_tokens:  # 4是需要添加的特殊符号的个数
                if len(input_ids) > (max_length - num_special_tokens) // 2 \
                        and len(output_ids) > (max_length - num_special_tokens) // 2:
                    # 如果都超过了最大长度的一半,那都截取到最大长度的一半
                    half_length = (max_length - num_special_tokens) // 2
                    if left_truncate:
                        input_ids = input_ids[-half_length:]
                    else:
                        input_ids = input_ids[:half_length]
                    output_ids = output_ids[:half_length]
                else:
                    # 从input_ids和output_ids中比较长的那一个截断,input_ids可以选择从左边或右边阶段,output_ids默认从右边截断
                    if len(input_ids) >= len(output_ids):
                        if left_truncate:
                            input_ids = input_ids[-(max_length -
                                                    num_special_tokens - len(output_ids)):]
                        else:
                            input_ids = input_ids[:max_length -
                                                  num_special_tokens - len(output_ids)]
                    else:
                        output_ids = output_ids[:max_length -
                                                num_special_tokens - len(input_ids)]
            assert len(input_ids) + len(output_ids) <= max_length - \
                num_special_tokens
        if gpt_data:
            input_ids = [cls_id] + input_ids
            sep = 0
        else:
            input_ids = [cls_id] + input_ids + [mask_id]
            sep = len(input_ids)
            mask_pos = input_ids.index(mask_id)
            if mask_pos == -1:
                print('No mask')
            position_ids = list(range(len(input_ids)))
            block_position_ids = [0] * len(input_ids)
        # 获得mask所在的位置，用于后面output positionid的构造
        if for_generation:
            if gpt_data:
                sep = 0
                position_ids = list(range(max_length))
            else:
                sep = len(input_ids)
                if "1d" in rotary_type:
                    position_ids = list(range(len(input_ids) + max_output_length + 1))
                else:
                    position_ids = position_ids + \
                        [mask_pos] * (max_output_length +
                                      1)  # 后面input_ids要加一个sop_id
                block_position_ids = block_position_ids + \
                    list(range(1, max_output_length + 2))
                position_ids = [position_ids, block_position_ids]
                if undirectional_attention:
                    # block_position_ids = [0] * len(block_position_ids)
                    block_position_ids = list(range(len(block_position_ids)))
                    position_ids = list(range(len(block_position_ids)))
                    position_ids = [position_ids, block_position_ids]
                    sep = 0
                # 后面input_ids要加一个sop_id
                max_length = len(input_ids) + max_output_length + 1
            generation_attention_mask = np.ones([max_length, max_length])
            generation_attention_mask[:sep, sep:] = 0
            for i in range(sep, max_length):
                generation_attention_mask[i, i + 1:] = 0
            if gpt_data:
                max_output_length = max_length - len(input_ids)
            input_ids = input_ids + [sop_id]
            inputs = {'input_ids': torch.Tensor([input_ids]).long(),
                      'position_ids': torch.Tensor([position_ids]).long(),
                      'generation_attention_mask': torch.Tensor([[generation_attention_mask]]).long()
                      }
            return max_output_length, BatchEncoding(inputs)
        else:
            # labels = output_ids
            output_ids = output_ids + [eop_id]
            labels = output_ids
            # # 拼接输入输出
            tokens = input_ids + [sop_id] + \
                output_ids
            # mask label
            labels = [-100] * len(input_ids) + labels + [-100]
            # 最大长度不全
            if len(tokens) < max_length:
                pad_length = max_length - len(tokens)
                tokens += [pad_id] * pad_length
                labels.extend([-100] * pad_length)
            if gpt_data:
                position_ids = list(range(max_length))
            else:
                if "1d" in rotary_type:
                    position_ids = list(range(len(tokens)))
                else:
                    # position_ids在mask之后全部补mask_pos
                    position_ids = position_ids + \
                        [mask_pos] * (len(tokens) - len(position_ids))
                # block_position_ids在mask之后补1 2 3 4 5..
                block_position_ids = block_position_ids + \
                    list(range(1, len(tokens) - len(block_position_ids) + 1))
                position_ids = [position_ids, block_position_ids]
            if undirectional_attention:
                sep = 0
                position_ids = list(range(max_length))
                # block_position_ids = [0] * max_length
                block_position_ids = list(range(max_length))
                position_ids = [position_ids, block_position_ids]
            assert len(tokens) == len(
                labels) == max_length
            return {'input_ids': tokens,
                    'position_ids': position_ids,
                    'attention_mask': sep,
                    'labels': labels}

    def skip_line(self, line):
        try:
            data = json.loads(line.rstrip('\n\r'))
        except Exception:
            return True
        if not data.get('input', '') or not data.get('output', ''):
            return True
        if not self.kwargs.get("online_packed", True) and isinstance(data.get('input', ''), list):
            return False
        if not isinstance(data.get('input', ''), str) or not isinstance(data.get('output'), str):
            return True
        return False

    def _load_dataset_from_jsonl(self):
        start = time.time()
        self.data_list = []

        self.global_num_samples = 0
        self.local_num_samples = 0
        fins = {}
        if os.path.isdir(self.data_path):
            for root, dirs, files in os.walk(self.data_path):
                for file in files:
                    filename = os.path.join(root, file)
                    fins[filename] = open(filename, 'r')
        else:
            fins[self.data_path] = open(self.data_path, 'r')
        for filename, fin in fins.items():
            for line in fin:
                if self.skip_line(line):
                    continue
                self.global_num_samples += 1
                if self.shard_data and (self.global_num_samples - 1) % self.world_size != self.global_rank:
                    continue
                data = json.loads(line.rstrip('\n\r'))
                self.data_list.append(data)
                self.local_num_samples += 1
        for fin in fins.values():
            fin.close()
        cost = time.time() - start
        print(f'Rank: {self.global_rank}, Number of total samples: {self.global_num_samples}, \
                Number of samples on this shard: {self.local_num_samples}, processing cost {cost} s')

    def _load_dataset_from_mds(self):
        self.data_list = []

        self.global_num_samples = 0
        self.local_num_samples = 0

        ds = mds.load_dataset(
            self.data_path, split=self.kwargs.get("split", "train"))
        print(f"Got a dataset of {self.data_path}: {ds}")
        for data in ds:
            self.global_num_samples += 1
            if self.shard_data and (self.global_num_samples - 1) % self.world_size != self.global_rank:
                continue
            self.local_num_samples += 1
            self.data_list.append(data)

        print(f'Number of total samples: {self.global_num_samples}, \
                Number of samples on this shard: {self.local_num_samples}')

    def __len__(self):
        return self.global_num_samples

    def __getitem__(self, idx):
        cls_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        mask_id = self.tokenizer.convert_tokens_to_ids(self.mask)
        data = self.data_list[idx]

        features = self.build_feature_from_sample(data,
                                                  self.tokenizer,
                                                  self.max_length,
                                                  self.max_input_length,
                                                  self.max_output_length,
                                                  self.no_append_glm_mask,
                                                  left_truncate=self.left_truncate,
                                                  sop_id=self.sop_id,
                                                  eop_id=self.eop_id,
                                                  mask_id=mask_id,
                                                  cls_id=cls_id,
                                                  pad_id=pad_id,
                                                  gpt_data=self.gpt_data,
                                                  eos_token=self.eos_token,
                                                  old_version_tokenizer=self.old_version_tokenizer,
                                                  undirectional_attention=self.undirectional_attention,
                                                  add_cls=self.add_cls,
                                                  isolation_position_ids=self.isolation_position_ids,
                                                  rotary_type=self.rotary_type
                                                  )
        return_colossal_format = self.kwargs.get(
            'return_colossal_format', False)
        # 如果要使用colossalai的训练，这里一定要在idx_data里返回模型需要
        # 所有字段。第二个字段是为了适配colossalai engine的用法，必须返回label
        idx_data = {
            'input_ids': torch.Tensor(features['input_ids']).long(),
            'position_ids': torch.Tensor(features['position_ids']).long(),
            'attention_mask': torch.Tensor(np.array([features['attention_mask']])).long(),
            'labels': torch.Tensor(features['labels']).long()}
        if return_colossal_format:
            return idx_data, {'labels': idx_data['labels']}
        else:
            return idx_data


def main():
    from solutions.antllm.antllm.models.glm.tokenization_glm import GLMTokenizer
    tokenizer = GLMTokenizer.from_pretrained(
        '/workspace/chatgpt/models_0602/sft/AntGLM-10B-SFT-20230602')
    features = GLMInstructionDataset.build_feature_from_sample(
        {'input': '     closet_diff a b c', 'output': 'abc'}, tokenizer)
    __import__('pudb').set_trace()
    print(features)


if "__main__" == __name__:
    main()
