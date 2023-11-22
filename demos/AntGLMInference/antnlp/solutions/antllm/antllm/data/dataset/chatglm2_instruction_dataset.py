import os

import torch
from transformers.tokenization_utils_base import BatchEncoding
from solutions.antllm.antllm.data.dataset.glm_instruction_dataset import GLMInstructionDataset


class ChatGLM2InstructionDataset(GLMInstructionDataset):
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
                 eos_token='<|endoftext|>',
                 **kwargs):
        super(GLMInstructionDataset, self).__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.max_length = max_length
        self.undirectional_attention = undirectional_attention
        self.left_truncate = left_truncate
        self.sop_id = self.tokenizer.get_command("sop")
        self.eos_token = eos_token
        self.eop_id = self.tokenizer.get_command("<eos>")
        self.mask = kwargs.get('glm_mask', '[gMASK]')
        self.no_append_glm_mask = no_append_glm_mask
        self.gpt_data = gpt_data
        self.kwargs = kwargs
        self.shard_data = shard_data
        self.world_size = world_size
        self.global_rank = global_rank
        if os.environ.get("MDS_ENABLE", "false").upper() == "TRUE":
            print("Loading a dataset through mds.load_dataset")
            self._load_dataset_from_mds()
        else:
            print("Loading a dataset through jsonl")
            self._load_dataset_from_jsonl()

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
                                  for_generation=False,
                                  for_eval_classification=False,
                                  pad_id=None,
                                  gpt_data=False,
                                  undirectional_attention=False,
                                  eos_token='<|endofpiece|>'
                                  ):
        sop_id = sop_id if sop_id else tokenizer.get_command("sop")
        eop_id = eop_id if eop_id else tokenizer.get_command("<eos>")
        mask_id = mask_id if mask_id else tokenizer.get_command("[gMASK]")
        pad_id = pad_id if pad_id else tokenizer.pad_token_id

        data['input'] = data['input'].replace('\\n', '\n')

        if not gpt_data and for_eval_classification:
            first_token = tokenizer.decode(
                tokenizer(data['input'])['input_ids'][1])
            new_data = {}
            new_data['output'] = data['input'][len(
                first_token):] + data['output']
            new_data['input'] = first_token
            data = new_data

        input_ids = tokenizer(data['input'])['input_ids'][2:]
        if for_generation:
            # 预留特殊字符的长度
            if len(input_ids) > max_input_length:
                if left_truncate:
                    input_ids = input_ids[-max_input_length:]
                else:
                    input_ids = input_ids[:max_input_length]
        else:
            num_special_tokens = 3
            data['output'] = data['output'].replace('\\n', '\n')

            output_ids = tokenizer(data['output'])['input_ids'][2:]
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
            input_ids = [sop_id] + input_ids
        else:
            input_ids = [mask_id] + [sop_id] + input_ids

        # 获得mask所在的位置，用于后面output positionid的构造
        if for_generation:
            max_length = len(input_ids) + max_output_length
            position_ids = len(input_ids)
            generation_attention_mask = [1] * len(input_ids)

            inputs = {'input_ids': torch.Tensor([input_ids]).long(),
                      'position_ids': torch.Tensor([position_ids]).long(),
                      'attention_mask': torch.Tensor([generation_attention_mask]).long()
                      }
            return max_output_length, BatchEncoding(inputs)
        else:
            # labels = output_ids
            output_ids = output_ids + [eop_id]
            labels = output_ids
            # # 拼接输入输出
            tokens = input_ids + output_ids
            # mask label
            labels = [-100] * len(input_ids) + labels
            attention_mask = [1] * len(tokens)
            # 最大长度不全
            if len(tokens) < max_length:
                pad_length = max_length - len(tokens)
                tokens += [pad_id] * pad_length
                attention_mask += [0] * pad_length
                labels.extend([-100] * pad_length)

            position_ids = list(range(max_length))

            assert len(tokens) == len(labels) == max_length

            return {
                'input_ids': tokens,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }

    def __getitem__(self, idx):
        pad_id = self.tokenizer.pad_token_id
        mask_id = self.tokenizer.get_command("[gMASK]")
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
                                                  pad_id=pad_id,
                                                  gpt_data=self.gpt_data,
                                                  eos_token=self.eos_token,
                                                  )
        return_colossal_format = self.kwargs.get(
            'return_colossal_format', False)
        # 如果要使用colossalai的训练，这里一定要在idx_data里返回模型需要
        # 所有字段。第二个字段是为了适配colossalai engine的用法，必须返回label
        idx_data = {
            'input_ids': torch.Tensor(features['input_ids']).long(),
            'position_ids': torch.Tensor(features['position_ids']).long(),
            'attention_mask': torch.Tensor(features['attention_mask']).long(),
            'labels': torch.Tensor(features['labels']).long()}
        if return_colossal_format:
            return idx_data, {'labels': idx_data['labels']}
        else:
            return idx_data


def main():
    from solutions.antllm.antllm.models.chatglm2.tokenization_chatglm2 import ChatGLMTokenizer
    tokenizer = ChatGLMTokenizer.from_pretrained('/home/jiangpeijie.jpj/weights/chatglm2-6B')
    print(tokenizer.encode(text='     closet_diff a b c', add_special_tokens=True, truncation=True, max_length=30))
    print(tokenizer.encode(text='a', add_special_tokens=False, truncation=True, max_length=30))
    features = ChatGLM2InstructionDataset.build_feature_from_sample(
        {'input': '     closet_diff a b c', 'output': 'a'}, tokenizer,
        max_length=30,
    )
    print(features)


if "__main__" == __name__:
    main()
