import numpy as np

import torch
from .glm_instruction_dataset import GLMInstructionDataset
from transformers.tokenization_utils_base import BatchEncoding


class GLMFoTDataset(GLMInstructionDataset):
    '''
    GLM结构模型所使用的Causal Dataset
    数据格式:
    ```json
    {"input": ["中国首都是哪？", "蚂蚁集团总部在哪里？"], "output": ["北京", "杭州"]}
    ```
    其中`input`代表输入数据，`output`代表其对应的输出。
    '''

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
                                  eos_token='<|endofpiece|>',
                                  old_version_tokenizer=False,
                                  undirectional_attention=False,
                                  max_fot_length=32000,
                                  add_cls=True,
                                  isolation_position_ids=False,
                                  rotary_type: str = 'none',
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
        data['input'] = data['input'].replace('\\n', '\n')
        if old_version_tokenizer:
            data['input'] = data['input'].replace('\n', '<n>')
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
            if len(input_ids) > max_fot_length:
                if left_truncate:
                    input_ids = input_ids[-max_fot_length:]
                else:
                    input_ids = input_ids[:max_fot_length]
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

            position_ids = []
            repeat_times = len(input_ids) // max_input_length
            for i in range(repeat_times):
                position_ids += list(range(max_input_length))
            position_ids += list(range(len(input_ids) - max_input_length * repeat_times))

            # remain_pos = len(input_ids) % max_input_length
            # for i in range(max_input_length):
            #     if i < remain_pos:
            #         position_ids += [i] * (repeat_times + 1)
            #     else:
            #         position_ids += [i] * repeat_times

            mask_pos = position_ids[input_ids.index(mask_id)]
            if mask_pos == -1:
                print('No mask')

            block_position_ids = [0] * len(position_ids)
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

                repeat_times = max_output_length // (max_input_length - 1)
                additional_block_position_ids = []
                for i in range(repeat_times):
                    additional_block_position_ids += list(range(1, max_input_length))
                additional_block_position_ids += list(range(
                    1, max_output_length + 2 - (max_input_length - 1) * repeat_times))
                block_position_ids = block_position_ids + \
                    additional_block_position_ids

                position_ids = [position_ids, block_position_ids]
                # 后面input_ids要加一个sop_id
                max_length = len(input_ids) + max_output_length + 1

            generation_attention_mask = np.ones([max_length, max_length])
            generation_attention_mask[:sep, sep:] = 0
            for i in range(sep, max_length):
                generation_attention_mask[i, i + 1:] = 0
            # generation_attention_mask = np.tril(generation_attention_mask)
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
                # position_ids在mask之后全部补mask_pos
                if "1d" in rotary_type:
                    position_ids = list(range(len(tokens)))
                else:
                    position_ids = position_ids + \
                        [mask_pos] * (len(tokens) - len(position_ids))
                # block_position_ids在mask之后补1 2 3 4 5..
                remain_block_length = len(tokens) - len(block_position_ids)
                repeat_times = remain_block_length // (max_input_length - 1)
                additional_block_position_ids = []
                for i in range(repeat_times):
                    additional_block_position_ids += list(range(1, max_input_length))
                additional_block_position_ids += list(range(
                    1, remain_block_length + 1 - (max_input_length - 1) * repeat_times))
                block_position_ids = block_position_ids + \
                    additional_block_position_ids
                position_ids = [position_ids, block_position_ids]

            attention_mask = np.ones([len(tokens), len(tokens)])
            attention_mask[:sep, sep:] = 0
            for i in range(sep, max_length):
                attention_mask[i, i + 1:] = 0

            assert len(tokens) == len(labels)
            return {'input_ids': tokens,
                    'position_ids': position_ids,
                    'attention_mask': attention_mask,
                    'labels': labels}
