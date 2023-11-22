import json

import numpy as np

import torch
from .glm_instruction_dataset import GLMInstructionDataset
from transformers.tokenization_utils_base import BatchEncoding


class GLMEmbeddingDataset(GLMInstructionDataset):
    '''
    GLM结构模型所使用的Embedding Dataset
    数据格式:
    ```json
    {"input": "这只猫真可爱", "output": "这只猫张的非常乖，我很喜欢"}
    ```
    其中`input`代表输入数据，`output`代表其对应的正例。
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
                                  old_version_tokenizer=False
                                  ):
        sop_id = sop_id if sop_id else tokenizer.convert_tokens_to_ids(
            tokenizer.sop_token)
        eop_id = eop_id if eop_id else tokenizer.convert_tokens_to_ids(
            tokenizer.eop_token)
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
            # 预留特殊字符的长度
            if len(input_ids) > max_input_length - num_special_tokens:
                if left_truncate:
                    input_ids = input_ids[-(max_output_length - num_special_tokens):]
                else:
                    input_ids = input_ids[:max_input_length - num_special_tokens]

            if len(output_ids) > max_output_length - num_special_tokens:
                if left_truncate:
                    output_ids = output_ids[-(max_output_length - num_special_tokens):]
                else:
                    output_ids = output_ids[:max_output_length - num_special_tokens]

            assert len(input_ids) <= max_input_length - num_special_tokens
            assert len(output_ids) <= max_output_length - num_special_tokens

        if gpt_data:
            input_ids = [cls_id] + input_ids
            sep = 0
        else:
            input_ids = [cls_id] + input_ids + [mask_id]
            sep = len(input_ids)
            mask_pos = input_ids.index(mask_id)
            if mask_pos == -1:
                print('Input no mask')
            position_ids = list(range(len(input_ids)))
            block_position_ids = [0] * len(input_ids)

            if not for_generation:
                output_ids = [cls_id] + output_ids + [mask_id]
                output_sep = len(output_ids)
                output_mask_pos = output_ids.index(mask_id)
                if output_mask_pos == -1:
                    print('Output no mask')
                output_position_ids = list(range(len(output_ids)))
                output_block_position_ids = [0] * len(output_ids)

        # 获得mask所在的位置，用于后面output positionid的构造
        if for_generation:
            if gpt_data:
                sep = 0
                position_ids = list(range(max_length))
            else:
                sep = len(input_ids)
                position_ids = position_ids + \
                    [mask_pos] * (max_output_length +
                                  1)  # 后面input_ids要加一个sop_id
                block_position_ids = block_position_ids + \
                    list(range(1, max_output_length + 2))
                position_ids = [position_ids, block_position_ids]
                # 后面input_ids要加一个sop_id
                max_length = len(input_ids) + max_output_length + 1
            generation_attention_mask = np.ones([max_length, max_length])
            generation_attention_mask[:sep, sep:] = 0
            for i in range(sep, max_length):
                generation_attention_mask[i, i + 1:] = 0
            if gpt_data:
                max_output_length = max_length - len(input_ids)
            input_ids = input_ids + [sop_id]
            inputs = {'query_ids': torch.Tensor([input_ids]).long(),
                      'query_position_ids': torch.Tensor([position_ids]).long(),
                      'query_attention_mask': torch.Tensor([[generation_attention_mask]]).long()
                      }
            return max_output_length, BatchEncoding(inputs)
        else:
            # 制作q、p的ids
            query_ids = input_ids + [sop_id]
            passage_ids = output_ids + [sop_id]

            # 制作对应的mask
            query_mask = [1] * (sep + 1) + [0] * (max_input_length - sep - 1)
            passage_mask = [1] * (output_sep + 1) + [0] * (max_output_length - output_sep - 1)

            # 最大长度不全
            if len(query_ids) < max_input_length:
                pad_length = max_input_length - len(query_ids)
                query_ids += [pad_id] * pad_length
            if len(passage_ids) < max_output_length:
                pad_length = max_output_length - len(passage_ids)
                passage_ids += [pad_id] * pad_length

            if gpt_data:
                query_position_ids = list(range(max_input_length))
                passage_position_ids = list(range(max_output_length))
            else:
                # position_ids在mask之后全部补mask_pos
                query_position_ids = position_ids + \
                    [mask_pos] * (len(query_ids) - len(position_ids))
                # block_position_ids在mask之后补1 2 3 4 5..
                query_block_position_ids = block_position_ids + \
                    list(range(1, len(query_ids) - len(block_position_ids) + 1))
                query_position_ids = [query_position_ids, query_block_position_ids]

                passage_position_ids = output_position_ids + \
                    [mask_pos] * (len(passage_ids) - len(output_position_ids))
                passage_block_position_ids = output_block_position_ids + \
                    list(range(1, len(passage_ids) - len(output_block_position_ids) + 1))
                passage_position_ids = [passage_position_ids, passage_block_position_ids]

            assert len(query_ids) == max_input_length
            assert len(passage_ids) == max_output_length

            return {
                'query_ids': query_ids,
                'query_position_ids': query_position_ids,
                'query_attention_mask': sep,
                'query_mask': query_mask,
                'passage_ids': passage_ids,
                'passage_position_ids': passage_position_ids,
                'passage_attention_mask': output_sep,
                'passage_mask': passage_mask,
            }

    def _load_dataset_from_jsonl(self):
        self.data_list = []
        fin = open(self.data_path, 'r')
        cls_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        mask_id = self.tokenizer.convert_tokens_to_ids(self.mask)

        all_query_ids = []
        all_query_position_ids = []
        all_query_attention_mask = []
        all_query_mask = []

        all_passage_ids = []
        all_passage_position_ids = []
        all_passage_attention_mask = []
        all_passage_mask = []

        self.global_num_samples = 0
        self.local_num_samples = 0
        for line in fin:
            if not line.strip():
                continue
            self.global_num_samples += 1
            if self.shard_data and (self.global_num_samples - 1) % self.world_size != self.global_rank:
                continue
            self.local_num_samples += 1
            data = json.loads(line.rstrip('\n\r'))
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
                                                      old_version_tokenizer=self.kwargs.get(
                                                          'old_version_tokenizer', False),
                                                      )

            all_query_ids.append(features['query_ids'])
            all_query_position_ids.append(features['query_position_ids'])
            all_query_attention_mask.append(features['query_attention_mask'])
            all_query_mask.append(features['query_mask'])

            all_passage_ids.append(features['passage_ids'])
            all_passage_position_ids.append(features['passage_position_ids'])
            all_passage_attention_mask.append(features['passage_attention_mask'])
            all_passage_mask.append(features['passage_mask'])

        fin.close()
        self.encoded_data = {
            'query_ids': torch.Tensor(all_query_ids).long(),
            'query_position_ids': torch.Tensor(all_query_position_ids).long(),
            'query_attention_mask': torch.Tensor(all_query_attention_mask).long(),
            'query_mask': torch.Tensor(all_query_mask).long(),
            'passage_ids': torch.Tensor(all_passage_ids).long(),
            'passage_position_ids': torch.Tensor(all_passage_position_ids).long(),
            'passage_attention_mask': torch.Tensor(all_passage_attention_mask).long(),
            'passage_mask': torch.Tensor(all_passage_mask).long()
        }
        print(f'Number of total samples: {self.global_num_samples}, \
                Number of samples on this shard: {self.local_num_samples}')
