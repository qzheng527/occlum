import json
import torch
import numpy as np
import logging
from typing import List, Dict  # noqa
from solutions.antllm.antllm.data.dataset.glm_instruction_dataset import GLMInstructionDataset
from transformers.tokenization_utils_base import BatchEncoding
from torch.utils.data._utils.collate import default_collate


class GLMEvalGenDataset(GLMInstructionDataset):
    '''
    GLM结构模型所使用的评估Dataset
    数据格式:
    {"input": "清华大学在哪里", "references": ["北京"]}
    '''

    def __init__(self,
                 data_path,
                 tokenizer,
                 name="",
                 max_input_length=550,
                 max_output_length=550,
                 max_length=1024,
                 batch_size=4,
                 no_append_glm_mask=False,
                 gpt_data=False,
                 world_size=1,
                 unidirectional=False,
                 rotary_1d=False,
                 global_rank=0,
                 left_truncate=False,
                 shard_data=False,
                 device=None,
                 **kwargs):
        self.name = name
        self.device = device
        self.rotary_1d = rotary_1d
        self.batch_size = batch_size
        self.unidirectional = unidirectional
        super().__init__(data_path=data_path,
                         tokenizer=tokenizer,
                         max_input_length=max_input_length,
                         max_output_length=max_output_length,
                         max_length=max_length,
                         no_append_glm_mask=no_append_glm_mask,
                         gpt_data=gpt_data,
                         world_size=world_size,
                         global_rank=global_rank,
                         left_truncate=left_truncate,
                         shard_data=shard_data,
                         **kwargs)

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
                                  unidirectional=False,
                                  rotary_1d=False,
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
            if unidirectional:
                input_ids = [cls_id] + [mask_id] + input_ids
                sep = 2
                mask_pos = input_ids.index(mask_id)
                assert mask_pos == 1
                position_ids = list(range(2)) + [1] * (len(input_ids) - 2)
                block_position_ids = [0] * 2 + \
                    list(range(1, len(input_ids) - 1))
            else:
                input_ids = [cls_id] + input_ids + [mask_id]
                sep = len(input_ids)
                mask_pos = input_ids.index(mask_id)
                position_ids = list(range(len(input_ids)))
                block_position_ids = [0] * len(input_ids)
            if mask_pos == -1:
                print('No mask')
        # 获得mask所在的位置，用于后面output positionid的构造
        if for_generation:
            if gpt_data:
                sep = 0
                position_ids = list(range(max_length))
            else:
                position_ids = position_ids + \
                    [mask_pos] * (max_output_length +
                                  1)  # 后面input_ids要加一个sop_id
                if unidirectional:
                    sep = 2
                    block_position_ids = block_position_ids + \
                        list(range(len(input_ids) - 1,
                             len(input_ids) + max_output_length))
                else:
                    sep = len(input_ids)
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
            if rotary_1d:
                position_ids[0] = list(range(len(position_ids[0])))
            inputs = {'input_ids': input_ids,
                      'position_ids': position_ids,
                      'generation_attention_mask': [generation_attention_mask]
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
            # padding过程迁移到batch处理中
            return {'input_ids': tokens,
                    'position_ids': position_ids,
                    'block_position_ids': block_position_ids,
                    'attention_mask': sep,
                    'labels': labels}

    def skip_line(self, line):
        try:
            data = json.loads(line.rstrip('\n\r'))
        except Exception:
            return True
        if not data['input'] or not data['references']:
            return True
        return False

    def _load_dataset_from_jsonl(self):
        self.data_list = []
        cls_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        mask_id = self.tokenizer.convert_tokens_to_ids(self.mask)

        all_input_ids = []
        all_position_ids = []
        generation_attention_mask = []
        datas = []
        self.global_num_samples = 0
        self.local_num_samples = 0

        with open(self.data_path, "r", encoding="utf-8") as fin:
            for line in fin:
                if self.skip_line(line):
                    continue
                data = json.loads(line.rstrip('\n\r'))
                self.global_num_samples += 1
                if self.shard_data and (self.global_num_samples - 1) % self.world_size != self.global_rank:
                    continue
                self.local_num_samples += 1
                max_output_length, features = self.build_feature_from_sample(data,
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
                                                                             for_generation=True,
                                                                             pad_id=pad_id,
                                                                             gpt_data=self.gpt_data,
                                                                             unidirectional=self.unidirectional,
                                                                             rotary_1d=self.rotary_1d,
                                                                             old_version_tokenizer=self.kwargs.get(
                                                                                 'old_version_tokenizer', False),
                                                                             )
                self.max_output_length = max_output_length
                all_input_ids.append(features['input_ids'])
                datas.append(data)
                all_position_ids.append(
                    features['position_ids'])
                generation_attention_mask.append(
                    features['generation_attention_mask'])
        self.encoded_data = {'input_ids': all_input_ids,
                             'position_ids': all_position_ids,
                             'generation_attention_mask': generation_attention_mask,
                             'extra': datas}
        if self.global_rank == 0:
            logging.info(f'Number of total samples: {self.global_num_samples}, \
                    Number of samples on this shard: {self.local_num_samples}')

    def collate_batch(self, batch_data) -> Dict[str, torch.Tensor]:
        """
        batch_data:[{"input_ids":[],
                        }...]
        """
        # pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        # mask_id = self.tokenizer.convert_tokens_to_ids('[gMASK]')
        max_ids_length = max([len(data["input_ids"]) for data in batch_data])
        extra_info = [data.pop("extra") for data in batch_data]

        for data in batch_data:
            cur_ids_length = len(data["input_ids"])
            data["input_ids"] = torch.LongTensor(
                [0] * (max_ids_length - cur_ids_length) + data["input_ids"]).to(self.device)

            # pad postition ids with left pad
            # 0, 1, 2, 3, 4 ... -> 0, ..., 0, 1, 2, 3, 4, ...
            padded_position_ids = data["position_ids"]
            padded_position_ids[0] = [
                0] * (max_ids_length - cur_ids_length) + padded_position_ids[0]
            padded_position_ids[1] = [
                0] * (max_ids_length - cur_ids_length) + padded_position_ids[1]
            data["position_ids"] = torch.LongTensor(
                padded_position_ids).to(self.device)

            # pad generation attention mask with left and bottom pad
            new_attention_mask = np.zeros(
                (1, max_ids_length + self.max_output_length,
                 max_ids_length + self.max_output_length)
            )
            new_attention_mask[
                :,
                max_ids_length - cur_ids_length:,
                max_ids_length - cur_ids_length:,
            ] = data["generation_attention_mask"]
            data["generation_attention_mask"] = torch.LongTensor(
                new_attention_mask).to(self.device)

        batch_data = default_collate(batch_data)
        batch_data["extra"] = extra_info
        return batch_data

    def __len__(self):
        return self.local_num_samples

    def __getitem__(self, idx):
        idx_data = {}
        for key in self.encoded_data:
            idx_data[key] = self.encoded_data[key][idx]
            # if key not in ['references', 'options']:
            #     idx_data[key] = torch.Tensor(idx_data[key]).long().to(torch.device("cuda"))
        return_colossal_format = self.kwargs.get(
            'return_colossal_format', False)
        # 所有字段。第二个字段是为了适配colossalai engine的用法，必须返回label
        if return_colossal_format:
            return idx_data, {'labels': idx_data['labels']}
        else:
            return idx_data


class GLMEvalClsDataset(GLMEvalGenDataset):
    '''
    GLM结构模型所使用的评估Dataset
    数据格式:
    {"input": "清华大学在哪里", "references": ["北京"], "options":["地球上","中国"]}
    '''

    def __init__(self,
                 data_path,
                 tokenizer,
                 name="",
                 max_input_length=550,
                 max_output_length=550,
                 max_length=1024,
                 batch_size=4,
                 no_append_glm_mask=False,
                 gpt_data=False,
                 world_size=1,
                 unidirectional=False,
                 rotary_1d=False,
                 global_rank=0,
                 left_truncate=False,
                 shard_data=False,
                 device=None,
                 **kwargs):
        super().__init__(
            data_path=data_path,
            tokenizer=tokenizer,
            name=name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            max_length=max_length,
            batch_size=batch_size,
            no_append_glm_mask=no_append_glm_mask,
            gpt_data=gpt_data,
            world_size=world_size,
            unidirectional=unidirectional,
            rotary_1d=rotary_1d,
            global_rank=global_rank,
            left_truncate=left_truncate,
            shard_data=shard_data,
            device=device,
            **kwargs)

    def _load_dataset_from_jsonl(self):
        self.data_list = []
        cls_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        mask_id = self.tokenizer.convert_tokens_to_ids(self.mask)

        all_input_ids = []
        all_position_ids = []
        attention_masks = []
        datas = []
        block_position_ids = []
        labels = []
        self.global_num_samples = 0
        self.local_num_samples = 0
        with open(self.data_path, "r", encoding="utf-8") as fin:
            for line in fin:
                if self.skip_line(line):
                    continue
                data = json.loads(line.rstrip('\n\r'))
                self.global_num_samples += 1
                if self.shard_data and (self.global_num_samples - 1) % self.world_size != self.global_rank:
                    continue
                self.local_num_samples += 1

                sample_batch_input_ids = []
                sample_batch_label_ids = []
                sample_batch_all_position_ids = []
                sample_batch_attention_masks = []
                sample_batch_block_position_ids = []
                for option in data["options"]:
                    # 将每一个option作为输出拼接，后续用于计算loss
                    data["output"] = option
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
                                                              for_generation=False,
                                                              pad_id=pad_id,
                                                              gpt_data=self.gpt_data,
                                                              unidirectional=self.unidirectional,
                                                              rotary_1d=self.rotary_1d,
                                                              old_version_tokenizer=self.kwargs.get(
                                                                  'old_version_tokenizer', False),
                                                              )

                    sample_batch_input_ids.append(features['input_ids'])
                    sample_batch_label_ids.append(features["labels"])
                    sample_batch_all_position_ids.append(
                        features['position_ids'])
                    sample_batch_attention_masks.append(
                        [features['attention_mask']])
                    sample_batch_block_position_ids.append(
                        features["block_position_ids"])

                datas.append(data)
                all_input_ids.append(sample_batch_input_ids)
                labels.append(sample_batch_label_ids)
                all_position_ids.append(sample_batch_all_position_ids)
                attention_masks.append(sample_batch_attention_masks)
                block_position_ids.append(sample_batch_block_position_ids)
        self.encoded_data = {'input_ids': all_input_ids,
                             'position_ids': all_position_ids,
                             'block_position_ids': block_position_ids,
                             'attention_mask': attention_masks,
                             'labels': labels,
                             'extra': datas}
        if self.global_rank == 0:
            logging.info(f'Number of total samples: {self.global_num_samples}, \
                Number of samples on this shard: {self.local_num_samples}')

    def collate_batch(self, batch_data) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        mask_id = self.tokenizer.convert_tokens_to_ids('[gMASK]')
        max_length = max(
            [max([len(input_id) for input_id in data["input_ids"]]) for data in batch_data])
        extra_info = []
        batch_data_unfolded = []
        for data in batch_data:
            # 将每个sample编码输入平铺开并进行padding
            extra_info.append(data.pop("extra"))
            block_position_ids = data.pop("block_position_ids")
            for i in range(len(data["input_ids"])):
                batch = {}
                mask_pos = data["input_ids"][i].index(mask_id)
                pad_length = max_length - len(data["input_ids"][i])
                batch["input_ids"] = torch.Tensor(
                    data["input_ids"][i] + [pad_id] * pad_length).long().to(self.device)
                batch["attention_mask"] = torch.Tensor(
                    data["attention_mask"][i]).long().to(self.device)
                batch["labels"] = torch.Tensor(
                    data["labels"][i] + [-100] * pad_length).long().to(self.device)
                if self.rotary_1d:
                    # TODO 有待确认
                    data["position_ids"][i] += [pad_id] * \
                        (len(batch["input_ids"]) - len(data["position_ids"][i]))
                else:
                    data["position_ids"][i] += [mask_pos] * \
                        (len(batch["input_ids"]) - len(data["position_ids"][i]))
                if self.unidirectional:
                    block_position_ids[i] = block_position_ids[i] + list(
                        range(block_position_ids[i][-1] + 1, len(batch["input_ids"]) - len(
                            block_position_ids[i]) + block_position_ids[i][-1] + 1))
                else:
                    block_position_ids[i] = block_position_ids[i] + list(
                        range(1, len(batch["input_ids"]) - len(block_position_ids[i]) + 1))
                batch["position_ids"] = torch.Tensor(
                    [data["position_ids"][i], block_position_ids[i]]).long().to(self.device)
                batch_data_unfolded.append(batch)
        batch_data_unfolded = default_collate(batch_data_unfolded)
        batch_data_unfolded["extra"] = extra_info
        return batch_data_unfolded

    def __getitem__(self, idx):
        idx_data = {}
        for key in self.encoded_data:
            idx_data[key] = self.encoded_data[key][idx]
            # if key == "attention_mask":
            #     idx_data[key] = [idx_data[key]]
        return_colossal_format = self.kwargs.get(
            'return_colossal_format', False)
        # 所有字段。第二个字段是为了适配colossalai engine的用法，必须返回label
        if return_colossal_format:
            return idx_data, {'labels': idx_data['labels']}
        else:
            return idx_data


class GLMPretrainEvalDataset(GLMEvalGenDataset):
    '''
    GLM结构模型所使用的评估Dataset
    数据格式:
    {"input": "清华大学在哪里", "references": ["北京"], "options":["地球上","中国"]}
    '''

    def __init__(self,
                 data_path,
                 tokenizer,
                 name="",
                 max_input_length=550,
                 max_output_length=550,
                 max_length=1024,
                 batch_size=4,
                 no_append_glm_mask=False,
                 gpt_data=False,
                 world_size=1,
                 unidirectional=False,
                 rotary_1d=False,
                 global_rank=0,
                 left_truncate=False,
                 shard_data=False,
                 device=None,
                 **kwargs):
        super().__init__(
            data_path=data_path,
            tokenizer=tokenizer,
            name=name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            max_length=max_length,
            batch_size=batch_size,
            no_append_glm_mask=no_append_glm_mask,
            gpt_data=gpt_data,
            world_size=world_size,
            unidirectional=unidirectional,
            rotary_1d=rotary_1d,
            global_rank=global_rank,
            left_truncate=left_truncate,
            shard_data=shard_data,
            device=device,
            **kwargs)
        self.name = name
        self.device = device

    @staticmethod
    def build_feature_from_sample(
        prompt,
        tokenizer,
        pos=-1,
        sop_id=None,
        eop_id=None,
        mask_id=None,
        cls_id=None,
        pad_id=None,
        rotary_1d=False,
        max_length=1024
    ):
        # input_ids0 = [50002] + tokenizer.EncodeAsIds(prompt).tokenization
        # print('prompt', prompt)
        input_ids0 = [50002] + tokenizer.encode(prompt)
        assert len(input_ids0) < 4090
        if pos < 0:
            pos = len(input_ids0) + 1
            input_ids = input_ids0 + [50007] + [50006]
            sep = len(input_ids) - 1  # sop pos
        else:
            input_ids = input_ids0[:pos] + [50007] + \
                [50006] + input_ids0[pos:-1]
            # input_ids = input_ids0[:pos] + [50007] + [50006] + input_ids0[pos:]
            sep = pos + 1  # sop pos
        position_ids = np.arange(len(input_ids))
        if not rotary_1d:
            position_ids[sep:] = sep - 1
        block_position_ids = [0] * (sep - 1)
        block_position_ids += list(range(len(input_ids) -
                                   len(block_position_ids)))
        position_ids = [position_ids.tolist(), block_position_ids]
        seq_len = len(input_ids)
        att_mask = np.tril(np.ones([seq_len, seq_len]))
        att_mask[:, :sep] = 1

        labels = [-100] * (pos + 1) + input_ids0[pos:]

        inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": att_mask,
            "labels": labels
        }
        # inputs = BatchEncoding(inputs)

        return inputs, pos

    def _load_dataset_from_jsonl(self):
        self.data_list = []
        cls_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        mask_id = self.tokenizer.convert_tokens_to_ids(self.mask)

        all_input_ids = []
        all_position_ids = []
        attention_masks = []
        datas = []
        block_position_ids = []
        labels = []
        self.global_num_samples = 0
        self.local_num_samples = 0
        with open(self.data_path, "r", encoding="utf-8") as fin:
            for line in fin:
                if self.skip_line(line):
                    continue
                data = json.loads(line.rstrip('\n\r'))
                self.global_num_samples += 1
                if self.shard_data and (self.global_num_samples - 1) % self.world_size != self.global_rank:
                    continue
                self.local_num_samples += 1

                sample_batch_input_ids = []
                sample_batch_label_ids = []
                sample_batch_all_position_ids = []
                sample_batch_attention_masks = []
                sample_batch_block_position_ids = []
                for option in data["options"]:
                    # 将每一个option作为输出拼接，后续用于计算loss
                    # data["output"] = option
                    prompt = data['input'] + option
                    features, pos = self.build_feature_from_sample(prompt,
                                                                   self.tokenizer,
                                                                   pos=2,
                                                                   sop_id=self.sop_id,
                                                                   eop_id=self.eop_id,
                                                                   mask_id=mask_id,
                                                                   cls_id=cls_id,
                                                                   pad_id=pad_id,
                                                                   rotary_1d=self.rotary_1d
                                                                   )

                    sample_batch_input_ids.append(features['input_ids'])
                    sample_batch_label_ids.append(features["labels"])
                    sample_batch_all_position_ids.append(
                        features['position_ids'][0])
                    sample_batch_attention_masks.append(
                        [pos])
                    sample_batch_block_position_ids.append(
                        features["position_ids"][1])

                datas.append(data)
                all_input_ids.append(sample_batch_input_ids)
                labels.append(sample_batch_label_ids)
                all_position_ids.append(sample_batch_all_position_ids)
                attention_masks.append(sample_batch_attention_masks)
                block_position_ids.append(sample_batch_block_position_ids)
        self.encoded_data = {'input_ids': all_input_ids,
                             'position_ids': all_position_ids,
                             "block_position_ids": block_position_ids,
                             'attention_mask': attention_masks,
                             'extra': datas,
                             "labels": labels}
        print(f'Number of total samples: {self.global_num_samples}, \
                Number of samples on this shard: {self.local_num_samples}')

    def collate_batch(self, batch_data) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        mask_id = self.tokenizer.convert_tokens_to_ids('[gMASK]')
        # max_length = max([len(data["input_ids"]) for data in batch_data])
        # extra_info = []
        """
        for data in batch_data:
            # padding
            # mask_pos = data["input_ids"].index(mask_id)
            #extra_info.append(data.pop("extra"))
            # pad_length = max_length - len(data["input_ids"])
            data["input_ids"] = torch.Tensor(
                data["input_ids"]).long().to(self.device)
            data["attention_mask"] = torch.Tensor(
                data["attention_mask"]).long().to(self.device)
            data["labels"] = torch.Tensor(
                data["labels"]).long().to(self.device)
            block_position_ids = data.pop("block_position_ids")
            # data["position_ids"] += [mask_pos] * (len(data["input_ids"]) - len(data["position_ids"]))
            # block_position_ids = block_position_ids + list(range(1, len(data["input_ids"]) - len(block_position_ids) + 1)) # noqa
            data["position_ids"] = torch.Tensor(
                [data["position_ids"], block_position_ids]).long().to(self.device)
            #data["position_ids"] = torch.cat(
            #    (data["position_ids"], block_position_ids), 1).squeeze(0).long().to(self.device)
        batch_data = default_collate(batch_data)
        batch_data["extra"] = extra_info
        return batch_data
        """

        extra_info = []
        batch_data_unfolded = []
        max_length = max(
            [max([len(input_id) for input_id in data["input_ids"]]) for data in batch_data])
        for data in batch_data:
            # 将每个sample编码输入平铺开并进行padding
            extra_info.append(data.pop("extra"))
            block_position_ids = data.pop("block_position_ids")
            for i in range(len(data["input_ids"])):
                batch = {}
                mask_pos = data["input_ids"][i].index(mask_id)
                pad_length = max_length - len(data["input_ids"][i])
                batch["input_ids"] = torch.Tensor(
                    data["input_ids"][i] + [pad_id] * pad_length).long().to(self.device)
                batch["attention_mask"] = torch.Tensor(
                    data["attention_mask"][i]).long().to(self.device)
                batch["labels"] = torch.Tensor(
                    data["labels"][i] + [-100] * pad_length).long().to(self.device)
                if self.rotary_1d:
                    # TODO 有待确认
                    data["position_ids"][i] += [pad_id] * \
                        (len(batch["input_ids"]) - len(data["position_ids"][i]))
                else:
                    data["position_ids"][i] += [mask_pos] * \
                        (len(batch["input_ids"]) - len(data["position_ids"][i]))
                block_position_ids[i] = block_position_ids[i] + list(
                    range(1, len(batch["input_ids"]) - len(block_position_ids[i]) + 1))
                batch["position_ids"] = torch.Tensor(
                    [data["position_ids"][i], block_position_ids[i]]).long().to(self.device)

                batch_data_unfolded.append(batch)
        batch_data_unfolded = default_collate(batch_data_unfolded)
        batch_data_unfolded["extra"] = extra_info
        return batch_data_unfolded

    def __getitem__(self, idx):
        idx_data = {}
        for key in self.encoded_data:
            idx_data[key] = self.encoded_data[key][idx]
            # if key == "attention_mask":
            #     idx_data[key] = [idx_data[key]]
        return_colossal_format = self.kwargs.get(
            'return_colossal_format', False)
        # 所有字段。第二个字段是为了适配colossalai engine的用法，必须返回label
        if return_colossal_format:
            return idx_data, {'labels': idx_data['labels']}
        else:
            return idx_data


def main():
    # from solutions.antllm.antllm.models.glm.tokenization_glm import GLMTokenizer
    from solutions.antllm.antllm.inference.glm_predictor import GLMForInference
    from torch.utils.data import DataLoader
    model = GLMForInference("glm-super-mini-model")
    # tokenizer = GLMTokenizer.from_pretrained("glm-super-mini-model")
    dataset = GLMEvalGenDataset("eval_data/GSM8k/test_prompts.1k.json",
                                model.tokenizer,
                                name="")
    dataloader = DataLoader(dataset, batch_size=2)
    model = torch.nn.DataParallel(model.model, device_ids=[0, 1])
    for batch in dataloader:
        batch.pop("extra")
        output = model.module.generate(
            **batch,
            max_new_tokens=100,
        )
        print(output)


if "__main__" == __name__:
    main()
