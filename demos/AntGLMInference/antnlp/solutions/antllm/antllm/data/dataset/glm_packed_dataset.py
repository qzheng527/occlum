import random
import json
import os
import time
import numpy as np

import torch
from transformers.tokenization_utils_base import BatchEncoding

from .glm_instruction_dataset import GLMInstructionDataset


def truncate_sentence(input_ids, output_ids, max_length, left_truncate=False):
    num_special_tokens = 4
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
                    input_ids = input_ids[
                        :max_length - num_special_tokens - len(output_ids)]
            else:
                output_ids = output_ids[:max_length -
                                        num_special_tokens - len(input_ids)]
    return input_ids, output_ids


def build_mask_matrix(seq_length, sep):
    # https://github.com/pytorch/pytorch/issues/101932, fix triu/tril bf16 support
    m = torch.ones((1, seq_length, seq_length))
    mask = torch.arange(
        1, m.shape[-1] + 1).reshape(1, -1, 1).to(m.device)
    ids = torch.arange(
        1, m.shape[-1] + 1).reshape(1, 1, -1).expand(1, m.shape[-1], -1).to(m.device)
    m = (ids <= mask).type_as(m)

    m[0, :, :int(sep)] = 1
    m = m.squeeze(0)
    return m


class GLMPackedDataset(GLMInstructionDataset):
    '''
    GLM结构模型所使用的Causal Dataset
    数据格式:
    ```json
    {"input": ["中国首都是哪？", "蚂蚁集团总部在哪里？"], "output": ["北京", "杭州"]}
    ```
    其中`input`代表输入数据，`output`代表其对应的输出。
    '''

    def pack_data(self, datas, tmp_pack_dir, add_cls=True):
        num_special_tokens = 4

        cat_input_data = []
        cat_output_data = []
        cat_data_length = 0
        packed_datas = []
        start = time.time()
        if not os.path.exists(tmp_pack_dir):
            os.makedirs(tmp_pack_dir, exist_ok=True)
        tmp_pack_file = os.path.join(
            tmp_pack_dir, f'pack_rank_{self.global_rank}.jsonl')
        # 将数据pack并写入到临时文件里
        for i, data in enumerate(datas):
            if i % self.world_size != self.global_rank:
                continue
            data["input"] = data["input"].replace("\\n", "\n")
            data["output"] = data["output"].replace("\\n", "\n")
            if self.old_version_tokenizer:
                data["input"] = data["input"].replace("\n", "<n>")
                data["output"] = data["output"].replace("\n", "<n>")

            input_ids = self.tokenizer(data["input"])["input_ids"][1:-1]
            output_ids = self.tokenizer(data["output"])["input_ids"][1:-1]

            if add_cls:
                num_special_tokens_to_allocate = num_special_tokens * \
                    (len(cat_input_data) + 1)
            else:
                num_special_tokens_to_allocate = (
                    num_special_tokens - 1) * (len(cat_input_data) + 1) + 1
            if cat_input_data and len(input_ids) + len(output_ids) > (
                    self.max_length - num_special_tokens_to_allocate - cat_data_length):
                packed_datas.append(
                    {'input': cat_input_data, 'output': cat_output_data})
                cat_input_data = []
                cat_output_data = []
                cat_data_length = 0

            if cat_data_length == 0 and len(input_ids) + len(output_ids) > self.max_length - num_special_tokens:
                input_ids, output_ids = truncate_sentence(
                    input_ids, output_ids, self.max_length, left_truncate=self.left_truncate)
                packed_datas.append(
                    {'input': [data['input']], 'output': [data['output']]})
                continue

            # input_ids = [self.cls_id] + input_ids + [self.mask_id]
            # output_ids = output_ids + [self.eop_id]
            # tokens = input_ids + [self.sop_id] + output_ids
            tokens = input_ids + output_ids
            cat_input_data.append(data["input"])
            cat_output_data.append(data["output"])
            cat_data_length += len(tokens)

            if i % 5000 == 0:
                cost = time.time() - start
                print(f'packed {i} samples, cost {cost} s')
        if cat_input_data:
            packed_datas.append(
                {'input': cat_input_data, 'output': cat_output_data})
        fout = open(tmp_pack_file, 'w')
        for data in packed_datas:
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')
        fout.close()
        try:
            torch.distributed.barrier()
        except Exception:
            print("The distributed is not init, please use init_process_group.")

        # 读取pack之后的数据
        self.global_num_samples = 0
        self.local_num_samples = 0
        fins = {}
        for root, dirs, files in os.walk(tmp_pack_dir):
            for file in files:
                filename = os.path.join(root, file)
                fins[filename] = open(filename, 'r')

        data_list = []
        for filename, fin in fins.items():
            for i, line in enumerate(fin):
                # if self.skip_line(line):
                #     continue
                self.global_num_samples += 1
                if self.shard_data and (self.global_num_samples - 1) % self.world_size != self.global_rank:
                    continue
                data = json.loads(line.rstrip('\n\r'))
                data_list.append(data)
                self.local_num_samples += 1
        for fin in fins.values():
            fin.close()
        return data_list

    def _load_dataset_from_jsonl(self):
        online_packed = self.kwargs.get('online_packed', True)
        if not online_packed:
            super(GLMPackedDataset, self)._load_dataset_from_jsonl()
            return

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
                # if self.shard_data and (self.global_num_samples - 1) % self.world_size != self.global_rank:
                #     continue
                data = json.loads(line.rstrip('\n\r'))
                self.data_list.append(data)
                self.local_num_samples += 1
        for fin in fins.values():
            fin.close()
        cost = time.time() - start
        print(f'load cost {cost} s')
        start = time.time()
        print(f'Packing data')
        if self.kwargs.get('shuffle', False):
            random.shuffle(self.data_list)
        self.data_list = self.pack_data(self.data_list, self.kwargs.get(
            'tmp_pack_dir', '/tmp/packed_data'), add_cls=True)
        cost = time.time() - start
        print(f'pack cost {cost} s')
        print(f'Rank: {self.global_rank}, Number of total samples: {self.global_num_samples}, \
                Number of samples on this shard: {self.local_num_samples}, processing cost {cost} s')

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
                                  add_cls=True,
                                  eos_token='<|endofpiece|>',
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

        data_lenght = len(data['input'])
        data['input'] = [item.replace('\\n', '\n') for item in data['input']]
        if old_version_tokenizer:
            data['input'] = [item.replace('\n', '<n>')
                             for item in data['input']]

        if not gpt_data and for_eval_classification:
            first_token = tokenizer.decode(
                tokenizer(data['input'][0])['input_ids'][1])
            new_data = {}
            new_data['output'] = [data['input'][i][len(
                first_token):] + data['output'][i] for i in range(data_lenght)]
            new_data['input'] = [first_token] * data_lenght
            data = new_data

        input_ids = [tokenizer(item)['input_ids'][1:-1]
                     for item in data['input']]
        total_input_length = sum([len(item) for item in input_ids])
        if for_generation:
            data['output'] = [item.replace('\\n', '\n')
                              for item in data['output']]
            if old_version_tokenizer:
                data['output'] = [item.replace('\n', '<n>')
                                  for item in data['output']]
            output_ids = [tokenizer(item)['input_ids'][1:-1]
                          for item in data['output']]

            assert len(input_ids) == len(output_ids) + 1

        else:
            if gpt_data:
                num_special_tokens = 3
            else:
                num_special_tokens = 4
            data['output'] = [item.replace('\\n', '\n')
                              for item in data['output']]
            if old_version_tokenizer:
                data['output'] = [item.replace('\n', '<n>')
                                  for item in data['output']]
            output_ids = [tokenizer(item)['input_ids'][1:-1]
                          for item in data['output']]
            total_output_length = sum([len(item) for item in output_ids])

            # 预留特殊字符的长度
            # 4是需要添加的特殊符号的个数

            if add_cls:
                allocated_tokens = num_special_tokens * len(input_ids)
            else:
                allocated_tokens = (num_special_tokens -
                                    1) * len(input_ids) + 1
            if total_input_length + total_output_length > max_length - allocated_tokens:
                if total_input_length > (max_length - num_special_tokens) // 2 \
                        and total_output_length > (max_length - num_special_tokens) // 2:
                    # 如果都超过了最大长度的一半,那都截取到最大长度的一半
                    half_length = (max_length - num_special_tokens) // 2
                    if left_truncate:
                        input_ids[-1] = input_ids[-1][total_input_length - half_length:]
                    else:
                        input_ids[-1] = input_ids[-1][:-
                                                      (total_input_length - half_length)]
                    output_ids[-1] = output_ids[-1][:-
                                                    (total_output_length - half_length)]
                else:
                    # 从input_ids和output_ids中比较长的那一个截断,input_ids可以选择从左边或右边阶段,output_ids默认从右边截断
                    if total_input_length >= total_output_length:
                        if left_truncate:
                            input_ids[-1] = input_ids[-1][
                                -(max_length - (num_special_tokens - 1) * len(input_ids) - 1 - total_output_length):]
                        else:
                            input_ids[-1] = input_ids[-1][
                                :max_length - (num_special_tokens - 1) * len(input_ids) - 1 - total_output_length]
                    else:
                        output_ids[-1] = output_ids[-1][
                            :max_length - (num_special_tokens - 1) * len(input_ids) - 1 - total_input_length]
            total_input_length = sum([len(item) for item in input_ids])
            total_output_length = sum([len(item) for item in output_ids])
            if total_input_length + total_output_length > max_length - allocated_tokens:
                print(f'total_input_length: {total_input_length}')
                print(f'total_output_length: {total_output_length}')
                print(f'max_length: {max_length}')
                print(f'num_special_tokens: {num_special_tokens}')
            assert total_input_length + total_output_length <= max_length - allocated_tokens

        if gpt_data:
            input_ids[0] = [cls_id] + input_ids[0]
            sep = 0  # noqa
        else:
            if add_cls:
                input_ids = [[cls_id] + item + [mask_id] for item in input_ids]
            else:
                input_ids[0] = [cls_id] + input_ids[0]
                input_ids = [item + [mask_id] for item in input_ids]
            sep = 0  # noqa
            mask_pos = 0
            if mask_pos == -1:
                print('No mask')

        # 获得mask所在的位置，用于后面output positionid的构造
        if for_generation:
            packed_input_ids = []
            position_ids = []
            block_position_ids = []
            attention_mask_list = []

            if gpt_data:
                # sep = 0
                position_ids = list(range(max_length))
            else:
                max_index = 0
                for i in range(len(output_ids)):
                    data = input_ids[i] + [sop_id] + output_ids[i] + [eop_id]
                    if undirectional_attention:
                        attention_mask = build_mask_matrix(len(data), 0)
                    else:
                        attention_mask = build_mask_matrix(len(data), len(input_ids[i]))
                    attention_mask_list.append(attention_mask)
                    packed_input_ids += data

                    position_ids += list(
                        range(max_index, max_index + len(input_ids[i]))) + \
                        (len(output_ids[i]) + 2) * [max_index + len(input_ids[i]) - 1]
                    block_position_ids += [0] * len(input_ids[i]) + list(range(1, len(output_ids[i]) + 3))
                    if isolation_position_ids is False:
                        max_index = position_ids[-1] + 1

                # add the input features for last input
                if undirectional_attention:
                    attention_mask = build_mask_matrix(len(input_ids[-1]) + 1 + max_output_length, 0)
                else:
                    attention_mask = build_mask_matrix(len(input_ids[-1]) + 1 + max_output_length, len(input_ids[-1]))
                attention_mask_list.append(attention_mask)

                packed_input_ids += input_ids[-1] + [sop_id]
                position_ids += list(
                    range(max_index, max_index + len(input_ids[-1]))) + [max_index + len(input_ids[-1]) - 1]
                block_position_ids += [0] * len(input_ids[-1]) + [1]

                total_input_length = len(packed_input_ids)
                # sep = 0
                position_ids += [position_ids[-1]] * (max_output_length + 1)
                block_position_ids += list(range(2, max_output_length + 3))
                
                # support for 1d rotary
                if "1d" in rotary_type:
                    position_ids = list(range(len(position_ids)))

                position_ids = [position_ids, block_position_ids]
                # 后面input_ids要加一个sop_id
                max_length = total_input_length + max_output_length

            generation_attention_mask = np.tril(torch.ones([max_length, max_length]))
            total_len = 0
            for i in range(len(attention_mask_list)):
                attention_mask = attention_mask_list[i]
                generation_attention_mask[total_len:total_len + attention_mask.shape[0],
                                          total_len:total_len + attention_mask.shape[0]] = attention_mask
                total_len += len(attention_mask_list[i])

            # only do left truncate
            if len(packed_input_ids) > max_input_length:
                packed_input_ids = packed_input_ids[-max_input_length:]
                max_length = len(packed_input_ids) + max_output_length
                position_ids = [position_ids[0][-max_length:], position_ids[1][-max_length:]]
                generation_attention_mask = generation_attention_mask[-max_length:, -max_length:]

            if gpt_data:
                max_output_length = max_length - total_input_length

            inputs = {'input_ids': torch.Tensor([packed_input_ids]).long(),
                      'position_ids': torch.Tensor([position_ids]).long(),
                      'generation_attention_mask': torch.Tensor([[generation_attention_mask]]).long()
                      }
            return max_output_length, BatchEncoding(inputs)
        else:
            # labels = output_ids
            output_ids = [item + [eop_id] for item in output_ids]
            # # 拼接输入输出
            tokens = []
            attention_mask_list = []
            position_id_list = []
            block_position_id_list = []
            for input, output in zip(input_ids, output_ids):
                data = input + [sop_id] + output
                if undirectional_attention:
                    attention_mask = build_mask_matrix(len(data), 0)
                else:
                    attention_mask = build_mask_matrix(len(data), len(input))
                attention_mask_list.append(attention_mask)
                tokens += data

                if gpt_data:
                    position_ids = list(range(max_length))
                else:
                    position_ids = list(range(len(input)))
                    block_position_ids = [0] * len(input)
                    position_ids = position_ids + \
                        [len(input) - 1] * (len(output) + 1)
                    block_position_ids = block_position_ids + \
                        list(range(1, len(output) + 2))

                    # support for 1d rotary
                    if "1d" in rotary_type:
                        position_ids = list(range(len(position_ids)))

                    # position_ids = list(range(len(tokens)))
                    # block_position_ids = list(range(len(tokens)))
                    position_id_list.append(position_ids)
                    block_position_id_list.append(block_position_ids)
                    # position_ids = [position_ids, block_position_ids]
            # mask label
            labels = []
            for i in range(len(input_ids)):
                labels += [-100] * len(input_ids[i]) + output_ids[i] + [-100]
            # concat position_ids, attention_mask, block_position_ids
            total_len = 0

            pack_attention_mask = torch.tril(
                torch.ones([max_length, max_length]))

            pack_position_ids = []
            pack_block_position_ids = []
            total_len = 0
            max_index = 0
            for i in range(len(attention_mask_list)):
                attention_mask = attention_mask_list[i]
                pack_attention_mask[total_len:total_len + attention_mask.shape[0],
                                    total_len:total_len + attention_mask.shape[0]] = attention_mask
                position_ids = [pid + max_index for pid in position_id_list[i]]
                block_position_ids = block_position_id_list[i]
                pack_position_ids.extend(position_ids)
                pack_block_position_ids.extend(block_position_ids)
                if isolation_position_ids is False:
                    max_index = pack_position_ids[-1] + 1
                total_len += len(attention_mask_list[i])
            position_ids = [pack_position_ids, pack_block_position_ids]

            # 最大长度不全
            if len(tokens) < max_length:
                pad_length = max_length - len(tokens)
                tokens += [pad_id] * pad_length
                labels.extend([-100] * pad_length)
                position_ids[0] += [0] * pad_length
                position_ids[1] += [0] * pad_length

            assert len(tokens) == len(
                labels) == max_length
            return {'input_ids': tokens,
                    'position_ids': position_ids,
                    'attention_mask': pack_attention_mask.numpy(),
                    'labels': labels}
            # return {'input_ids': tokens,
            #         'position_ids': position_ids,
            #         'attention_mask': sep,
            #         'labels': labels}
