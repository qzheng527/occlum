import json

import torch
from torch.utils.data import Dataset


class ChatGLMInstructionDataset(Dataset):
    """
        基于`GLMInstructionDataset`实现，
        `ChatGLMInstructionDataset`主要针对`ChatGLM`的数据格式进行优化:

        对样本按以下格式进行编码：
            `["_"] + input + [gMask] + <bos> + target + <eos>`
    """

    def __init__(self,
                 data_path,
                 tokenizer,
                 max_input_length: int = 550,
                 max_output_length: int = 550,
                 max_length: int = 1024,
                 world_size: int = 1,
                 rank: int = 0,
                 generate_attention_mask: bool = False,
                 generate_position_ids: bool = True,
                 **kwargs):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.max_length = max_length
        self.world_size = world_size
        self.rank = rank
        self.generate_attention_mask = generate_attention_mask
        self.generate_position_ids = generate_position_ids
        self.gMASK_token = '[gMASK]'

        self.MASK_token_id = 150000
        self.gMASK_token_id = 150001
        self.bos_token_id = 150004
        self.eos_token_id = 150005
        self.start_token_id = 20005
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eop_token

        self.kwargs = kwargs
        self._load_dataset_from_jsonl()

    def _check(self, tokens, position_ids, block_position_ids, labels):
        assert len(tokens) == len(labels) == len(
            position_ids) == len(block_position_ids)
        assert position_ids[:self.max_input_length] == list(
            range(self.max_input_length))
        assert max(position_ids) <= len(tokens) - 1
        assert len(position_ids) == len(block_position_ids)
        for i in range(self.max_input_length, len(labels)):
            if labels[i] == -100:
                break
            assert labels[i] == tokens[i + 1]
        assert set(block_position_ids[:self.max_input_length]) == set([0])
        assert block_position_ids[self.max_input_length:] == list(
            range(1, len(tokens) - self.max_input_length + 1))

    def _load_dataset_from_jsonl(self):
        self.data_list = []
        fin = open(self.data_path, 'r')
        pad_id = self.pad_token_id

        all_input_ids = []
        all_position_ids = []
        all_block_position_ids = []
        all_attention_mask = []
        all_labels = []
        for i, line in enumerate(fin):
            data = json.loads(line.rstrip('\n\r'))

            # input token前面添加一个cls，后面添加一个mask
            # "_" + input + "[gMASK]"
            tokenizer_outs = self.tokenizer(
                data['input'], padding=False, add_special_tokens=False, return_attention_mask=True)
            input_ids = tokenizer_outs.input_ids
            attention_mask = tokenizer_outs.attention_mask

            # if input_ids[0] != self.start_token_id:
            #     input_ids = [self.start_token_id] + input_ids
            # input token 截断
            if len(input_ids) > self.max_input_length - 2:
                input_ids = input_ids[:self.max_input_length - 2]
                attention_mask = attention_mask[:self.max_input_length - 2]
            # input token前面添加一个"_"，后面添加一个mask
            input_ids = input_ids + [self.gMASK_token_id]
            mask_position = len(input_ids) - 1
            attention_mask = attention_mask + [1]
            # input token补全
            if len(input_ids) < self.max_input_length:
                input_ids = input_ids + \
                    [pad_id] * (self.max_input_length - len(input_ids))
                attention_mask = attention_mask + \
                    [0] * (self.max_input_length - len(attention_mask))
            attention_mask = torch.tensor(attention_mask)
            attention_mask = attention_mask.unsqueeze(0).expand(
                self.max_input_length + self.max_output_length, -1)

            # 获得mask所在的位置，用于后面output positionid的构造
            output_ids = self.tokenizer(
                data['output'], add_special_tokens=False, padding=False).input_ids
            # 过滤可能的"_"符号
            if output_ids[0] == self.start_token_id:
                output_ids = output_ids[1:]

            # 限制output ids长度
            # output token前面添加一个<bos>，后面添加一个<eos>
            if len(output_ids) > self.max_output_length - 2:
                output_ids = output_ids[:self.max_output_length - 2]
            output_ids = [self.bos_token_id] + output_ids + [self.eos_token_id]
            # label需要错位
            labels = output_ids[1:]
            # 拼接输入输出
            tokens = input_ids + output_ids
            # mask label
            labels = [-100] * len(input_ids) + labels + [-100]
            labels[mask_position] = self.start_token_id

            # mask补全
            if self.generate_attention_mask is True:
                generation_attention_mask = torch.cat([
                    attention_mask.new_zeros(
                        (self.max_input_length, self.max_output_length)),
                    torch.tril(attention_mask.new_ones(
                        (self.max_output_length, self.max_output_length)))
                ], dim=0)
                attention_mask = torch.cat(
                    (attention_mask, generation_attention_mask), dim=1)

            # 最大长度不全
            if len(tokens) < self.max_length:
                pad_length = self.max_length - len(tokens)
                tokens += [pad_id] * pad_length
                labels += [-100] * pad_length

                if self.generate_attention_mask is True:
                    attention_length = attention_mask.size(0)
                    attention_mask = torch.cat([
                        attention_mask,
                        attention_mask.new_zeros(
                            (self.max_length - attention_length, attention_length))
                    ], dim=0)
                    attention_mask = torch.cat([
                        attention_mask,
                        attention_mask.new_zeros(
                            (self.max_length, self.max_length - attention_length))
                    ], dim=1)

            # input段的position_ids和block_position_ids
            # chatglm的position_ids于glm存在区别，直接是0 1 2 3 4 .. (max_len - 1)
            # 建议仅在内存极度紧张的情况下，设置generate_position_ids为False
            if self.generate_position_ids is True:
                position_ids = list(range(len(input_ids))) + [mask_position + 1] + \
                    [mask_position] * (len(tokens) - len(input_ids) - 1)

                block_position_ids = [0] * len(input_ids)
                # block_position_ids在mask之后补1 2 3 4 5..
                block_position_ids = block_position_ids + \
                    list(range(1, len(tokens) - len(block_position_ids) + 1))
                self._check(tokens, position_ids, block_position_ids, labels)

                all_position_ids.append([position_ids, block_position_ids])
                all_block_position_ids.append(block_position_ids)

            all_input_ids.append(tokens)
            all_labels.append(labels)
            # 控制attention mask生成，建议将generate_attention_mask设置为False，
            # 将其生成提到train step中以节约内存
            if self.generate_attention_mask is True:
                attention_mask = attention_mask.unsqueeze(0).cpu().tolist()
                all_attention_mask.append(attention_mask)

        fin.close()
        self.encoded_data = {
            'input_ids': torch.Tensor(all_input_ids).long(),
            'labels': torch.Tensor(all_labels).long()
        }

        if self.generate_position_ids is True:
            self.encoded_data['position_ids'] = torch.Tensor(all_position_ids).long()
        if self.generate_attention_mask is True:
            self.encoded_data["attention_mask"] = ~torch.Tensor(all_attention_mask).long()
        print(f'Number of samples: {len(self.encoded_data["input_ids"])}')

    def __len__(self):
        return len(self.encoded_data['input_ids'])

    def __getitem__(self, idx):
        idx_data = {key: self.encoded_data[key][idx]
                    for key in self.encoded_data}
        return_colossal_format = self.kwargs.get(
            'return_colossal_format', False)
        # 如果要使用colossalai的训练，这里一定要在idx_data里返回模型需要
        # 所有字段。第二个字段是为了适配colossalai engine的用法，必须返回label
        if return_colossal_format:
            return idx_data, {'labels': idx_data['labels']}
        else:
            return idx_data
