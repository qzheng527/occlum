import json
import os
import warnings

import torch
from typing import Dict
from torch.utils.data import Dataset
# from transformers.tokenization_utils_base import BatchEncoding
from torch.utils.data._utils.collate import default_collate


class LlamaEvalDataset(Dataset):
    '''
    llama2模型所使用的评估Dataset
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
                 global_rank=0,
                 left_truncate=False,
                 shard_data=False,
                 device=None,
                 **kwargs):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.max_length = max_length
        self.left_truncate = left_truncate
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
        self.name = name
        self.device = device
        self.batch_size = batch_size

    def __len__(self):
        return self.local_num_samples

    def skip_line(self, line):
        try:
            data = json.loads(line.rstrip('\n\r'))
        except Exception:
            return True
        if not data['input'] or not data['references']:
            return True
        return False

    def _load_dataset_from_jsonl(self):
        all_input_ids = []
        attention_mask = []
        datas = []
        # all_labels = []
        self.global_num_samples = 0
        self.local_num_samples = 0
        with open(self.data_path, "r") as fin:
            for line in fin:
                if self.skip_line(line):
                    continue
                data = json.loads(line.rstrip('\n\r'))
                self.global_num_samples += 1
                if self.shard_data and (self.global_num_samples - 1) % self.world_size != self.global_rank:
                    continue
                self.local_num_samples += 1
                features = self.tokenizer(data["input"], return_tensors='pt')
                datas.append(data)
                all_input_ids.append(features['input_ids'])
                attention_mask.append(features['attention_mask'])
        self.encoded_data = {'input_ids': all_input_ids,
                             'attention_mask': attention_mask,
                             "extra": datas}
        print(f'Number of total samples: {self.global_num_samples}, \
                Number of samples on this shard: {self.local_num_samples}')

    def collate_batch(self, batch_data) -> Dict[str, torch.Tensor]:
        max_ids_length = max([data["input_ids"].shape[1] for data in batch_data])
        extra_info = [data.pop("extra") for data in batch_data]

        # input_data_shape = [data["input_ids"].shape[1] for data in batch_data]
        # attention_shape = [data["attention_mask"].shape[1] for data in batch_data]
        # print(batch_data)
        # print(f"max_ids_length:{max_ids_length}, batch_size:{len(batch_data)}")
        for data in batch_data:
            data["input_ids"] = data["input_ids"].squeeze(0).tolist()
            data["attention_mask"] = data["attention_mask"].squeeze(0).tolist()
            cur_ids_length = len(data["input_ids"])

            data["input_ids"] = torch.LongTensor(
                [0] * (max_ids_length - cur_ids_length) + data["input_ids"]).to(self.device)
            data["attention_mask"] = torch.LongTensor(
                [0] * (max_ids_length - cur_ids_length) + data["attention_mask"]).to(self.device)

        batch_data = default_collate(batch_data)
        batch_data["extra"] = extra_info

        return batch_data

    def __getitem__(self, idx):
        idx_data = {}
        for key in self.encoded_data:
            idx_data[key] = self.encoded_data[key][idx]
        return_colossal_format = self.kwargs.get(
            'return_colossal_format', False)
        # 所有字段。第二个字段是为了适配colossalai engine的用法，必须返回label
        if return_colossal_format:
            return idx_data, {'labels': idx_data['labels']}
        else:
            return idx_data


class LlamaEvalClsDataset(Dataset):
    '''
    llama2模型所使用的评估Dataset
    数据格式:
    {"input": "清华大学在哪里", "references": ["北京"], "options":["地球上","北京"]}
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
                 global_rank=0,
                 left_truncate=False,
                 shard_data=False,
                 device=None,
                 **kwargs):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.max_length = max_length
        self.left_truncate = left_truncate
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
        self.name = name
        self.device = device
        self.batch_size = batch_size

    def __len__(self):
        return self.local_num_samples

    def skip_line(self, line):
        try:
            data = json.loads(line.rstrip('\n\r'))
        except Exception:
            return True
        if not data['input'] or not data['references']:
            return True
        return False

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
                                  pad_id=None
                                  ):
        sop_id = sop_id if sop_id else tokenizer.convert_tokens_to_ids(
            tokenizer.bos_token)
        eop_id = eop_id if eop_id else tokenizer.convert_tokens_to_ids(
            tokenizer.eos_token)
        # pad_id = pad_id if pad_id else tokenizer.convert_tokens_to_ids(
        #     tokenizer.pad_token)

        # import pdb; pdb.set_trace()
        input_ids = tokenizer(data['input'])['input_ids'][1:-1]  
        output_ids = tokenizer(data['output'])['input_ids'][1:-1] 
        if len(input_ids) + len(output_ids) > max_length:
            warnings.warn("input + output 长度超过max_length，需要截断")
            if len(input_ids) > max_length // 2 and len(output_ids) > max_length // 2:
                # 如果都超过了最大长度的一半,那都截取到最大长度的一半
                half_length = max_length // 2
                if left_truncate:
                    input_ids = input_ids[-half_length:] 
                else:
                    input_ids = input_ids[:half_length] 
                output_ids = output_ids[:half_length] 
            else:
                # 从input_ids和output_ids中比较长的那一个截断,input_ids可以选择从左边或右边阶段,output_ids默认从右边截断
                if len(input_ids) >= len(output_ids):
                    if left_truncate:
                        input_ids = input_ids[-(max_length - len(output_ids)):] 
                    else:
                        input_ids = input_ids[:max_length - len(output_ids)] 
                else:
                    output_ids = output_ids[:max_length - len(input_ids)]
        assert len(input_ids) + len(output_ids) <= max_length

        output_ids = output_ids + [eop_id]
        labels = output_ids
        # # 拼接输入输出
        tokens = input_ids + [sop_id] + \
            output_ids
        # mask label
        labels = [-100] * len(input_ids) + labels + [-100]
        # padding过程迁移到batch处理中
        return {'input_ids': tokens,
                'attention_mask': [1] * len(tokens),
                'labels': labels}

    def _load_dataset_from_jsonl(self):
        sop_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)
        eop_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        pad_id = 0

        all_input_ids = []
        attention_masks = []
        datas = []
        labels = []
        self.global_num_samples = 0
        self.local_num_samples = 0
        with open(self.data_path, "r") as fin:
            for line in fin:
                if self.skip_line(line):
                    continue
                data = json.loads(line.rstrip('\n\r'))
                self.global_num_samples += 1
                if self.shard_data and (self.global_num_samples - 1) % self.world_size != self.global_rank:
                    continue
                self.local_num_samples += 1

                sample_batch_input_ids = []
                sample_batch_attention_masks = []
                sample_batch_label_ids = []
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
                                                              sop_id=sop_id,
                                                              eop_id=eop_id,
                                                              pad_id=pad_id
                                                              )      
                    sample_batch_input_ids.append(features['input_ids'])
                    sample_batch_attention_masks.append(features['attention_mask'])
                    sample_batch_label_ids.append(features["labels"])
                
                datas.append(data)
                all_input_ids.append(sample_batch_input_ids)
                labels.append(sample_batch_label_ids)
                attention_masks.append(sample_batch_attention_masks)

        self.encoded_data = {'input_ids': all_input_ids,
                             'attention_mask': attention_masks,
                             'labels': labels,
                             "extra": datas}

        print(f'Number of total samples: {self.global_num_samples}, \
                Number of samples on this shard: {self.local_num_samples}')

    def collate_batch(self, batch_data) -> Dict[str, torch.Tensor]:
        pad_id = 0  # self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token) if self.tokenizer.pad_token else 0
        max_input_length = max(
            [max([len(input_id) for input_id in data["input_ids"]]) for data in batch_data])
        max_label_length = max(
            [max([len(label) for label in data["labels"]]) for data in batch_data])
        extra_info = []
        batch_data_unfolded = []

        for data in batch_data:
            # 将每个sample编码输入平铺开并进行padding
            extra_info.append(data.pop("extra")) 
            for i in range(len(data["input_ids"])):
                batch = {}
                cur_input_length = len(data["input_ids"][i])
                batch["input_ids"] = torch.LongTensor(
                    [pad_id] * (max_input_length - cur_input_length) + data["input_ids"][i]).to(self.device)
                batch["attention_mask"] = torch.LongTensor(
                    [0] * (max_input_length - cur_input_length) + data["attention_mask"][i]).to(self.device)

                cur_label_length = len(data["labels"][i])
                batch["labels"] = torch.LongTensor(
                    [-100] * (max_label_length - cur_label_length) + data["labels"][i]).to(self.device)
                batch_data_unfolded.append(batch)

        batch_data_unfolded = default_collate(batch_data_unfolded)
        batch_data_unfolded["extra"] = extra_info
        return batch_data_unfolded

    def __getitem__(self, idx):
        idx_data = {}
        for key in self.encoded_data:
            idx_data[key] = self.encoded_data[key][idx]
        return_colossal_format = self.kwargs.get(
            'return_colossal_format', False)
        # 所有字段。第二个字段是为了适配colossalai engine的用法，必须返回label
        if return_colossal_format:
            return idx_data, {'labels': idx_data['labels']}
        else:
            return idx_data


class LlamaPretrainEvalDataset(Dataset):
    '''
    llama2模型所使用的评估Dataset
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
                 global_rank=0,
                 left_truncate=False,
                 shard_data=False,
                 device=None,
                 chunk_id=0,
                 chunk_num=1,
                 **kwargs):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.max_length = max_length
        self.left_truncate = left_truncate
        self.mask = kwargs.get('glm_mask', '[gMASK]')
        self.no_append_glm_mask = no_append_glm_mask
        self.gpt_data = gpt_data
        self.kwargs = kwargs
        self.shard_data = shard_data
        self.world_size = world_size
        self.global_rank = global_rank
        self.chunk_id = chunk_id
        self.chunk_num = chunk_num
        if os.environ.get("MDS_ENABLE", "false").upper() == "TRUE":
            print("Loading a dataset through mds.load_dataset")
            self._load_dataset_from_mds()
        else:
            print("Loading a dataset through jsonl")
            self._load_dataset_from_jsonl()
        self.name = name
        self.device = device
        self.batch_size = batch_size

    def __len__(self):
        return self.local_num_samples

    def skip_line(self, line):
        try:
            data = json.loads(line.rstrip('\n\r'))
        except Exception:
            return True
        if not data['input'] or not data['references']:
            return True
        return False

    def build_feature_from_sample(prompt, tokenizer):
        # import pdb; pdb.set_trace()
        # input_ids0 = [50002] + tokenizer.EncodeAsIds(prompt).tokenization
        inputs0 = tokenizer(prompt, return_tensors='pt')
        # import pdb; pdb.set_trace()
        inputs = {
            "input_ids": torch.Tensor(inputs0['input_ids'][..., :-1]).long(),
            "attention_mask": torch.Tensor(inputs0['attention_mask'][..., :-1]).long(),
            "labels": torch.Tensor(inputs0['input_ids'][..., 1:].cpu().numpy().tolist()).long().squeeze(0)
        }

        # inputs = BatchEncoding(inputs)

        return inputs

    def _load_dataset_from_jsonl(self):
        all_input_ids = []
        attention_mask = []
        datas = []
        # all_labels = []
        self.global_num_samples = 0
        self.local_num_samples = 0

        with open(self.data_path, "r") as fin:
            for line in fin:
                if self.skip_line(line):
                    continue
                data = json.loads(line.rstrip('\n\r'))
                self.global_num_samples += 1
                if self.shard_data and (self.global_num_samples - 1) % self.world_size != self.global_rank:
                    continue
                self.local_num_samples += 1
                features = self.tokenizer(data["input"], return_tensors='pt')
                datas.append(data)
                all_input_ids.append(features['input_ids'])
                attention_mask.append(features['attention_mask'])
        self.encoded_data = {'input_ids': all_input_ids,
                             'attention_mask': attention_mask,
                             "extra": datas}
        print(f'Number of total samples: {self.global_num_samples}, \
                Number of samples on this shard: {self.local_num_samples}')

    def collate_batch(self, batch_data) -> Dict[str, torch.Tensor]:
        # extra_info = [data.pop("extra") for data in batch_data]
        # batch_data = default_collate(batch_data)
        # batch_data["extra"] = extra_info
        data = batch_data[0]
        data['input_ids'] = torch.Tensor(
            data['input_ids']).long().to(self.device)
        data['attention_mask'] = torch.Tensor(
            data['attention_mask']).long().to(self.device)
        data['extra'] = [data['extra']]
        return data

    def __getitem__(self, idx):
        idx_data = {}
        for key in self.encoded_data:
            idx_data[key] = self.encoded_data[key][idx]
        return_colossal_format = self.kwargs.get(
            'return_colossal_format', False)
        # 所有字段。第二个字段是为了适配colossalai engine的用法，必须返回label
        if return_colossal_format:
            return idx_data, {'labels': idx_data['labels']}
        else:
            return idx_data


def main():
    from torch.utils.data import DataLoader
    # from solutions.antllm.antllm.models.llama2.modeling_llama import LlamaForCausalLM
    from solutions.antllm.antllm.models.llama2.tokenization_llama import LlamaTokenizer
    # from accelerate import load_checkpoint_and_dispatch
    # from accelerate import init_empty_weights
    model_name_or_path = "/mntnlp/common_base_model/llama2-7b"
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    dataset = LlamaEvalDataset(
        "/workspace/chatgpt/data/评测数据集/GSM8k/test_prompts.1k.json",
        tokenizer
    )
    ddataloader = DataLoader(dataset, batch_size=3,
                             collate_fn=dataset.collate_batch)
    for batch in ddataloader:
        print(batch)
        break


if "__main__" == __name__:
    main()
