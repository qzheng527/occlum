import json

from torch.utils.data import Dataset


class InstructionDataset(Dataset):
    '''
    GPT结构模型所使用的Dataset
    数据格式:
    {"input": "清华大学在哪里", "output": "北京"}
    '''
    def __init__(self, data_path, tokenizer, max_length=550, **kwargs):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.kwargs = kwargs
        self._load_dataset_from_jsonl()

    def _load_dataset_from_jsonl(self):
        self.data_list = []
        fin = open(self.data_path, 'r')
        for i, line in enumerate(fin):
            data = json.loads(line.rstrip('\n\r'))
            self.data_list.append(data['input'] + data['output'])
        fin.close()
        self.encoded_data = self.tokenizer(
            self.data_list, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
        print(f'Number of samples: {len(self.data_list)}')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        idx_data = {key: self.encoded_data[key][idx]
                    for key in self.encoded_data}
        return_colossal_format = self.kwargs.get(
            'return_colossal_format', False)
        idx_data = {**idx_data, 'labels': self.encoded_data['input_ids'][idx]}
        # 如果要使用colossalai的训练，这里一定要在idx_data里返回模型需要
        # 所有字段。第二个字段是为了适配colossalai engine的用法，必须返回label
        if return_colossal_format:
            return idx_data, {'labels': self.encoded_data['input_ids'][idx]}
        else:
            return idx_data
