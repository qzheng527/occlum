import json

import torch
from torch.utils.data import Dataset
from transformers.utils import logging

logger = logging.get_logger(__name__)


class GLMDPODataset(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        max_input_length=550,
        max_output_length=550,
        max_length=1024,
        left_truncate=False,
        **kwargs,
    ):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.max_length = max_length
        self.left_truncate = left_truncate
        self.sop_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sop_token)
        self.eop_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eop_token)
        self.mask = kwargs.get("glm_mask", "[gMASK]")
        self.kwargs = kwargs
        self._load_dataset_from_jsonl()

    @staticmethod
    def build_feature_from_sample(
        input_str,
        output_str,
        tokenizer,
        max_length=1024,
        max_input_length=512,  # 仅在生成预测样本时生效,最大不超过1024
        max_output_length=512,  # 仅在生成预测样本是生效,最大不超过1024
        left_truncate=False,
        sop_id=None,
        eop_id=None,
        mask_id=None,
        cls_id=None,
        pad_id=None,
        old_version_tokenizer=False,
    ):
        num_special_tokens = 4
        sop_id = (
            sop_id if sop_id else tokenizer.convert_tokens_to_ids(tokenizer.sop_token)
        )
        eop_id = (
            eop_id if eop_id else tokenizer.convert_tokens_to_ids(tokenizer.eop_token)
        )
        mask_id = mask_id if mask_id else tokenizer.convert_tokens_to_ids("[gMASK]")
        cls_id = (
            cls_id if cls_id else tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
        )
        pad_id = (
            pad_id if pad_id else tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        )
        input_str = input_str.replace("\\n", "\n")
        input_ids = tokenizer(input_str)["input_ids"][1:-1]

        output_str = output_str.replace("\\n", "\n")
        output_ids = tokenizer(output_str)["input_ids"][1:-1]

        if not left_truncate:
            if len(input_ids) > max_input_length - 2:
                input_ids = input_ids[: max_input_length - 2]
        else:
            if len(input_ids) > max_input_length - 2:
                input_ids = input_ids[len(input_ids) - max_input_length + 2 :]
        
        max_output_length = max_length - max_input_length - 2

        if not left_truncate:
            if len(output_ids) > max_output_length:
                output_ids = output_ids[:max_output_length]
        else:
            if len(output_ids) > max_output_length:
                output_ids = output_ids[-max_output_length:]

        assert len(input_ids) + len(output_ids) <= max_length - num_special_tokens

        input_ids = [cls_id] + input_ids + [mask_id]
        sep = len(input_ids)
        mask_pos = input_ids.index(mask_id)
        if mask_pos == -1:
            logger.info("No mask")
        position_ids = list(range(len(input_ids)))
        block_position_ids = [0] * len(input_ids)

        # 获得mask所在的位置，用于后面output positionid的构造
        # labels = output_ids
        output_ids = output_ids + [eop_id]
        labels = output_ids
        # # 拼接输入输出
        tokens = input_ids + [sop_id] + output_ids
        # mask label
        labels = [-100] * len(input_ids) + labels + [-100]
        
        # 最大长度补全
        if len(tokens) < max_length:
            pad_length = max_length - len(tokens)
            tokens += [pad_id] * pad_length
            # labels.extend([-100] * pad_length)
            labels.extend([pad_id] * pad_length)

        # position_ids在mask之后全部补mask_pos
        position_ids = position_ids + [mask_pos] * (len(tokens) - len(position_ids))
        # block_position_ids在mask之后补1 2 3 4 5..
        block_position_ids = block_position_ids + list(
            range(1, len(tokens) - len(block_position_ids) + 1)
        )
        position_ids = [position_ids, block_position_ids]
        assert len(tokens) == len(labels) == max_length

        return {
            "input_ids": tokens,
            "position_ids": position_ids,
            "attention_mask": sep,
            "labels": labels,
        }

    def _load_dataset_from_jsonl(self):
        self.data_list = []
        fin = open(self.data_path, "r")
        cls_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        mask_id = self.tokenizer.convert_tokens_to_ids(self.mask)

        all_input_ids_chosen = []
        all_position_ids_chosen = []
        all_attention_mask_chosen = []
        all_labels_chosen = []

        all_input_ids_rejected = []
        all_position_ids_rejected = []
        all_attention_mask_rejected = []
        all_labels_rejected = []

        self.global_num_samples = 0
        self.local_num_samples = 0
        for i, line in enumerate(fin):
            self.global_num_samples += 1
            self.local_num_samples += 1
            data = json.loads(line.rstrip("\n\r"))
            chosen_features = self.build_feature_from_sample(
                data["prompt"],
                data["chosen"],
                self.tokenizer,
                self.max_length,
                self.max_input_length,
                self.max_output_length,
                left_truncate=self.left_truncate,
                sop_id=self.sop_id,
                eop_id=self.eop_id,
                mask_id=mask_id,
                cls_id=cls_id,
                pad_id=pad_id,
            )
            rejected_features = self.build_feature_from_sample(
                data["prompt"],
                data["rejected"],
                self.tokenizer,
                self.max_length,
                self.max_input_length,
                self.max_output_length,
                left_truncate=self.left_truncate,
                sop_id=self.sop_id,
                eop_id=self.eop_id,
                mask_id=mask_id,
                cls_id=cls_id,
                pad_id=pad_id,
            )

            all_input_ids_chosen.append(chosen_features["input_ids"])
            all_position_ids_chosen.append(chosen_features["position_ids"])
            all_attention_mask_chosen.append(chosen_features["attention_mask"])
            all_labels_chosen.append(chosen_features["labels"])

            all_input_ids_rejected.append(rejected_features["input_ids"])
            all_position_ids_rejected.append(rejected_features["position_ids"])
            all_attention_mask_rejected.append(rejected_features["attention_mask"])
            all_labels_rejected.append(rejected_features["labels"])

        fin.close()
        self.encoded_data = {
            "input_ids_chosen": torch.Tensor(all_input_ids_chosen).long(),
            "position_ids_chosen": torch.Tensor(all_position_ids_chosen).long(),
            "attention_mask_chosen": torch.Tensor(all_attention_mask_chosen).long(),
            "labels_chosen": torch.Tensor(all_labels_chosen).long(),
            "input_ids_rejected": torch.Tensor(all_input_ids_rejected).long(),
            "position_ids_rejected": torch.Tensor(all_position_ids_rejected).long(),
            "attention_mask_rejected": torch.Tensor(all_attention_mask_rejected).long(),
            "labels_rejected": torch.Tensor(all_labels_rejected).long(),
        }
        logger.info(
            f"Number of total samples: {self.global_num_samples}, \
                Number of samples on this shard: {self.local_num_samples}"
        )

    def __len__(self):
        return self.global_num_samples

    def __getitem__(self, idx):
        idx_data = {key: self.encoded_data[key][idx] for key in self.encoded_data}
        return idx_data
