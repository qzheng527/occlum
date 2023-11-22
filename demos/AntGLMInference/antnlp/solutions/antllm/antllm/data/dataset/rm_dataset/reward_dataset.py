from typing import Callable

from tqdm import tqdm

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, IterableDataset

from transformers.data.data_collator import DataCollatorMixin
from solutions.antllm.antllm.utils.modeling_glm_rm_utils import build_glm_inputs_from_sample


def is_rank_0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


class RewardDataset(Dataset):
    """
    Dataset for reward model
    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
        max_length: max length of input
    """

    def __init__(
        self, dataset, tokenizer: Callable, max_length: int, return_dict=False
    ) -> None:
        super().__init__()
        self.return_dict = return_dict
        self.chosen = []
        self.rejected = []
        for data in tqdm(dataset, disable=not is_rank_0()):
            prompt = data["prompt"]

            chosen = prompt + " " + data["chosen"]
            chosen_token = tokenizer(
                chosen,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.chosen.append(
                {
                    "input_ids": chosen_token["input_ids"],
                    "attention_mask": chosen_token["attention_mask"]
                    if "attention_mask" in chosen_token
                    else None,
                }
            )

            rejected = prompt + " " + data["rejected"]
            rejected_token = tokenizer(
                rejected,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.rejected.append(
                {
                    "input_ids": rejected_token["input_ids"],
                    "attention_mask": rejected_token["attention_mask"]
                    if "attention_mask" in rejected_token
                    else None,
                }
            )

    def __len__(self):
        length = len(self.chosen)
        return length

    def __getitem__(self, idx):
        if self.return_dict:
            return {
                "input_ids_chosen": self.chosen[idx]["input_ids"],
                "attention_mask_chosen": self.chosen[idx]["attention_mask"],
                "input_ids_rejected": self.rejected[idx]["input_ids"],
                "attention_mask_rejected": self.rejected[idx]["attention_mask"],
            }
        else:
            return (
                self.chosen[idx]["input_ids"],
                self.chosen[idx]["attention_mask"],
                self.rejected[idx]["input_ids"],
                self.rejected[idx]["attention_mask"],
            )


class GLMRewardDataset(Dataset):
    """
    Dataset for reward model
    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        max_input_length: int,
        mask="[gMASK]",
        return_dict=False,
        truncation_side="left",
        dynamic_padding=False,
        eos_token="<|endoftext|>",
        rotary_type="none",
        **kwargs
    ) -> None:
        super().__init__()
        self.chosen = []
        self.rejected = []
        self.return_dict = return_dict
        self.rotary_type = rotary_type

        for data in tqdm(dataset, disable=not is_rank_0()):
            prompt = data["prompt"].replace("\\n", "\n").rstrip()
            chosen = data["chosen"].replace("\\n", "\n").rstrip()
            rejected = data["rejected"].replace("\\n", "\n").rstrip()

            processed_chosen = build_glm_inputs_from_sample(
                prompt=prompt,
                response=chosen,
                tokenizer=tokenizer,
                max_length=max_length,
                max_input_length=max_input_length,
                mask=mask,
                truncation_side=truncation_side,
                dynamic_padding=dynamic_padding,
                eos_token=eos_token,
                rotary_type=rotary_type
            )

            self.chosen.append(processed_chosen)

            processed_rejected = build_glm_inputs_from_sample(
                prompt=prompt,
                response=rejected,
                tokenizer=tokenizer,
                max_length=max_length,
                max_input_length=max_input_length,
                mask=mask,
                truncation_side=truncation_side,
                dynamic_padding=dynamic_padding,
                eos_token=eos_token,
                rotary_type=rotary_type
            )

            self.rejected.append(processed_rejected)

    def __len__(self):
        length = len(self.chosen)
        return length

    def __getitem__(self, idx):
        if self.return_dict:
            return {
                "input_ids_chosen": self.chosen[idx]["input_ids"],
                "attention_mask_chosen": self.chosen[idx]["attention_mask"],
                "position_ids_chosen": self.chosen[idx]["position_ids"],
                "input_ids_rejected": self.rejected[idx]["input_ids"],
                "attention_mask_rejected": self.rejected[idx]["attention_mask"],
                "position_ids_rejected": self.rejected[idx]["position_ids"]
            }
        else:
            return (
                self.chosen[idx]["input_ids"],
                self.chosen[idx]["attention_mask"],
                self.chosen[idx]["position_ids"],
                self.rejected[idx]["input_ids"],
                self.rejected[idx]["attention_mask"],
                self.rejected[idx]["position_ids"]
            )


class GLMIterableRewardDataset(IterableDataset):
    """
    Dataset for reward model
    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        max_input_length: int,
        mask="[gMASK]",
        return_dict=False,
        truncation_side="left",
        dynamic_padding=False,
        eos_token="<|endoftext|>",
        rotary_type="none"
    ) -> None:
        super().__init__()
        self.return_dict = return_dict
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_input_length = max_input_length
        self.truncation_side = truncation_side
        self.mask = mask
        self.dynamic_padding = dynamic_padding
        self.eos_token = eos_token
        self.rotary_type = rotary_type

    def __iter__(self):
        for data in self.dataset:
            prompt = data["prompt"].replace("\\n", "\n").rstrip()
            chosen = data["chosen"].replace("\\n", "\n").rstrip()
            rejected = data["rejected"].replace("\\n", "\n").rstrip()

            processed_chosen = build_glm_inputs_from_sample(
                prompt=prompt,
                response=chosen,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                max_input_length=self.max_input_length,
                mask=self.mask,
                truncation_side=self.truncation_side,
                dynamic_padding=self.dynamic_padding,
                eos_token=self.eos_token,
                rotary_type=self.rotary_type
            )

            processed_rejected = build_glm_inputs_from_sample(
                prompt=prompt,
                response=rejected,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                max_input_length=self.max_input_length,
                mask=self.mask,
                truncation_side=self.truncation_side,
                dynamic_padding=self.dynamic_padding,
                eos_token=self.eos_token,
                rotary_type=self.rotary_type
            )

            if self.return_dict:
                yield {
                    "input_ids_chosen": processed_chosen["input_ids"],
                    "attention_mask_chosen": processed_chosen["attention_mask"],
                    "position_ids_chosen": processed_chosen["position_ids"],
                    "input_ids_rejected": processed_rejected["input_ids"],
                    "attention_mask_rejected": processed_rejected["attention_mask"],
                    "position_ids_rejected": processed_rejected["position_ids"]
                }
            else:
                yield (
                    processed_chosen["input_ids"],
                    processed_chosen["attention_mask"],
                    processed_chosen["position_ids"],
                    processed_rejected["input_ids"],
                    processed_rejected["attention_mask"],
                    processed_rejected["position_ids"]
                )


class RewardDatasetForPointWise(Dataset):
    """
    Dataset for reward model
    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
        max_length: max length of input
    """

    def __init__(
        self, dataset, tokenizer: Callable, max_length: int, num_head: int, return_dict=False, weights=None
    ) -> None:
        super().__init__()
        self.return_dict = return_dict
        self.answer = []
        self.labels = []
        for data in tqdm(dataset, disable=not is_rank_0()):
            prompt = data["prompt"]
            answer = prompt + " " + data["answer"]
            answer_token = tokenizer(
                answer,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.answer.append(
                {
                    "input_ids": answer_token["input_ids"],
                    "attention_mask": answer_token["attention_mask"]
                    if "attention_mask" in answer_token
                    else None,
                }
            )

            sub_label = []
            if weights is None:
                for i in range(0, num_head):
                    sub_label.append(data[f"label_{i+1}"])
            else:
                for i in range(0, num_head):
                    sub_label.append(np.multiply(weights[i], data[f"label_{i+1}"]))                
            
            self.labels.append(
                {
                    "labels": torch.tensor(sub_label)
                }
            )

    def __len__(self):
        length = len(self.answer)
        return length

    def __getitem__(self, idx):
        if self.return_dict:
            return {
                "input_ids": self.answer[idx]["input_ids"],
                "attention_mask": self.answer[idx]["attention_mask"],
                "labels": self.labels[idx]["labels"],
            }
        else:
            return (
                self.answer[idx]["input_ids"],
                self.answer[idx]["attention_mask"],
                self.labels[idx]["labels"],
            )


class GLMRewardDatasetForPointWise(Dataset):
    """
    Dataset for reward model
    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        max_input_length: int,
        num_head: int,
        mask="[gMASK]",
        return_dict=False,
        weights=None,
        truncation_side="right",
        dynamic_padding=False,
        eos_token="<|endoftext|>",
        rotary_type="none"
    ) -> None:
        super().__init__()
        self.return_dict = return_dict
        self.answer = []
        self.labels = []

        for data in tqdm(dataset, disable=not is_rank_0()):
            prompt = data["prompt"].replace("\\n", "\n").rstrip()
            answer = data["answer"].replace("\\n", "\n").rstrip()

            processed_answer = build_glm_inputs_from_sample(
                prompt=prompt, 
                response=answer, 
                tokenizer=tokenizer, 
                max_length=max_length, 
                max_input_length=max_input_length, 
                mask=mask,
                truncation_side=truncation_side,
                dynamic_padding=dynamic_padding,
                eos_token=eos_token,
                rotary_type=rotary_type
            )

            self.answer.append(processed_answer)

            if num_head == 1:
                sub_label = data["label"]
                self.labels.append(
                    {
                        "labels": torch.tensor([sub_label])
                    }
                )
            else:
                sub_label = []
                if weights is None:
                    for i in range(0, num_head):
                        sub_label.append(data[f"label_{i+1}"])
                else:
                    for i in range(0, num_head):
                        sub_label.append(np.multiply(weights[i], data[f"label_{i+1}"]))

                self.labels.append(
                    {
                        "labels": torch.tensor(sub_label)
                    }
                )

    def __len__(self):
        length = len(self.answer)
        return length

    def __getitem__(self, idx):
        if self.return_dict:
            return {
                "input_ids_answer": self.answer[idx]["input_ids"],
                "attention_mask_answer": self.answer[idx]["attention_mask"],
                "position_ids_answer": self.answer[idx]["position_ids"],
                "labels": self.labels[idx]["labels"]
            }
        else:
            return (
                self.answer[idx]["input_ids"],
                self.answer[idx]["attention_mask"],
                self.answer[idx]["position_ids"],
                self.labels[idx]["labels"]
            )


class GLMIterableRewardDatasetForPointWise(IterableDataset):
    """
    Dataset for reward model
    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        max_input_length: int,
        num_head: int,
        mask="[gMASK]",
        return_dict=False,
        weights=None,
        truncation_side="right",
        dynamic_padding=False,
        eos_token="<|endoftext|>",
        rotary_type="none"
    ) -> None:
        super().__init__()
        self.return_dict = return_dict
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_input_length = max_input_length
        self.truncation_side = truncation_side
        self.mask = mask
        self.num_head = num_head
        self.weights = weights,
        self.dynamic_padding = dynamic_padding
        self.eos_token = eos_token
        self.rotary_type = rotary_type

    def __iter__(self):
        for data in self.dataset:
            prompt = data["prompt"].replace("\\n", "\n").rstrip()
            answer = data["answer"].replace("\\n", "\n").rstrip()

            processed_answer = build_glm_inputs_from_sample(
                prompt=prompt,
                response=answer,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                max_input_length=self.max_input_length,
                mask=self.mask,
                truncation_side=self.truncation_side,
                dynamic_padding=self.dynamic_padding,
                eos_token=self.eos_token,
                rotary_type=self.rotary_type
            )

            if self.num_head == 1:
                sub_label = data["label"]
            else:
                sub_label = []
                if self.weights is None:
                    for i in range(0, self.num_head):
                        sub_label.append(data[f"label_{i+1}"])
                else:
                    for i in range(0, self.num_head):
                        sub_label.append(np.multiply(self.weights[i], data[f"label_{i+1}"]))

            if self.return_dict:
                yield {
                    "input_ids_answer": processed_answer["input_ids"],
                    "attention_mask_answer": processed_answer["attention_mask"],
                    "position_ids_answer": processed_answer["position_ids"],
                    "labels": torch.tensor(sub_label)
                }
            else:
                yield (
                    processed_answer["input_ids"],
                    processed_answer["attention_mask"],
                    processed_answer["position_ids"],
                    torch.tensor(sub_label)
                )


class DynamicPaddingCollator(DataCollatorMixin):
    def __init__(self, pad_id=None, data_type="pairwise"):
        super().__init__()
        self.pad_id = pad_id
        self.data_type = data_type

    def collate_pointwise(self, features):
        max_length = -1
        for row in range(len(features)):
            input_ids = features[row]["input_ids_answer"]
            pads = torch.where(input_ids == self.pad_id)[0]
            length = input_ids.shape[0] if pads.shape[0] == 0 else pads[0].item()
            max_length = length if length > max_length else max_length

        for row in range(len(features)):
            features[row]['input_ids_answer'] = features[row]['input_ids_answer'][:max_length]
            features[row]['position_ids_answer'] = features[row]['position_ids_answer'][:, :max_length]

    def collate_pairwise(self, features):
        chosen_max_length = -1
        rejected_max_length = -1
        for row in range(len(features)):
            input_ids_chosen = features[row]["input_ids_chosen"]
            chosen_pads = torch.where(input_ids_chosen == self.pad_id)[0]
            chosen_length = input_ids_chosen.shape[0] if chosen_pads.shape[0] == 0 else chosen_pads[0].item()
            chosen_max_length = chosen_length if chosen_length > chosen_max_length else chosen_max_length

            input_ids_rejected = features[row]["input_ids_rejected"]
            rejected_pads = torch.where(input_ids_rejected == self.pad_id)[0]
            rejected_length = input_ids_rejected.shape[0] if rejected_pads.shape[0] == 0 else rejected_pads[0].item()
            rejected_max_length = rejected_length if rejected_length > rejected_max_length else rejected_max_length

        for row in range(len(features)):
            features[row]['input_ids_chosen'] = features[row]['input_ids_chosen'][:chosen_max_length]
            features[row]['position_ids_chosen'] = features[row]['position_ids_chosen'][:, :chosen_max_length]

            features[row]['input_ids_rejected'] = features[row]['input_ids_rejected'][:rejected_max_length]
            features[row]['position_ids_rejected'] = features[row]['position_ids_rejected'][:, :rejected_max_length]

    def __call__(self, features):
        if self.pad_id is not None:
            if self.data_type == "pairwise":
                self.collate_pairwise(features)
            elif self.data_type == "pointwise":
                self.collate_pointwise(features)
            else:
                raise ValueError(f"unknown data type: {self.data_type}")
        else:
            print('Ignore dynamic_padding, while dynamic_padding, pad_id muast be set')
        batch = {}
        for feature in features:
            for key in feature:
                if key not in batch:
                    batch[key] = []
                batch[key].append(feature[key].unsqueeze(0))
        batch = {key: torch.cat(value) for key, value in batch.items()}
        return batch


class GLMDynamicPaddingCollator(DataCollatorMixin):
    def __init__(self, pad_id=None, data_type="pairwise", mask_id=None, rotary_type="none"):
        super().__init__()
        self.pad_id = pad_id
        self.data_type = data_type
        self.mask_id = mask_id
        self.rotary_type = rotary_type

    def collate_pointwise(self, features):
        max_length = -1
        for row in range(len(features)):
            input_ids = features[row]["input_ids_answer"]
            length = input_ids.shape[0]
            max_length = length if length > max_length else max_length

        data_type = features[0]["input_ids_answer"].dtype
        device = features[0]["input_ids_answer"].device
        for row in range(len(features)):
            mask_pos = torch.where(features[row]["input_ids_answer"] == self.mask_id)[0][0].item()
            pad_len = max_length - features[row]["input_ids_answer"].shape[0]

            features[row]["input_ids_answer"] = torch.cat(
                (
                    features[row]['input_ids_answer'],
                    torch.tensor([self.pad_id] * pad_len, dtype=data_type, device=device)
                )
            )
            if "1d" in self.rotary_type:
                position_ids = torch.tensor(list(range(max_length)), dtype=data_type, device=device)
            else:
                position_ids = torch.cat(
                    (
                        features[row]["position_ids_answer"][0],
                        torch.tensor([mask_pos] * pad_len, dtype=data_type, device=device)
                    )
                )

            start_pos = features[row]["position_ids_answer"][1][-1].item() + 1
            block_position_ids = torch.cat(
                (
                    features[row]["position_ids_answer"][1],
                    torch.tensor(list(range(start_pos, start_pos + pad_len)), dtype=data_type, device=device)
                )
            )
            features[row]["position_ids_answer"] = torch.stack((position_ids, block_position_ids), dim=0)
            assert features[row]["input_ids_answer"].shape[0] == features[row]["position_ids_answer"].shape[1]

    def collate_pairwise(self, features):
        chosen_max_length = -1
        rejected_max_length = -1
        data_type = features[0]["input_ids_chosen"].dtype
        device = features[0]["input_ids_chosen"].device
        for row in range(len(features)):
            input_ids_chosen = features[row]["input_ids_chosen"]
            chosen_length = input_ids_chosen.shape[0]
            chosen_max_length = chosen_length if chosen_length > chosen_max_length else chosen_max_length

            input_ids_rejected = features[row]["input_ids_rejected"]
            rejected_length = input_ids_rejected.shape[0]
            rejected_max_length = rejected_length if rejected_length > rejected_max_length else rejected_max_length

        for row in range(len(features)):
            mask_pos = torch.where(features[row]["input_ids_chosen"] == self.mask_id)[0][0].item()
            
            # chosen
            chosen_pad_len = chosen_max_length - features[row]["input_ids_chosen"].shape[0]
            features[row]["input_ids_chosen"] = torch.cat(
                (
                    features[row]['input_ids_chosen'],
                    torch.tensor([self.pad_id] * chosen_pad_len, dtype=data_type, device=device)
                )
            )

            if "1d" in self.rotary_type:
                position_ids_chosen = torch.tensor(list(range(chosen_max_length)), dtype=data_type, device=device)
            else:
                position_ids_chosen = torch.cat(
                    (
                        features[row]["position_ids_chosen"][0],
                        torch.tensor([mask_pos] * chosen_pad_len, dtype=data_type, device=device)
                    )
                )

            chosen_start_pos = features[row]["position_ids_chosen"][1][-1].item() + 1
            block_position_ids_chosen = torch.cat(
                (
                    features[row]["position_ids_chosen"][1],
                    torch.tensor(list(range(chosen_start_pos, chosen_start_pos + chosen_pad_len)), dtype=data_type, device=device)  # noqa
                )
            )
            features[row]["position_ids_chosen"] = torch.stack((position_ids_chosen, block_position_ids_chosen), dim=0)
            assert features[row]["input_ids_chosen"].shape[0] == features[row]["position_ids_chosen"].shape[1]

            # rejected
            rejected_pad_len = rejected_max_length - features[row]["input_ids_rejected"].shape[0]
            features[row]["input_ids_rejected"] = torch.cat(
                (
                    features[row]["input_ids_rejected"],
                    torch.tensor([self.pad_id] * rejected_pad_len, dtype=data_type, device=device)
                )
            )
            if "1d" in self.rotary_type:
                position_ids_rejected = torch.tensor(list(range(rejected_max_length)), dtype=data_type, device=device)
            else:
                position_ids_rejected = torch.cat(
                    (
                        features[row]["position_ids_rejected"][0],
                        torch.tensor([mask_pos] * rejected_pad_len, dtype=data_type, device=device)
                    )
                )
    
            rejected_start_pos = features[row]["position_ids_rejected"][1][-1].item() + 1
            block_position_ids_rejected = torch.cat(
                (
                    features[row]["position_ids_rejected"][1],
                    torch.tensor(list(range(rejected_start_pos, rejected_start_pos + rejected_pad_len)), dtype=data_type, device=device)  # noqa
                )
            )
            features[row]["position_ids_rejected"] = torch.stack((position_ids_rejected, block_position_ids_rejected), dim=0)  # noqa
            assert features[row]["input_ids_rejected"].shape[0] == features[row]["position_ids_rejected"].shape[1]

    def __call__(self, features):
        if self.pad_id is not None:
            if self.data_type == "pairwise":
                self.collate_pairwise(features)
            elif self.data_type == "pointwise":
                self.collate_pointwise(features)
            else:
                raise ValueError(f"unknown data type: {self.data_type}")
        else:
            print('Ignore dynamic_padding, while dynamic_padding, pad_id muast be set')
        batch = {}
        for feature in features:
            for key in feature:
                if key not in batch:
                    batch[key] = []
                batch[key].append(feature[key].unsqueeze(0))
        batch = {key: torch.cat(value) for key, value in batch.items()}
        return batch
