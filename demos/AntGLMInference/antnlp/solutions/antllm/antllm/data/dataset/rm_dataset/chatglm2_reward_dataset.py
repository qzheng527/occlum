from typing import Callable
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
from solutions.antllm.antllm.data.dataset.rm_dataset.reward_dataset import is_rank_0


def build_chatglm2_inputs_from_sample(
    prompt: str,
    response: str,
    tokenizer: Callable,
    max_length: int,
    max_input_length: int,
    truncation_side="right"
):
    pad_id = tokenizer.pad_token_id  # 0

    prompt = "[Round {}]\n\n问：{}\n\n答：".format(1, prompt)
    
    input_ids = tokenizer.encode(text=prompt, add_special_tokens=False)

    # 截断 prompt
    if truncation_side == "right":
        if len(input_ids) > max_input_length - 2:  # gmask and sop tokens
            input_ids = input_ids[: max_input_length - 2]
    else:
        if len(input_ids) > max_input_length - 2:
            input_ids = input_ids[len(input_ids) - max_input_length + 2 :]

    max_output_length = max_length - max_input_length - 1

    response_ids = tokenizer.encode(text=response, add_special_tokens=False)

    if truncation_side == "right":
        if len(response_ids) > max_output_length:
            response_ids = response_ids[:max_output_length]
    else:
        if len(response_ids) > max_output_length:
            response_ids = response_ids[-max_output_length:]

    response_ids = tokenizer.build_inputs_with_special_tokens(input_ids[:], response_ids)
    attention_mask = [1] * len(response_ids)
    position_ids = list(range(len(response_ids)))

    # left padding default
    if len(response_ids) < max_length:
        response_pad_length = max_length - len(response_ids)
        response_ids = [pad_id] * response_pad_length + response_ids
        attention_mask = [pad_id] * response_pad_length + attention_mask
        position_ids = [pad_id] * response_pad_length + position_ids

    assert len(response_ids) == len(position_ids) == len(attention_mask) == max_length
        
    return {
        "input_ids": torch.Tensor(response_ids).long(),
        "attention_mask": torch.Tensor(attention_mask).long(),
        "position_ids": torch.Tensor(position_ids).long()
    }


class ChatGLM2RewardDataset(Dataset):
    """
    ChatGLM2 Dataset for reward model
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
        return_dict=False,
        truncation_side="right"
    ) -> None:
        super().__init__()
        self.chosen = []
        self.rejected = []
        self.return_dict = return_dict

        for data in tqdm(dataset, disable=not is_rank_0()):
            prompt = data["prompt"].replace("\\n", "\n").rstrip()
            chosen = data["chosen"].replace("\\n", "\n").rstrip()
            rejected = data["rejected"].replace("\\n", "\n").rstrip()

            processed_chosen = build_chatglm2_inputs_from_sample(
                prompt=prompt,
                response=chosen,
                tokenizer=tokenizer,
                max_length=max_length,
                max_input_length=max_input_length,
                truncation_side=truncation_side
            )

            self.chosen.append(processed_chosen)

            processed_rejected = build_chatglm2_inputs_from_sample(
                prompt=prompt,
                response=rejected,
                tokenizer=tokenizer,
                max_length=max_length,
                max_input_length=max_input_length,
                truncation_side=truncation_side
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


if __name__ == "__main__":
    # {'[MASK]': 64789, '[gMASK]': 64790, '[sMASK]': 64791, 'sop': 64792, 'eop': 64793}
    # v1: build input pairs with format `X [gMASK] <sop> Y1 <eop>` and `X [gMASK] <sop> Y2 <eop>`
    # v2: build input pairs with format `[gMASK] sop X Y1 </s>` and `[gMASK] sop X Y2 </s>`
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("/mnt1/yingting.wyt/llms/chatglm2-6b", trust_remote_code=True)
    t = "\n\n角色介绍：\n张三：公司高管，经验丰富，喜欢掌控一切。"
    out = tokenizer(t, padding=True,
                    add_special_tokens=True,
                    return_attention_mask=True,)
    print(out["input_ids"])
    print(out["attention_mask"])
    print(tokenizer.eos_token_id)  # 2
    model = AutoModel.from_pretrained("/mnt1/yingting.wyt/llms/chatglm2-6b", trust_remote_code=True)
    for name, param in model.named_parameters():
        if name.find("layers") >= 0:
            print(name)
