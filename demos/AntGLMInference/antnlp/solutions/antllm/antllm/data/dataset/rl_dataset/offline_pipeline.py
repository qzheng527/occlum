from typing import List
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedTokenizer
from trlx.pipeline import BasePipeline, register_datapipeline


@register_datapipeline
class GlmPipeline(BasePipeline):
    """
    Tokenizes prompts, unless they are already tokenized, and truncates them to `max_prompt_length` from the right
    """

    def __init__(self, prompts: List[str], max_prompt_length: int, tokenizer: PreTrainedTokenizer):
        super().__init__()

        model_inputs = tokenizer(
            prompts, padding=False, add_special_tokens=True
        )

        prompts_tokens = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        self.tokenizer = tokenizer
        self.prompts = [
            {
                "input_ids": tokens[: -1],
                "attention_mask": mask[: -1],
                "idx": idx
            } for idx, (tokens, mask) in enumerate(zip(prompts_tokens, attention_mask))
        ]

    def __getitem__(self, ix: int):
        return self.prompts[ix]

    def __len__(self) -> int:
        return len(self.prompts)

    def collate_fn(self):
        collate_fn = DataCollatorWithPadding(self.tokenizer) if self.tokenizer else torch.vstack
        return collate_fn
    
    def create_loader(self, batch_size: int, shuffle=False) -> DataLoader:
        collate_fn = DataCollatorWithPadding(self.tokenizer) if self.tokenizer else torch.vstack
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)