import json
import os
import time
from typing import Iterable
import torch

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from trlx.data.ppo_types import PPORLBatch, PPORLElement
from trlx.pipeline import BaseRolloutStore


class GLMPPORolloutStorage(BaseRolloutStore):
    """
    Rollout storage for training PPO
    """

    def __init__(self, pad_token_id, sop_token_id):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.sop_token_id = sop_token_id
        self.history: Iterable[PPORLElement] = [None]

    def push(self, exps: Iterable[PPORLElement]):
        self.history += exps

    def clear_history(self):
        self.history = []

    def export_history(self, location: str):
        assert os.path.exists(location)

        fpath = os.path.join(location, f"epoch-{str(time.time())}.json")

        def exp_to_dict(exp):
            {k: v.cpu().tolist() for k, v in exp.__dict__.items()}

        data = [exp_to_dict(exp) for exp in self.history]
        with open(fpath, "w") as f:
            f.write(json.dumps(data, indent=2))

    def __getitem__(self, index: int) -> PPORLElement:
        return self.history[index]

    def __len__(self) -> int:
        return len(self.history)

    def collate_fn(self):
        def collate_fn(elems: Iterable[PPORLElement]):
            remove_sop_pad = pad_sequence(
                [elem.query_tensor[:-1] for elem in elems],
                padding_value=self.pad_token_id,
                batch_first=True,
            )
            sops = torch.full((remove_sop_pad.size(0), 1), self.sop_token_id)
            query_padded = torch.cat((remove_sop_pad, sops), -1)

            return PPORLBatch(
                # right padding of already right-padded queries
                query_padded,
                # Right pad the rest, to have a single horizontal query/response split
                pad_sequence(
                    [elem.response_tensor for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
                pad_sequence(
                    [elem.logprobs for elem in elems],
                    padding_value=0.0,
                    batch_first=True,
                ),
                pad_sequence([elem.values for elem in elems], padding_value=0.0, batch_first=True),
                pad_sequence(
                    [elem.rewards for elem in elems],
                    padding_value=0.0,
                    batch_first=True,
                ),
            )
        return collate_fn
    
    def create_loader(
        self,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        def collate_fn(elems: Iterable[PPORLElement]):
            remove_sop_pad = pad_sequence(
                [elem.query_tensor[:-1] for elem in elems],
                padding_value=self.pad_token_id,
                batch_first=True,
            )
            sops = torch.full((remove_sop_pad.size(0), 1), self.sop_token_id)
            query_padded = torch.cat((remove_sop_pad, sops), -1)

            return PPORLBatch(
                # right padding of already right-padded queries
                query_padded,
                # Right pad the rest, to have a single horizontal query/response split
                pad_sequence(
                    [elem.response_tensor for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
                pad_sequence(
                    [elem.logprobs for elem in elems],
                    padding_value=0.0,
                    batch_first=True,
                ),
                pad_sequence([elem.values for elem in elems], padding_value=0.0, batch_first=True),
                pad_sequence(
                    [elem.rewards for elem in elems],
                    padding_value=0.0,
                    batch_first=True,
                ),
            )

        return DataLoader(self, batch_size, shuffle=shuffle, collate_fn=collate_fn)
