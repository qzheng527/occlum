#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"

import random
from typing import Dict, Sequence

import numpy as np
import torch

from solutions.antllm.antllm.datav2.batch_collators import (MODE,
                                                            BaseBatchCollator,
                                                            pad_batch)
from solutions.antllm.antllm.datav2.data_utils import contains_sentence_end
from solutions.antllm.antllm.utils.dist_utils import get_rank, get_world_size

from .featurizer import (FEAT_ATTENTION_MASK, FEAT_LOSS_MASK, FEAT_POSITION_ID,
                         FEAT_TARGETS, FEAT_TOKENS, TASK_BERT, TASK_GPT,
                         TASK_SENTENCE, GLMBertFeaturizer, GLMGPTFeaturizer,
                         GLMSentenceInfillingFeaturizer)


def pad_position_id_batch(position_id_batch, pad=0):
    seq_lengths = [len(seq[0]) for seq in position_id_batch]
    if seq_lengths.count(seq_lengths[0]) != len(seq_lengths):
        max_length = max(seq_lengths)
        position_id_batch = [
            np.concatenate(
                (
                    position_ids,
                    np.full(
                        (2, max_length - np.asarray(position_ids).shape[1]), fill_value=pad, dtype=np.longlong
                    ),
                ),
                axis=1,
            )
            for position_ids in position_id_batch
        ]
    return position_id_batch


class GLMBlockCollator(BaseBatchCollator):
    """The batch collator used for GLM pretrain.

    Args:
        tokenizer (Required): the GLM tokenizer.
        max_seq_length (int, Required): the maximum length of the token sequence.
        bert_prob (float, Optional): the probablity of the bert MLM task. Default: 0.3.
        bert_ratio (float, Optional): the ratio of length for bert MLM task. Default: 0.15
        gap_sentence_prob (float, Optional): the probability of sentence infilling. Default: 0
        gap_sentence_ratio (float, Optional): the ratio of length for sentence infilling task. Default: 0.15
        gpt_min_ratio (float, Optional): the minimum ratio of length for generation. Default: 0.5
        average_block_length (int, Optional): the average length of masked block in bert MLM task. Default: 3
        max_block_length (int, Optional): the maximum length of masked block. Default: 40
    """

    def __init__(self,
                 tokenizer,
                 max_seq_length,
                 bert_prob=0.3,
                 bert_ratio=0.15,
                 gap_sentence_prob=0.0,
                 gap_sentence_ratio=0.15,
                 gpt_infill_prob=0.5,
                 gpt_min_ratio=0.5,
                 average_block_length=3,
                 max_block_length=40,
                 block_mask_prob=0.0,
                 context_mask_ratio=0.0,
                 context_mask_range=3,
                 short_seq_prob=0.0,
                 single_span_prob=0.0,
                 block_position_encoding=True,
                 encoder_decoder=False,
                 shuffle_blocks=True,
                 sentinel_token=False,
                 task_mask=False,
                 random_position=False,
                 masked_lm=False,
                 use_prefix_mode=True,
                 eod_token=None) -> None:
        super().__init__(name="glm_block")
        self.eod_token = eod_token if eod_token else tokenizer.pad_token_id
        self.tokenizer = tokenizer

        self.count = 0
        self.max_seq_length = max_seq_length

        assert 0.0 <= bert_prob <= 1.0
        self.bert_prob = bert_prob
        self.gap_sentence_prob = gap_sentence_prob
        self.gpt_prob = 1 - bert_prob - gap_sentence_prob
        assert self.gpt_prob >= -1e-10

        self.infill_prob = gpt_infill_prob
        self.gpt_min_ratio = gpt_min_ratio
        self.bert_ratio = bert_ratio
        self.gap_sentence_ratio = gap_sentence_ratio

        self.block_mask_prob = block_mask_prob
        self.context_mask_ratio = context_mask_ratio
        self.context_mask_range = context_mask_range
        self.short_seq_prob = short_seq_prob
        self.single_span_prob = single_span_prob
        self.block_position_encoding = block_position_encoding
        self.encoder_decoder = encoder_decoder
        self.shuffle_blocks = shuffle_blocks
        self.sentinel_token = sentinel_token

        self.task_mask = task_mask
        self.generation_mask = "[gMASK]" if task_mask else "[MASK]"
        self.generation_mask = self.tokenizer.convert_tokens_to_ids(
            self.generation_mask
        )
        self.gap_sentence_mask = "[sMASK]" if task_mask else "[MASK]"
        self.gap_sentence_mask = self.tokenizer.convert_tokens_to_ids(
            self.gap_sentence_mask
        )
        self.random_position = random_position
        self.masked_lm = masked_lm
        self.use_prefix_mode = use_prefix_mode

        # bert featurizer
        self.glm_bert_featurizer = GLMBertFeaturizer(name="bert",
                                                     tokenizer=self.tokenizer,
                                                     max_seq_length=self.max_seq_length,
                                                     bert_ratio=bert_ratio,
                                                     average_block_length=average_block_length,
                                                     max_block_length=max_block_length,
                                                     encoder_decoder=encoder_decoder,
                                                     masked_lm=masked_lm)

        # sentence filling
        self.glm_sent_featurizer = GLMSentenceInfillingFeaturizer(name="sent",
                                                                  tokenizer=self.tokenizer,
                                                                  max_seq_length=self.max_seq_length,
                                                                  gap_sentence_ratio=self.gap_sentence_ratio,
                                                                  encoder_decoder=self.encoder_decoder,
                                                                  shuffle_blocks=self.shuffle_blocks,
                                                                  sentinel_token=self.sentinel_token,
                                                                  task_mask=self.task_mask,
                                                                  random_position=self.random_position,
                                                                  masked_lm=self.masked_lm,
                                                                  eod_token=self.eod_token
                                                                  )

        # GPT task
        self.glm_gpt_featurizer = GLMGPTFeaturizer(name="gpt",
                                                   tokenizer=self.tokenizer,
                                                   max_seq_length=self.max_seq_length,
                                                   gpt_min_ratio=self.gpt_min_ratio,
                                                   use_prefix_mode=self.use_prefix_mode,
                                                   encoder_decoder=self.encoder_decoder,
                                                   shuffle_blocks=self.shuffle_blocks,
                                                   sentinel_token=self.sentinel_token,
                                                   task_mask=self.task_mask,
                                                   random_position=self.random_position,
                                                   masked_lm=self.masked_lm,
                                                   eod_token=self.eod_token)

    def collate(self, samples: Sequence[Dict]) -> Dict:
        """the key of the tokenized content should be content
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id, num_workers = worker_info.id, worker_info.num_workers
        else:
            worker_id, num_workers = 0, 1
        rng = random.Random(
            (self.count * num_workers + worker_id) * get_world_size() + get_rank()
        )
        self.count += 1

        token_batch, target_batch, loss_mask_batch, position_id_batch = [], [], [], []
        source_batch, target_batch = [], []

        if rng.random() < self.short_seq_prob:
            samples = self.split_samples(samples, rng)

        rand = rng.random()
        single_span = rand < self.single_span_prob
        rand = 0.0 if single_span else rng.random()
        attention_mask = []

        if rand < self.bert_prob:
            self.task = TASK_BERT
            for sample in samples:
                feats = self.glm_bert_featurizer.featurize(sample, single_span=single_span, rng=rng)  # noqa
                if self.encoder_decoder:
                    source_batch.append(feats[FEAT_TOKENS])
                    target_batch.append(feats[FEAT_TARGETS])
                    loss_mask_batch.append(feats[FEAT_LOSS_MASK])
                else:
                    token_batch.append(feats[FEAT_TOKENS])
                    target_batch.append(feats[FEAT_TARGETS])
                    loss_mask_batch.append(feats[FEAT_LOSS_MASK])
                    position_id_batch.append(feats[FEAT_POSITION_ID])
                attention_mask.append(feats[FEAT_ATTENTION_MASK])
        elif rand < self.bert_prob + self.gap_sentence_prob:
            self.task = TASK_SENTENCE
            for sample in samples:
                feats = self.glm_sent_featurizer.featurize(sample, rng=rng)
                token_batch.append(feats[FEAT_TOKENS])
                target_batch.append(feats[FEAT_TARGETS])
                loss_mask_batch.append(feats[FEAT_LOSS_MASK])
                position_id_batch.append(feats[FEAT_POSITION_ID])
                attention_mask.append(feats[FEAT_ATTENTION_MASK])
        else:
            self.task = TASK_GPT
            max_generation_length = rng.randint(
                int(self.gpt_min_ratio * min(map(lambda x: len(x["text"]), samples))),
                max(map(lambda x: len(x["text"]), samples)) - 2,
            )
            for sample in samples:
                feats = self.glm_gpt_featurizer.featurize(
                    sample, max_generation_length=max_generation_length, rng=rng)
                token_batch.append(feats[FEAT_TOKENS])
                target_batch.append(feats[FEAT_TARGETS])
                loss_mask_batch.append(feats[FEAT_LOSS_MASK])
                position_id_batch.append(feats[FEAT_POSITION_ID])
                attention_mask.append(feats[FEAT_ATTENTION_MASK])

        if self.encoder_decoder:
            return {
                "text": torch.tensor(source_batch, dtype=torch.long),
                "target": torch.tensor(target_batch, dtype=torch.long),
                "loss_mask": torch.tensor(loss_mask_batch, dtype=torch.long),
            }
        else:
            token_batch = pad_batch(token_batch, self.tokenizer.pad_token_id)
            target_batch = pad_batch(target_batch, self.tokenizer.pad_token_id)
            loss_mask_batch = pad_batch(loss_mask_batch, 1)
            position_id_batch = pad_position_id_batch(position_id_batch, 0)

        return {
            "text": torch.tensor(token_batch, dtype=torch.long),
            "target": torch.tensor(target_batch, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask_batch, dtype=torch.long),
            "position_id": torch.tensor(position_id_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            MODE: self.task
        }

    def split_samples(self, samples, rng):
        target_length = rng.randrange(32, self.max_seq_length - 1)
        num_splits = (self.max_seq_length - 1) // target_length
        new_samples = []
        cls_id = self.tokenizer.cls_token_id
        eos_id = self.tokenizer.eos_token_id
        for sample in samples:
            tokens, loss_masks = sample["text"][1:], sample["loss_mask"][1:]
            for _ in range(num_splits):
                if target_length >= len(tokens):
                    new_tokens, new_loss_masks = tokens, loss_masks
                else:
                    random_start = rng.randrange(0, len(tokens) - target_length)
                    while random_start > 0 and (
                        tokens[random_start] == eos_id
                        or not (
                            contains_sentence_end(tokens[random_start - 1], self.tokenizer)
                            or tokens[random_start - 1] == eos_id
                        )
                    ):
                        random_start -= 1
                    random_end = random_start + target_length
                    while random_end > random_start and not (
                        contains_sentence_end(tokens[random_end - 1], self.tokenizer)
                        or tokens[random_end - 1] == eos_id
                    ):
                        random_end -= 1
                    if random_end - random_start < target_length // 2:
                        random_end = random_start + target_length
                    new_tokens, new_loss_masks = (
                        tokens[random_start:random_end],
                        loss_masks[random_start:random_end],
                    )
                new_tokens = np.concatenate(([cls_id], new_tokens))
                new_loss_masks = np.concatenate(([0], new_loss_masks))
                new_samples.append({"text": new_tokens, "loss_mask": new_loss_masks})
        return new_samples
