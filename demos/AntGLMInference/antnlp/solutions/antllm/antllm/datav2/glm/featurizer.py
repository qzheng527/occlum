#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"

import copy
import random
from typing import Dict

import numpy as np
import torch
from scipy.stats import poisson
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from solutions.antllm.antllm.datav2.data_utils import SENTENCE_END
from solutions.antllm.antllm.datav2.featurizers import (BaseFeaturizer,
                                                        sample_spans)
from solutions.antllm.antllm.utils.utils import contains_any, index_in_list


def ensure_glm_tokenizer(tokenizer: PreTrainedTokenizer):
    """Check whether the tokenizer is a valid glm tokenizer

    Args:
        tokenizer (PreTrainedTokenizer): A pretrained tokenizer.
    """
    assert isinstance(
        tokenizer, PreTrainedTokenizer), "the tokenizer is not a PreTrainedTokenizer"
    assert hasattr(tokenizer, "sop_token_id") and \
        hasattr(tokenizer, "eop_token_id") and \
        hasattr(tokenizer, "cls_token_id") and \
        hasattr(tokenizer, "mask_token_id") and \
        hasattr(tokenizer, "pad_token_id"), \
        "make sure the tokenizer has below token ids: \
            sop_token_id, eop_token_id, cls_token_id, mask_token_id, pad_token_id"


def contains_sentence_end(tok):
    return contains_any(tok, SENTENCE_END)


FEAT_TOKENS = "tokens"
FEAT_INPUT_IDS = "input_ids"
FEAT_TARGETS = "targets"
FEAT_LABELS = "labels"
FEAT_LOSS_MASK = "loss_mask"
FEAT_POSITION_ID = "position_id"
FEAT_ATTENTION_MASK = "attention_mask"


MODE_TRAIN = "train"
MODE_GENERATION = "generation"


class GLMPrefixLMFeaturizer(BaseFeaturizer):
    """A generic featurizer for prefix language modeling. 
    Will be used in SFT featurizer and GPT featurizer in pretraining stage.

    Args:
        name (str, Required): the name of the featurizer
    """

    def __init__(self,
                 name,
                 tokenizer,
                 max_len,
                 prefix_key="prefix",
                 content_key="content",
                 loss_mask_key="loss_mask",
                 prefix_len=0,
                 add_cls=True,
                 add_eos=True,
                 mode=MODE_TRAIN,
                 block_position_encoding=True,
                 padding=True) -> None:
        super().__init__(name)

        self.tokenizer = tokenizer
        ensure_glm_tokenizer(self.tokenizer)

        self.max_len = max_len

        self.prefix_key = prefix_key
        self.content_key = content_key
        self.loss_mask_key = loss_mask_key

        self.prefix_len = prefix_len

        self.add_cls = add_cls
        self.add_eos = add_eos
        self.mode = mode
        assert self.mode in (
            MODE_TRAIN, MODE_GENERATION), f"only \"{MODE_TRAIN}\" and \"{MODE_GENERATION}\" mode are supported."

        self.padding = padding
        self.block_position_encoding = block_position_encoding

    def featurize(self, sample: Dict, **kwargs) -> Dict:
        """if prefix cannot be found in sample, will use prefix_len to get the prefix token ids.

        For sft, raw input ids and output ids will be used as prefix and content, so we need add cls and eos token.
        For pretraining,
        """
        self.prefix_len = kwargs.get("prefix_len", self.prefix_len)
        content_ids = sample[self.content_key]
        if isinstance(content_ids, np.ndarray):
            content_ids = content_ids.tolist()

        if self.prefix_key in sample:
            prefix_ids = sample[self.prefix_key]
            if isinstance(prefix_ids, np.ndarray):
                prefix_ids = prefix_ids.tolist()
        else:
            prefix_ids = content_ids[:self.prefix_len]
            content_ids = content_ids[self.prefix_len:]

        loss_mask = sample.get(self.loss_mask_key, None)
        use_loss_mask = False
        if loss_mask is not None:
            use_loss_mask = True
            if isinstance(loss_mask, np.ndarray):
                loss_mask = loss_mask.tolist()
            target_mask = loss_mask[self.prefix_len:]
        else:
            target_mask = [0] * len(content_ids)

        if self.add_cls:
            prefix_ids = [self.tokenizer.cls_token_id] + \
                prefix_ids

        sep = len(prefix_ids) + 1
        mask_pos = len(prefix_ids)

        # 获得mask所在的位置，用于后面output positionid的构造
        if self.mode == MODE_GENERATION:
            position_ids = list(range(len(prefix_ids) + 1)) + [mask_pos] * (self.max_len + 1)  # 后面input_ids要加一个sop_id
            block_position_ids = [0] * len(prefix_ids) + list(range(self.max_len + 2))
            position_ids = [position_ids, block_position_ids]
            # 后面input_ids要加一个sop_id
            max_length = len(prefix_ids) + self.max_len + 2
            generation_attention_mask = np.ones([max_length, max_length])
            generation_attention_mask[:sep, sep:] = 0
            for i in range(sep, max_length):
                generation_attention_mask[i, i + 1:] = 0

            prefix_ids = prefix_ids + [self.tokenizer.gmask_token_id] + [self.tokenizer.sop_token_id]
            inputs = {
                'input_ids': torch.Tensor([prefix_ids]).long(),
                'position_ids': torch.Tensor([position_ids]).long(),
                'generation_attention_mask': torch.Tensor([[generation_attention_mask]]).long()
            }
            return self.max_len, BatchEncoding(inputs)
        else:
            if len(prefix_ids) + 1 + len(content_ids) > self.max_len:
                truncated_content_len = self.max_len - len(prefix_ids) - 1  # gmask
                assert truncated_content_len > 0
                content_ids = content_ids[:truncated_content_len]

            if self.add_eos:
                content_ids = content_ids + [self.tokenizer.eos_token_id]
                target_mask = target_mask + [1]

            tokens = prefix_ids + [self.tokenizer.gmask_token_id] + [self.tokenizer.sop_token_id] + content_ids[:-1]
            # mask label
            if not use_loss_mask:
                labels = [-100] * (len(prefix_ids) + 1) + content_ids
            else:
                labels = prefix_ids + [self.tokenizer.gmask_token_id] + content_ids
                loss_mask = [1] * (len(prefix_ids) + 1) + target_mask

            position_ids = list(range(len(prefix_ids) + 1)) + [mask_pos] * len(content_ids)
            if self.block_position_encoding:
                block_position_ids = [0] * len(prefix_ids) + list(range(len(content_ids) + 1))
            else:
                block_position_ids = [0] * len(prefix_ids) + [1] * (len(content_ids) + 1)

            if self.padding and len(tokens) < self.max_len:
                pad_len = self.max_len - len(tokens)
                tokens += [self.tokenizer.pad_token_id] * pad_len
                if not use_loss_mask:
                    labels += [-100] * pad_len
                else:
                    labels = [self.tokenizer.pad_token_id] * pad_len
                    loss_mask = [1] * pad_len
                position_ids += [0] * pad_len
                block_position_ids += [0] * pad_len

            position_ids = [position_ids, block_position_ids]
            assert len(tokens) == len(labels) <= self.max_len
            if use_loss_mask:
                return {
                    'input_ids': tokens,
                    'position_ids': position_ids,
                    'attention_mask': sep,
                    'labels': labels,
                    "loss_mask": loss_mask
                }
            else:
                return {
                    'input_ids': tokens,
                    'position_ids': position_ids,
                    'attention_mask': sep,
                    'labels': labels
                }


class GLMSeq2SeqFeaturizer(BaseFeaturizer):
    """The featurizer used for GLM SFT.

    Args:
    """

    def __init__(self,
                 name,
                 tokenizer: PreTrainedTokenizer,
                 need_tokenize=True,
                 mode="train",
                 max_length=1024,
                 max_input_length=512,
                 max_output_length=512,
                 left_truncate=True,
                 instruction_key="instruction",
                 input_key="input",
                 output_key="output"
                 ) -> None:
        super().__init__(name)
        self.mode = mode
        assert self.mode in (
            "train", "generation"), "only \"train\" and \"generation\" mode are supported."

        self.need_tokenize = need_tokenize
        self.tokenizer = tokenizer

        self.max_length = max_length
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.left_truncate = left_truncate

        self.instruction_key = instruction_key
        self.input_key = input_key
        self.output_key = output_key

        self.prefixlm_featurizer = GLMPrefixLMFeaturizer(
            name=name, tokenizer=self.tokenizer, max_len=self.max_length
            if self.mode == MODE_TRAIN else self.max_output_length,
            mode=self.mode)

    def featurize(self, sample, **kwargs):
        if self.need_tokenize:
            input = ""
            if self.instruction_key in sample:
                input += sample[self.instruction_key]
            if self.input_key in sample:
                input += sample[self.input_key]

            input = input.replace('\\n', '\n')
            input_ids = self.tokenizer(input)['input_ids'][1:-1]
        else:
            input_ids = []
            if self.instruction_key in sample:
                input_ids.extend(sample[self.instruction_key])
            if self.input_key in sample:
                input_ids.extend(sample[self.input_key])

        if self.mode == "generation":
            # 预留特殊字符的长度
            if len(input_ids) > self.max_input_length:
                if self.left_truncate:
                    input_ids = input_ids[-self.max_input_length:]
                else:
                    input_ids = input_ids[:self.max_input_length]
        else:
            num_special_tokens = 4  # cls, gmask, sop, eos

            if self.need_tokenize:
                output = sample[self.output_key]
                output = output.replace('\\n', '\n')
                output_ids = self.tokenizer(output)[
                    'input_ids'][1:-1]
            else:
                output_ids = sample[self.output_key]

            if len(input_ids) + len(output_ids) > self.max_length - num_special_tokens:  # 4是需要添加的特殊符号的个数
                if len(input_ids) > (self.max_length - num_special_tokens) // 2 \
                        and len(output_ids) > (self.max_length - num_special_tokens) // 2:
                    # 如果都超过了最大长度的一半,那都截取到最大长度的一半
                    half_length = (self.max_length - num_special_tokens) // 2
                    if self.left_truncate:
                        input_ids = input_ids[-half_length:]
                    else:
                        input_ids = input_ids[:half_length]
                    output_ids = output_ids[:half_length]
                else:
                    # 从input_ids和output_ids中比较长的那一个截断,input_ids可以选择从左边或右边阶段,output_ids默认从右边截断
                    if len(input_ids) >= len(output_ids):
                        if self.left_truncate:
                            input_ids = input_ids[-(self.max_length -
                                                    num_special_tokens - len(output_ids)):]
                        else:
                            input_ids = input_ids[:self.max_length -
                                                  num_special_tokens - len(output_ids)]
                    else:
                        output_ids = output_ids[:self.max_length -
                                                num_special_tokens - len(input_ids)]
            assert len(input_ids) + len(output_ids) <= self.max_length - \
                num_special_tokens

        return self.prefixlm_featurizer.featurize({"prefix": input_ids, "content": output_ids})


class GLMCausalLMFeaturizer(BaseFeaturizer):
    """featurizer used for GPT task in GLM

    Args:
    """

    def __init__(self, name) -> None:
        super().__init__(name)

    def featurize(self, sample: Dict, **kwargs) -> Dict:
        pass


TASK_BERT = "bert"
TASK_SENTENCE = "sentence"
TASK_GPT = "gpt"


class GLMBlockInfillingFeaturizer(BaseFeaturizer):
    """Provide basic functionalities of blank infilling task for GLM
    """

    def __init__(self,
                 name,
                 tokenizer,
                 max_seq_length,
                 bert_ratio=0.15,
                 block_mask_prob=0.0,
                 context_mask_ratio=0.0,
                 context_mask_range=3,
                 block_position_encoding=True,
                 encoder_decoder=False,
                 shuffle_blocks=True,
                 sentinel_token=False,
                 task_mask=False,
                 random_position=False,
                 masked_lm=False,
                 eod_token=None) -> None:
        super().__init__(name)
        self.tokenizer = tokenizer
        ensure_glm_tokenizer(self.tokenizer)

        self.max_seq_length = max_seq_length
        self.bert_ratio = bert_ratio
        self.block_mask_prob = block_mask_prob
        self.context_mask_ratio = context_mask_ratio
        self.context_mask_range = context_mask_range
        self.block_position_encoding = block_position_encoding
        self.encoder_decoder = encoder_decoder
        self.shuffle_blocks = shuffle_blocks
        self.sentinel_token = sentinel_token
        self.task_mask = task_mask
        self.random_position = random_position
        self.masked_lm = masked_lm
        self.eod_token = eod_token if eod_token else tokenizer.pad_token_id

        self.generation_mask = "[gMASK]" if task_mask else "[MASK]"
        self.generation_mask = self.tokenizer.convert_tokens_to_ids(
            self.generation_mask
        )
        self.gap_sentence_mask = "[sMASK]" if task_mask else "[MASK]"
        self.gap_sentence_mask = self.tokenizer.convert_tokens_to_ids(
            self.gap_sentence_mask
        )

    def generate_blank_data(self,
                            sample,
                            masked_lengths,
                            attention_mask,
                            rng: random.Random,
                            task="bert"
                            ):
        rng.shuffle(masked_lengths)
        tokens, loss_masks = sample["text"], sample["loss_mask"]
        assert tokens[0] == self.tokenizer.cls_token_id
        block_spans = self.sample_span_in_document(tokens, masked_lengths, rng)
        if len(block_spans) < len(masked_lengths):
            return None
        if self.masked_lm:
            data = self.make_masked_data(
                tokens, loss_masks, attention_mask, block_spans, rng
            )
        else:
            data = self.make_block_data(
                tokens, loss_masks, attention_mask, block_spans, rng, task=task
            )
        return data

    def sample_span_in_document(self,
                                tokens,
                                masked_lengths,
                                rng: random.Random):
        rng.shuffle(masked_lengths)
        mask_spans = []
        mask_index = 0
        indices = [-1] + np.where(tokens == self.eod_token)[0].tolist()
        last_index = len(tokens)
        documents = []
        # split the documents
        for index in reversed(indices):
            start_index = index
            if (
                start_index + 1 < len(tokens)
                and tokens[start_index + 1] == self.tokenizer.cls_token_id
            ):
                start_index += 1
            length = last_index - start_index - 1
            if last_index == len(tokens) and length > 0:
                length -= 1
            documents.append((start_index + 1, length))
            last_index = index
        documents.sort(key=lambda x: x[1])

        for i, (offset, length) in enumerate(documents):
            if i == len(documents) - 1:
                current_masked_length, current_count = 0, 0
                while (
                    mask_index + current_count < len(masked_lengths)
                    and masked_lengths[mask_index + current_count]
                    + current_masked_length
                    + current_count
                    <= length
                ):
                    current_masked_length += masked_lengths[mask_index + current_count]
                    current_count += 1
                if current_count > 0:
                    spans = sample_spans(
                        masked_lengths[mask_index: mask_index + current_count],
                        length,
                        rng,
                        offset=offset,
                    )
                    mask_spans += spans
                if mask_index + current_count < len(masked_lengths) - 1:
                    print(
                        length,
                        masked_lengths[mask_index:],
                        masked_lengths[:mask_index],
                        indices,
                    )
            else:
                current_masked_total = int(length * self.bert_ratio)
                current_masked_length, current_count = 0, 0
                while (
                    mask_index + current_count < len(masked_lengths)
                    and masked_lengths[mask_index + current_count]
                    + current_masked_length
                    <= current_masked_total
                ):
                    current_masked_length += masked_lengths[mask_index + current_count]
                    current_count += 1
                if current_count > 0:
                    spans = sample_spans(
                        masked_lengths[mask_index: mask_index + current_count],
                        length,
                        rng,
                        offset=offset,
                    )
                    mask_spans += spans
                    mask_index += current_count
        return mask_spans

    def make_masked_data(self,
                         tokens,
                         loss_masks,
                         block_spans):
        """Build MLM data.
        """
        position_ids = np.arange(len(tokens), dtype=np.longlong)
        targets = copy.deepcopy(tokens)
        mask_id = self.tokenizer.mask_token_id
        mlm_masks = np.zeros(len(tokens), dtype=np.longlong)
        for start, end in block_spans:
            for idx in range(start, end):
                tokens[idx] = mask_id
            mlm_masks[start:end] = 1
        loss_masks = loss_masks * mlm_masks
        return tokens, targets, loss_masks, position_ids

    def make_block_data(self,
                        tokens,
                        loss_masks,
                        attention_mask,
                        block_spans,
                        rng: random.Random,
                        task=TASK_BERT):
        """Build block data used in GLM
        """
        text_length = len(tokens)
        position_ids = np.ones(len(tokens), dtype=np.longlong)
        for start, end in block_spans:
            position_ids[start + 1: end] = 0
        position_ids = np.cumsum(position_ids) - 1
        if self.random_position and position_ids[-1] < self.max_seq_length - 1:
            position_bias = self.max_seq_length - position_ids[-1]
            position_bias = rng.randrange(0, position_bias)
            position_ids = position_ids + position_bias
        if self.encoder_decoder or not self.shuffle_blocks:
            block_spans.sort(key=lambda x: x[0])
        else:
            rng.shuffle(block_spans)
        if self.sentinel_token:
            block_spans = [
                (start, end, idx) for idx, (start, end) in enumerate(block_spans)
            ]
        else:
            block_spans = [(start, end, 0) for start, end in block_spans]
        target_tokens, target_position_ids, target_block_position_ids, targets = (
            [],
            [],
            [],
            [],
        )
        for start, end, idx in block_spans:
            target_tokens.append([self.tokenizer.sop_token_id])
            span_tokens = copy.deepcopy(tokens[start:end])
            if self.block_mask_prob > 0.0 and task == TASK_BERT:
                for sub_idx in range(len(span_tokens)):
                    if random.random() < self.block_mask_prob:
                        span_tokens[sub_idx] = self.tokenizer.dblock_token_id
            target_tokens.append(span_tokens)
            targets.append(tokens[start:end])
            targets.append([self.tokenizer.eop_token_id])
            if not self.sentinel_token:
                target_position_id = position_ids[start:end]
                target_position_ids.append(target_position_id)
                target_position_ids.append([target_position_id[0]])
            else:
                target_position_ids.append([self.max_seq_length] * (end - start + 1))
            if self.block_position_encoding:
                target_block_position_ids.append(
                    np.arange(1, end - start + 2, dtype=np.longlong)
                )
            else:
                target_block_position_ids.append([1] * (end - start + 1))
        block_spans.sort(key=lambda x: x[0])
        source_tokens, source_position_ids, local_spans = [], [], []
        last, current_length = 0, 0
        for start, end, idx in block_spans:
            if task == TASK_GPT:
                mask_id = self.generation_mask
            elif task == TASK_SENTENCE:
                mask_id = self.gap_sentence_mask
            else:
                mask_id = self.tokenizer.mask_token_id
            local_spans.append((current_length, current_length + start - last))
            source_tokens.append(tokens[last:start])
            source_tokens.append([mask_id])
            source_position_ids.append(position_ids[last:start])
            source_position_ids.append([position_ids[start]])
            current_length += start - last + 1
            last = end
        if last < len(tokens):
            local_spans.append((current_length, current_length + len(tokens) - last))
            source_tokens.append(tokens[last:])
            source_position_ids.append(position_ids[last:])
        source_length = sum(map(len, source_tokens))
        if attention_mask is not None:
            assert source_length == attention_mask
        if target_tokens and self.eod_token in np.concatenate(target_tokens).tolist():
            print("Found EOS in target", self.tokenizer.decode(tokens))
            raise RuntimeError
        if self.encoder_decoder:
            target_tokens = target_tokens + [self.tokenizer.eop_token_id]
            loss_masks = np.ones(len(target_tokens), dtype=np.longlong)
            return source_tokens, target_tokens, loss_masks
        else:
            tokens = np.concatenate(source_tokens + target_tokens)
            if task == TASK_BERT and self.context_mask_ratio > 0:
                mask_candidates = set()
                for start, end in local_spans:
                    if start != 0:
                        local_end = min(end, start + self.context_mask_range)
                        mask_candidates.update(range(start, local_end))
                    if end != 0:
                        local_start = max(start, end - self.context_mask_range)
                        mask_candidates.update(range(local_start, end))
                mask_pos = rng.sample(
                    mask_candidates, int(self.context_mask_ratio * text_length)
                )
                for pos in mask_pos:
                    tokens[pos] = self.tokenizer.dblock_token_id
            targets = np.concatenate(source_tokens + targets)
            loss_masks = np.ones(len(tokens), dtype=np.longlong)
            loss_masks[:source_length] = 0
            position_ids = np.concatenate(source_position_ids + target_position_ids)
            block_position_ids = np.concatenate(
                [np.zeros(source_length, dtype=np.longlong)] + target_block_position_ids
            )
            position_ids = np.stack([position_ids, block_position_ids], axis=0)
            if attention_mask is not None:
                return tokens, targets, loss_masks, position_ids
            else:
                return tokens, targets, loss_masks, position_ids, source_length

    def featurize(self, sample: Dict, **kwargs) -> Dict:
        return super().featurize(sample, **kwargs)


class GLMBertFeaturizer(GLMBlockInfillingFeaturizer):
    """bert featurizer in GLM
    """

    def __init__(self,
                 name,
                 tokenizer,
                 max_seq_length,
                 bert_ratio=0.15,
                 average_block_length=3,
                 max_block_length=40,
                 single_span: bool = False,
                 encoder_decoder=False,
                 masked_lm=False,
                 rng: random.Random = None,
                 block_mask_prob=0.0,
                 context_mask_ratio=0.0,
                 context_mask_range=3,
                 block_position_encoding=True,
                 shuffle_blocks=True,
                 sentinel_token=False,
                 task_mask=False,
                 random_position=False
                 ) -> None:
        super().__init__(name,
                         tokenizer,
                         max_seq_length,
                         bert_ratio=bert_ratio,
                         masked_lm=masked_lm,
                         block_mask_prob=block_mask_prob,
                         context_mask_ratio=context_mask_ratio,
                         context_mask_range=context_mask_range,
                         block_position_encoding=block_position_encoding,
                         encoder_decoder=encoder_decoder,
                         shuffle_blocks=shuffle_blocks,
                         sentinel_token=sentinel_token,
                         task_mask=task_mask,
                         random_position=random_position)
        self.bert_ratio = bert_ratio
        self.average_block_length = average_block_length
        self.max_block_length = max_block_length

        self.block_length_distribution = [
            poisson.pmf(i, average_block_length) for i in range(1, max_block_length)
        ]
        self.single_span = single_span
        self.rng = rng if rng else random.Random()

        self.encoder_decoder = encoder_decoder
        self.masked_lm = masked_lm

    def featurize(self, sample: Dict, **kwargs) -> Dict:
        self.single_span = kwargs.get("single_span", self.single_span)
        self.rng = kwargs.get("rng", self.rng)

        if self.single_span:
            masked_lengths = [
                self.rng.choices(
                    range(1, len(self.block_length_distribution) + 1),
                    weights=self.block_length_distribution,
                )[0]
            ]
            masked_count = masked_lengths[0]
        else:
            masked_lengths, masked_count = [], 0
            while masked_count < int(self.bert_ratio * len(sample["text"])):
                block_length = self.rng.choices(
                    range(1, len(self.block_length_distribution) + 1),
                    weights=self.block_length_distribution,
                )[0]
                masked_lengths.append(block_length)
                masked_count += block_length
        if self.masked_lm:
            sep = len(sample["text"])
        else:
            sep = len(sample["text"]) - masked_count + len(masked_lengths)

        data = self.generate_blank_data(
            sample, masked_lengths, sep, self.rng, task="bert"
        )
        feat = {}
        if data is not None:
            if self.encoder_decoder:
                source_tokens, target_tokens, loss_masks = data
                feat[FEAT_TOKENS] = source_tokens
                feat[FEAT_TARGETS] = target_tokens
                feat[FEAT_LOSS_MASK] = loss_masks
            else:
                tokens, targets, loss_masks, position_ids = data
                feat[FEAT_TOKENS] = tokens
                feat[FEAT_TARGETS] = targets
                feat[FEAT_LOSS_MASK] = loss_masks
                feat[FEAT_POSITION_ID] = position_ids
            feat[FEAT_ATTENTION_MASK] = sep
        return feat


class GLMSentenceInfillingFeaturizer(GLMBlockInfillingFeaturizer):
    """Build sentence infilling featurizer.
    """

    def __init__(self,
                 name,
                 tokenizer,
                 max_seq_length,
                 gap_sentence_ratio=0.15,
                 encoder_decoder=False,
                 shuffle_blocks=True,
                 sentinel_token=False,
                 task_mask=False,
                 random_position=False,
                 masked_lm=False,
                 eod_token=None,
                 rng: random.Random = None,
                 bert_ratio=0.15,
                 block_mask_prob=0.0,
                 context_mask_ratio=0.0,
                 context_mask_range=3,
                 block_position_encoding=True) -> None:
        super().__init__(name=name,
                         tokenizer=tokenizer,
                         max_seq_length=max_seq_length,
                         encoder_decoder=encoder_decoder,
                         shuffle_blocks=shuffle_blocks,
                         sentinel_token=sentinel_token,
                         task_mask=task_mask,
                         random_position=random_position,
                         masked_lm=masked_lm,
                         eod_token=eod_token,
                         bert_ratio=bert_ratio,
                         block_mask_prob=block_mask_prob,
                         context_mask_ratio=context_mask_ratio,
                         context_mask_range=context_mask_range,
                         block_position_encoding=block_position_encoding)
        self.gap_sentence_ratio = gap_sentence_ratio
        self.rng = rng if rng else random.Random()

    def featurize(self, sample: Dict, **kwargs) -> Dict:
        self.rng: random.Random = kwargs.get("rng", self.rng)
        tokens, loss_masks = sample["text"], sample["loss_mask"]
        sentence_spans = []

        last_index = (
            1 if tokens[0] == self.tokenizer.cls_token_id else 0
        )
        for i in range(len(tokens)):
            if contains_sentence_end(self.tokenizer.convert_ids_to_tokens(int(tokens[i]))):
                if last_index < i + 1:
                    sentence_spans.append((last_index, i + 1))
                last_index = i + 1
            elif tokens[i] == self.tokenizer.eos_token_id:
                last_index = i + 1
        if last_index < len(tokens):
            sentence_spans.append((last_index, len(tokens)))

        self.rng.shuffle(sentence_spans)
        block_spans, block_length = [], 0
        for start, end in sentence_spans:
            block_spans.append((start, end))
            block_length += end - start
            if block_length >= int(self.gap_sentence_ratio * len(tokens)):
                break
        data = self.make_block_data(
            tokens, loss_masks, None, block_spans, self.rng, task=TASK_SENTENCE
        )
        tokens, targets, loss_masks, position_ids, sep = data
        return {
            FEAT_TOKENS: tokens,
            FEAT_TARGETS: targets,
            FEAT_LOSS_MASK: loss_masks,
            FEAT_POSITION_ID: position_ids,
            FEAT_ATTENTION_MASK: sep
        }


class GLMGPTFeaturizer(GLMBlockInfillingFeaturizer):
    """The GPT featurizer in GLM
    """

    def __init__(self,
                 name,
                 tokenizer,
                 max_seq_length,
                 max_generation_length=0,
                 gpt_infill_prob=0.5,
                 gpt_min_ratio=0.5,
                 use_prefix_mode=True,
                 encoder_decoder=False,
                 shuffle_blocks=True,
                 sentinel_token=False,
                 task_mask=False,
                 random_position=False,
                 masked_lm=False,
                 eod_token=None,
                 rng: random.Random = None,
                 bert_ratio=0.15,
                 block_mask_prob=0.0,
                 context_mask_ratio=0.0,
                 context_mask_range=3,
                 block_position_encoding=True
                 ) -> None:
        super().__init__(name=name,
                         tokenizer=tokenizer,
                         max_seq_length=max_seq_length,
                         encoder_decoder=encoder_decoder,
                         shuffle_blocks=shuffle_blocks,
                         sentinel_token=sentinel_token,
                         task_mask=task_mask,
                         random_position=random_position,
                         masked_lm=masked_lm,
                         eod_token=eod_token,
                         bert_ratio=bert_ratio,
                         block_mask_prob=block_mask_prob,
                         context_mask_ratio=context_mask_ratio,
                         context_mask_range=context_mask_range,
                         block_position_encoding=block_position_encoding)
        self.max_generation_length = max_generation_length if max_generation_length > 0 else max_seq_length
        self.use_prefix_mode = use_prefix_mode
        self.gpt_infill_prob = gpt_infill_prob
        self.gpt_min_ratio = gpt_min_ratio
        self.rng = rng if rng else random.Random()

        self.prefix_lm_featurizer = GLMPrefixLMFeaturizer(name=name,
                                                          tokenizer=tokenizer,
                                                          max_len=max_seq_length + 1,  # gmask
                                                          add_cls=False,
                                                          add_eos=False,
                                                          padding=False)

    def featurize(self, sample: Dict, **kwargs) -> Dict:
        self.max_generation_length = kwargs.get("max_generation_length", self.max_generation_length)
        self.rng = kwargs.get("rng", self.rng)
        feat = {}
        if not self.use_prefix_mode:
            generation_length = min(
                self.max_generation_length, len(sample["text"]) - 2
            )
            feat[FEAT_ATTENTION_MASK] = len(sample["text"]) - generation_length + 1
            multiple_doc = index_in_list(
                sample["text"], self.tokenizer.eos_token_id
            ) not in [-1, len(sample["text"]) - 1]
        else:
            first_eod_pos = index_in_list(
                sample["text"], self.tokenizer.eos_token_id
            )
            multiple_doc = first_eod_pos not in [-1, len(sample["text"]) - 1]

            if first_eod_pos >= 0:
                start = int(self.gpt_min_ratio * first_eod_pos)
                end = first_eod_pos - 2
                if start > end:
                    generation_length = (
                        start + len(sample["text"]) - first_eod_pos
                    )
                else:
                    generation_length = (
                        self.rng.randint(start, end)
                        + len(sample["text"])
                        - first_eod_pos
                    )
            else:
                generation_length = self.rng.randint(
                    int(self.gpt_min_ratio * len(sample["text"])),
                    len(sample["text"]) - 2,
                )
            feat[FEAT_ATTENTION_MASK] = len(sample["text"]) - generation_length + 1

        if self.use_prefix_mode or multiple_doc or self.rng.random() < self.gpt_infill_prob:
            division = len(sample["text"]) - generation_length
            tokens, loss_masks = sample["text"], sample["loss_mask"]
            source_tokens, target_tokens = tokens[:division], tokens[division:]
            target_masks = loss_masks[division:]

            res = self.prefix_lm_featurizer.featurize(
                {"prefix": source_tokens, "content": target_tokens, "loss_mask": target_masks})

            feat[FEAT_TOKENS] = res["input_ids"]
            feat[FEAT_TARGETS] = res["labels"]
            feat[FEAT_LOSS_MASK] = res["loss_mask"]
            feat[FEAT_POSITION_ID] = res["position_ids"]
        else:
            (
                tokens,
                targets,
                loss_masks,
                position_ids,
            ) = self.generate_blank_data(
                sample,
                [generation_length],
                feat[FEAT_ATTENTION_MASK],
                self.rng,
                task=TASK_GPT,
            )
            feat[FEAT_TOKENS] = tokens
            feat[FEAT_TARGETS] = targets
            feat[FEAT_LOSS_MASK] = loss_masks
            feat[FEAT_POSITION_ID] = position_ids
        return feat
