#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"


import os
import random
import time
from argparse import Namespace
from unittest import TestCase, main

import numpy as np
from tqdm import tqdm

from solutions.antllm.antllm.data import corpora
from solutions.antllm.antllm.data.dataset.glm_instruction_dataset import \
    GLMInstructionDataset
from solutions.antllm.antllm.data.datasets import BlockDataset
from solutions.antllm.antllm.data.lazy_loader import (LazyLoader, LazyWriter,
                                                      exists_lazy,
                                                      exists_scatter,
                                                      get_scatter_path)
from solutions.antllm.antllm.datav2.data_utils import (truncate_left,
                                                       truncate_right)
from solutions.antllm.antllm.datav2.glm import GLMBlockCollator
from solutions.antllm.antllm.datav2.glm.dataset import (GLMBlockDataset,
                                                        GLMSeq2SeqDataset)
from solutions.antllm.antllm.datav2.glm.featurizer import (
    GLMBertFeaturizer, GLMGPTFeaturizer, GLMSentenceInfillingFeaturizer,
    GLMSeq2SeqFeaturizer)
from solutions.antllm.antllm.models.glm.tokenization_glm import GLMTokenizer
from solutions.antllm.antllm.utils.blocklm_utils import ConstructBlockStrategy


def build_legacy_blockdatasset(
        path,
        tokenizer,
        pre_tokenize,
        max_seq_len,
        data_parallel_rank=0,
        global_rank=0,
        loader_scatter=None,
        no_lazy_loader=False,
        half_lazy_loader=False):

    class LegacyReader(corpora.PromptReader):
        PATH = path
        is_json = False

        def process_line(self, data, tokenizer, tokenize):
            text = data  # ['text']
            prompt, text = self.process_sample(
                "", tokenizer, tokenize), self.process_sample(text, tokenizer, tokenize)
            return [prompt], [text]

    if not (
        loader_scatter is not None
        and exists_scatter(path, data_type="prompt", scatter_num=loader_scatter)
        and exists_scatter(path, data_type="text", scatter_num=loader_scatter)
    ):
        if not (
            exists_lazy(path, data_type="prompt")
            and exists_lazy(path, data_type="text")
        ):
            # create cached version of dataset for lazy loading if it doesn't exist
            if global_rank == 0:
                prompt_writer = LazyWriter(
                    path, data_type="prompt", is_array=pre_tokenize
                )
                text_writer = LazyWriter(
                    path, data_type="text", is_array=pre_tokenize
                )
                writers = {"prompt": prompt_writer, "text": text_writer}
                reader = LegacyReader(
                    writers=writers, tokenizer=tokenizer, tokenize=pre_tokenize
                )
                reader.process()
                prompt_writer.close()
                text_writer.close()
            else:
                while not os.path.exists(
                    LazyWriter.get_len_path(path, data_type="prompt")
                ):
                    time.sleep(1)
    map_fn = (lambda x: x.tolist()) if pre_tokenize else None
    if loader_scatter is not None:
        if not (
            exists_scatter(path, data_type="prompt", scatter_num=loader_scatter)
            and exists_scatter(path, data_type="text", scatter_num=loader_scatter)
        ):
            if global_rank == 0:
                texts = LazyLoader(
                    path,
                    data_type="text",
                    map_fn=map_fn,
                    mem_map=True,
                    is_array=pre_tokenize,
                )
                prompts = LazyLoader(
                    path,
                    data_type="prompt",
                    map_fn=map_fn,
                    mem_map=True,
                    is_array=pre_tokenize,
                )
                t_len_path = os.path.join(path, f"len_{len(texts)}.npy")
                done_len_path = t_len_path + ".done"
                if not os.path.exists(done_len_path):
                    if global_rank == 0:
                        indices = np.arange(len(texts))
                        random.shuffle(indices)
                        np.save(t_len_path, indices)
                        wt = open(done_len_path, "w")
                        wt.close()
                    else:
                        while not os.path.exists(done_len_path):
                            time.sleep(5)

                indices = np.load(t_len_path)

                print(f"load indices: {indices.shape}.")
                segment_length = (len(indices) - 1) // loader_scatter + 1
                for i in range(loader_scatter):
                    print(f"Start process scatter {i}")
                    scatter_path = get_scatter_path(path, scatter_rank=i)
                    prompt_writer = LazyWriter(
                        scatter_path, data_type="prompt", is_array=pre_tokenize
                    )
                    text_writer = LazyWriter(
                        scatter_path, data_type="text", is_array=pre_tokenize
                    )
                    for idx in tqdm(
                        indices[i * segment_length: (i + 1) * segment_length]
                    ):
                        prompt_writer.write(prompts[idx])
                        text_writer.write(texts[idx])
                    prompt_writer.close()
                    text_writer.close()
            else:
                while not (
                    exists_scatter(
                        path, data_type="prompt", scatter_num=loader_scatter
                    )
                    and exists_scatter(
                        path, data_type="text", scatter_num=loader_scatter
                    )
                ):
                    time.sleep(1)
        scatter_path = get_scatter_path(
            path, scatter_rank=data_parallel_rank % loader_scatter
        )
        print(f"Rank {global_rank} is using scatter from {scatter_path}")
        prompts = LazyLoader(
            scatter_path,
            data_type="prompt",
            map_fn=map_fn,
            mem_map=True,
            is_array=pre_tokenize,
            load_memory=no_lazy_loader,
            half_load=half_lazy_loader,
        )
        texts = LazyLoader(
            scatter_path,
            data_type="text",
            map_fn=map_fn,
            mem_map=True,
            is_array=pre_tokenize,
            load_memory=no_lazy_loader,
            half_load=half_lazy_loader,
        )
    else:
        prompts = LazyLoader(
            path,
            data_type="prompt",
            map_fn=map_fn,
            mem_map=True,
            is_array=pre_tokenize,
            load_memory=no_lazy_loader,
            half_load=half_lazy_loader,
        )
        texts = LazyLoader(
            path,
            data_type="text",
            map_fn=map_fn,
            mem_map=True,
            is_array=pre_tokenize,
            load_memory=no_lazy_loader,
            half_load=half_lazy_loader,
        )
    text = corpora.PromptDataset(
        prompt_loader=prompts,
        text_loader=texts,
        tokenizer=tokenizer,
        to_tokenize=not pre_tokenize,
        name="test"
    )

    dataset = BlockDataset(
        text,
        tokenizer,
        max_seq_len=max_seq_len,
        sample_across_doc=False,
        filter_english=False,
        non_sentence_start=0.0,
    )
    return dataset


class TestGLMDataset(TestCase):
    def setUp(self):
        self.base_dir = os.path.dirname(__file__)
        self.tokenizer = GLMTokenizer.from_pretrained(
            os.path.join(self.base_dir, '../../../..', 'zhen_sp5/')
        )

    def test_truncate(self):
        s = "我爱北京天安门。天安门上太阳升"
        ori_tokens = self.tokenizer.encode(s)
        trunc_left_tokens = truncate_left(ori_tokens, 6, True, False, 3, self.tokenizer)
        trunc_right_tokens = truncate_right(ori_tokens, 6, True, False, 3, self.tokenizer)
        self.assertEquals(len(trunc_left_tokens), 5)
        self.assertEquals(len(trunc_right_tokens), 5)

    def test_glm_s2s_featurizer(self):
        featurizer = GLMSeq2SeqFeaturizer(
            name="s2s",
            need_tokenize=True,
            tokenizer=self.tokenizer,
            max_length=32,
            max_input_length=16,
            max_output_length=16
        )
        sample = {
            "input": "以下是一道小学数学题：由三只小狗和两只小猫组成的动物园里，共有多少只动物？ ",
            "output": """1.我们可以先将小狗和小猫的数量加起来，得到总数。\n2.将小狗和小猫的数量分别记为a和b，因此总数可以表示为a+b。\
            \n3.将a和b分别代入题目中，得到a=3，b=2。\n4.将a和b代入公式a+b，得到3+2=5。\n5.因此，由三只小狗和两只小猫组成的动物园里共有5只动物。"""
        }
        res = featurizer.featurize(sample)

        old_res = GLMInstructionDataset.build_feature_from_sample(sample, self.tokenizer,
                                                                  max_length=32,
                                                                  max_input_length=16,
                                                                  max_output_length=16,
                                                                  mask_id=self.tokenizer.convert_tokens_to_ids(
                                                                      '[gMASK]'),
                                                                  for_generation=False,
                                                                  left_truncate=True)

        self.assertTrue(len(res["input_ids"]) > 0)
        self.assertTrue(len(old_res["labels"]) > 0)

    def test_glm_block_feturizer(self):
        max_seq_len = 30
        content = """1.我们可以先将小狗和小猫的数量加起来，得到总数。\n2.将小狗和小猫的数量分别记为a和b，因此总数可以表示为a+b。\
        \n3.将a和b分别代入题目中，得到a=3，b=2。\n4.将a和b代入公式a+b，得到3+2=5。\n5.因此，由三只小狗和两只小猫组成的动物园里共有5只动物。"""
        tokens = self.tokenizer.encode(content)
        tokens = [self.tokenizer.cls_token_id] + \
            tokens + [self.tokenizer.eos_token_id]
        loss_mask = [0] + [1] * (len(tokens) - 1)

        sample = {
            "text": tokens[:max_seq_len],
            "loss_mask": loss_mask[:max_seq_len]
        }

        bert_featurizer = GLMBertFeaturizer("bert", self.tokenizer, max_seq_len)
        bert_feat = bert_featurizer.featurize(sample)
        self.assertTrue("tokens" in bert_feat)

        sent_featurizer = GLMSentenceInfillingFeaturizer("sent", self.tokenizer, max_seq_len)
        sent_feat = sent_featurizer.featurize(sample)
        self.assertTrue("tokens" in sent_feat)

        gpt_featurizer = GLMGPTFeaturizer("gpt", self.tokenizer, max_seq_len, 20)
        gpt_feat = gpt_featurizer.featurize(sample)
        self.assertTrue("tokens" in gpt_feat)

    def test_glms2sdataset(self):
        ds = GLMSeq2SeqDataset(
            name="s2s",
            data_path=os.path.join(self.base_dir, "toy_sft_data.jsonl"),
            tokenizer=self.tokenizer,
            need_tokenize=True,
            pre_tokenize=True,
            load_memory=True
        )
        self.assertTrue(len(ds) > 0)

    def test_glmblockdataset(self):
        ds = GLMBlockDataset(
            name="block",
            data_path=os.path.join(self.base_dir, "toy_pretrain_data.jsonl"),
            tokenizer=self.tokenizer,
            max_len=20,
            left_truncate_prob=0.2,
            lazy_loader_opt="naive",
            load_memory=True,
            scatter_num=8,
            rank=0,
            data_parallel_rank=0
        )
        self.assertTrue(len(ds) > 0)

    def test_glmblockdataset_compatibility(self):
        ds_new = GLMBlockDataset(
            name="toy",
            data_path=os.path.join(self.base_dir, "toy_pretrain"),
            tokenizer=self.tokenizer,
            max_len=2048,
            left_truncate_prob=0,
            lazy_loader_opt="v1",
            load_old_format=True,
            scatter_num=4,
            rank=0,
            data_parallel_rank=0
        )

        ds_old = build_legacy_blockdatasset(
            os.path.join(self.base_dir, "toy_pretrain"),
            self.tokenizer, True, 2048, loader_scatter=4)

        self.assertTrue(len(ds_new) == 3)
        self.assertTrue(len(ds_old) == 3000)

    def test_glm_batch_collator(self):
        max_seq_len = 30
        content = """1.我们可以先将小狗和小猫的数量加起来，得到总数。\n2.将小狗和小猫的数量分别记为a和b，因此总数可以表示为a+b。\
        \n3.将a和b分别代入题目中，得到a=3，b=2。\n4.将a和b代入公式a+b，得到3+2=5。\n5.因此，由三只小狗和两只小猫组成的动物园里共有5只动物。"""
        tokens = self.tokenizer.encode(content)
        tokens = [self.tokenizer.cls_token_id] + \
            tokens + [self.tokenizer.eos_token_id]
        loss_mask = [0] + [1] * (len(tokens) - 1)

        sample = {
            "text": tokens[:max_seq_len],
            "loss_mask": loss_mask[:max_seq_len]
        }

        args = Namespace()
        args.eod_token = self.tokenizer.eos_token_id

        bert_batch_collator = GLMBlockCollator(
            tokenizer=self.tokenizer,
            max_seq_length=max_seq_len,
            bert_prob=1,
            bert_ratio=0.15,
            gap_sentence_prob=0,
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
            task_mask=True,
            random_position=False,
            masked_lm=False,
            use_prefix_mode=True,
            eod_token=self.tokenizer.eos_token_id
        )
        new_bert_feat = bert_batch_collator([sample])

        old_bert_block_strategy = ConstructBlockStrategy(
            args=args,
            tokenizer=self.tokenizer,
            max_seq_length=max_seq_len,
            bert_prob=1,
            gap_sentence_prob=0,
            gpt_infill_prob=0.5,
            gpt_min_ratio=0.5,
            bert_ratio=0.15,
            gap_sentence_ratio=0.15,
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
            task_mask=True,
            random_position=False,
            masked_lm=False,
            use_prefix_mode=True,
        )

        old_bert_feat = old_bert_block_strategy.construct_blocks([sample])

        self.assertListEqual(list(new_bert_feat["text"][0]), list(old_bert_feat["text"][0]))

        sent_batch_collator = GLMBlockCollator(
            tokenizer=self.tokenizer,
            max_seq_length=max_seq_len,
            bert_prob=0,
            bert_ratio=0.15,
            gap_sentence_prob=1,
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
            task_mask=True,
            random_position=False,
            masked_lm=False,
            use_prefix_mode=True,
            eod_token=self.tokenizer.eos_token_id
        )

        new_sent_feat = sent_batch_collator([sample])

        old_sent_block_strategy = ConstructBlockStrategy(
            args=args,
            tokenizer=self.tokenizer,
            max_seq_length=max_seq_len,
            bert_prob=0,
            gap_sentence_prob=1,
            gpt_infill_prob=0.5,
            gpt_min_ratio=0.5,
            bert_ratio=0.15,
            gap_sentence_ratio=0.15,
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
            task_mask=True,
            random_position=False,
            masked_lm=False,
            use_prefix_mode=True,
        )

        old_sent_feat = old_sent_block_strategy.construct_blocks([sample])
        self.assertListEqual(list(new_sent_feat["text"][0]), list(old_sent_feat["text"][0]))

        gpt_batch_collator = GLMBlockCollator(
            tokenizer=self.tokenizer,
            max_seq_length=max_seq_len,
            bert_prob=0,
            bert_ratio=0.15,
            gap_sentence_prob=0,
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
            task_mask=True,
            random_position=False,
            masked_lm=False,
            use_prefix_mode=True,
            eod_token=self.tokenizer.eos_token_id
        )

        new_gpt_feat = gpt_batch_collator([sample])

        old_gpt_block_strategy = ConstructBlockStrategy(
            args=args,
            tokenizer=self.tokenizer,
            max_seq_length=max_seq_len,
            bert_prob=0,
            gap_sentence_prob=0,
            gpt_infill_prob=0.5,
            gpt_min_ratio=0.5,
            bert_ratio=0.15,
            gap_sentence_ratio=0.15,
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
            task_mask=True,
            random_position=False,
            masked_lm=False,
            use_prefix_mode=True,
        )

        old_gpt_feat = old_gpt_block_strategy.construct_blocks([sample])
        self.assertTrue(len(list(new_gpt_feat["text"][0])) > 0)
        self.assertTrue(len(list(old_gpt_feat["text"][0])) > 0)


if __name__ == '__main__':
    main()
