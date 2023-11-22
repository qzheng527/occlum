# coding: utf-8
# Copyright (c) Antfin, Inc. All rights reserved.

import argparse
from pytorch_lightning.trainer.trainer import Trainer
from alps.util import logger
from .antglm import AntGLMModule
from solutions.antllm.antllm.models.petuning.tuner.unipelt import PEUniPELTModel
from solutions.antllm.examples.petuning.io_utils.petuning_odps_dataset import (
    create_dataset,
)
from .io_utils import CustomWriter
from typing import List
from alps.pytorch.lightning.plugins.environments import AntClusterEnvironment
from alps.pytorch.lightning.plugins.io import AntCheckpointIO
from alps.pytorch.lightning.callbacks import AntLogCallback
from alps.pytorch.lightning.core import AntDataModule
from solutions.antllm.antllm.models.glm.modeling_glm import GLMForConditionalGeneration
from solutions.antllm.antllm.models.glm.tokenization_glm import GLMTokenizer

"""
这个例子中，将用训练出来的PE增量部分和原模型组合来进行预测。
模型的读取将分为两步：1. 读取原模型 2.在原模型的基础上插上增量部分
"""

parser = argparse.ArgumentParser(description="Lightning Huggingface Training")

parser.add_argument(
    "--model_name_or_path",
    default="/mntnlp/zhuangyou/glm/AntGLM-10B-RLHF-20230602",
    type=str,
    help="和transformers的同名变量一致，基础模型的结构和权重",
)
parser.add_argument(
    "--delta_tuning_save_dir",
    default="/mntnlp/zhuangyou/petuning_example/antglm_unipelt",
    # 如果没有跑unipelt_train_antglm.py，可以先用这个
    type=str,
    help="增量参数保存路径",
)
parser.add_argument(
    "--input_table",
    default="apmktalgo_dev.my_tx_eval_v2",
    type=str,
    help="输入表",
)
parser.add_argument(
    "--output_table",
    default="apmktalgo_dev.my_tx_eval_v2_out",
    type=str,
    help="输出表",
)
parser.add_argument(
    "--read_fields", default=["input"], type=List[str], help="从训练表读取哪些字段"
)
parser.add_argument(
    "--max_seq_len", default=200, type=int, help="整个seq的最大token数量，超过的话会被tokenzier截断"
)
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--num_devices", default=1, type=int, help="gpu数量")
args = parser.parse_args()
# 从训练的导出路径读取tokenizer
tokenizer = GLMTokenizer.from_pretrained(
    args.delta_tuning_save_dir, trust_remote_code=True
)
tokenizer.padding_side = "right"
tokenizer.pad_token = "<pad>"


# dataset的全局setup，必须写成global，不然后面后面分布式预测会出错
def setup_fn(cls):
    cls.max_seq_length = args.max_seq_len
    cls.tokenizer = tokenizer


# dataset中对于每条数据的前置处理，必须写成global，不然后面后面分布式预测会出错
def transform(cls, data):
    sentence = data[args.read_fields[0]] + "\n回答：[gMASK]"
    # 通过padding属性将训练数据统一填充到max_seq_length大小
    feature = cls.tokenizer(
        sentence,
        max_length=cls.max_seq_length,
        padding=False,
        truncation=True,
        return_tensors="pt",
    )
    feature = cls.tokenizer.build_inputs_for_generation(feature, max_gen_length=1024)

    return feature


def main(args):
    # 读取原模型权重
    model = GLMForConditionalGeneration.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )

    # 读取unipelta增量权重
    model = PEUniPELTModel.restore(model, args.delta_tuning_save_dir)

    # 实例化为antglm打造的lightning module
    module = AntGLMModule(tokenizer=tokenizer, model=model)

    plugins = [AntClusterEnvironment(), AntCheckpointIO()]
    # 创建输出结果的writer
    pred_writer = CustomWriter(
        output_table=args.output_table,  # 输出表名
        write_interval="epoch",  # 每个epoch结束写一次
    )

    callbacks = [pred_writer, AntLogCallback()]
    trainer = Trainer(
        accelerator="gpu",
        strategy="deepspeed_stage_2",
        devices=args.num_devices,  # gpu数量
        plugins=plugins,
        callbacks=callbacks,
        limit_predict_batches=0.1,  # 演示用，只预测10%的数据
    )
    local_rank_env = int(trainer.global_rank)
    print("the local_rank is {}".format(local_rank_env))

    predict_dataset = create_dataset(
        table=args.input_table,
        fields=args.read_fields,  # 读取表的字段
        setup_fn=setup_fn,
        transform=transform,
        is_train=False,
        num_devices=args.num_devices,  # gpu数量
        batch_size=args.batch_size,
    )

    predict_dataset.set_shard(shard_id=local_rank_env, shard_num=args.device_nums)

    logger.info(f"predict_dataset size {len(predict_dataset)}")
    data_module = AntDataModule(
        predict_dataset=predict_dataset,
        train_batch_size=args.batch_size,
        num_workers=1,
    )

    logger.info("start predict #########################")
    trainer.predict(model=module, datamodule=data_module, return_predictions=False)
    trainer.strategy.barrier()  # 等待所有线程结束后退出
    logger.info("end predict #########################")


if __name__ == "__main__":
    main(args)
