# coding: utf-8
# Copyright (c) Antfin, Inc. All rights reserved.

import argparse
from pytorch_lightning.trainer.trainer import Trainer
from alps.util import logger
from .bloomz import BloomzModule
from transformers import AutoTokenizer, AutoModelForCausalLM
from solutions.antllm.examples.petuning.io_utils.petuning_odps_dataset import (
    create_dataset,
)
from typing import List
from alps.pytorch.lightning import seed_everything
from solutions.antllm.antllm.models.petuning.tuner.roem import PEROEMModel
from alps.pytorch.lightning.plugins.environments import AntClusterEnvironment
from alps.pytorch.lightning.plugins.io import AntCheckpointIO
from alps.pytorch.lightning.callbacks import AntLogCallback
from alps.pytorch.lightning.core import AntDataModule

"""
这个例子对bloomz模型进行了roem微调
作为演示，只跑了训练样本5%的数据
"""


parser = argparse.ArgumentParser(description="Lightning Huggingface Training")

parser.add_argument(
    "--model_name_or_path",
    default="/mntnlp/zhuangyou/bloomz/models--bigscience--bloomz-7b1-mt",
    type=str,
    help="和transformers的同名变量一致",
)
parser.add_argument(
    "--delta_tuning_save_dir",
    default="/mntnlp/zhuangyou/petuning_example/bloomz_roem",
    type=str,
    help="增量参数保存路径",
)
parser.add_argument(
    "--input_table",
    default="apmktalgo_dev.glm_sharegpt_zh_train_data_yumu_v1",
    type=str,
    help="训练表",
)
parser.add_argument(
    "--read_fields", default=["question", "answer"], type=List[str], help="从训练表读取哪些字段"
)
parser.add_argument(
    "--max_seq_len", default=500, type=int, help="整个seq的最大token数量，超过的话会被tokenzier截断"
)
parser.add_argument("--batch_size", default=4, type=int, help="batch size")
parser.add_argument("--max_epochs", default=1, type=int, help="epoch数")
parser.add_argument("--num_devices", default=1, type=int, help="gpu数量")
parser.add_argument("--num_nodes", default=1, type=int, help="节点数量")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
tokenizer.pad_token = "<pad>"


# dataset的全局setup
def setup_fn(cls):
    cls.max_seq_length = args.max_seq_len
    cls.tokenizer = tokenizer


# dataset中对于每条数据的前置处理,
# bloomz是causal架构的，所以这样写
def transform(cls, data):
    # bloomz 前面加<s>代表序列开始
    sentence = "<s>" + data[args.read_fields[0]] + data[args.read_fields[1]]
    # 通过padding属性将训练数据统一填充到self.max_seq_length大小

    feature = cls.tokenizer(
        sentence,
        max_length=cls.max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    return feature


def main(args):
    seed_everything(42)

    # 实例化原版模型
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    logger.info(f"base bloomz load from {args.model_name_or_path}")

    # 实例化PEModel

    adalora_model = PEROEMModel(
        model=model,
        model_name="bloomz",
    )
    model = adalora_model.get_model()
    logger.info("PE ROEM initialized")

    module = BloomzModule(tokenizer=tokenizer, model=model)

    # 添加多机多卡环境适配及模型保存支持oss与pangu

    plugins = [AntClusterEnvironment(), AntCheckpointIO()]

    # 使用AntLogCallback取代官方进度条适配蚂蚁日志系统

    callbacks = [AntLogCallback()]

    trainer = Trainer(
        max_epochs=args.max_epochs,
        devices=args.num_devices,  # gpu数量
        precision=16,
        strategy="deepspeed_stage_2",
        accelerator="gpu",
        num_nodes=1,  # 节点数
        limit_val_batches=0,
        # 添加plugins
        plugins=plugins,
        # 添加log和profile
        # 添加callback
        callbacks=callbacks,
        # 关闭官方进度条
        enable_progress_bar=True,
        # 仅作演示，所以只训练5%的数据
        limit_train_batches=0.05,
    )
    local_rank_env = int(trainer.global_rank)
    print("the local_rank is {}".format(local_rank_env))
    logger.info("create dataset ...")
    train_dataset = create_dataset(
        table=args.input_table,
        fields=args.read_fields,  # 读取表的字段
        setup_fn=setup_fn,
        transform=transform,
        is_train=True,
        num_devices=args.num_devices,  # gpu数量
        batch_size=args.batch_size,
    )

    logger.info("create data module ...")

    data_module = AntDataModule(
        train_dataset=train_dataset,
        train_batch_size=args.batch_size,
        num_workers=1,
    )
    logger.info("start train #########################")
    trainer.fit(module, data_module)
    trainer.strategy.barrier()  # 等待所有线程结束后退出
    logger.info("end train #########################")

    # 最后的最后，保存成transformers PretrainedModel的形状
    model.save_pretrained(args.delta_tuning_save_dir)
    logger.info(f"model saved to {args.delta_tuning_save_dir}")
    tokenizer.save_pretrained(args.delta_tuning_save_dir)
    logger.info(f"tokenizer saved to {args.delta_tuning_save_dir}")


if __name__ == "__main__":
    main(args)
