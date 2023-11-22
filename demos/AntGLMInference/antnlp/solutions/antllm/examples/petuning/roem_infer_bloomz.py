# coding: utf-8
# Copyright (c) Antfin, Inc. All rights reserved.

import argparse
from pytorch_lightning.trainer.trainer import Trainer
from alps.util import logger
from .bloomz import BloomzModule
from transformers import AutoTokenizer, AutoModelForCausalLM
from .io_utils import CustomWriter
from solutions.antllm.examples.petuning.io_utils.petuning_odps_dataset import (
    create_dataset,
)
from typing import List
from alps.pytorch.lightning.plugins.environments import AntClusterEnvironment
from alps.pytorch.lightning.plugins.io import AntCheckpointIO
from alps.pytorch.lightning.callbacks import AntLogCallback
from alps.pytorch.lightning.core import AntDataModule

"""
这个例子中，我们恢复roem训练出来的模型来进行预测。
由于roem是reparameterization的方法，没有额外参数引入，所以只要一步加载模型就行
"""

parser = argparse.ArgumentParser(description="Lightning Huggingface Training")


parser.add_argument(
    "--delta_tuning_save_dir",
    # 如果没有跑roem_train_bloomz.py，可以先用这个
    default="/mntnlp/zhuangyou/petuning_example/bloomz_roem",
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
    "--read_fields", default=["input"], type=List[str], help="从输入表读取哪些字段"
)
parser.add_argument(
    "--max_seq_len", default=200, type=int, help="整个seq的最大token数量，超过的话会被tokenzier截断"
)
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--num_devices", default=1, type=int, help="gpu数量")
args = parser.parse_args()

# 从训练的导出路径读取模型
auto_tokenizer = AutoTokenizer.from_pretrained(args.delta_tuning_save_dir)
auto_tokenizer.pad_token = "<pad>"
auto_tokenizer.padding_side = "left"


# dataset的全局setup
def setup_fn(cls):
    cls.max_seq_length = args.max_seq_len
    cls.tokenizer = auto_tokenizer


# dataset中对于每条数据的前置处理
def transform(cls, data):
    sentence = "<s>" + data[args.read_fields[0]] + "\n回答："
    # 通过padding属性将训练数据统一填充到self.max_seq_length大小
    feature = cls.tokenizer(
        sentence,
        max_length=cls.max_seq_length,
        padding=False,
        truncation=True,
        return_tensors="pt",
    )
    return feature


def main(args):
    # 从训练的导出路径读取模型
    model = AutoModelForCausalLM.from_pretrained(args.delta_tuning_save_dir)
    logger.info(f"bloomz load from {args.delta_tuning_save_dir}")

    # 实例化为bloomz打造的lightning module
    module = BloomzModule(tokenizer=auto_tokenizer, model=model)

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
