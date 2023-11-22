# -*- coding: utf-8 -*-
import os
import sys
import json
import subprocess
import torch
import shutil
import logging
import time
import datetime

import antllm
from adabench.impl.run_impl import run_upload, run_download
from solutions.antllm.antllm.api.data_utils import download
from adabench.utils.util import get_user_name
from solutions.antllm.antllm.utils.benchmark import notify_benchmark_server

logger = logging.getLogger(__name__)

os.environ["WANDB_DISABLED"] = "true"


def build_exec_command(config, use_atorch=True):
    if ('peft_type' in config and config['peft_type'] != '') or use_atorch is not True:
        training_args = '{0} --deepspeed {1} '.format(
            os.path.join(antllm.__path__[0], "commands/sft/train_deepspeed.py"),
            os.path.join(antllm.__path__[0], "api/configs/deepspeed.json"))
    else:
        training_args = os.path.join(antllm.__path__[0], "commands/sft/train_atorch.py") + ' '
    for k, v in config.items():
        if v:
            training_args += f"--{k} {v} "
        else:
            training_args += f"--{k} "
    gpu_num = torch.cuda.device_count()
    if not os.environ.get('WORLD_SIZE') or int(os.environ['WORLD_SIZE']) == 1:
        master_port = os.environ['MASTER_PORT'] if os.environ.get('MASTER_PORT') else '12346'
        cmd = "python -m torch.distributed.run " \
            f"--nnode=1 --nproc_per_node={gpu_num} " \
            f"--node_rank=0 --master_addr=127.0.0.1 " \
            f"--master_port={master_port} "
    else:
        cmd = "python -m torch.distributed.run " \
            f"--nnode=$WORLD_SIZE --nproc_per_node={gpu_num} " \
            f"--node_rank=$RANK --master_addr=$MASTER_ADDR " \
            f"--master_port=$MASTER_PORT "
    cmd += training_args
    return cmd


def get_pretrained_model_path(pretrained_model_name_or_path):
    if os.path.exists(pretrained_model_name_or_path):
        return pretrained_model_name_or_path
    return os.path.join('/adabench_mnt/llm/', pretrained_model_name_or_path)


def is_valid_peft_checkpoint_dir(path):
    model_exists = config_exists = False
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name == 'adapter_model.bin':
                model_exists = True
            if name == 'adapter_config.json':
                config_exists = True
            if model_exists and config_exists:
                return True
    return False


def send_message_if_need(run_id, submitor, message, status):
    ret = ""
    if os.environ.get('WORLD_SIZE') is not None and int(os.environ.get('WORLD_SIZE')) > 1:
        if os.environ.get('RANK') is not None and int(os.environ.get('RANK')) == 0:
            ret = notify_benchmark_server(run_id, submitor, message, status)
    else:
        ret = notify_benchmark_server(run_id, submitor, message, status)
    logger.info("request result %s", str(ret))


def main():
    '''
    根据配置文件，执行sft训练
    Args:
        config_path: 配置文件路径，配置文件为json格式，必须包含以下信息：
        {
            "train_args": {...},
            "dataset_id": "train_dataset_id",
            "run_id": "run_id"
        }
    '''
    config_path = sys.argv[1]
    with open(config_path) as fi:
        train_conf = json.load(fi)
    # 通知benchmark server训练开始
    send_message_if_need(train_conf["run_id"], get_user_name(), "train start", "Running")
    # 根据dataset_id下载数据集
    download(train_conf['dataset_id'], train_conf['dataset_id'])

    # 生成模型训练参数
    logger.info(f'star training for {train_conf["run_id"]}')
    train_args = train_conf['train_args']
    model_output = os.path.join('/adabench_mnt/api/model_output', train_conf['run_id'])
    os.makedirs(model_output, exist_ok=True)
    if train_conf['resume_task_id']:
        last_model_output = os.path.join('/adabench_mnt/api/model_output', train_conf['resume_task_id'])
        if not os.path.exists(last_model_output):
            logger.info('resume from finished run, download checkpoint first')
            run_download(train_conf['resume_task_id'])
            tar_model_name = train_conf["resume_task_id"] + "_model.tar.gz"
            if not os.path.exists(tar_model_name):
                raise Exception(f'fail to download model {tar_model_name}')
            untar_cmd = f'tar zxf {tar_model_name} -C {model_output}'
            subprocess.run(untar_cmd, shell=True, check=True)
        else:
            logger.info(f'resume from unfinish run dir: {last_model_output}')
            shutil.copytree(last_model_output, model_output, dirs_exist_ok=True)
        if not is_valid_peft_checkpoint_dir(model_output):
            logger.warning(f'checkpoint is not exists for run_id: {train_conf["resume_task_id"]}, train from strach.')
        else:
            train_conf['train_args']['resume_from_checkpoint'] = 'true'

    train_args['output_dir'] = model_output
    train_args['train_data'] = os.path.join(train_conf['dataset_id'], 'train.jsonl')
    validation_path = os.path.join(train_conf['dataset_id'], 'dev.jsonl')
    if os.path.exists(validation_path) and os.path.getsize(validation_path) > 0:
        train_args['test_data'] = validation_path
        train_args['do_eval'] = ''
    else:
        train_args['evaluation_strategy'] = 'no'
    train_args['pretrained_model_name_or_path'] = get_pretrained_model_path(
        train_args['pretrained_model_name_or_path'])

    cmd = build_exec_command(train_conf['train_args'], use_atorch=train_conf.get('use_atorch'))
    logger.info(f'exec cmd: {cmd}')
    # 执行训练
    logger.info('start training')
    status = "Success"
    message = "train success"
    returncode = 0
    try:
        start_time = time.time()
        ret = subprocess.run(cmd, shell=True)
        total_time = time.time() - start_time
        if ret.returncode != 0:
            returncode = ret.returncode
            logger.error("train failed, return code %s", ret.returncode)
            if ret.returncode == -9:
                raise RuntimeError(
                    "The program ran into a kernel error with exitcode 9."
                    " may be due to insufficient memory or cpu, please increase memory and cpu."
                )
            else:
                raise Exception("train failed, return code {}".format(ret.returncode))
        else:
            message = "train success: time used {}".format(str(datetime.timedelta(seconds=total_time)))
            logger.info(message)

        # 上传模型
        logger.info('upload model')
        if os.environ.get('WORLD_SIZE') is not None and int(os.environ.get('WORLD_SIZE')) > 1:
            if os.environ.get('RANK') is not None and int(os.environ.get('RANK')) == 0:
                success = run_upload(train_conf['run_id'], train_args['output_dir'], gzip=True)
                if not success:
                    raise Exception('failed to upload model')
                shutil.rmtree(model_output)
        else:
            success = run_upload(train_conf['run_id'], train_args['output_dir'], gzip=True)
            if not success:
                raise Exception('failed to upload model')
    except Exception as e:
        message = "failed reason: {}, return code {}".format(str(e), returncode)
        status = "Failed"
        send_message_if_need(train_conf["run_id"], get_user_name(), message, status)
        raise e
    finally:
        send_message_if_need(train_conf["run_id"], get_user_name(), message, status)
    return 0


if __name__ == '__main__':
    main()
