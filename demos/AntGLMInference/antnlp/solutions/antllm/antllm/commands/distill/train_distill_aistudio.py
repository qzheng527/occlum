# -*- coding: utf-8 -*-
import os
import sys
import json
import subprocess
import shutil
import logging
import time
import datetime
import antllm
from adabench.impl.run_impl import run_upload
from solutions.antllm.antllm.api.data_utils import download
from adabench.utils.util import get_user_name
from solutions.antllm.antllm.utils.benchmark import notify_benchmark_server

logger = logging.getLogger(__name__)

os.environ["WANDB_DISABLED"] = "true"


def exec_file_adapt_hard_target(train_args, model_output):
    """
    进行一些路径拼接，同时配置到train_args里面
    @param train_args: dict形式，训练的参数
    @param model_output: 训练结果的存放位置
    """
    if train_args["teacher_fine_tune"]["do_fine_tune"]:
        train_args["teacher_fine_tune"]["fine_tuned_path"] = os.path.join(model_output, train_args["teacher_fine_tune"][
            "fine_tuned_path"])
    train_args["train_config"]["output_dir"] = os.path.join(model_output, train_args["train_config"]["output_dir"])
    if train_args["student_fine_tune"]["do_fine_tune"]:
        train_args["student_fine_tune"]["output_dir"] = os.path.join(model_output,
                                                                     train_args["student_fine_tune"]["output_dir"])

    if train_args["teacher_fine_tune"]["do_fine_tune"]:
        if "training_config" in train_args["teacher_fine_tune"]:
            training_config_path = os.path.join(model_output, "teacher_ft_training_config.json")
            json.dump(train_args["teacher_fine_tune"]["training_config"], open(training_config_path, "w"))
            train_args["teacher_fine_tune"]["training_config_path"] = training_config_path
        if "deepspeed_config" in train_args["teacher_fine_tune"]:
            deepspeed_config_path = os.path.join(model_output, "teacher_ft_deepspeed_config.json")
            json.dump(train_args["teacher_fine_tune"]["deepspeed_config"], open(deepspeed_config_path, "w"))
            train_args["teacher_fine_tune"]["deepspeed_config_path"] = deepspeed_config_path

    if train_args["trainer_type"] in ["GlmTrainer", ] and train_args["student_fine_tune"]["do_fine_tune"]:
        if "training_config" in train_args["student_fine_tune"]:
            training_config_path = os.path.join(model_output, "student_ft_training_config.json")
            json.dump(train_args["student_fine_tune"]["training_config"], open(training_config_path, "w"))
            train_args["student_fine_tune"]["training_config_path"] = training_config_path
        if "deepspeed_config" in train_args["student_fine_tune"]:
            deepspeed_config_path = os.path.join(model_output, "student_ft_deepspeed_config.json")
            json.dump(train_args["student_fine_tune"]["deepspeed_config"], open(deepspeed_config_path, "w"))
            train_args["student_fine_tune"]["deepspeed_config_path"] = deepspeed_config_path


def exec_file_adapt_soft_target(train_args, model_output):
    """
        进行一些路径拼接，同时配置到train_args里面
        @param train_args: dict形式，训练的参数
        @param model_output: 训练结果的存放位置
        """
    training_config_path = os.path.join(model_output, "training_config_path.json")
    json.dump(train_args["training_config"], open(training_config_path, "w"))
    train_args["training_config_path"] = training_config_path
    deepspeed_config_path = os.path.join(model_output, "deepspeed_config_path.json")
    json.dump(train_args["deepspeed_config"], open(deepspeed_config_path, "w"))
    train_args["deepspeed_config_path"] = deepspeed_config_path


def get_pretrained_model_path(pretrained_model_name_or_path):
    if os.path.exists(pretrained_model_name_or_path):
        return pretrained_model_name_or_path
    return os.path.join('/adabench_mnt/llm/', pretrained_model_name_or_path)


def get_pretrained_student_model_path(pretrained_model_name_or_path):
    if os.path.exists(pretrained_model_name_or_path):
        return pretrained_model_name_or_path
    return os.path.join('/adabench_mnt/distill/', pretrained_model_name_or_path)


def send_message_if_need(run_id, submitor, message, status):
    ret = ""
    if os.environ.get('WORLD_SIZE') is not None and int(os.environ.get('WORLD_SIZE')) > 1:
        if os.environ.get('RANK') is not None and int(os.environ.get('RANK')) == 0:
            ret = notify_benchmark_server(run_id, submitor, message, status)
    else:
        ret = notify_benchmark_server(run_id, submitor, message, status)
    logger.info("request result %s", str(ret))


def main():
    # config_path: 配置文件路径，配置文件为json格式，必须包含以下信息：
    # {
    #     "train_args": {...},
    #     "dataset_id": "train_dataset_id",
    #     "run_id": "run_id"
    # }
    config_path = sys.argv[1]
    with open(config_path) as fi:
        train_conf = json.load(fi)

    # 生成模型训练参数
    logger.info(f'star training for {train_conf["run_id"]}')
    train_args = train_conf['train_args']
    model_output = os.path.join('/adabench_mnt/api/model_output', train_conf['run_id'])
    os.makedirs(model_output, exist_ok=True)

    train_data_dir = os.path.join(model_output, "train_data")
    download(train_conf['dataset_id'], train_data_dir)

    train_args['output_dir'] = model_output
    # 适配现有数据路径，最终需要修改
    train_args['train_data'] = os.path.join(train_data_dir, "resource")
    train_args['teacher_model'] = get_pretrained_model_path(
        train_args['teacher_model'])
    train_args['student_model'] = get_pretrained_student_model_path(
        train_args['student_model'])
    if "reasoning_model" in train_args:
        train_args["reasoning_model"] = get_pretrained_model_path(train_args['reasoning_model'])
    if train_args["distill_method"] == "hard_target":
        exec_file_adapt_hard_target(train_args, model_output)
    elif train_args["distill_method"] == "soft_target":
        exec_file_adapt_soft_target(train_args, model_output)

    conf_path = os.path.join(model_output, "train_conf.json")
    json.dump(train_args, open(conf_path, "w"))

    shell_path = os.path.join(antllm.__path__[0], "api/distill.py")
    cmd = 'python {} {} remote_distill'.format(shell_path, conf_path)

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
    main(sys.argv[1])
