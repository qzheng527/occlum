# coding=utf-8
# @Date: 2023-06-14
import sys

if sys.argv[-1] == "remote_distill":  # noqa
    import antllm  # noqa
import subprocess
from typing import Dict, Any  # noqa
from solutions.antllm.antllm.api.fine_tune import AntLLMk8sConf
from typing import List
from tqdm import tqdm
from itertools import chain
import threading
import torch
import os
import json
import shutil
import logging
import random
import math
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from solutions.antllm.antllm.api.define import (
    AISTUDIO_SYSTEM_CMD,
    ALLOWED_MODEL_NAMES,
    STUDENT_MODEL_NAMES,
)
from solutions.antllm.antllm.api.error import JobPrepareError
from solutions.antllm.antllm.api.fine_tune import FineTune
from solutions.antllm.antllm.inference.glm_predictor import GLMForInference
from solutions.antllm.antllm.evaluation.metrics.generation.bleu_metrics import HuggingfaceBLEU
from solutions.antllm.antllm.utils.benchmark import PeftSolutionRunPredict, get_request, submit_aistudio_task_v2  # noqa
import numpy as np
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)

metric_map = {"bleu": HuggingfaceBLEU}

os.environ["WANDB_DISABLED"] = "true"


class Distill:
    '''
    以训练好的蚂蚁大模型以及有监督数据做student模型的微调

    Args:
        - model (string): 蒸馏小模型模型地址
        - teacher_model (string): 蚂蚁官方发布的模型名或者模型地址
        - distill_config_path [str, dict]: 模型训练配置文件路径或dict，可选参数
    '''

    def __init__(
            self,
            model: str,
            teacher_model: str,
            reasoning_model: str = None,
            distill_config: [str, dict] = None
    ) -> None:
        self.model = model
        self.teacher_model = teacher_model
        self.reasoning_model = reasoning_model
        self.logger = logging.getLogger(__name__)
        if distill_config is None:
            default_conf_path = os.path.join(os.path.dirname(__file__), "configs/distill/hard_target_seq2seq.json")
            self.distill_config = json.load(open(default_conf_path))
        elif isinstance(distill_config, str):
            self.distill_config = json.load(open(distill_config))
        else:
            self.distill_config = distill_config
        if self.distill_config["distill_method"] == "hard_target" and \
                self.distill_config["trainer_type"] == "COT_Trainer" and self.reasoning_model is None:
            self.reasoning_model = teacher_model

    def train_local(
            self,
            data_folder: str,
            output_dir: str,
            peft: str = None,
            epoch: int = 2,
            resume_from_checkpoint: bool = False,
            dynamic_batch: bool = False
    ) -> bool:

        distill_method = self.distill_config["distill_method"]

        if distill_method == "soft_target":
            distill_process = SoftTargetDistillStudent(
                self.model, self.teacher_model,
                training_config_path=self.distill_config["training_config_path"],
                deepspeed_config_path=self.distill_config["deepspeed_config_path"],
                distill_config=self.distill_config)
            return distill_process.train_local(
                data_folder=data_folder,
                output_dir=output_dir,
                peft=peft, epoch=epoch,
                resume_from_checkpoint=resume_from_checkpoint,
                dynamic_batch=dynamic_batch
            )
        elif distill_method == "hard_target":
            distill_process = HardTargetDistill(
                teacher_model=self.teacher_model,
                student_model=self.model,
                reasoning_model=self.reasoning_model,
                distill_config=self.distill_config
            )
            return distill_process.train_local(distill_data=data_folder, output_dir=output_dir)
        else:
            raise ValueError('distill_method must in "soft_target" or "hard_target"')

    def init_remote_run(self):
        import adabench.core.run as adabench_run
        run = adabench_run.Run.new_run({})
        run.execute_context = {
            'base_llm': ALLOWED_MODEL_NAMES[self.teacher_model]
        }
        return run.run_id

    def train_remote(self, dataset_id: str, k8s_conf: AntLLMk8sConf):
        '''
        远程蒸馏训练接口
         Args:
            - dataset_id (string): 使用数据上传接口获得的dataset_id
            - k8s_conf (string): k8s任务相关配置，见`antllm/api/object_classes.py:AntLLMk8sConf`
        Return: 返回taskid，基于此taskid可以用来下载模型产出物。
        '''
        if self.teacher_model not in ALLOWED_MODEL_NAMES:
            raise JobPrepareError(
                f'model {self.teacher_model} is not in allowed model list: {list(ALLOWED_MODEL_NAMES.keys())}')

        if self.model not in STUDENT_MODEL_NAMES:
            raise JobPrepareError(
                f'student model {self.model} is not in allowed model list: {list(STUDENT_MODEL_NAMES.keys())}')

        self.distill_config["teacher_model"] = ALLOWED_MODEL_NAMES[self.teacher_model]
        self.distill_config["student_model"] = STUDENT_MODEL_NAMES[self.model]
        if self.distill_config["distill_method"] == "hard_target" and \
                self.distill_config["trainer_type"] == "COT_Trainer":
            self.distill_config["reasoning_model"] = ALLOWED_MODEL_NAMES[self.reasoning_model]

        # 设置最小内存, 10B最小300G, 5B最小150G
        if '-10B-' in self.teacher_model:
            k8s_conf.memory = k8s_conf.memory if k8s_conf.memory is not None else max(300, 100 * k8s_conf.gpu_num)
        elif '-5B-' in self.teacher_model:
            k8s_conf.memory = k8s_conf.memory if k8s_conf.memory is not None else max(150, 100 * k8s_conf.gpu_num)
        # 更新训练参数
        distill_method = self.distill_config["distill_method"]

        if distill_method == "hard_target":
            if self.distill_config["teacher_fine_tune"]["do_fine_tune"]:
                training_config_path = self.distill_config["teacher_fine_tune"]["training_config_path"]
                if training_config_path is None:
                    training_config_path = os.path.join(os.path.dirname(__file__), "configs/fine_tune.json")
                self.distill_config["teacher_fine_tune"]["training_config"] = json.load(open(training_config_path))

                deepspeed_config_path = self.distill_config["teacher_fine_tune"]["deepspeed_config_path"]
                if deepspeed_config_path is None:
                    deepspeed_config_path = os.path.join(os.path.dirname(__file__), "configs/deepspeed.json")
                self.distill_config["teacher_fine_tune"]["deepspeed_config"] = json.load(open(deepspeed_config_path))

            if self.distill_config["trainer_type"] == "GlmTrainer" and "student_fine_tune" in self.distill_config and \
                    self.distill_config.get("do_fine_tune", False):
                training_config_path = self.distill_config["student_fine_tune"]["training_config_path"]
                if training_config_path is None:
                    training_config_path = os.path.join(os.path.dirname(__file__), "configs/fine_tune.json")
                self.distill_config["student_fine_tune"]["training_config"] = json.load(open(training_config_path))

                deepspeed_config_path = self.distill_config["student_fine_tune"]["deepspeed_config_path"]
                if deepspeed_config_path is None:
                    deepspeed_config_path = os.path.join(os.path.dirname(__file__), "configs/deepspeed.json")
                self.distill_config["student_fine_tune"]["deepspeed_config"] = json.load(open(deepspeed_config_path))
        elif distill_method == "soft_target":
            training_config_path = self.distill_config["training_config_path"]
            if training_config_path is None:
                training_config_path = os.path.join(os.path.dirname(__file__), "configs/fine_tune.json")
            self.distill_config["training_config"] = json.load(open(training_config_path))

            deepspeed_config_path = self.distill_config["deepspeed_config_path"]
            if deepspeed_config_path is None:
                deepspeed_config_path = os.path.join(os.path.dirname(__file__), "configs/deepspeed.json")
            self.distill_config["deepspeed_config"] = json.load(open(deepspeed_config_path))

        # 初始化训练任务，用于训练产出物管理
        run_id = self.init_remote_run()
        # 训练命令
        cmd = f"{AISTUDIO_SYSTEM_CMD} && " + \
              "antllm_train_model_distill train_config.json"
        if k8s_conf.init_command:
            cmd = f'{k8s_conf.init_command} && {cmd}'
        # 提交aistudio训练任务, channel和origin是提交给服务端进行统计用的参数，表示调用方来自antllm,使用train_remote函数触发
        ret = submit_aistudio_task_v2(
            {
                'train_args': self.distill_config, 'dataset_id': dataset_id, 'run_id': run_id
            },
            k8s_conf,
            cmd,
            channel="antllm",
            origin="Distill.train_remote"
        )
        if not ret:
            raise Exception("submit failed")

        self.logger.info(f'task is running, task id: {run_id}, please use task id to download training output')
        return run_id


class SoftTargetDistillStudent(FineTune):
    '''
    以训练好的蚂蚁大模型以及有监督数据做student模型的微调

    Args:
        - model (string): 蚂蚁官方发布的模型名或者模型地址
        - teacher_model (string): 蚂蚁官方发布的模型名或者模型地址
    '''

    def __init__(
            self,
            model: str,
            teacher_model: str,
            training_config_path: str = None,
            deepspeed_config_path: str = None,
            distill_config: dict = None
    ) -> None:
        super().__init__(model, training_config_path=training_config_path, deepspeed_config_path=deepspeed_config_path)
        self.teacher_model = teacher_model
        self.distill_config = distill_config
        try:
            if self.distill_config is None:
                distill_config_path = os.path.join(os.path.dirname(__file__), "configs/distill/soft_target.json")
                with open(distill_config_path) as f:
                    self.distill_config = json.load(f)
        except Exception:
            self.logger.exception('read distill_config_path file failed')

        self.logit_weight = self.distill_config.get('logit_weight')
        self.hard_target_weight = self.distill_config.get('hard_target_weight')
        self.hidden_state_cos_weight = self.distill_config.get('hidden_state_cos_weight')
        self.hidden_state_mse_weight = self.distill_config.get('hidden_state_mse_weight')
        self.hidden_states_mes_mapping = self.distill_config.get('hidden_states_mes_mapping')
        self.temperature = self.distill_config.get('temperature')

    def train_local(
            self,
            data_folder: str,
            output_dir: str,
            peft: str = None,
            epoch: int = 2,
            resume_from_checkpoint: bool = False,
            dynamic_batch: bool = False
    ) -> bool:
        r'''
        本地soft target 蒸馏训练接口，模型训练结果和日志会保存到指定目录中

        Args:
            - train_fpath (string): 训练数据地址路径
            - output_dir (string): 输出路径
            - validation_fpath (string): 训练数据地址路径
            - peft (string): 建议直接使用全量微调做student模型的训练。 使用高效微调（PEFT）方法进行部分参数训练，可选参数有：
            ``'None'`` | ``'lora'`` | ``'adalora'`` | ``'prefix'``。
            其中``'None'``：不使用PEFT方法，进行全量参数训练；``'lora'``：使用LoRA方法进行微调；
            ``'adalora'``：使用AdaLoRA方法进行微调；``'prefix'``：使用Prefix Tuning进行微调；
            默认使用``'None'``，即进行全量微调。
            - epoch (int): 训练轮次
            - dynamic_batch (boolean): 是否动态配置batch size大小

        Example:
        ```python
        # Load the tuner
        tuner = SoftTargetDistillStudent(model="llm_path", teacher_model="llm_path2")

        # Train local with fully fine-tune
        tuner.train_local(
            "train_data_path",
            "output_dir",
            validation_fpath="valid_data_path"
        )

        # Train local with lora
        tuner.train_local(
            "train_data_path",
            "output_dir",
            validation_fpath="valid_data_path",
            peft="lora"
        )
        ```
        '''
        if not os.path.exists(self.model):
            raise FileNotFoundError(f"The LLM model not found: {self.model}")

        if not os.path.exists(output_dir):
            self.logger.warning(f"The output directory {output_dir} is not exist, make a new one.")
            os.makedirs(output_dir, )
        train_fpath = os.path.join(data_folder, "distill_train.jsonl")
        if os.path.exists(os.path.join(data_folder, "distill_eval.jsonl")):
            validation_fpath = os.path.join(data_folder, "distill_eval.jsonl")
        else:
            validation_fpath = None
        # 检查训练数据
        self._check_data(train_fpath)
        if validation_fpath is not None:
            self._check_data(validation_fpath)

        if not torch.cuda.is_available():
            gpu_num = 0
        else:
            gpu_num = torch.cuda.device_count()

        # 训练脚本命令
        training_python_file = os.path.join(
            os.path.dirname(__file__), "../commands/sft/train_deepspeed_distill.py")

        if gpu_num > 0:
            deepspeed_cmd = "python -m torch.distributed.run " \
                            f"--nnode=1 --nproc_per_node={gpu_num} --node_rank=0 " \
                            f"--master_addr=127.0.0.1 --master_port={self.config.local_training_port} " \
                            f"{training_python_file} "
        else:
            self.logger.warn(f"There was no GPU available, exit the training process.")
            return True

        # 更新训练参数
        self.training_config["train_data"] = train_fpath
        self.training_config["pretrained_model_name_or_path"] = self.model
        self.training_config["num_train_epochs"] = epoch
        self.training_config["output_dir"] = output_dir
        self.training_config["logit_weight"] = self.logit_weight
        self.training_config["hard_target_weight"] = self.hard_target_weight
        self.training_config["hidden_state_cos_weight"] = self.hidden_state_cos_weight
        self.training_config["hidden_state_mse_weight"] = self.hidden_state_mse_weight
        self.training_config["hidden_states_mes_mapping"] = self.hidden_states_mes_mapping
        self.training_config["temperature"] = self.temperature
        self.training_config["teacher_model_path"] = self.teacher_model

        if dynamic_batch is True or int(self.training_config.get("per_device_train_batch_size", 0)) < 1:
            dynamic_batch_size = self._get_batch_size(peft=peft)
            self.training_config["per_device_train_batch_size"] = dynamic_batch_size

        if validation_fpath is not None:
            self.training_config["test_data"] = validation_fpath
            self.training_config["do_eval"] = ""
        if resume_from_checkpoint is True:
            self.training_config["resume_from_checkpoint"] = "true"
        if peft is not None:
            self.training_config["peft_type"] = peft
            self.training_config["no_save_base_model"] = ""

        training_args = self._generate_training_args_from_config()
        training_args += f" 2>&1 | tee -a {output_dir}/log.txt"

        cmd = "set -o pipefail;"
        cmd += deepspeed_cmd + training_args
        self.logger.info("save the finetune config.")
        self.config.save_config(output_dir)

        # Excute the local training comand
        self.logger.info("excute cmd: " + cmd)
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            if e.returncode == -9:
                raise RuntimeError(
                    "The program ran into a kernel error with exitcode 9."
                    " may be due to insufficient memory or cpu, please increase memory and cpu."
                )
            with open(f"{output_dir}/log.txt", "r") as f:
                lines = f.readlines()
                error_traceback = lines[-2]
                if "Signal 9" in error_traceback:
                    raise RuntimeError(
                        "The program ran into a kernel error with exitcode 9."
                        " may be due to insufficient memory or cpu, please increase memory and cpu."
                    )
                else:
                    raise e

        self.logger.info("train finished")

        return True

    def train_remote(self, dataset_id: str, k8s_conf: AntLLMk8sConf,
                     peft='lora', epoch=2, resume_task_id=None):
        raise NotImplementedError


class HardTargetDistill:
    def __init__(self, teacher_model, student_model, reasoning_model=None, distill_config: [str, dict] = None):

        if distill_config is None:
            base_dir = os.path.dirname(__file__)
            distill_config = os.path.join(base_dir, "configs/distill/hard_target_seq2seq.json")
        if isinstance(distill_config, str):
            self.config = json.load(open(distill_config))
        else:
            self.config = distill_config
        self.trainer_type = self.config["trainer_type"]
        assert self.trainer_type in ["Seq2SeqTrainer", "GlmTrainer", "COT_Trainer"]
        self.teacher_path = teacher_model
        self.student_path = student_model
        self.reasoning_path = reasoning_model

    def train_local(self, distill_data, output_dir=None):
        if output_dir is not None:
            self.config["teacher_fine_tune"]["fine_tuned_path"] = output_dir + "_fine_tuned_teacher"
            self.config["train_config"]["output_dir"] = output_dir
            self.config["student_fine_tune"]["output_dir"] = output_dir + "_finetune"
        # 进行蒸馏的原始数据
        self.config["data_config"]["raw_file_train"] = os.path.join(distill_data, "distill_train.jsonl")
        self.config["data_config"]["raw_file_eval"] = os.path.join(distill_data, "distill_eval.jsonl")
        # cot reason数据
        self.config["data_config"]["to_reason_train_file"] = os.path.join(distill_data, "distill_to_reason_train.jsonl")
        self.config["data_config"]["to_reason_validation_file"] = os.path.join(distill_data,
                                                                               "distill_to_reason_eval.jsonl")
        # 大模型标注后的数据
        self.config["data_config"]["train_file"] = os.path.join(distill_data, "train.jsonl")
        self.config["data_config"]["validation_file"] = os.path.join(distill_data, "eval.jsonl")
        # 如果需要进行小模型精调
        if "student_fine_tune" in self.config and self.config["student_fine_tune"].get("do_fine_tune", False):
            # 小模型精调数据
            self.config["data_config"]["student_ft_train_file"] = os.path.join(distill_data, "student_train.jsonl")
            self.config["data_config"]["student_ft_validation_file"] = os.path.join(distill_data, "student_eval.jsonl")
        # 如果进行大模型精调训练
        if self.config["teacher_fine_tune"]["do_fine_tune"]:
            teacher_fine_tune_conf = self.config["teacher_fine_tune"]
            use_atorch = self.config["teacher_fine_tune"].get("use_atorch", False)
            fine_tuned_model_path = teacher_fine_tune_conf.get("fine_tuned_path", "./fine_tuned_teacher")
            GLMTeacher.finetune(base_model_path=self.teacher_path,
                                fine_tuned_model_path=fine_tuned_model_path,
                                train_fpath=os.path.join(distill_data, "llm_finetune_train.jsonl"),
                                validation_fpath=os.path.join(distill_data, "llm_finetune_eval.jsonl"),
                                training_config_path=teacher_fine_tune_conf.get("training_config_path", None),
                                deepspeed_config_path=teacher_fine_tune_conf.get("deepspeed_config_path", None),
                                epoch=teacher_fine_tune_conf["num_train_epochs"],
                                use_atorch=use_atorch)
            logger.info("finish llm fine_tuning")
            # hyper_parameters_conf = os.path.join(self.teacher_path, "hyper_parameters.json")
            # if os.path.exists(hyper_parameters_conf):
            #     shutil.copy(os.path.join(self.teacher_path, "hyper_parameters.json"),
            #                 os.path.join(fine_tuned_model_path, "hyper_parameters.json"))
            if not use_atorch:
                fine_tuned_model_path = os.path.join(fine_tuned_model_path, "epochs")
            self.teacher_path = get_last_checkpoint(fine_tuned_model_path)

        # 构造teacher，支持多卡并行进行伪标签数据构造
        teacher_model = MultiThreadingGLMInference(self.teacher_path)

        # 如果使用COT蒸馏
        if self.trainer_type == "COT_Trainer":
            # 需要配置cot蒸馏的参数
            if "cot_distill" not in self.config:
                raise KeyError(
                    "when using COT_Trainer, you need to specify your chain of thought config")
            reasoning_model = MultiThreadingGLMInference(self.reasoning_path)
            # self.config.get("teacher_predict_config", {})通过配置读取大模型做预测时的参数
            teacher = COT_GLMTeacher(teacher_model, reasoning_model, self.config.get("teacher_predict_config", {}),
                                     self.config.get("cot_distill", {}))
        else:
            # self.config.get("teacher_predict_config", {})通过配置读取大模型做预测时的参数
            teacher = GLMTeacher(teacher_model, self.config.get("teacher_predict_config", {}))

        if self.trainer_type in ["Seq2SeqTrainer", "COT_Trainer"]:
            student_model = AutoModelForSeq2SeqLM.from_pretrained(self.student_path)
            student_tokenizer = AutoTokenizer.from_pretrained(self.student_path)
            student = Student(student_model, student_tokenizer)
            distiller = Seq2SeqModelDistill(teacher, student, self.config)
        elif self.trainer_type == "GlmTrainer":
            distiller = GlmModelDistill(teacher, self.student_path, self.config)
        else:
            raise RuntimeError("trainer_type not in [Seq2SeqTrainer, GlmTrainer]")
        logger.info("finish forming teacher and student")
        distiller.distill()
        logger.info("finish distilling process")
        return True


class Teacher:
    def __init__(self, model, predict_config):
        self.model = model
        self.predict_config = predict_config

    @classmethod
    def finetune(cls, base_model_path, fine_tuned_model_path, train_fpath, validation_fpath, epoch):
        raise NotImplementedError

    def predicts(self, items: List[str]) -> List[str]:
        """
            predict函数对输出的样本进行预测，得到模型的输出结果
            @param items: 以字符串列表的形式传入待预测的样本
            @return 以字符串列表的返回对每条样本的预测结果
        """
        raise NotImplementedError

    def form_mimic_data(self, data_config: dict):
        """
        @param data_config为待预测的文件路径、预测结果存入的文件路径
        """
        raise NotImplementedError


class Student:
    def __init__(self, model, tokenizer):
        """
        @param model: student model
        @param tokenizer: student model对应的tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer


class ModelDistillBase:
    def __init__(self, teacher, student, distill_config):
        """
        @param teacher:  teacher模型
        @param student:  student模型
        @param distill_config: 模型蒸馏的配置信息
        """
        self.teacher = teacher
        self.student = student
        self.distill_config = distill_config
        self.trainer, self.ft_trainer = self.form_trainer()

    def compute_metrics(self, eval_preds: list):
        """
            对模型预测结果的指标计算
            @type eval_preds: 以list形式输入每条数据的模型预测结果与ground truth
        """
        raise NotImplementedError

    def form_trainer(self):
        """
            构建trainer的函数，子类需要通过本函数实现trainer函数的构造
            trainer需要包含train函数用于执行模型蒸馏、save_model函数用于将模型存储于特定路径
            返回值为trainer本身
        """
        raise NotImplementedError

    def distill(self):
        """
            进行模型整理的函数
            返回为完成模型蒸馏的小模型
        """
        self.trainer.train()
        if self.ft_trainer:
            self.ft_trainer.train()
        return True

    def save_distill_model(self, save_path):
        """
            小模型存储函数，将小模型存储到特定的路径上
        """
        self.trainer.save_model(save_path)


class GlmModelDistill(ModelDistillBase):
    def __init__(self, teacher, student, distill_config):
        super(GlmModelDistill, self).__init__(teacher, student, distill_config)

    def form_trainer(self):
        logger.info("start forming mimic data")
        self.teacher.form_mimic_data(self.distill_config["data_config"])
        logger.info("finish forming mimic data")
        self.teacher.model = None
        trainer = GlmTrainer(self.student,
                             self.distill_config["data_config"]["train_file"],
                             self.distill_config["data_config"]["validation_file"],
                             self.distill_config["train_config"]
                             )
        distill_use_atorch = self.distill_config["train_config"].get("use_atorch", False)
        logger.info("distilling trainer is ready")
        if "student_fine_tune" in self.distill_config and self.distill_config["student_fine_tune"].get("do_fine_tune",
                                                                                                       False):
            distill_student_path = self.distill_config["train_config"]["output_dir"]
            if not distill_use_atorch:
                distill_student_path = os.path.join(distill_student_path, "epochs")
            student_fine_tune_config = self.distill_config["student_fine_tune"]
            student_fine_tune_config["model_path"] = distill_student_path
            ft_trainer = GlmTrainer(
                distill_student_path,
                self.distill_config["data_config"]["student_ft_train_file"],
                self.distill_config["data_config"]["student_ft_validation_file"],
                student_fine_tune_config,
                load_last_student=True
            )
            logger.info("student finetune model trainer is ready")
            return trainer, ft_trainer

        return trainer, None


class Seq2SeqModelDistill(ModelDistillBase):
    def __init__(self, teacher, student, distill_config):
        super(Seq2SeqModelDistill, self).__init__(teacher, student, distill_config)

    def compute_metrics(self, eval_preds):
        """
            对模型预测结果的指标计算
            @type eval_preds: 以list形式输入每条数据的模型预测结果与ground truth
        """
        metric = metric_map[self.distill_config.get("metric", "bleu")]()
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.student.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.student.tokenizer.pad_token_id)
        decoded_labels = self.student.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_labels = ["None" if not item else item for item in decoded_labels]
        result = metric.compute([[item] for item in decoded_preds], [[item] for item in decoded_labels], None)
        result = {"BLEU": result}
        prediction_lens = [np.count_nonzero(pred != self.student.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    def make_datasets(self):
        '''
        data_config example
        {
            "train_file": "",
            "validation_file": "",
            "max_source_length": 256,
            "max_target_length": 128,
            "inpout_key": "input",
            "output_key": "output",
            "preprocessing_num_workers": 4
        }
        '''
        data_config = self.distill_config["data_config"]

        def preprocess_function(examples):
            inpout_key = data_config["input_key"]
            output_key = data_config["output_key"]

            inputs, targets = [], []
            for i in range(len(examples[inpout_key])):
                if examples[inpout_key][i] and examples[output_key][i]:
                    inputs.append(examples[inpout_key][i])
                    targets.append(examples[output_key][i])

            model_inputs = self.student.tokenizer(
                inputs, max_length=data_config["max_source_length"], truncation=True
            )

            # Tokenize targets
            labels = self.student.tokenizer(targets, max_length=data_config["max_target_length"], truncation=True)
            model_inputs["labels"] = []
            for target_input in labels["input_ids"]:
                model_inputs["labels"].append(target_input[1:])
            return model_inputs

        logger.info("start forming mimic data")
        self.teacher.form_mimic_data(data_config)
        logger.info("finish forming mimic data")
        self.teacher.model = None

        data_files = {}
        if "train_file" in data_config:
            data_files["train"] = data_config["train_file"]
        if "validation_file" in data_config:
            data_files["validation"] = data_config["validation_file"]
        if "student_ft_train_file" in data_config:
            data_files["student_ft_train"] = data_config["student_ft_train_file"]
        if "student_ft_validation_file" in data_config:
            data_files["student_ft_validation"] = data_config["student_ft_validation_file"]
        raw_datasets = load_dataset("json", data_files=data_files)
        column_names = raw_datasets["train"].column_names

        datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=data_config["preprocessing_num_workers"],
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.student.tokenizer,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )
        return datasets, data_collator

    def form_trainer(self):
        training_args = Seq2SeqTrainingArguments(**self.distill_config["train_config"])
        datasets, data_collator = self.make_datasets()
        trainer = Seq2SeqTrainer(
            model=self.student.model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            tokenizer=self.student.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        logger.info("distilling trainer is ready")

        if "student_fine_tune" in self.distill_config and self.distill_config["student_fine_tune"].get("do_fine_tune",
                                                                                                       False):
            student_ft_training_args = Seq2SeqTrainingArguments(
                **{k: v for k, v in self.distill_config["student_fine_tune"].items() if k != "do_fine_tune"})

            ft_trainer = Seq2SeqTrainer(
                model=self.student.model,
                args=student_ft_training_args,
                train_dataset=datasets["student_ft_train"],
                eval_dataset=datasets["student_ft_validation"],
                tokenizer=self.student.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics
            )
            logger.info("student model fine-tune trainer is ready")
            return trainer, ft_trainer
        return trainer, None


class GLMForDatasetInference:
    '''
        对基础预测类，做预测函数的扩充，扩充dataset_predict函数用于支持多线程形式的模型预测
    '''

    def __init__(self, path, adapter_path=None, gpu_index=None, multi_gpu=False, torch_dtype=torch.float16):
        self.path = path
        self.adapter_path = adapter_path
        self.gpu_index = gpu_index
        self.multi_gpu = multi_gpu
        self.torch_dtype = torch_dtype
        self.infer = None

    def init_model(self):
        self.infer = GLMForInference(path=self.path, adapter_path=self.adapter_path, gpu_index=self.gpu_index,
                                     multi_gpu=self.multi_gpu, torch_dtype=self.torch_dtype)

    def del_model(self):
        self.infer = None

    def dataset_predict(self, data_list, result_list, max_output_tokens=50, predict_batch_size=1):
        '''
            该函数主要用于支持多线程的文本预测
        @param predict_batch_size: 预测batch size
        @param data_list: 待预测的文本列表
        @param result_list: 存放预测结果的列表
        @param max_output_tokens: 设定的最大预测token数
        @return:
        '''

        tmp_result_list = []
        num_batch = math.ceil(len(data_list) / predict_batch_size)
        for i in tqdm(range(num_batch)):
            batch = data_list[i * predict_batch_size: (i + 1) * predict_batch_size]
            result = self.infer.generate_batch(batch, max_output_tokens=max_output_tokens)
            tmp_result_list.extend([item.texts[0] if item.texts[0] else "无" for item in result])
        result_list[:] = tmp_result_list


class MultiThreadingGLMInference:
    def __init__(self, model_path, adapter_path=None, torch_dtype=torch.float16):
        # 判断执行环境：是否满足开启多线程模型预测
        self.multi_predictor_flag = torch.cuda.is_available() and torch.cuda.device_count() > 1
        # 每张卡会部署一个predictor
        self.predictor_list = []
        if self.multi_predictor_flag:
            self.num_gpu = torch.cuda.device_count()
            for i in range(self.num_gpu):
                # 每张卡会部署一个predictor
                predictor = GLMForDatasetInference(model_path, adapter_path=adapter_path, gpu_index=i, multi_gpu=False,
                                                   torch_dtype=torch_dtype)
                self.predictor_list.append(predictor)
        else:
            # 如果不满足多线程预测条件，则使用一个predictor
            predictor = GLMForDatasetInference(model_path, adapter_path=adapter_path, multi_gpu=False,
                                               torch_dtype=torch_dtype)
            self.predictor_list.append(predictor)

    def dataset_inference(self, data_list, max_output_length=50, predict_batch_size=1):
        num_data = len(data_list)
        # 按是否能进行多线程预测来判断每个data segment的数据量
        if self.multi_predictor_flag:
            # 每个gpu均分预测数据
            num_per_segment = int(num_data // self.num_gpu)
            seg_input = [[], ] * self.num_gpu
        else:
            # 如果不能多线程，则只有一个segment，数据量为全量的数据
            num_per_segment = num_data
            seg_input = [[], ]

        # 如果数据量比分块数还少，将num_per_segment强制置为1
        if num_per_segment == 0:
            num_per_segment = 1
        split_index = 0
        seg_index = 0
        while split_index < len(seg_input):
            seg_input[split_index] = data_list[seg_index: seg_index + num_per_segment]
            split_index += 1
            seg_index += num_per_segment
        # 把遗留的数据放在最后一个数据分割中
        if seg_index < num_data:
            seg_input[-1].extend(data_list[seg_index: num_data])
        # 线程列表
        threads = []
        # 初始化结果列表，供填充模型预测结果
        result_list = [["", ] * len(data) for data in seg_input]
        if self.multi_predictor_flag:
            for i in range(self.num_gpu):
                predictor = self.predictor_list[i]
                predictor.init_model()
                data = seg_input[i]
                result = result_list[i]
                newThread = threading.Thread(target=predictor.dataset_predict,
                                             args=(data[:], result, max_output_length, predict_batch_size))
                threads.append(newThread)
            # 多个线程同时进行预测
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            # 不满足使用多线程则使用单线程进行预测
            self.predictor_list[0].init_model()
            self.predictor_list[0].dataset_predict(seg_input[0], result_list[0], max_output_length)

        for predictor in self.predictor_list:
            predictor.del_model()

        return list(chain(*result_list))


class GLMTeacher(Teacher):
    def __init__(self, model, predict_config):
        super(GLMTeacher, self).__init__(model, predict_config)

    @classmethod
    def finetune(cls, base_model_path, fine_tuned_model_path, train_fpath, validation_fpath,
                 training_config_path=None, deepspeed_config_path=None, epoch=1, use_atorch=False):
        tuner = FineTune(base_model_path, training_config_path, deepspeed_config_path)
        flag = tuner.train_local(
            train_fpath=train_fpath,
            validation_fpath=validation_fpath,
            output_dir=fine_tuned_model_path,
            epoch=epoch,
            use_atorch=use_atorch
        )
        assert flag
        assert os.path.exists(fine_tuned_model_path)
        logger.info("finish llm fine_tuning")

    def predicts(self, data_list):
        return self.model.dataset_inference(data_list,
                                            max_output_length=self.predict_config.get("max_output_length", 50),
                                            predict_batch_size=self.predict_config.get("batch_size", 1))

    def file_predict(self, raw_file, result_file, input_key, output_key):
        with open(raw_file) as f:
            lines = f.readlines()
        # 如果数据已经包含了标准的结果，则直接复制文件
        if len(lines) > 0 and output_key in json.loads(lines[0]):
            shutil.copy(raw_file, result_file)
        # 如果数据没有包含标准结果，则使用大模型对数据进行预测，形成带噪label
        else:
            input_data_list = []
            fw = open(result_file, "w")
            for line in lines:
                info = json.loads(line)
                input_data_list.append(info)
            logger.info("start predict {}".format(os.path.basename(raw_file)))
            output_data_list = self.predicts([item[input_key] for item in input_data_list])
            logger.info("finish predict {}".format(os.path.basename(raw_file)))
            for item in zip(input_data_list, output_data_list):
                input_data, output_text = item
                fw.write(json.dumps({**input_data, output_key: output_text}, ensure_ascii=False) + "\n")
            fw.close()

    def form_mimic_data(self, data_config):

        self.file_predict(data_config["raw_file_train"], data_config["train_file"], data_config["input_key"],
                          data_config["output_key"])
        self.file_predict(data_config["raw_file_eval"], data_config["validation_file"], data_config["input_key"],
                          data_config["output_key"])


class COT_GLMTeacher(GLMTeacher):
    def __init__(self, model, reasoner, predict_config, cot_config):
        super(GLMTeacher, self).__init__(model, predict_config)
        self.reasoner = reasoner
        self.cot_config = cot_config

    def get_reasons(self, data_list):
        return self.reasoner.dataset_inference(data_list,
                                               max_output_length=self.predict_config.get("max_output_length", 50),
                                               predict_batch_size=self.predict_config.get("batch_size", 1))

    def reason_predict(self, raw_file, result_file, input_key, output_key, tmp_label_key, few_shot_templates):
        with open(raw_file) as f:
            lines = f.readlines()
        # 如果数据已经包含了标准的结果，则直接复制文件
        if len(lines) > 0 and output_key in json.loads(lines[0]):
            shutil.copy(raw_file, result_file)
        # 如果数据没有包含标准结果，则使用大模型对数据进行预测，形成带噪label
        else:
            num_template = len(few_shot_templates)
            input_data_list = []
            to_pred_list = []
            index_list = []
            fw = open(result_file, "w")
            for index, line in enumerate(lines):
                info = json.loads(line)
                input_data_list.append(info)
                if random.random() > self.cot_config.get("reasoning_rate", 0.5):
                    continue
                to_pred_list.append(
                    few_shot_templates[index % num_template].format(info[input_key], info[tmp_label_key]))
                index_list.append(index)
            logger.info("start cot predict {}".format(os.path.basename(raw_file)))
            output_data_list = self.get_reasons(to_pred_list)
            logger.info("finish cot predict {}".format(os.path.basename(raw_file)))

            # 以数据的index为键构造index与模型输出的dict，方便查找模型对样本的解释结果
            reason_dict = {index: reason for reason, index in zip(output_data_list, index_list)}
            for index, data in enumerate(input_data_list):
                if index in reason_dict:
                    # CoT样本拼接
                    input_text = self.cot_config["student_reason_input_template"].format(data[input_key])
                    output_text = self.cot_config["student_reason_output_template"].format(reason_dict[index],
                                                                                           data[tmp_label_key])
                    fw.write(json.dumps({input_key: input_text, output_key: output_text}, ensure_ascii=False) + "\n")
                # 伪标签数据需要保留
                fw.write(json.dumps({input_key: data[input_key], output_key: data[tmp_label_key]},
                                    ensure_ascii=False) + "\n")
            fw.close()

    def form_mimic_data(self, data_config):
        few_shot_templates = self.cot_config["few_shot_templates"]

        self.file_predict(data_config["raw_file_train"], data_config["to_reason_train_file"],
                          data_config["input_key"], data_config["tmp_label_key"])
        self.file_predict(data_config["raw_file_eval"], data_config["to_reason_validation_file"],
                          data_config["input_key"], data_config["tmp_label_key"])
        self.reason_predict(data_config["to_reason_train_file"], data_config["train_file"],
                            data_config["input_key"], data_config["output_key"], data_config["tmp_label_key"],
                            few_shot_templates)
        self.reason_predict(data_config["to_reason_validation_file"], data_config["validation_file"],
                            data_config["input_key"], data_config["output_key"], data_config["tmp_label_key"],
                            few_shot_templates)


class GlmTrainer:
    def __init__(self, student_path, train_fpath, validation_fpath, config, load_last_student=False):
        self.student_path = student_path
        self.output_dir = config["output_dir"]
        self.training_config_path = config.get("training_config_path", None)
        self.deepspeed_config_path = config.get("deepspeed_config_path", None)
        self.epoch = config.get("epoch", 1)
        self.train_fpath = train_fpath
        self.validation_fpath = validation_fpath
        self.load_last_student = load_last_student
        self.use_atorch = config.get("use_atorch", False)

    def train(self):
        if self.load_last_student:
            self.student_path = get_last_checkpoint(self.student_path)
        logger.info("self.load_last_student: {}".format(self.load_last_student))
        logger.info("student_path: {}".format(self.student_path))
        tuner = FineTune(self.student_path, self.training_config_path, self.deepspeed_config_path)
        flag = tuner.train_local(
            train_fpath=self.train_fpath,
            validation_fpath=self.validation_fpath,
            output_dir=self.output_dir,
            epoch=self.epoch,
            use_atorch=self.use_atorch
        )
        # if os.path.exists(os.path.join(self.student_path, "hyper_parameters.json")):
        #     shutil.copy(os.path.join(self.student_path, "hyper_parameters.json"),
        #                 os.path.join(self.output_dir, "hyper_parameters.json"))
        assert flag
        assert os.path.exists(self.output_dir)


def main(config_path):
    config = json.load(open(config_path))
    output_dir = config["output_dir"]
    train_data = config["train_data"]
    student_model = config["student_model"]
    teacher_model = config["teacher_model"]
    distiller = Distill(model=student_model, teacher_model=teacher_model,
                        reasoning_model=config.get("reasoning_model", None), distill_config=config)
    distiller.train_local(data_folder=train_data, output_dir=os.path.join(output_dir, "distill_student"))


if __name__ == '__main__':
    main(sys.argv[1])
