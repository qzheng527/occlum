# coding=utf-8
# @Date: 2023-06-14
import os
import json
import torch
import logging
import subprocess
from typing import Dict, Any # noqa
from jsonschema import validate

from .error import (
    InvalidParamError,
    JobPrepareError,
    FileFormatError
)
from .define import (
    AISTUDIO_SYSTEM_CMD,
    ALLOWED_MODEL_NAMES,
    DATA_SCHEMA,
    PACKED_DATA_SCHEMA,
    MODEL_AND_GPU_TO_BATCH_MAP,
    MODEL_AND_GPU_WITH_PEFT_TO_BATCH_MAP,
    AISTUDIO_JOB_DETAIL
)
from .configs.configure import FineTuneConfig
from solutions.antllm.antllm.models.glm.modeling_glm import check_gpu_sm75_or_greater
from solutions.antllm.antllm.models.peft import SUPPORTED_PEFT_TYPES
from solutions.antllm.antllm.utils.aistudio_utils import AntLLMk8sConf
from solutions.antllm.antllm.utils.benchmark import PeftSolutionRunPredict, get_request, submit_aistudio_task_v2  # noqa

os.environ["WANDB_DISABLED"] = "true"

glogger = logging.getLogger(__name__)


# 查询FineTune.batch_predict发起的任务状态
# 也可以使用adabench_cli run-desc --run_id ${your_run_id}查看
def get_status(run_id):
    url = os.path.join(PeftSolutionRunPredict.BENCHMARK_SERVER_URL, "get_job_and_pod_info")
    params = {"id": run_id}
    ret = get_request(url, params)
    if not ret or not ret.get("result", {}):
        glogger.error("get run info faiuled")
        return "get job info failed"
    glogger.info("run %s result %s", run_id, ret)
    return ret.get("result", {}).get("job_status", "")


# 查询FineTune.batch_predict发起的任务日志
# 也可以使用adabench_cli run-log --run_id ${your_run_id}查看
def get_run_log(run_id):
    url = os.path.join(PeftSolutionRunPredict.BENCHMARK_SERVER_URL, "run_log")
    params = {"run_id": run_id}
    ret = get_request(url, params)
    if not ret or not ret.get("result", {}):
        glogger.error("get run log faiuled")
        return "get job info failed"
    glogger.info("run %s result %s", run_id, ret["stdout"])
    return ret.get("result", {}).get("stdout", "")


class FineTune:
    '''
    蚂蚁大模型有监督微调（Supervised-Fine-Tune，SFT）API，
    支持`train_local`和`train_remote`两种方案
    '''
    def __init__(
        self,
        model: str,
        training_config_path: str = None,
        deepspeed_config_path: str = None,
        peft_config_path: str = None
    ) -> None:
        """
        训练的自定义配置文件路径通过在 FineTune 初始化时传入
        三种默认配置文件见：https://code.alipay.com/ai-dls/antnlp/tree/master/solutions/antllm/antllm/api/configs
        可在命令行执行`train_deepspeed --help`查看各参数详细说明。

        :param model(string): 蚂蚁官方发布的模型名或者模型地址
        :param training_config_path:  整体训练的配置，如epoch，learning_rate等。
        fixme：deepspeed的参数暂不支持配置
        :param deepspeed_config_path: 配置deepspeed框架相关参数，如optimizer优化器选择、精度控制等。
        :param peft_config_path: 配置高效微调的参数。
        """
        self.model = model

        # 加载配置文件
        self.config = FineTuneConfig(training_config_path, deepspeed_config_path, peft_config_path)
        self.training_config = self.config.training_config
        self.deepspeed_config = self.config.deepspeed_config
        self.peft_config = self.config.peft_config

        # 设置日志运行级别
        self.logger = logging.getLogger(__name__)

    def _check_data(self, path, use_packed_training=False) -> None:
        if not os.path.exists(path) or not os.path.isfile(path):
            raise FileNotFoundError(f"The file {path} is not exists or not a vaild json file.")

        if use_packed_training and self.config.training_config["online_packed"] == "false":
            data_schema = PACKED_DATA_SCHEMA
        else:
            data_schema = DATA_SCHEMA

        self.logger.info(f"Begin check file: {path}.")
        with open(path, "r", encoding="utf-8") as f:
            try:
                for line in f:
                    validate(instance=json.loads(line), schema=data_schema)
            except Exception:
                self.logger.exception(f'check file data failed')
                raise FileFormatError(f'file format error, please check: {path}')
        self.logger.info(f"Check success.")

    def train_local(
        self,
        train_fpath: str,
        output_dir: str,
        validation_fpath: str = None,
        peft: str = None,
        epoch: int = 2,
        resume_from_checkpoint: bool = False,
        use_long_glm: bool = False,
        use_packed_training: bool = False,
        dynamic_batch: bool = False,
        use_atorch: bool = True
    ) -> bool:
        r'''
        本地 SFT 训练接口，模型训练结果和日志会保存到指定目录中

        Args:
            - train_fpath (string): 训练数据地址路径
            - output_dir (string): 输出路径
            - validation_fpath (string): 训练数据地址路径
            - peft (string): 使用高效微调（PEFT）方法进行部分参数训练，可选参数有：
            ``'None'`` | ``'lora'`` | ``'adalora'`` | ``'prefix'``。
            其中``'None'``：不使用PEFT方法，进行全量参数训练；``'lora'``：使用LoRA方法进行微调；
            ``'adalora'``：使用AdaLoRA方法进行微调；``'prefix'``：使用Prefix Tuning进行微调；
            默认使用``'None'``，即进行全量微调。
            - epoch (int): 训练轮次
            - resume_from_checkpoint (boolean): 是否需要恢复训练
            - use_long_glm (boolean): 是否进行扩展上下文训练
            - use_packed_training (boolean): 是否进行数据packed训练
            - dynamic_batch (boolean): 是否动态配置batch size大小
            - use_atorch (string): 是否使用atorch框架训练,默认为True,否则使用deepspeed
        
        Example:
        ```python
        # Load the tuner
        tuner = FineTune(model="llm_path")

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
            os.makedirs(output_dir,)
        if peft is None and dynamic_batch is True:
            self.logger.warning(f"dynamic batch is disabled in full finetune mode")
            dynamic_batch = False
        
        # 检查训练数据
        self._check_data(train_fpath, use_packed_training)
        if validation_fpath is not None:
            self._check_data(validation_fpath, use_packed_training)

        if not torch.cuda.is_available():
            gpu_num = 0
        else:
            gpu_num = torch.cuda.device_count()

        # 训练脚本命令
        if peft is None and use_atorch is True:
            training_python_file = os.path.join(
                os.path.dirname(__file__), "../commands/sft/train_atorch.py")
            # 兼容atorch训练参数
            self.training_config['save_policy'] = self.training_config.pop('save_strategy')
        else:
            training_python_file = os.path.join(
                os.path.dirname(__file__), "../commands/sft/train_deepspeed.py")
        
        if gpu_num > 0:
            dist_cmd = "python -m torch.distributed.run " \
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

        if dynamic_batch is True or int(self.training_config.get("per_device_train_batch_size", 0)) < 1:
            dynamic_batch_size = self._get_batch_size(peft=peft)
            self.training_config["per_device_train_batch_size"] = dynamic_batch_size

        if validation_fpath is not None:
            self.training_config["test_data"] = validation_fpath
            self.training_config["do_eval"] = ""
        else:
            self.training_config['evaluation_strategy'] = "no"
        if resume_from_checkpoint is True:
            self.training_config["resume_from_checkpoint"] = "true"
        if peft is not None:
            self.training_config["peft_type"] = peft
            self.training_config["no_save_base_model"] = ""
        if use_long_glm is True:
            self.training_config["use_long_glm"] = ""
        if use_packed_training is True:
            self.training_config["use_packed_data"] = "true"

        training_args = self._generate_training_args_from_config(peft, use_atorch)
        training_args += f" 2>&1 | tee -a {output_dir}/log.txt"

        cmd = "set -o pipefail;"
        cmd += dist_cmd + training_args
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

    def train_remote(
        self,
        dataset_id: str,
        k8s_conf: AntLLMk8sConf,
        peft: str = 'lora',
        epoch: int = 2,
        resume_task_id=None,
        use_long_glm: bool = False,
        use_packed_training: bool = False,
        use_atorch: bool = True
    ):
        '''
        远程 SFT 训练接口
         Args:
            - dataset_id (string): 使用数据上传接口获得的dataset_id
            - k8s_conf (string): k8s任务相关配置，见`antllm/api/object_classes.py:AntLLMk8sConf`
            - peft (string): 使用高效微调（PEFT）方法进行部分参数训练，可选参数有：
            ``'None'`` | ``'lora'`` | ``'adalora'`` | ``'prefix'``。
            其中``'None'``：不使用PEFT方法，进行全量参数训练；``'lora'``：使用LoRA方法进行微调；
            ``'adalora'``：使用AdaLoRA方法进行微调；``'prefix'``：使用Prefix Tuning进行微调；
            默认使用``'lora'``。
            - epoch (int): 训练轮次
            - resume_task_id(string): 从某次run中恢复继续train。默认None(不做恢复)，支持传入run_id，即该次run中恢复
            - use_long_glm (boolean): 是否进行扩展上下文训练
            - use_packed_training (boolean): 是否进行数据packed训练
            - use_atorch (string): 是否使用atorch框架训练,默认为True,否则使用deepspeed
        Return: 返回taskid，基于此taskid可以用来下载模型产出物。
        Example:
        ```python
        # Load the tuner
        tuner = FineTune(model="llm_model_name")
        # upload train validation data
        dataset_id = easy_upload(train_fpath, validation_fpath)
        # k8s resources configure
        k8s_conf = AntLLMk8sConf(app_name='gbank', gpu_num=8)

        # Train remote with fully fine-tune
        task_id = tuner.train_remote(
            dataset_id,
            k8s_conf,
            peft=None
        )

        # Train remote with lora
        task_id = tuner.train_remote(
            dataset_id,
            k8s_conf,
            peft="lora"
        )
        ```
        '''
        if self.model not in ALLOWED_MODEL_NAMES:
            raise JobPrepareError(
                f'model {self.model} is not in allowed model list: {list(ALLOWED_MODEL_NAMES.keys())}')
        if peft is None:
            self.logger.warning('ALL-Parameter finetune, make sure the resources you apply for are sufficient!')
        elif peft not in SUPPORTED_PEFT_TYPES:
            raise InvalidParamError(f'{peft} is not supported, use peft types: {SUPPORTED_PEFT_TYPES}')
        if resume_task_id and peft in ['prefix', 'prompt', 'ptuing']:
            self.logger.warning(f'peft mode {peft} does not support resume, will train from strach')
        
        # 针对全量微调设置最小内存, 10B最小300G, 5B最小150G
        if peft is None: 
            if '-10B-' in self.model:
                k8s_conf.memory = k8s_conf.memory if k8s_conf.memory is not None else max(300, 100 * k8s_conf.gpu_num)
            elif '-5B-' in self.model:
                k8s_conf.memory = k8s_conf.memory if k8s_conf.memory is not None else max(150, 100 * k8s_conf.gpu_num)
        # 更新训练参数
        self.training_config["pretrained_model_name_or_path"] = ALLOWED_MODEL_NAMES[self.model]
        self.training_config["num_train_epochs"] = epoch
        if peft is not None:
            self.training_config["peft_type"] = peft
            self.training_config["no_save_base_model"] = ""
        if use_long_glm is True:
            self.training_config["use_long_glm"] = ""
        if use_packed_training is True:
            self.training_config["use_packed_data"] = "true"

        # 管理训练参数
        train_args = {}
        train_args.update(self.training_config)
        if peft is not None:
            train_args['no_save_deepspeed_checkpoint'] = ''
            train_args.update(self.peft_config)
        
        # 兼容atorch训练参数
        if peft is None and use_atorch is True:
            train_args['save_policy'] = train_args.pop('save_strategy')

        # 初始化训练任务，用于训练产出物管理
        run_id = self.init_remote_run()
        # 训练命令
        cmd = f"{AISTUDIO_SYSTEM_CMD} && " + \
            "antllm_train_sft_aistudio train_config.json"
        if k8s_conf.init_command:
            cmd = f'{k8s_conf.init_command} && {cmd}'
        # 提交aistudio训练任务, channel和origin是提交给服务端进行统计用的参数，表示调用方来自antllm,使用train_remote函数触发
        aistudio_task_id = submit_aistudio_task_v2(
            {
                'train_args': train_args, 'dataset_id': dataset_id, 'run_id': run_id, 'use_atorch': use_atorch,
                'resume_task_id': resume_task_id if resume_task_id is not None else ''
            },
            k8s_conf,
            cmd,
            tags_str="basemodel={},type={},dev_pattern=AntNLP".format(self.model, peft),
            channel="antllm",
            origin="FineTune.train_remote"
        )
        if not aistudio_task_id:
            raise Exception("submit failed")
        self.logger.info(f'Task status and log url:{os.path.join(AISTUDIO_JOB_DETAIL, str(aistudio_task_id))}')
        self.logger.info(f'Please use id: {run_id} to download training output')
        return run_id

    def batch_predict(self, run_id: str, predictions: str, k8s_conf: AntLLMk8sConf) -> str:
        '''
        lora解决方案云端预测
        Args:
            - run_id, benchmark的run_id(必须是跑过云端训练的run_id)
            - predictions,  可以是一个本地路径，表示测试文件(test.jsonl)所在的位置，如/home/admin/my_predictions.jsonl
              注意，你的test.jsonl必须是符合supervised_finetune这个task要求的格式！！！！！
              可以是一个dataset_id, 如my_dataset，不必用id://开头
            - k8s_conf 资源调度相关的设置
        Return run_id if success else ""
        '''
        if self.model not in ALLOWED_MODEL_NAMES:
            raise JobPrepareError(
                f'model {self.model} is not in allowed model list: {list(ALLOWED_MODEL_NAMES.keys())}')
        llm_path = os.path.join('/adabench_mnt/llm/', ALLOWED_MODEL_NAMES[self.model])
        lora_predict = PeftSolutionRunPredict(run_id, predictions, self.training_config, self.deepspeed_config,
                                              llm_path,
                                              k8s_conf,
                                              origin="FineTune.batch_predict")
        run_id = lora_predict.run()
        if not run_id:
            self.logger.error("trigger predict faield: %s %s", run_id, predictions)
            return ""
        return run_id

    def load_config(self, model_name_or_path: str) -> Dict[str, Any]:
        if not os.path.exists(model_name_or_path):
            raise FileExistsError(f"The config file: {model_name_or_path}, not exists.")

        with open(model_name_or_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        return config

    def init_remote_run(self):
        import adabench.core.run as adabench_run
        run = adabench_run.Run.new_run({})
        run.execute_context = {
            'base_llm': ALLOWED_MODEL_NAMES[self.model]
        }
        return run.run_id

    def _generate_training_args_from_config(self, peft=None, use_atorch=False) -> str:
        training_args = ""
        if not use_atorch or peft is not None:
            training_args += f"--deepspeed {self.config.deepspeed_config_path} "
            training_args += f"--no_save_deepspeed_checkpoint "

            if peft is not None :
                for k, v in self.peft_config.items():
                    if v:
                        training_args += f"--{k} {v} "
                    else:
                        training_args += f"--{k} "

        for k, v in self.training_config.items():
            if v:
                training_args += f"--{k} {v} "
            else:
                training_args += f"--{k} "

        return training_args

    def _get_batch_size(self, peft: str = None) -> int:
        """
        根据模型和GPU显存大小获取最佳`batch_size`配置，当无法匹配时调用动态计算算法。
        """
        MODEL_TYPE = None
        GPU_TYPE = None

        p = torch.cuda.get_device_properties(0)
        gpu_memory = p.total_memory / (1 << 20)
        if gpu_memory >= 80000:
            GPU_TYPE = 80
        elif gpu_memory >= 40000:
            GPU_TYPE = 40
        elif gpu_memory >= 32000:
            GPU_TYPE = 32
        elif gpu_memory >= 16000:
            GPU_TYPE = 16
        else:
            raise JobPrepareError(f"The finetune is not support for the GPU with memory {gpu_memory} MiB.")

        use_flash_attention = False
        if check_gpu_sm75_or_greater():
            try:
                from atorch.modules.transformer.layers import HFGLMSelfAttentionFA # noqa
                use_flash_attention = True
            except ImportError:
                self.logger.warning(f"The current python environment does not support flash attention.")
        else:
            self.logger.warning(f"Current device is not allowed to use falsh attention.")

        if "5B" in self.model.upper():
            MODEL_TYPE = "5B"
        elif "10B" in self.model.upper():
            MODEL_TYPE = "10B"
        elif "SUPER-MINI" in self.model.upper():
            return 10
        elif use_flash_attention is False:
            self.logger.info(f"Can not find model type {MODEL_TYPE}, use the dynamic compute process.")
            batch_size = self._compute_dynamic_batch_size(peft)
            return batch_size
        else:
            raise JobPrepareError(f"Flash attention is not support for model type: {self.model}")

        if self.training_config["max_length"] != 1024:
            batch_size = self._compute_dynamic_batch_size(peft)
            return batch_size            

        if use_flash_attention is True:
            SELECT_TYPE = f"{MODEL_TYPE}_MODEL_{GPU_TYPE}G_MEMORY_1024_LENGTH_ATORCH"
        else:
            SELECT_TYPE = f"{MODEL_TYPE}_MODEL_{GPU_TYPE}G_MEMORY_1024_LENGTH"

        if peft is not None:
            batch_size = MODEL_AND_GPU_WITH_PEFT_TO_BATCH_MAP[SELECT_TYPE]
        else:
            batch_size = MODEL_AND_GPU_TO_BATCH_MAP[SELECT_TYPE]
        
        assert batch_size > 0, ValueError(
            f"The max batch_size {batch_size} must greater than 0. "
            f"Please use a smaller model or allocate more GPU memory."
        )
        return batch_size

    def _compute_dynamic_batch_size(self, peft: str = None) -> int:
        """
        SFT API中提供了动态batch size计算功能，帮助用户最大化利用显存，可以通过在`train_local`中设置`dynamic_batch=True`开启该功能。
        注意：该计算逻辑只适合于标准的GLM网络，如果使用Flash Attention加速则暂不支持

        **计算原理**:

        通常一个GPU模型训练程序的显存占用分为四个部分：模型自身大小、前向传播、后向传播以及优化器显存占用。
        在Deepspeed场景下优化器会将梯度卸载至CPU进行计算，因此只需计算前三个部分即可。
        需要注意，在CUDA启动和计算后向传播过程中会有不同的CUDA CONTEXT，即torch所必须的CUDA环境占用内存，
        占用内存大小和GPU型号、CUDA版本以及torch版本有关，这里统一默认设置为2048 MiB。

        **参数计算**：
        - 模型自身大小：根据模型`hidden_size`、`num_layers`、`num_heads`等参数进行计算，再加上额外的CUDA CONTEXT。对于AntGLM，其计算公式为:
            `vocab_size * hidden_size + num_layers * hidden_size * hidden_size * (4 + 4 + 3 + 1)`

            其中`4+4`代表`MLP`中的`FFN`网络，`3`代表`attention`中的`query_key_value`计算，
            `1`代表`attention`中的全连接。注意，这个计算公式中忽略了`LayerNorm`中的少量参数。

        - 前向传播：针对AntGLM，将前向激活值计算分为以下部分
            - MLP层中的激活值：两次线性计算（`4 + 1`）、一次激活计算（`4`）和一次dropout（`0.5`）: 

                `mlp_forward_size = max_length * hidden_size * (4 + 1 + 4 + 0.5)`

            - 注意力权重激活值：`attention`计算过程中的注意力图计算，和序列长度高度相关，包含两次权重计算，一次全链接和一次`dropout`: 

                `attention_score = head_size * max_length^2 * 2`
                `+ max_length * hidden_size`
                `+ 1.5 * max_length^2 * head_size`

            
            - `qkv`映射函数激活值，会受到`peft`方法的影响（公式中的后半部分，包含`dropout`）：
                
                `qkv = max_length * hidden_size * 3` 
                `+ (max_length * hidden_size * 1.5 + max_length * rank)_{lora}`
            
            - `attention`中所有的激活值（`qkv`映射函数激活值和注意力权重激活值）以及dropout和一个全联接层的计算：
                
                `attention &= attention_score + qkv`
                `+ max_length * hidden_size * 2 + hidden_size * max_length * 0.5`
            
            - 残差激活值，每个`GLMLayer`会进行两次残差计算：

                `res_connect = max_length * hidden_size * 3 + hidden_size`
            
            - 最终计算公式：
                
                `forward_memory_per_batch =`
                `(attention * 0.75 + mlp_forward_size * 0.65 + res_connect)`
                `* num_layers \div 1024^2 * tensor_byte_size`

                
                其中`0.75`和`0.65`是放缩系数，由实际运算估计得出，这是torch在计算图构建过程中会做一定的优化，
                释放一些不需要保存的激活值以节约显存，因此该值会略微大于实际的显存占用值，最终的`batch size`可能略小于真实的最大值。

            - 计算样例：

                ```python
                >>> mlp_forward_size = max_length * hidden_size * (4 + 1 + 4 + 0.5)
                >>> attention_score = head_size * max_length ** 2 * 2 + max_length * hidden_size + \\
                ...     1.5 * max_length ** 2 * head_size # dropout
                >>> qkv = max_length * hidden_size * 3 + lora -> (max_length * hidden_size * 1.5 + max_length * rank)
                >>> attention = attention score + qkv + max_length * hidden_size * 2 + \\
                ...     hidden_size * max_length * 0.5 # dropout
                >>> res_connect = max_length * hidden_size * 3 + hidden_size
                >>> forward_memory = (attention * 0.75 + mlp_forward_size * 0.65 + res_connect) * num_layers \\
                ...     / 1024 ** 2 * tensor_byte_size
                ```

        - 后向传播：后向梯度计算部分需要有以下显存占用：
            - 类似于CUDA CONTEXT的显存占用，设计为固定值2048 MiB，P100上实际值为1900 MiB
            - 需要训练的模型参数
            - 最后一层输出用于回传的前向激活值

        **参考资料**

        - [Pytorch动态计算图规则](https://www.pytorchmaster.com/2-3%2C%E5%8A%A8%E6%80%81%E8%AE%A1%E7%AE%97%E5%9B%BE/)
        - [PyTorch显存机制分析](https://zhuanlan.zhihu.com/p/424512257)
        - [deepspeed原理](https://www.deepspeed.ai/training/)
        - Torch官方对于这个问题的讨论：
        [GPU memory estimation given a network]
        (https://discuss.pytorch.org/t/gpu-memory-estimation-given-a-network/1713)、
        [How to calculate the GPU memory that a model uses?]
        (https://discuss.pytorch.org/t/how-to-calculate-the-gpu-memory-that-a-model-uses/157486)、
        [GPU memory that model uses](https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822)
        """

        if not torch.cuda.is_available():
            self.logger.warning(f"There is no GPU, the batch size will be set to 1.")
            return 1

        # Get the memory for each GPU
        p = torch.cuda.get_device_properties(0)
        gpu_memory = p.total_memory / (1 << 20)
        
        config_path = os.path.join(self.model, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Calculate the parameter size of model
        hidden_size, num_layers, vocab_size = config["hidden_size"], config["num_layers"], config["vocab_size"]
        head_size, max_length = config["num_attention_heads"], int(self.training_config["max_length"])
        if "fp16" in self.training_config or "bf16" in self.training_config:
            tensor_byte_size = 2
        elif "int8" in self.training_config:
            tensor_byte_size = 1
        else:
            tensor_byte_size = 4

        # Calculate the model size
        model_vocab_size = vocab_size * hidden_size
        model_param_size = num_layers * hidden_size * hidden_size * 12
        cuda_context_memory = 1024

        model_size = (model_vocab_size + model_param_size)
        model_size = model_size / 1024 ** 2 * tensor_byte_size + cuda_context_memory
        
        # lora rank size
        rank = 100

        # Calculate forward param size with the rule define in the top docstring
        mlp_forward_size = max_length * hidden_size * (4 + 1 + 4 + 0.5)

        attention_score = head_size * max_length ** 2 * 2 + max_length * hidden_size + 1.5 * max_length ** 2 * head_size
        qkv = max_length * hidden_size * 3
        if peft == "lora":
            qkv += max_length * hidden_size * 1.5 + max_length * rank
        attention = attention_score + qkv + max_length * hidden_size * 2 + hidden_size * max_length * 0.5
        res_connect = max_length * hidden_size * 3 + hidden_size

        # Calculate the size of each token i.e. the memory used by the forward process,
        # and the size of memory used by backward
        forward_memory = (attention * 0.75 + mlp_forward_size * 0.65 + res_connect) * num_layers \
            / 1024 ** 2 * tensor_byte_size

        # The backward cuda context
        backward_fixed_memory = 2048
        backward_dynamic_memory = max_length * vocab_size / 1024 ** 2 * tensor_byte_size
        if peft is not None:
            lora_param_size = num_layers * hidden_size * rank * (1 + 3) / 1024 ** 2
            last_activation_memory = max_length * hidden_size * tensor_byte_size / 1024 ** 2
            backward_fixed_memory += lora_param_size * tensor_byte_size + last_activation_memory
        else:
            backward_fixed_memory += model_size
        remain_memory = gpu_memory - model_size - backward_fixed_memory
        memory_per_batch = forward_memory + backward_dynamic_memory
        
        batch_size = remain_memory // memory_per_batch
        self.logger.info(f"The memory of the model is {model_size:.2f} MiB")
        self.logger.info(f"The memory for each GPU device is {gpu_memory:.2f} MiB")
        self.logger.info(f"The GPU memory allocated by each batch is {memory_per_batch:.2f} MiB")
        self.logger.info(f"The batch size used for training is {batch_size}")

        assert batch_size > 0, ValueError(f"The max batch_size {batch_size} must greater than 0.")

        return int(batch_size)
