from abc import ABC, abstractmethod
import os
import logging
import copy
import json
import adabench.utils.util as util
import shutil
import tempfile
from adabench.api import dataset_upload_v2
from adabench.core.run import get_run

glogger = logging.getLogger(__name__)


def get_request(url, params={}):
    ret = util.get_remote(url, params=params)
    if ret.status_code != 200:
        glogger.error("url %s http code %s params %s", url, ret.status_code, params)
        return {}
    if ret.json()["code"] != 0:
        glogger.error("url %s error message %s params %s", url, ret.json()["message"], params)
        return {}
    return ret.json()


def post_request(url, params):
    ret = util.post_remote(url, json=params)
    if ret.status_code != 200:
        glogger.error("url %s http code  %s, params %s", url, ret.status_code, json.dumps(params, indent=4))
        return {}
    if ret.json()["code"] != 0:
        glogger.error("url %s error message %s, params %s", url, ret.json()["message"], json.dumps(params, indent=4))
        return {}
    return ret.json()


class PeftSolutionRunBase(ABC):
    # lora解决方案云端solition run的需要的参数
    LORA_REMOTE_RUN_PARAMS_TEMPLATE = {
        "alg_task_id": "multi_task",
        "solution_id": "peft_demo",
        "batch_size": 32,
        "config_name": "conf/default.json",
        "dataset_id": "",
        "deploy_unit": "gbank",
        "image": "",
        "quota": {
            "cpu": 8,
            "memory": 81920,
            "gpu_count": 1
        },
        "solution_config": {},
        "submitor" : "",
        "target_step" : "",
        "labels": "custom.k8s.alipay.com/gpu-type-name:gpu_{}",
        "remote_model_path": "",
        "gpu_kubemaker_priority": ""
    }

    BENCHMARK_SERVER_URL = "http://zarkmeta.alipay.com/benchmark_server"

    def run(self, need_check_dataset=True, **kwargs):
        if need_check_dataset:
            ret = self.varify_dataset_has_request_file("test" if self.target_step == "predict" else "train",
                                                       self.dataset_id)
            if not ret:
                return ""
        ret, solution_info = self.set_up_run_config()
        if not ret:
            return ""
        url = os.path.join(self.BENCHMARK_SERVER_URL, "alg_solution/run")
        ret = post_request(url, solution_info).get("result", {})
        if not ret:
            return ""
        self.new_run_id = ret.get("run_id", "")
        glogger.info("new run_id is %s", self.new_run_id)
        return self.new_run_id

    def varify_dataset_has_request_file(self, file_name, dataset_id):
        if not dataset_id:
            glogger.error("dataset_id is empty")
            return False
        url = os.path.join(self.BENCHMARK_SERVER_URL, "dataset/list")
        ret = get_request(url, {"dataset_id": dataset_id}).get("result", [])
        if not ret or len(ret) != 1:
            glogger.error("dataset %s list result abnormal %s" , dataset_id, len(ret))
            return False
        return True if file_name in ret[0]["resources"] else False

    @abstractmethod
    def set_up_run_config(self):
        pass

    
class PeftSolutionRunPredict(PeftSolutionRunBase):
    def __init__(self, model, predictions, train_config, deep_config, llm_path, k8s_conf, **kwargs) -> None:

        self.old_run_id = ""
        self.dataset_id = ""
        self.target_step = "predict"
        self.run_info = {}
        self.new_run_id = ""
        self.remote_model_path = ""
        self.config_name = ""
        self.k8s_conf = k8s_conf
        self.solution_config = {
            "train_args": train_config,
            "llm": {
                "path": llm_path
            },
            "ds_config": deep_config
        }

        if os.path.isfile(predictions):
            with tempfile.TemporaryDirectory() as tmp_path:
                test_file = os.path.join(tmp_path, "test.jsonl")
                shutil.copy(predictions, test_file)
                self.dataset_id = dataset_upload_v2(tmp_path,
                                                    author=util.get_user_name(),
                                                    alg_task="supervised_finetune",
                                                    privilege='public',
                                                    channel="antllm",
                                                    origin="easy_upload")
        else:
            self.dataset_id = predictions

        self.old_run_id = model
        self.origin = kwargs.get("origin", "")
        super(PeftSolutionRunPredict, self).__init__()

    def get_run_info_from_run_id(self):
        ret = get_run(self.old_run_id)
        if not ret.is_ok():
            glogger.error("get run failed %s", ret.message)
            return ""
        return ret.result

    def get_config_by_run_context(self, run_info):
        try:
            remote_model_path = run_info.get("execute_context",
                                             {}).get("step_info", [])[0]["context"]["result"]["model"]
            config_name = run_info.get("execute_context", {}).get("params", {}).get("config", "") 
        except Exception as e:
            glogger.error("get info failed %s  %s", e, run_info)
            return "", ""
        return remote_model_path, config_name

    def set_up_run_config(self):
        run_info = self.get_run_info_from_run_id()
        if not run_info:
            return False, {}
        self.remote_model_path, self.config_name = self.get_config_by_run_context(run_info)
        if not self.remote_model_path or not self.config_name:
            return False, {}

        solution_info = copy.deepcopy(self.LORA_REMOTE_RUN_PARAMS_TEMPLATE)
        solution_info["remote_model_path"] = self.remote_model_path
        solution_info["config_name"] = self.config_name
        solution_info["dataset_id"] = self.dataset_id
        solution_info["submitor"] = util.get_user_name()
        solution_info["target_step"] = self.target_step
        solution_info["labels"] = solution_info["labels"].format(self.k8s_conf.gpu_type)
        solution_info["gpu_kubemaker_priority"] = self.k8s_conf.priority
        solution_info["solution_config"] = self.solution_config
        solution_info["origin"] = self.origin
        solution_info["channel"] = "antllm"
        glogger.info("solution config %s", json.dumps(solution_info, indent=4, ensure_ascii=False))
        return True, solution_info  


class PeftSolutionRunTrain(PeftSolutionRunBase):
    def __init__(self, dataset_id, solution_config, k8s_conf, **kwargs):
        self.dataset_id = dataset_id
        self.solution_config = solution_config
        self.gpu_count = k8s_conf.gpu_num
        self.target_step = "train"
        self.deploy_unit = k8s_conf.app_name
        self.batch_size = solution_config.get("train_args", {}).get("per_device_train_batch_size", 2)
        self.k8s_conf = k8s_conf
        self.origin = kwargs.get("origin", "")
        super(PeftSolutionRunTrain, self).__init__()

    def set_up_run_config(self):
        solution_info = copy.deepcopy(self.LORA_REMOTE_RUN_PARAMS_TEMPLATE)
        solution_info["solution_config"] = self.solution_config
        solution_info["dataset_id"] = self.dataset_id
        solution_info["submitor"] = util.get_user_name()
        solution_info["target_step"] = self.target_step
        solution_info["deploy_unit"] = self.deploy_unit
        solution_info["quota"]["gpu_count"] = self.gpu_count
        solution_info["quota"]["cpu"] = 12 * self.gpu_count
        solution_info["quota"]["memory"] = 100 * 1024 * self.gpu_count
        solution_info["quota"]["disk"] = 100 * 1024 * self.gpu_count
        solution_info["batch_size"] = self.batch_size
        solution_info["labels"] = solution_info["labels"].format(self.k8s_conf.gpu_type)
        solution_info["gpu_kubemaker_priority"] = self.k8s_conf.priority
        solution_info["origin"] = self.origin

        solution_info["channel"] = "antllm"
        solution_info["model"] = self.solution_config.get("llm", {}).get("path", "").rsplit("/", 1)[-1]

        glogger.info("solution config %s", json.dumps(solution_info, indent=4, ensure_ascii=False))
        return True, solution_info


def submit_aistudio_task_v2(train_args, k8s_conf, cmd,
                            tags_str: str = "basemodel=unknow_model,type=unknow_type,dev_pattern=AntNLP", **kwargs):
    from solutions.antllm.antllm.utils.aistudio_utils import IMAGE
    params = {
        "train_args": train_args,
        "cmd": cmd,
        "image": IMAGE,
        "submitor": util.get_user_name(),
        "extras": {"tags": tags_str}
    }
    params.update(vars(k8s_conf))
    params.update(kwargs)
    glogger.info("params %s", json.dumps(params, indent=4))

    url = os.path.join(PeftSolutionRunBase.BENCHMARK_SERVER_URL, "aistudio_train")
    ret = post_request(url, params).get("message", "")
    glogger.info("aistudio task id %s", ret)
    return ret


def notify_benchmark_server(run_id, submitor, message, status):
    url = os.path.join(PeftSolutionRunBase.BENCHMARK_SERVER_URL, "aistudio_commit_task")
    params = {
        "run_id": run_id,
        "submitor": submitor,
        "message": message,
        "status": status
    }
    glogger.info("params %s", json.dumps(params, indent=4))

    return get_request(url, params)
    