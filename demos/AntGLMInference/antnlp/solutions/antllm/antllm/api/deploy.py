# coding=utf-8
# @Date: 2023-06-15
import os
import json
import requests
import logging
import adabench.core.run as adabench_run
import adabench.impl.run_impl as run_util
import adabench.utils.util as adabench_util
from solutions.antllm.antllm.utils.benchmark import get_request
from enum import Enum
from pydantic import BaseModel
from .error import TriggerDeployError, GetBaseLLMError, GetDeployStatusError, BaseLLMNotExistError
from typing import Union
BENCHMARK_SERVER_HOST = 'zarkmeta.alipay.com'
glogger = logging.getLogger()


class GPUType(Enum):
    A100 = "PHYSICAL-GPU-A100"
    A10 = "PHYSICAL-GPU-A10"
    Default = ''  # 根据训练的模型自动推导


class DeployStatus(Enum):
    running = "running"
    success = "success"
    fail = "fail"


class DeployParams(BaseModel):
    scene_name: str = ''  # 场景名，用于生成http/client调用地址
    version: str = ''  # maya版本，用于生成http/client调用地址
    biz_domain: str = ''  # aistudio租户
    gpu_type: GPUType = GPUType.Default  # GPU型号
    pre_idc: str = ''  # 预发机房，一般填写一个机房
    prod_idc: str = ''  # 生产机房，一个到多个，多个机房用逗号分隔
    replica: int = 1  # 实例个数
    worker_count: int = 1  # 单实例下python worker个数

    """基座模型的名称。如果是自定义基座部署或本地训练的LORA部署，需要填写此参数。
    基座填写方式可以为：
    1. nas地址，例如 nas://domain-name:/path-to-llm
    2. 可被wget下载的http/https地址，例如 https://domain-name/path-to-llm/model.tar(or .tar.gz)
    """
    base_llm: str = ''


class DeployInfo(BaseModel):
    status: DeployStatus  # 部署状态
    message: str  # 部署查询信息
    service_id: str  # 服务ID   
    scene_name: str  # 场景名
    version: str  # 版本
    detail: dict  # 部署过程详细信息


class DeployManager():
    def deploy_base(self, base_llm: str, scene_name: str, version: str, biz_domain: str,
                    gpu_type: Union[GPUType, str], pre_idc: str = '', prod_idc: str = '',
                    gpu_count: int = 1, replica: int = 1, worker_count: int = 1,
                    app_path: str = 'app/antglm/api_template', **kwargs):
        """
        部署自定义基座大模型到maya
        :param base_llm, 必填，部署时使用的基座模型远端地址。基座只支持nas盘和http(s)两种云端模式。
                         nas盘格式: nas://{domain-name}:/path/to/llm
                         http(s)格式： http://{domain-name}/path/to/llm.tar 文件名任意取，只能为.tar或.tar.gz包，
                                      包内可直接存放大模型文件或包含一个目录，大模型文件放置于子目录内。
        :param scene_name, 必填，部署后的场景名
        :param version， 必填，部署后的版本名
        :param biz_domain，必填，部署的aistudio租户名英文ID，可通过aistudio IDE右上角我的租户查看
        :param gpu_type，必填，部署的GPU卡类型, 可以是GPUType类型, 可以是str类型, 当是str类型时可以由用户指定显卡型号
        :param pre_idc，选填，部署的预发机房, 默认会从空闲机房自动选择IDC
        :param prod_idc，选填，部署的生产机房，默认会从空闲机房自动选择IDC
        :param gpu_count，选填，部署的卡个数，默认1
        :param replica，选填，部署实例个数，默认1
        :param worker_count，选填，每个实例下python worker进程个数，默认1
        :param app_path, 选填，部署使用的app代码
        :return str, 部署task id:

        部署自定义基座大模型到zark(默认支持a10双卡部署10b大模型+TGI镜像推理加速)
        :param base_llm, 必填,部署时使用的基座模型远端地址。基座http(s)云端模式。
                         只需要是个可以下载的地址即可, 基座模型需要支持safe tensor转换
        :param scene_name, 必填, 映射为zark app name
        :param version, 必填, 映射为zark instance name
        :param biz_domain, 必填, 映射为zark deploy unit
        :param gpu_type, 必填，部署的GPU卡类型
        :param pre_idc, 无效参数, 可不填
        :param prod_idc, 必填, 映射为zark idcs, 指本次部署的生产机房
        :param gpu_count，选填，部署的卡个数，默认1
        :param replica，选填，部署实例个数，默认1
        :param worker_count，选填，每个实例下python worker进程个数，默认1
        :param **kwargs, 扩展参数, 详细见README.md
        """
        if kwargs.get("platform") and kwargs["platform"] == "zark":
            return self.deploy_zark(base_llm, scene_name, version, biz_domain,
                                    gpu_type, prod_idc, gpu_count, replica,
                                    worker_count, **kwargs)  
        if base_llm.find('://') == -1:
            raise Exception('base_llm ONLY support NAS or http(s) URI')
        else:
            glogger.info(f'Deploy remote {base_llm} model to maya')

        post_json = {
            'base_llm': base_llm,
            'scene_name': scene_name,
            'version': version,
            'biz_domain': biz_domain,
            'gpu_type': gpu_type if isinstance(gpu_type, str) else gpu_type.value,
            'gpu_count': gpu_count,
            'pre_idc': pre_idc,
            'prod_idc': prod_idc,
            'replica': replica,
            'worker_count': worker_count,
            'app_path': app_path,
            'user': adabench_util.get_user_name(),
            'channel': 'antllm',
            'origin': 'DeployManager.deploy'
        }
        url = f'http://{BENCHMARK_SERVER_HOST}/benchmark_server/deploy_aistudio'
        response = requests.post(url=url, json=post_json)
        if response.status_code != 200 or response.json()['code'] != 0:
            msg = 'trigger aistudio deploy failed:{}'.format(response.text)
            raise TriggerDeployError(msg)
        glogger.info("https://aistudio.alipay.com/project/job/detail/%s", response.json()['result']['task_id'])
        return response.json()['result']['task_id']

    def deploy(self, 
               model: str,
               adapter_name: str,
               params: DeployParams = DeployParams(),
               app_path: str = 'app/antglm/api_template'):
        """
        部署云端训练产出的lora权重到maya
        :param model, 必填，本地模型路径，或则云端运行的run_id
        :param adapter_name, 必填，部署后的adapter name
        :param params, 部署参数，默认使用公共底座
        :param base_llm, 部署时使用的公共基座模型名，如果是私有的大模型底座，请填写对应的scene_name, version,
        :param app_path, 部署使用的app代码，一般不用变
        :return str, 部署task id:
        """
        base_llms = self.get_base_llms()
        # 公共底座lora更新
        if params.base_llm:
            params.scene_name, params.version = self.get_scene_name_and_version(params.base_llm, base_llms)
        elif not os.path.isdir(model) and not params.scene_name and not params.version:
            params.base_llm = self.get_llm_from_run_id(model)
            if not params.base_llm:
                raise BaseLLMNotExistError(f'base llm not exist in {model} context')
            params.scene_name, params.version = self.get_scene_name_and_version(params.base_llm, base_llms)
        # 存在scene_name和version为私有底座更新
        else:
            if not params.scene_name or not params.version:
                raise Exception("must have biz_domain and scene_name and version when using private llm")

        if os.path.isdir(model):
            glogger.info(f'Deploy local {model} to maya.')
            adapter_config = os.path.join(model, 'adapter_config.json')
            with open(adapter_config, 'r') as fp:
                base_llm = os.path.basename(json.load(fp)['base_model_name_or_path'])
            run = adabench_run.Run.new_run({})
            run.execute_context = {
                'base_llm': base_llm
            }
            run_id = run.run_id
            run_util.run_upload(run_id, model)
        else:
            glogger.info(f'Deploy remote {model} model to maya')
            run_id = model

        post_json = json.loads(params.json())
        post_json['user'] = adabench_util.get_user_name()
        post_json['peft_model'] = [{
            'adapter_name': adapter_name,
            'model': run_id
        }]
        post_json['app_path'] = app_path

        post_json['channel'] = "antllm"
        post_json["origin"] = "DeployManager.deploy"

        url = f'http://{BENCHMARK_SERVER_HOST}/benchmark_server/update_lora'
        response = requests.post(url=url, json=post_json)
        if response.status_code != 200 or response.json()['code'] != 0:
            msg = 'trigger aistudio deploy failed:{}'.format(response.text)
            raise TriggerDeployError(msg)
        return response.json()['result']['task_id']

    def deploy_status(self, task_id: str):
        """
        查询云端部署任务
        :param task_id, 部署任务id
        :return DeployInfo
        """
        service_id = ''
        scene_name = ''
        version = ''
        url = f'http://{BENCHMARK_SERVER_HOST}/meta_server/meta_task_status'
        response = requests.get(url=url, params={
            'task_id': task_id
        })

        if response.status_code != 200 or response.json()['code'] != 0:
            msg = 'get aistudio deploy status failed:{}'.format(response.text)
            raise GetDeployStatusError(msg)
        
        if response.json()['result']['status'] == 'fail':
            status = DeployStatus.fail
        elif response.json()['result']['status'] == 'success':
            status = DeployStatus.success
            service_id = response.json()['result']['task_context']['step_info'][-1]['result']['service_id']
            scene_name = response.json()['result']['task_context']['step_info'][-1]['result']['scene_name']
            version = response.json()['result']['task_context']['step_info'][-1]['result']['version']
        else:
            status = DeployStatus.running

        return DeployInfo(
            status=status,
            message=response.json()['result']['message'].encode().decode("unicode_escape"),
            detail=response.json()['result'],
            service_id=service_id,
            scene_name=scene_name,
            version=version
        )

    def get_base_llms(self):
        """
        查询基座模型信息
        :return dict 基座模型信息，其中key就是基座模型名称。scene_name/version就是默认部署的基座模型服务名称。例如:
        {
            "glm10b_rlhf_20230602": {
                "ais_public_service": {
                    "scene_name": "glm10b_rlhf_20230602",
                    "version": "v1"
                },
                "image": "reg.docker.alibaba-inc.com/aii/aistudio:aistudio-102991717-855067390-1687184787085"
            },
            "glm10b_sft_20230602": {
                "ais_public_service": {
                    "scene_name": "glm10b_sft_20230602",
                    "version": "v1"
                },
            "image": "reg.docker.alibaba-inc.com/aii/aistudio:aistudio-102276384-662218608-1687183793710"
            }
        }
        """
        url = f'http://{BENCHMARK_SERVER_HOST}/benchmark_server/base_llm_info'
        return get_request(url)["result"]

    def get_llm_from_run_id(self, model):
        url = 'http://{}/benchmark_server/run/{}'.format(BENCHMARK_SERVER_HOST, model)
        result = get_request(url).get("result")
        if not result:
            msg = 'get base llm info failed, run_id {}'.format(model)
            raise GetBaseLLMError(msg)
        return result.get('execute_context', {}).get('base_llm', '')

    def get_scene_name_and_version(self, llm, base_llms):
        if llm not in base_llms:
            raise BaseLLMNotExistError(f'base llm name {llm} not in {base_llms}')
        scene_name = base_llms[llm]["ais_public_service"]["scene_name"]
        version = base_llms[llm]["ais_public_service"]["version"]
        return scene_name, version

    def deploy_zark(self, model: str, app_name: str, instance_name: str,
                    deploy_unit: str, gpu_device: GPUType, idcs: str, 
                    gpu_device_count: int = 1, replica_num: int = 1, worker_count: int = 1,
                    **kwargs):
        if not model.startswith('http://') and not model.startswith('https://'):
            raise Exception('base llm only support http format when deploy to zark')
        gpu_device = "gpu_" + gpu_device.name.lower() if gpu_device.name != "Default" else "gpu_a10"
        
        post_json = {
            'platform': "zark",
            'model': model,
            'app_name': app_name,
            'instance_name': instance_name,
            'deploy_unit': deploy_unit,
            'gpu_device': gpu_device,
            'cpu_count': kwargs.get("cpu_count", 20),
            'mem_count': kwargs.get("mem_count", 51200),
            'disk_count': kwargs.get("disk_count", 100),
            'gpu_device_count': gpu_device_count,
            'idcs': idcs.split(","),
            'replica_num': replica_num,
            'worker_count': worker_count,
            'user': adabench_util.get_user_name(),
            'channel': 'antllm',
            'origin': 'DeployManager.deploy_zark'
        }
        post_json.update(**kwargs)
        url = f'http://{BENCHMARK_SERVER_HOST}/benchmark_server/deploy'
        response = requests.post(url=url, json=post_json)
        if response.status_code != 200 or response.json()['code'] != 0:
            msg = 'trigger aistudio deploy failed:{}'.format(response.text)
            raise TriggerDeployError(msg)
        return response.json()['result']['pmas_addr']