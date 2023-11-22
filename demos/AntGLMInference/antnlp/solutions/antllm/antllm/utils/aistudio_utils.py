# -*- coding: utf-8 -*-
import os
import json
import tempfile
from typing import Dict
from pypai.job import PythonJobBuilder
from pypai.conf import ExecConf, KMConf


# 大模型官方镜像
IMAGE = 'reg.docker.alibaba-inc.com/aii/aistudio:400318-20230717171959'

# from pypai.constants.file_constants import FileConstants
# aistudio任务提交会打包本地文件，默认包限制大小为50M，设为1G
# FileConstants.MAX_DATA_FILE_SIZE = 1024 * 1024 * 1024


class AntLLMk8sConf:
    def __init__(self, app_name='gpudefault', gpu_num=1, priority='low', gpu_type='a100', init_command=None,
                 cluster='', host_network=True, rdma=True, memory=None):
        """
        默认走公共池低保A100，提交方式参考 https://yuque.antfin-inc.com/aii/aistudio/lbqgq7ky7z1gcuzq
        app_name: 不填 或者 gpudefault
        priority: low
        cluster: 不填
        gpu_type: a100
        """
        self.app_name = app_name
        self.gpu_num = gpu_num
        self.priority = priority
        self.gpu_type = gpu_type.lower()
        self.init_command = init_command
        self.cluster = cluster.lower()
        self.host_network = host_network
        self.rdma = rdma
        self.memory = memory


def parse_resource_config(k8s_conf: AntLLMk8sConf):
    '''
    资源设置: cpu gpu mem disk 大小. 单位分别为 个 个 Mb Mb 代表job最大占用资源
    单机最小内存分配
    GPU机器资源配比: https://yuque.antfin-inc.com/aii/aistudio/gpumachine
    '''
    CPUS_PER_GPU = 12
    MEM_PER_GPU = 100 * 1024
    DISK_PER_GPU = 25 * 1024
    MIN_DISK_VOLUME = 100 * 1024
    GPUS_PER_NODE = 8
    if k8s_conf.cluster.upper() == 'ET15' and k8s_conf.gpu_type.upper() == 'P100':
        GPUS_PER_NODE = 2
    if k8s_conf.cluster.upper() == 'ET15' and k8s_conf.gpu_type.upper() == 'V100':
        MEM_PER_GPU = 60 * 1024
    if k8s_conf.cluster.upper() == 'STL':
        MEM_PER_GPU = 90 * 1024

    if k8s_conf.gpu_num > GPUS_PER_NODE:
        mem = GPUS_PER_NODE * MEM_PER_GPU
        cpu_num = GPUS_PER_NODE * CPUS_PER_GPU
        disk_m = max(GPUS_PER_NODE * DISK_PER_GPU, MIN_DISK_VOLUME)
        if k8s_conf.gpu_num > 2 * GPUS_PER_NODE:
            # 分布式训练大于2*GPUS_PER_NODE卡：gpu数需为GPUS_PER_NODE的倍数，即整机使用
            assert k8s_conf.gpu_num % GPUS_PER_NODE == 0, f'gpu num should divide by {GPUS_PER_NODE} in distribute mode'
            worker_num = k8s_conf.gpu_num / GPUS_PER_NODE - 1
            master = ExecConf(
                num=1, memory=mem, cpu=cpu_num, gpu_num=GPUS_PER_NODE, gpu_type=k8s_conf.gpu_type, disk_m=disk_m)
            worker = ExecConf(
                num=worker_num, memory=mem, cpu=cpu_num,
                gpu_num=GPUS_PER_NODE, gpu_type=k8s_conf.gpu_type, disk_m=disk_m)
        else:
            # 分布式训练大于GPUS_PER_NODE卡小于2*GPUS_PER_NODE卡：master使用GPUS_PER_NODE卡，worker使用剩余卡，全量微调内存使用量200G+
            worker_mem = (k8s_conf.gpu_num - GPUS_PER_NODE) * MEM_PER_GPU
            worker_cpu_num = (k8s_conf.gpu_num - GPUS_PER_NODE) * CPUS_PER_GPU
            worker_disk_m = max((k8s_conf.gpu_num - GPUS_PER_NODE) * DISK_PER_GPU, MIN_DISK_VOLUME)
            master = ExecConf(
                num=1, memory=mem, cpu=cpu_num, gpu_num=GPUS_PER_NODE, gpu_type=k8s_conf.gpu_type, disk_m=disk_m)
            worker = ExecConf(
                num=1, memory=worker_mem, cpu=worker_cpu_num,
                gpu_num=k8s_conf.gpu_num - GPUS_PER_NODE, gpu_type=k8s_conf.gpu_type, disk_m=worker_disk_m)
    else:
        # 每个gpu对应 100G内存、12个cpu、25G磁盘
        mem = k8s_conf.gpu_num * MEM_PER_GPU
        cpu_num = k8s_conf.gpu_num * CPUS_PER_GPU
        disk_m = max(k8s_conf.gpu_num * DISK_PER_GPU, MIN_DISK_VOLUME)
        master = ExecConf(
            num=1, memory=mem, cpu=cpu_num, gpu_num=k8s_conf.gpu_num, gpu_type=k8s_conf.gpu_type, disk_m=disk_m
        )
        worker = None

    return master, worker


def submit_aistudio_task(train_config: Dict, k8s_conf: AntLLMk8sConf, cmd: str) -> None:
    master, worker = parse_resource_config(k8s_conf)
    # 更多job参数见 https://aistudio.alipay.com/doc/python_job.html
    km_conf = KMConf(image=IMAGE, cluster=k8s_conf.cluster)
    # 打包训练配置
    with tempfile.TemporaryDirectory() as temp_dir:
        train_config_path = os.path.join(temp_dir, 'train_config.json')
        with open(train_config_path, 'w') as fo:
            json.dump(train_config, fo, ensure_ascii=False, indent=4)
        builder = PythonJobBuilder(
            rdma=k8s_conf.rdma,
            host_network=k8s_conf.host_network,
            source_root=temp_dir,
            command=cmd,
            main_file='',
            master=master,
            worker=worker,
            km_conf=km_conf,
            name='antllm_training',
            k8s_app_name=k8s_conf.app_name,
            k8s_priority=k8s_conf.priority,
            runtime='pytorch')
        # 任务发起后直接退出
        record_id = builder.run(enable_wait=False)
        return record_id


