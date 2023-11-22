#!/usr/bin/env bash
###############################################
# File Name: run_kube.sh
# Author: Liang Jiang
# mail: jiangliang0811@gmail.com
# Created Time: Sat 25 Mar 2023 05:51:30 PM CST
# Description: 
###############################################

set -e
set -x


script=$1
output_dir=$2
num_workers=$3

resume_timestamp=$4
if [ $resume_timestamp"x" == "x" ]; then
  resume_timestamp=$(date "+%Y%m%d-%H%M%S")
  resume_from_checkpoint=false
else
  resume_from_checkpoint=true
fi
output_dir=$output_dir/`basename $script`/$resume_timestamp

if [ $resume_from_checkpoint == "false" ]; then
  mkdir -p $output_dir/solutions
  cp -r solutions/antllm $output_dir/solutions
fi

kmcli run --name tianxuan-glm10b-if-$(date "+%Y%m%d-%H%M%S") --runtime 'pytorch' --user 140807 --rdma --enable-host-network --image 'reg.docker.alibaba-inc.com/atorch/atorch-dev:20230619torch210dev20230613cu118' --master 'cpu=32,disk=102400,memory=819200,gpu=8,gpu_type=a100' --worker 'cpu=32,disk=102400,memory=819200,gpu=8,gpu_type=a100' $num_workers --app 'gbank' --priority=high --env PYTHONPATH=${output_dir},NCCL_DEBUG=INFO,NCCL_SOCKET_IFNAME=eth0,NCCL_IB_GID_INDEX=3,LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 "if [ ! -d /workspace ]; then mkdir /workspace; fi; mount -t nfs -o vers=3,nolock,proto=tcp alipay-heyuan-31-bmc76.cn-heyuan-alipay.nas.aliyuncs.com:/  /workspace && cd ${output_dir} && pip install -U --no-deps http://alps-common.oss-cn-hangzhou-zmf.aliyuncs.com/users/sichuan/dist/atorch-0.1.7rc11-py3-none-any.whl && sh ${script} ${output_dir} ${resume_from_checkpoint}"
