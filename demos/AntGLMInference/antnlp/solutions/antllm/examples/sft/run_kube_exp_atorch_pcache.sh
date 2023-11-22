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

kmcli run \
  --name tianxuan-glm10b-if-$resume_timestamp \
  --runtime 'pytorch' \
  --user 140807 \
  --rdma \
  --enable-host-network \
  --image 'reg.docker.alibaba-inc.com/aii/aistudio:4300121-20230808211027' \
  --master 'cpu=32,disk=1024000,memory=819200,gpu=8' \
  --worker 'cpu=32,disk=1024000,memory=819200,gpu=8' $num_workers \
  --app 'gbank' \
  --priority=high \
  --label 'kubemaker.alipay.com/enable-privilege=true' \
  --env PYTHONPATH=${output_dir},NCCL_DEBUG=INFO,NCCL_SOCKET_IFNAME=eth0,NCCL_IB_GID_INDEX=3,LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
  "if [ ! -d /workspace ]; then mkdir /workspace; fi; mount -t nfs -o vers=3,nolock,proto=tcp alipay-heyuan-31-bmc76.cn-heyuan-alipay.nas.aliyuncs.com:/  /workspace && cd /workspace/gushuwei/pcache/sft/ && sh start-pcache.sh && cd - && cd ${output_dir} && sh ${script} ${output_dir} ${resume_from_checkpoint}"
