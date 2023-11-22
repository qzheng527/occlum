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

USER=140807
IMAGE=acr-wulanchabu-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/gpu/antglm:sft0818

kmcli run --name "sft-a100-$USER-$resume_timestamp" \
--runtime 'pytorch' \
--rdma --enable-host-network \
--image $IMAGE \
--master 'cpu=92,memory=1094387,gpu=8,disk=2021070,shared_memory=16384'  \
--worker 'cpu=92,memory=1094387,gpu=8,disk=2021070,shared_memory=16384' $num_workers \
--input znjh-cpfs:/ \
--priority=high \
--app wulanchabu \
--user $USER \
"ldconfig && nvidia-smi && cd /input/sft/antnlp_master/ && sh ${script} ${output_dir} ${resume_from_checkpoint}"


# --label 'kubemaker.alipay.com/enable-privilege=true' \
