#!/usr/bin/env bash
###############################################
# File Name: run_kube.sh
# Author: Liang Jiang
# mail: jiangliang0811@gmail.com
# Created Time: Sat 25 Mar 2023 05:51:30 PM CST
# Description:-
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

USER=140807
IMAGE=acr-wulanchabu-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/gpu/antglm:sft0818

kmcli run --name "sft-$resume_timestamp-`echo $RANDOM |cksum |cut -c 1-8`" \
--no-master \
--runtime 'elasticdl' \
--elasticdl-args "--distribution_strategy=AllreduceStrategy --relaunch_always --volume 'host_path=/etc/sysconfig/rdma/cluster,mount_path=/etc/sysconfig/rdma/cluster;host_path=/cpfs01/accl/efm/rprobe,mount_path=/usr/bin/rprobe'" \
--rdma --enable-host-network \
--image $IMAGE \
--worker 'cpu=92,memory=1094387,gpu=8,disk=2021070,shared_memory=16384' $num_workers \
--input znjh-cpfs:/ \
--priority=high \
--app wulanchabu \
--user $USER \
"ldconfig && nvidia-smi && source /input/accl/accl.sh && cd ${output_dir} && export PYTHONPATH=${output_dir} && sh ${script} ${output_dir} ${resume_from_checkpoint}"
