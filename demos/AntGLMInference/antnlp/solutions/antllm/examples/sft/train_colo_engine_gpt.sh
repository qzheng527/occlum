#!/usr/bin/env bash
###############################################
# File Name: train_colo_engine.sh
# Author: tianxuan.jl
# mail: tianxuan.jl@antgroup.com
# Created Time: Tue 07 Mar 2023 02:58:26 PM CST
# Description: colossalai gpt的脚本
###############################################

config_path=$1

WORLD_SIZE=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-12345}

output_dir=$2/`basename $config_path`/$(date "+%Y%m%d-%H%M%S")
mkdir -p $output_dir
log_file=$output_dir/log.txt
echo $output_dir
echo $output_dir > $log_file

if [ $WORLD_SIZE -gt 1 ]; then
  torchrun  --nnodes $WORLD_SIZE --node_rank $NODE_RANK --nproc_per_node=auto --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    `dirname $0`/train_colo_engine.py \
    --lm_type causal \
    --config $config_path \
    --output_dir $output_dir \
    --from_torch > $log_file 2>&1
else
  torchrun  --nproc_per_node=auto \
    `dirname $0`/train_colo_engine.py \
    --lm_type causal \
    --config $config_path \
    --output_dir $output_dir \
    --from_torch > $log_file 2>&1
fi
