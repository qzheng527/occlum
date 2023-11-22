#!/usr/bin/env bash

set -x

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64

WORLD_SIZE=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}
NODE_NAME=`echo $ILOGTAIL_PODNAME | awk -F 'ptjob-' '{print $2}'`
NODE_NAME=${NODE_NAME:-master-0}

NUM_PROCESSES=`echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}'`
if [ $NUM_PROCESSES -eq 0 ]; then
NUM_PROCESSES=`echo $NVIDIA_VISIBLE_DEVICES | awk -F ',' '{print NF}'`
fi

NUM_PROCESSES=`expr $NUM_PROCESSES - 1`
RM_DEVICE=${NUM_PROCESSES}

NUM_PROCESSES=`expr $NUM_PROCESSES \* $WORLD_SIZE`

# rl_model_path=/mnt/xiaohao.wzh/sft/glm-10b-2k-v9-bf16-20230506-140819-checkpoint-133230
rl_model_path=
rw_model_path=
cost_model_path=
prompt_path=
output_dir=
log_dir=${output_dir}/logs

if [ $NODE_RANK -eq 0 ]; then
    rm -rf ${output_dir}/logs/*
    mkdir -p ${log_dir}
fi

export PYTHONPATH=$PYTHONPATH:"../../../../../"


accelerate launch \
  --use_deepspeed \
  --num_machines ${WORLD_SIZE} \
  --num_processes ${NUM_PROCESSES} \
  --machine_rank $NODE_RANK \
  --deepspeed_config_file configs/ds_config_trlx_bf16.json \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --deepspeed_multinode_launcher "standard" \
  --same_network \
  train_ppo_reward_shaping.py \
  --rm_model_path $rw_model_path \
  --cost_model_path $cost_model_path \
  --lambda_value -0.2 \
  --ppo_model_path $rl_model_path \
  --prompt_path $prompt_path \
  --exp_cfg_path exps/exp_10b_10b_separate_multi_nodes_reward_shaping.yml \
  --save_dir ${output_dir} \
  --log_dir ${log_dir} \
  --mask_type '[gMASK]' \
  --rw_device $RM_DEVICE \
  --val_size 200 \
  --rm_mean_value \
  --rm_use_normalized_cost \
  2>&1 | tee $log_dir/log-${NODE_RANK}.txt
