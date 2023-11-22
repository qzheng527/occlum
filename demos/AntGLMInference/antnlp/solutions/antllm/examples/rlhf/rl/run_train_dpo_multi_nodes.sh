#!/usr/bin/env bash

set -x

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
export PYTHONPATH=$PYTHONPATH:"../../../../../"

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

torch_cmd="python -m torch.distributed.run --nnode=$WORLD_SIZE --nproc_per_node=$NUM_PROCESSES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"

pretrained_model_name_or_path=/mnt/chatgpt/experiments/tianxuan.jl/train_deepspeed_seq2seq_glm10b_2k_v9_add_math_toxicity_bf16_old_code.sh/20230623-000804/epochs/checkpoint-64512
reference_model_name_or_path=/mnt/chatgpt/experiments/tianxuan.jl/train_deepspeed_seq2seq_glm10b_2k_v9_add_math_toxicity_bf16_old_code.sh/20230623-000804/epochs/checkpoint-64512
output_dir=/mnt/xiaohao.wzh/rlhf/model/ckpt_10b_2k_v9_add_math_toxicity_rm-v8-k6-100k-detoxity-use-mean-no-position-bf16-detoxicity-dpo-3-nodes
train_data=/mnt/xiaohao.wzh/rm/data/v8_k6_toxicity_100k/train.jsonl
test_data=/mnt/xiaohao.wzh/rm/data/v8_k6_toxicity_100k/test.jsonl

timestamp=$1
output_dir=$output_dir/$timestamp

log_dir=${output_dir}/logs

if [ $NODE_RANK -eq 0 ]; then
    rm -rf ${log_dir}/*
    mkdir -p ${log_dir}
    rm -rf ${output_dir}/runs/
fi


$torch_cmd \
train_dpo.py \
--deepspeed configs/ds_config_bf16.json  \
--train_data $train_data \
--test_data $test_data \
--pretrained_model_name_or_path $pretrained_model_name_or_path \
--reference_model_name_or_path $reference_model_name_or_path \
--beta_coef 0.2 \
--report_to tensorboard \
--log_on_each_node false \
--no_save_deepspeed_checkpoint true \
--weight_decay 0.1 \
--output_dir $output_dir \
--resume_from_checkpoint false \
--do_train \
--do_eval \
--bf16 \
--bf16_full_eval \
--max_length 2048 \
--max_input_length 1024 \
--max_output_length 1024 \
--evaluation_strategy "epoch" \
--save_strategy "epoch" \
--save_steps 5000 \
--eval_steps 50000 \
--eval_accumulation_steps 10 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 2 \
--max_grad_norm 10.0 \
--num_train_epochs 10 \
--logging_steps 50 \
--learning_rate 5e-7 \
--warmup_steps 150 2>&1 | tee ${log_dir}/log-$NODE_RANK.txt