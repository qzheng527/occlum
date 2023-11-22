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
# torch_cmd="torchrun --nnodes=$WORLD_SIZE --nproc_per_node=$NUM_PROCESSES --node_rank $NODE_RANK --master_port $MASTER_PORT --master_addr $MASTER_ADDR"

pretrained=/mnt/tangjian.dtj/pretrained_models/glm-10b-2k-sft-v9-checkpoint-133230
output_dir=/mnt/tangjian.dtj/model/rw_model/glm-10b-2k-sft-v9-rm-v8-use-last-6-nodes
data_dir=/mnt/chatgpt/data/RM/v8


pretrained=/mnt/chatgpt/models_0602/sft/AntGLM-10B-SFT-Detoxcity-20230602
output_dir=/mnt/tangjian.dtj/model/rw_model/glm-10b-2k-sft-v9-detoxity-rm-v7-k8-100k-detoxity-freeze-2-use-mean-no-position-9-nodes
data_dir=/mnt/chatgpt/data/RM/v7_k8_100k_add_detoxity


pretrained=/mnt/chatgpt/experiments/tianxuan.jl/train_deepspeed_seq2seq_glm10b_2k_v12_bf16_old_code.sh/20230622-023018/epochs/checkpoint-88650/
output_dir=/mnt/tangjian.dtj/model/rw_model/glm-sft-v12-rm-v9-k6-100k-fix-ID-freeze-2-use-mean-no-position-no-normalized
data_dir=/mnt/chatgpt/data/RM/v9_k6_100k_fix_ID

log_dir=${output_dir}/logs

if [ $NODE_RANK -eq 0 ]; then
    rm -rf ${log_dir}/*
    mkdir -p ${log_dir}
    rm -rf ${output_dir}/runs/
fi


$torch_cmd \
    train_rm.py \
    --deepspeed ds_config.json \
    --dataset_dir $data_dir \
    --model_name_or_path $pretrained \
    --output_dir $output_dir \
    --do_train \
    --do_eval \
    --bf16 \
    --bf16_full_eval \
    --max_len 2048 \
    --max_input_len 1024 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --eval_accumulation_steps 10 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 5 \
    --logging_steps 20 \
    --learning_rate 5e-6 \
    --warmup_ratio 0.1 \
    --model_type glm \
    --overwrite_output_dir \
    --report_to tensorboard \
    --no_shuffle_dataloader \
    --mask_type '[gMASK]' \
    --truncation_side 'left' \
    --use_mean_value false \
    --use_position_id true \
    --use_normalized_reward false \
    --num_layers_unfrozen 2 \
    --dynamic_padding \
    --ddp_timeout 3600 \
    2>&1 | tee ${log_dir}/log-$NODE_RANK.txt