#!/usr/bin/env bash
###############################################
# File Name: train_colo_gpt.sh
# Author: tianxuan.jl
# mail: tianxuan.jl@antgroup.com
# Created Time: Tue 07 Mar 2023 02:58:26 PM CST
# Description: 基于deepspeed和peft运行chatgpt6B的脚本，支持16g显存下110最大输入长度和2batch size的finetune
###############################################

CUDA_VISIBLE_DEVICES=0

set -x

ds_config_path=$1

WORLD_SIZE=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-12345}

output_dir=$2/`basename $0`/$(date "+%Y%m%d-%H%M%S")
mkdir -p $output_dir
log_file=$output_dir/log.txt
echo $output_dir
cp -r $ds_config_path $output_dir
cp -r $0 $output_dir
echo $output_dir > $log_file

if [ $WORLD_SIZE -gt 1 ]; then
  deepspeed_cmd=deepspeed --num_nodes=$WORLD_SIZE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT
else
  deepspeed_cmd=deepspeed 
fi

$deepspeed_cmd --include localhost:0 \
`dirname $0`/../../antllm/commands/sft/train_deepspeed.py \
  --deepspeed $ds_config_path \
  --train_data '/workspace/data/llm/train.jsonl' \
  --test_data '/workspace/data/llm/test.jsonl' \
  --pretrained_model_name_or_path '/workspace/weights/glm5b-sft' \
  --lm_type seq2seq \
  --peft_type lora \
  --output_dir $output_dir \
  --do_train \
  --do_eval \
  --fp16 \
  --no_save_deepspeed_checkpoint \
  --max_length 110 \
  --max_input_length 50 \
  --max_output_length 50 \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --save_steps 1 \
  --eval_steps 10 \
  --eval_accumulation_steps 10 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 1 \
  --logging_steps 50 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.1 \
  --overwrite_output_dir  2>&1 | tee $log_file
