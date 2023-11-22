#!/usr/bin/env bash
###############################################
# File Name: train_colo_gpt.sh
# Author: tianxuan.jl
# mail: tianxuan.jl@antgroup.com
# Created Time: Tue 07 Mar 2023 02:58:26 PM CST
# Description: colossalai gpt的脚本
###############################################

set -x

ds_config_path=`dirname $0`/configs/ds_config_fp16.json

WORLD_SIZE=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-12346}

output_dir=$1/`basename $0`/$(date "+%Y%m%d-%H%M%S")
mkdir -p $output_dir
log_file=$output_dir/log.txt
echo $output_dir
cp -r $ds_config_path $output_dir
cp -r $0 $output_dir
cp -r solutions/antllm $output_dir
echo $output_dir > $log_file

NUM_PROCESSES=`echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}'`
if [ $NUM_PROCESSES -eq 0 ]; then
NUM_PROCESSES=`echo $NVIDIA_VISIBLE_DEVICES | awk -F ',' '{print NF}'`
fi

deepspeed_cmd="python -m torch.distributed.run --nnode=$WORLD_SIZE --nproc_per_node=$NUM_PROCESSES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"

$deepspeed_cmd \
`dirname $0`/../../antllm/commands/sft/train_deepspeed.py \
  --deepspeed $ds_config_path \
  --pretrained_model_name_or_path '/workspace/chatgpt/pretrained_models/glm-10b-ant-addcode/' \
  --train_data '/workspace/tianxuan.jl/code/antnlp/data/v9/SFT_train_v9_clean.json' \
  --test_data '/workspace/tianxuan.jl/code/antnlp/data/v9/SFT_val_v9_clean.json' \
  --lm_type seq2seq \
  --save_on_each_node false \
  --log_on_each_node false \
  --report_to tensorboard \
  --weight_decay 0.1 \
  --output_dir $output_dir \
  --do_train \
  --do_eval \
  --fp16 \
  --no_save_deepspeed_checkpoint \
  --max_length 1024 \
  --max_input_length 500 \
  --max_output_length 500 \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --save_steps 5000 \
  --eval_steps 50000 \
  --eval_accumulation_steps 10 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 10 \
  --logging_steps 50 \
  --learning_rate 3e-6 \
  --warmup_ratio 0.06 \
  --overwrite_output_dir  2>&1 | tee $log_file

