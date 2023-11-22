
#!/usr/bin/env bash
###############################################
# File Name: train_colo_gpt.sh
# Author: tianxuan.jl
# mail: tianxuan.jl@antgroup.com
# Created Time: Tue 07 Mar 2023 02:58:26 PM CST
# Description: colossalai gpt的脚本
###############################################


set -o pipefail

set -x

mkdir -p /root/.cache/huggingface/hub
echo 1 > /root/.cache/huggingface/hub/version.txt

pip install -U --no-deps http://alps-common.oss-cn-hangzhou-zmf.aliyuncs.com/users%2Flizhi%2Fatorch-0.18.0.dev0%2Bfix.fsdp.prefix-py3-none-any.whl
curl http://alps-common.oss-cn-hangzhou-zmf.aliyuncs.com/users%2Flizhi%2Fpatch-fsdp.sh | sh -


WORLD_SIZE=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
RANDOM_PORT=$[$RANDOM + 20000]
MASTER_PORT=${MASTER_PORT:-$RANDOM_PORT}

NODE_NAME=`echo $ILOGTAIL_PODNAME | awk -F 'ptjob-' '{print $2}'`
NODE_NAME=${NODE_NAME:-master-0}
output_dir=$1
resume_from_checkpoint=true
# output_dir=$1/$NODE_NAME
mkdir -p $output_dir
if [ $NODE_RANK == 0 ]; then
  if [ -d $output_dir ]; then
    echo 'Exist: ', $output_dir
  else
    echo 'Not exist: ', $output_dir
  fi
  log_file=$output_dir/log-$NODE_NAME.txt
  echo $output_dir
  cp -r $0 $output_dir
  cur_time=$(date "+%Y%m%d-%H%M%S")
  echo $cur_time >> $log_file
  echo $output_dir >> $log_file
fi

NUM_PROCESSES=`echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}'`
if [ $NUM_PROCESSES -eq 0 ]; then
NUM_PROCESSES=`echo $NVIDIA_VISIBLE_DEVICES | awk -F ',' '{print NF}'`
fi

deepspeed_cmd="python -m atorch.distributed.run --fault_tolerant --max_restarts=0 --nnode=$WORLD_SIZE --nproc_per_node=$NUM_PROCESSES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --log_dir=/tmp"

while :
do
  nvidia-smi
  $deepspeed_cmd \
  `dirname $0`/../../antllm/commands/sft/train_atorch.py \
    --train_data <path-of-your-train-data> \
    --test_data <path-of-your-test-data> \
    --pretrained_model_name_or_path <path-of-your-pretrained-model> \
    --lm_type chatglm2 \
    --max_grad_norm 1 \
    --report_to tensorboard \
    --log_on_each_node false \
    --weight_decay 0.1 \
    --output_dir $output_dir \
    --resume_from_checkpoint $resume_from_checkpoint \
    --dynamic_padding \
    --bf16 \
    --do_train \
    --do_eval \
    --max_length 2048 \
    --max_input_length 1024 \
    --max_output_length 1024 \
    --evaluation_strategy "epoch" \
    --save_policy "interval" \
    --extra_save_by_epoch \
    --save_total_limit 1 \
    --save_steps 2000 \
    --eval_steps 100 \
    --eval_accumulation_steps 10 \
    --auto_batch_size \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 5 \
    --gradient_accumulation_steps 1 \
    --custom_lr_scheduler_type log_warmup_linear_decay \
    --num_train_epochs 10 \
    --logging_steps 50 \
    --learning_rate 3e-6 \
    --warmup_ratio 0.06 \
    --seed 42 \
    --overwrite_output_dir  2>&1 | tee -a $log_file
  if [ $?"x" == "0x" ]; then
    break
  else
    exist_checkpoint=`ls $output_dir | grep checkpoint`
    if [ $exist_checkpoint"x" == "x" ]; then
      resume_from_checkpoint=false
    else
      resume_from_checkpoint=true
    fi
  fi
done
