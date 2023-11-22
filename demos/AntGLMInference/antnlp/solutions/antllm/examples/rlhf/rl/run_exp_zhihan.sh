set_n_least_used_CUDA_VISIBLE_DEVICES() {
    local n=${1:-"9999"}
    echo "GPU Memory Usage:"
    local FIRST_N_GPU_IDS=$(nvidia-smi --query-gpu=memory.used --format=csv \
        | tail -n +2 \
        | nl -v 0 \
        | tee /dev/tty \
        | sort -g -k 2 \
        | awk '{print $1}' \
        | head -n $n)
    export CUDA_VISIBLE_DEVICES=$(echo $FIRST_N_GPU_IDS | sed 's/ /,/g')
    echo "Now CUDA_VISIBLE_DEVICES is set to:"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
}

n_gpu=8

set_n_least_used_CUDA_VISIBLE_DEVICES $n_gpu


rl_model_path=/ossfs/workspace/pretrained_models/glm-2b-sft-v3
rw_model_path=/ossfs/workspace/chatgpt/model/reward_model/glm-2b-sft-mix

rl_model_path=/ossfs/workspace/mnt/tangjian.dtj/pretrained_models/glm-10b-sft-v8minus-checkpoint-5528
rw_model_path=/ossfs/workspace/mnt/tangjian.dtj/model/rw_model/glm-5b-sft-temp-v4-mix-en-noshuffle/checkpoint-4000 

rl_model_path=/ossfs/workspace/mnt/tangjian.dtj/pretrained_models/glm-10b-sft-v8minus-checkpoint-21672
rw_model_path=/ossfs/workspace/mnt/tangjian.dtj/model/rw_model/glm-5b-sft-v8minus-checkpoint-5528-en-noshuffle-new/checkpoint-8000

# model_name=glm-10b-sft-actor-lora-r16-layer6
# model_name=glm-10b-5b-freeze-2-max-512-norm
rl_model_path=/ossfs/workspace/mnt/tangjian.dtj/pretrained_models/glm-5b-sft-v8minus-checkpoint-37926-flash
rw_model_path=/ossfs/workspace/mnt/tangjian.dtj/model/rw_model/glm-5b-sft-flash-zh-clean-noshuffle-max1024/checkpoint-9000

rl_model_path=/ossfs/workspace/mnt/tangjian.dtj/pretrained_models/glm-10b-sft-v8minus-checkpoint-21672-flash-cache
rw_model_path=/ossfs/workspace/mnt/tangjian.dtj/model/rw_model/glm-5b-sft-flash-zh-clean-noshuffle-max1024/checkpoint-9000

rl_model_path=/ossfs/workspace/mnt/tangjian.dtj/pretrained_models/glm-10b-sft-v8minus-checkpoint-21672-flash-cache
rw_model_path=/ossfs/workspace/mnt/tangjian.dtj/model/rw_model/glm-5b-sft-flash-bf16-oasst_mix_v1

model_name=debug
# model_name=glm-5b-5b-max-1024-freeze-2-fp16-flash-v2
# model_name=glm-10b-5b-max-1024-freeze-2-fp16-oasst-mix-v1


output_dir=/ossfs/workspace/mnt/tangjian.dtj/model/rl_model/${model_name}
log_dir=${output_dir}/logs

rm -rf ${output_dir}/logs/*
mkdir -p ${log_dir}

export PYTHONPATH=$PYTHONPATH:"../../../../../"

NCCL_DEBUG=INFO accelerate launch \
  --config_file configs/default_accelerate_config.yaml \
  --num_processes 1 \
  train_ppo.py \
  --rm_model_path $rw_model_path \
  --ppo_model_path $rl_model_path \
  --prompt_path /ossfs/workspace/mnt/tangjian.dtj/data/RL/rl_v10.csv \
  --exp_cfg_path ./exps/exp_zhihan.yml \
  --save_dir ${output_dir} \
  --log_dir ${log_dir} \
  --mask_type "[gMASK]" \
  --rl_norm_reward \
  --val_size 50 \
  2>&1 | tee $log_dir/log.txt
