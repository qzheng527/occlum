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

rl_model_path=/mnt/xiaohao.wzh/sft/glm-10b-2k-v9-bf16-20230506-140819-checkpoint-133230
rw_model_path=/mnt/tangjian.dtj/model/rw_model/glm-10b-2k-sft-v9-rm-v8-use-last-6-nodes/checkpoint-4600
output_dir=/mnt/xiaohao.wzh/rlhf/model/ckpt_10b_10b_separate_test
log_dir=${output_dir}/logs


rm -rf ${output_dir}/logs/*
mkdir -p ${log_dir}

export PYTHONPATH=$PYTHONPATH:"../../../../../"

accelerate launch \
  --config_file configs/default_accelerate_config.yaml \
  --num_processes 7 \
  train_ppo.py \
  --rm_model_path $rw_model_path \
  --ppo_model_path $rl_model_path \
  --prompt_path /mnt/xiaohao.wzh/rlhf/data/rl_test.csv \
  --exp_cfg_path ./exps/exp_10b_10b_separate.yml \
  --save_dir ${output_dir} \
  --log_dir ${log_dir} \
  --mask_type '[gMASK]' \
  --val_size 5 \
  2>&1 | tee $log_dir/log.txt
