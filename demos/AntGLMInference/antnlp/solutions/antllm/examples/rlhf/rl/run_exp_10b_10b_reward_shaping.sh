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

rl_model_path=/mnt/tangjian.dtj/pretrained_models/sft_atorch_0717_full_32g_5b_checkpoint-20316-epoch-1
rw_model_path=/mnt/tangjian.dtj/model/rw_model/sft_0717_32g_5b_ckpt20316_rm_v10_freeze2/20230728-170736/checkpoint-29450/
cost_model_path=/mnt/tangjian.dtj/model/rw_model/sft_0717_32g_5b_ckpt20316_rm_v10_freeze2/20230728-170736/checkpoint-29450/
output_dir=/mnt/xiaohao.wzh/rlhf/model/ckpt_10b_10b_reward_shading_test_tt
log_dir=${output_dir}/logs


rm -rf ${output_dir}/logs/*
mkdir -p ${log_dir}

export PYTHONPATH=$PYTHONPATH:"../../../../../"

accelerate launch \
  --config_file configs/default_accelerate_config.yaml \
  --num_processes 7 \
  train_ppo_reward_shaping.py \
  --rm_model_path $rw_model_path \
  --cost_model_path $cost_model_path \
  --lambda_value 0.2 \
  --ppo_model_path $rl_model_path \
  --prompt_path /mnt/xiaohao.wzh/rlhf/data/rl_test.csv \
  --exp_cfg_path ./exps/exp_10b_10b_separate.yml \
  --save_dir ${output_dir} \
  --log_dir ${log_dir} \
  --mask_type '[gMASK]' \
  --val_size 50 \
  --rm_use_position_id \
  --rw_device 1 \
  2>&1 | tee $log_dir/log.txt
