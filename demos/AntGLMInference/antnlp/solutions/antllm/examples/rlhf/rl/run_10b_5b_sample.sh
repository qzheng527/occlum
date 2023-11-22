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

export PYTHONPATH=$PYTHONPATH:"../../../../../"
accelerate launch --main_process_port 50003 --config_file configs/default_accelerate_config.yaml train_ppo.py \
  --rm_model_path /mnt/tangjian.dtj/model/rw_model/glm-5b-sft-v8minus-checkpoint-5528-en-noshuffle-new/checkpoint-8000 \
  --prompt_path /mnt/tangjian.dtj/data/RL/rl_v8.csv \
  --exp_cfg_path ./exps/exp_10b_5b_sample.yml \
  --mask_type [sMASK] \
  --rl_norm_reward \
  --val_size 500
