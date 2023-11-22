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

# rl_model_path=/mnt/chatgpt/models_0602/sft/AntGLM-10B-SFT-Detoxcity-20230602
rl_model_path=/mnt/xiaohao.wzh/sft/truthful/train_deepspeed_seq2seq_bf16_xiaohao.sh/20230626-115729/checkpoint-24524/
rw_model_path=/mnt/tangjian.dtj/model/rw_model/glm-10b-2k-sft-v9-detoxity-rm-v7-k8-100k-detoxity-freeze-2-use-mean-no-position-9-nodes/checkpoint-2000/
output_dir=/mnt/xiaohao.wzh/rlhf/model/ckpt_10b_10b_lora_test_tt
log_dir=${output_dir}/logs


rm -rf ${output_dir}/logs/*
mkdir -p ${log_dir}

export PYTHONPATH=$PYTHONPATH:"../../../../../"

accelerate launch \
  --config_file configs/default_accelerate_config.yaml \
  --num_processes 1 \
  train_ppo.py \
  --rm_model_path $rw_model_path \
  --ppo_model_path $rl_model_path \
  --prompt_path /mnt/xiaohao.wzh/rlhf/data/rl_test.csv \
  --exp_cfg_path ./exps/exp_10b_10b_lora.yml \
  --save_dir ${output_dir} \
  --log_dir ${log_dir} \
  --mask_type '[gMASK]' \
  --val_size 5 \
  2>&1 | tee $log_dir/log.txt
