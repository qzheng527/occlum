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


current_dir=$(pwd)
parent_dir=$(dirname $current_dir)

export PYTHONPATH=$PYTHONPATH:"../../../../.."
export PYTHONPATH=$PYTHONPATH:$parent_dir
export WANDB_DISABLED=true

echo $PYTHONPATH

n_gpu=1

# pretrained=/ossfs/workspace/pretrained_models/glm-2b-ant
# pretrained=/ossfs/workspace/pretrained_models/glm-2b-sft-v3
# pretrained=/mnt/chatgpt/experiments/xinyu.kxy/glm_2b_v3/train_deepspeed_seq2seq_glm10b.sh/20230322-191437
# pretrained=/ossfs/workspace/personal/glm-10b-ant-addcode-temp
# pretrained=/mnt/chatgpt/experiments/tianxuan.jl/train_deepspeed_seq2seq_glm10b_v6minus.sh/20230325-203551/checkpoint-15000
# pretrained=/ossfs/workspace/mnt/tangjian.dtj/pretrained_models/glm-10b-sft-temp

pretrained=/ossfs/workspace/mnt/xiaohao.wzh/sft/glm-10b-20230420-190915-checkpoint-44410
pretrained=/ossfs/workspace/mnt/tangjian.dtj/pretrained_models/glm-5b-sft-v9-checkpoint-83270
# pretrained=/mnt/xiaohao.wzh/sft/glm-5b-20230425-002340-checkpoint-83270

# output_dir=/ossfs/workspace/chatgpt/model/reward_model/glm-10b-ant-static
# output_dir=/ossfs/workspace/chatgpt/model/reward_model/glm-2b-crank
# output_dir=/ossfs/workspace/chatgpt/model/reward_model/glm-2b-sft-mix
# output_dir=/ossfs/workspace/chatgpt/model/model/reward_model/glm-10b-sft-mix-en-noshuffle
# output_dir=/ossfs/workspace/mnt/tangjian.dtj/model/rw_model/glm-10b-sft-temp-v4-mix-en-noshuffle
# output_dir=/ossfs/workspace/mnt/tangjian.dtj/model/rw_model/debug
# output_dir=/ossfs/workspace/mnt/tangjian.dtj/model/rw_model/glm-5b-exp-v2

# output_dir=/ossfs/workspace/mnt/tangjian.dtj/model/rw_model/debug
output_dir=/ossfs/workspace/mnt/tangjian.dtj/model/rw_model/glm-5b-sft-v9-rm-v7-use-last


# output_dir=/mnt/xiaohao.wzh/rm/model/glm-5b-20230425-002340-checkpoint-83270-rm-zh

# data_dir=/ossfs/workspace/chatgpt/data/RM_static
data_dir=/ossfs/workspace/chatgpt/data/RM_test
# data_dir=/ossfs/workspace/chatgpt/data/RM_baike
# data_dir=/ossfs/workspace/chatgpt/data/RM_crank
# data_dir=/ossfs/workspace/chatgpt/data/RM_mix_label_crank
# data_dir=/ossfs/workspace/chatgpt/data/RM_label
# data_dir=/ossfs/workspace/chatgpt/data/RM_mix_label_crank_en
# data_dir=/ossfs/workspace/mnt/tangjian.dtj/data/RM/v4_mix_en
# data_dir=/ossfs/workspace/mnt/xiaohao.wzh/rm/data/oasst_mix_v1
# data_dir=/ossfs/workspace/chatgpt/data/RM_label
data_dir=/ossfs/workspace/mnt/chatgpt/data/RM/v7

pretrained=/ossfs/workspace/mnt/tangjian.dtj/pretrained_models/glm-10b-2k-sft-v9-checkpoint-133230
output_dir=/ossfs/workspace/mnt/tangjian.dtj/model/rw_model/glm-10b-2k-sft-v9-rm-v7-use-last
data_dir=/ossfs/workspace/mnt/chatgpt/data/RM/v7

pretrained=/ossfs/workspace/mnt/tangjian.dtj/pretrained_models/glm-10b-2k-sft-v9-checkpoint-133230
output_dir=/ossfs/workspace/mnt/tangjian.dtj/model/rw_model/glm-10b-2k-sft-v9-rm-v8-use-last
data_dir=/ossfs/workspace/mnt/chatgpt/data/RM/v8

pretrained=/mnt/xiaohao.wzh/sft/glm-10b-2k-v9-bf16-20230506-140819-checkpoint-133230
output_dir=/mnt/xiaohao.wzh/rm/model/rw_model/glm-10b-2k-v9-bf16-20230506-140819-checkpoint-133230-rm-v8_k6_100k-use_mean-no_position-bf16
data_dir=/mnt/chatgpt/data/RM/v8_k6_100k

pretrained=/mnt/xiaohao.wzh/sft/AntGLM-10B-SFT-Detoxcity-20230602
output_dir=/mnt/xiaohao.wzh/rm/model/rw_model/glm-10B-SFT-Detoxcity-20230602-rm-v8_k6_toxicity_100k-use_mean-no_position-bf16
data_dir=/mnt/xiaohao.wzh/rm/data/v8_k6_toxicity_100k

pretrained=/mnt/chatgpt/models_0602/sft/AntGLM-10B-SFT-Detoxcity-20230602
pretrained=/mnt/xiaohao.wzh/sft/AntGLM-10B-SFT-Detoxcity-20230602
output_dir=/mnt/tangjian.dtj/model/rw_model/debug
data_dir=/mnt/chatgpt/data/RM/test

mkdir -p $output_dir

set_n_least_used_CUDA_VISIBLE_DEVICES $n_gpu

deepspeed --num_gpus=$n_gpu \
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
  --evaluation_strategy "steps" \
  --eval_steps 250 \
  --save_steps 500 \
  --eval_accumulation_steps 20 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 1 \
  --logging_steps 20 \
  --learning_rate 5e-6 \
  --warmup_ratio 0.1 \
  --model_type glm \
  --overwrite_output_dir \
  --report_to tensorboard \
  --no_shuffle_dataloader \
  --mask_type '[gMASK]' \
  --use_mean_value false \
  --use_position_id true \
  --use_normalized_reward false \
  --num_layers_unfrozen 2 \
  --truncation_side 'left' \
  --num_head 1 \
  --data_type "pairwise" \
  --weights 0.6 0.3 0.2 0.1 \
  --dynamic_padding \
  2>&1 | tee $output_dir/log.txt