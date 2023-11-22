
set -e
set -x

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

pretrained_model_name_or_path=/mnt/chatgpt/experiments/tianxuan.jl/train_deepspeed_seq2seq_glm10b_2k_v9_add_math_toxicity_bf16_old_code.sh/20230623-000804/epochs/checkpoint-64512
reference_model_name_or_path=/mnt/chatgpt/experiments/tianxuan.jl/train_deepspeed_seq2seq_glm10b_2k_v9_add_math_toxicity_bf16_old_code.sh/20230623-000804/epochs/checkpoint-64512
output_dir=/mnt/xiaohao.wzh/rlhf/model/ckpt_10b_2k_v9_add_math_toxicity_rm-v8-k6-100k-detoxity-use-mean-no-position-bf16-detoxicity-dpo-1-nodes
train_data=/mnt/xiaohao.wzh/rm/data/v8_k6_toxicity_100k/train.jsonl
test_data=/mnt/xiaohao.wzh/rm/data/v8_k6_toxicity_100k/test.jsonl
timestamp=$(date "+%Y%m%d-%H%M%S")
output_dir=$output_dir/$timestamp

mkdir -p $output_dir
if [ -d $output_dir ]; then
  echo 'Exist: ', $output_dir
else
  echo 'Not exist: ', $output_dir
fi

log_dir=${output_dir}/logs
rm -rf ${log_dir}/*
mkdir -p ${log_dir}
rm -rf ${output_dir}/runs/

export PYTHONPATH=$PYTHONPATH:"../../../../../"
echo $PYTHONPATH

deepspeed --num_gpus=$n_gpu \
train_dpo.py \
--deepspeed configs/ds_config_bf16.json  \
--train_data $train_data \
--test_data $test_data \
--pretrained_model_name_or_path $pretrained_model_name_or_path \
--reference_model_name_or_path $reference_model_name_or_path \
--beta_coef 0.2 \
--report_to tensorboard \
--log_on_each_node false \
--no_save_deepspeed_checkpoint true \
--weight_decay 0.1 \
--output_dir $output_dir \
--resume_from_checkpoint false \
--do_train \
--do_eval \
--bf16 \
--bf16_full_eval \
--max_length 2048 \
--max_input_length 1024 \
--max_output_length 1024 \
--evaluation_strategy "steps" \
--save_strategy "steps" \
--save_steps 3000 \
--eval_steps 3000 \
--eval_accumulation_steps 10 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 2 \
--max_grad_norm 10.0 \
--num_train_epochs 1 \
--logging_steps 50 \
--learning_rate 5e-7 \
--warmup_steps 150 2>&1 | tee -a ${log_dir}/log.txt