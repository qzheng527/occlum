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

n_gpu=8
set_n_least_used_CUDA_VISIBLE_DEVICES $n_gpu


# model_name_or_path: 预训练模型路径
# output_dir: 评估模型路径

rm_model_path="path/to/rm"
eval_data_path="path/to/evaluation/data"
output_dir="path/to/output"

python eval_rm.py \
    --eval_data_path $eval_data_path \
    --model_name_or_path $rm_model_path \
    --output_dir $rm_model_path \
    --max_len 2048 \
    --max_input_len 1024 \
    --per_device_eval_batch_size 8 \
    --model_type glm \
    --bf16 \
    --do_predict \
    --use_mean_value false \
    --use_position_id true \
    --use_normalized_reward false \
    --dynamic_padding \
    --predict_output_path ${output_path} \
    --truncation_side "left" \
    --data_type "pairwise"