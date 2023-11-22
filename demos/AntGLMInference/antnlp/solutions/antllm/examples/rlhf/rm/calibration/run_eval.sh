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
export PYTHONPATH=$PYTHONPATH:../../../../../../
export WANDB_DISABLED=true

n_gpu=8
set_n_least_used_CUDA_VISIBLE_DEVICES $n_gpu


# rm_model_path: 预训练模型路径
# output_dir: 模型输出文件路径
rm_model_path="/ossfs/workspace/mnt/tangjian.dtj/model/rw_model/glm-10b-sft-v9-rm-v7-use-last/checkpoint-2500"
output_dir="/ossfs/workspace/mnt/tangjian.dtj/calibration/glm-10b-sft-v9-rm-v7-use-last"
eval_data_path="/ossfs/workspace/mnt/chatgpt/data/RM/v7/test.jsonl"

rm_model_path="/ossfs/workspace/mnt/tangjian.dtj/model/rw_model/glm-10b-2k-sft-v9-rm-v7-use-last-lora/checkpoint-3000"
output_dir="/ossfs/workspace/mnt/tangjian.dtj/calibration/glm-10b-2k-sft-v9-rm-v7-use-last-lora/"
eval_data_path="/ossfs/workspace/mnt/chatgpt/data/RM/v7/test.jsonl"

rm_model_path=/mnt/xiaohao.wzh/rm/model/rw_model/glm-10b-2k-v9-bf16-20230506-140819-checkpoint-133230-rm-v8_k6_100k-use_mean-no_position-bf16
output_dir=/mnt/xiaohao.wzh/rm/calibration/v8_k6_100k/glm-10b-2k-v9-bf16-20230506-140819-checkpoint-133230-rm-v8_k6_100k-use_mean-no_position-bf16
eval_data_path=/mnt/chatgpt/data/RM/v8_k6_100k/test.jsonl

rm_model_path=/mnt/tangjian.dtj/model/rw_model/glm-10b-2k-sft-v9-detoxity-rm-v7-k8-100k-detoxity-freeze-2-use-mean-no-position-9-nodes/checkpoint-2000/
output_dir=/mnt/tangjian.dtj/calibration/glm-10b-2k-sft-v9-detoxity-rm-v7-k8-100k-detoxity-freeze-2-use-mean-no-position-9-nodes
eval_data_path=/mnt/chatgpt/data/RM/v7_k8_100k_add_detoxity/test.jsonl

mkdir -p ${output_dir}

python eval_rm.py \
  --eval_data_path $eval_data_path \
  --model_name_or_path $rm_model_path \
  --output_dir $rm_model_path \
  --max_len 2048 \
  --max_input_len 1024 \
  --max_output_len 1024 \
  --per_device_eval_batch_size 8 \
  --model_type glm \
  --bf16 \
  --do_predict \
  --use_mean_value true \
  --use_position_id false \
  --predict_output_path ${output_dir} \
  --data_format "jsonl" \
  --weights 1 1 1 1 \
  --truncation_side "right"

python plot_calibration.py \
 --score_path ${output_dir}