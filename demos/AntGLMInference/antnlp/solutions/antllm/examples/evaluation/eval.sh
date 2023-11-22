#!/bin/bash
set -u
set -x
model_folder="/ossfs/workspace/nas2/models/AntGLM-10B-20230602"
# output_path="/ossfs/workspace/nas2/models/AntGLM-10B-20230602/eval"
gpu="0,1,2,3"
dataset_name="AGIEval,CEval,MMLU,BIG-Bench-Hard,GSM8k"
# dataset_name=""
batch_size=8
# model_name="llama2"
model_name="glm"
test_file="test_prompts.json"
data_path="/ossfs/workspace/nas_new/chatgpt/data/评测数据集"
dataset_config="solutions/antllm/antllm/evaluation/configs/datasets_des.json"
# dataset_config="solutions/antllm/antllm/evaluation/configs/datasets_des_llama2.json"
# dataset_config="solutions/antllm/antllm/evaluation/configs/datasets_des.pretrain.json"

python solutions/antllm/examples/evaluation/eval.py \
--model_folder "$model_folder" \
--datasets_folder "$data_path" \
--dataset_name "$dataset_name" \
--gpu "$gpu" \
--model_name "$model_name" \
--dataset_config "$dataset_config" \
--batch_size "$batch_size" \
--rotary_1d \
--output_folder "$output_path"
