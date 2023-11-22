#!/bin/bash
set -u
set -x
train_save_dir="/ossfs/workspace/nas2/models/antllm/glm10b_v2/AntGLM-10B-20230602"
data_path="/ossfs/workspace/nas_new/chatgpt/data/评测数据集"
output_path="./"

gpu="0,1,2,3"
dataset_name="MMLU,CEval,AGIEval"
batch_size=8
test_file="test_prompts.json"
base_model_path="/ossfs/workspace/nas_new/chatgpt/models_0602/glm/AntGLM-10B-20230602"
# dataset_config="solutions/antllm/antllm/evaluation/configs/datasets_des.pretrain.json"
dataset_config="/ossfs/workspace/nas2/xinyu.kxy/antnlp/solutions/antllm/antllm/evaluation/configs/datasets_des.json"


python solutions/antllm/examples/evaluation/async_eval.py \
--train_save_dir "$train_save_dir" \
--datasets_folder "$data_path" \
--dataset_name "$dataset_name" \
--gpu "$gpu" \
--batch_size "$batch_size" \
--test_file "$test_file" \
--base_model_path "$base_model_path" \
--output_folder "$output_path" \
--dataset_config "$dataset_config" \
--re_run \
--rotary_1d