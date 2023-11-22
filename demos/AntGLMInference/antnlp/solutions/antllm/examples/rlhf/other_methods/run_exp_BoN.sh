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



rl_model_path=/ossfs/workspace/rlhf_new_rm
rw_model_path=/ossfs/workspace/model/multi_doc_model/antglm-5b-sft/checkpoint-2000
model_name=glm-10b-5b

output_dir=/ossfs/workspace/model/BoN/antglm-10b-sft
log_dir=${output_dir}/logs
infer_dir=${output_dir}/infer_results

rm -rf ${output_dir}/logs/*
mkdir -p ${log_dir}

export PYTHONPATH=$PYTHONPATH:"../..":"../../../../../.."
accelerate launch \
  --config_file configs/default_accelerate_config.yaml \
  --num_processes 1 \
  BoN.py \
  --rm_model_path $rw_model_path \
  --ppo_model_path $rl_model_path \
  --prompt_path /ossfs/workspace/data/RL/multi_doc_infer_data.csv \
  --exp_cfg_path ./exps/exp_10b_5b_BoN.yml \
  --save_dir ${output_dir} \
  --log_dir ${log_dir} \
  --mask_type '[sMASK]' \
  --num_head 4
  --infer size 1000
  --infer_dir ${infer_dir}
  2>&1 | tee $log_dir/log.txt