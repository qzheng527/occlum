set -x

#rl_model_path=/mnt1/xuantai.hxd/glm-10b-2k-stf-v9/glm-10b-2k-sft-v9/ #/mnt/xiaohao.wzh/glm-10b-2k-sft-v9
#rw_model_path=/mnt1/xuantai.hxd/glm-10b-2k-stf-v9/glm-10b-2k-sft-v9/ #/mnt/xiaohao.wzh/glm-10b-2k-sft-v9
#cost_model_path=/mnt1/xuantai.hxd/glm-10b-2k-stf-v9/glm-10b-2k-sft-v9/ #/mnt/xiaohao.wzh/glm-10b-2k-sft-v9
rl_model_path=/mnt/chatgpt/experiments/tianxuan.jl/train_atorch_0831_192g_2b_mini_batch.sh/20230830-165832/checkpoint-19170-epoch-2/
rw_model_path=/mnt/tangjian.dtj/model/rw_model/sft_atorch_0831_192g_2b_mini_batch_rm_v11_helpful_add_toxic/20230901-172529/checkpoint-92890/
cost_model_path=/mnt/tangjian.dtj/model/rw_model/sft_atorch_0831_192g_2b_mini_batch_rm_v11_truthful_pointwise/20230901-192741/checkpoint-88090/
prompt_path=/mnt/chatgpt/data/RL/rl_v13/rl_v13.csv
output_dir=/mnt1/xiaohao.wzh/rlhf/model/atorch_compare
log_dir=${output_dir}/logs

rm -rf ${output_dir}/logs/*
mkdir -p ${log_dir}

export PYTHONPATH=$PYTHONPATH:"../../../../../"

python -m atorch.distributed.launch \
    --nnodes 1 \
    --nproc_per_node 8 \
    --master_port 5004  \
    atorch_train_ppo_reward_shaping.py \
    --actor_path $rl_model_path \
    --critic_path $rw_model_path \
    --cost_model_path $cost_model_path \
    --reward_model_path $rw_model_path \
    --lambda_value -0.6 \
    --rm_use_position_id \
    --cost_use_position_id \
    --cost_use_normalized_reward \
    --prompt_path $prompt_path \
    --checkpoint_dir ${output_dir} \
    --logdir ${log_dir} \
    --val_size 200 \
    --strategy ./atorch_strategies/strategy_zero_double_reward_model_for_10B.py \
    --exp_cfg_path ./atorch_exps/config_actor_10b_10b_sep_double_reward_model_frozen_block.yaml \
	  2>&1 | tee $log_dir/log.txt
