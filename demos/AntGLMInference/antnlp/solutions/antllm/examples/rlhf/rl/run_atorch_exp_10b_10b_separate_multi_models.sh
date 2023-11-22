set -x

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64

WORLD_SIZE=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}
NODE_NAME=`echo $ILOGTAIL_PODNAME | awk -F 'ptjob-' '{print $2}'`
NODE_NAME=${NODE_NAME:-master-0}

NUM_PROCESSES=`echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}'`
if [ $NUM_PROCESSES -eq 0 ]; then
NUM_PROCESSES=`echo $NVIDIA_VISIBLE_DEVICES | awk -F ',' '{print NF}'`
fi

torch_cmd="python -m atorch.distributed.launch --nnode=$WORLD_SIZE --nproc_per_node=$NUM_PROCESSES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"


# rl_model_path=/mnt/xiaohao.wzh/glm-10b-2k-sft-v9
# rw_model_path=/mnt/xiaohao.wzh/glm-10b-2k-sft-v9
# cost_model_path=/mnt/xiaohao.wzh/glm-10b-2k-sft-v9
rl_model_path=/mnt/chatgpt/experiments/tianxuan.jl/train_atorch_0831_192g_2b_mini_batch.sh/20230830-165832/checkpoint-19170-epoch-2/
rw_model_path=/mnt/tangjian.dtj/model/rw_model/sft_atorch_0831_192g_2b_mini_batch_rm_v11_helpful_add_toxic/20230901-172529/checkpoint-92890/
cost_model_path=/mnt/tangjian.dtj/model/rw_model/sft_atorch_0831_192g_2b_mini_batch_rm_v11_truthful_pointwise/20230901-192741/checkpoint-88090/
prompt_path=/mnt/chatgpt/data/RL/rl_v13/rl_v13.csv
output_dir=/mnt/xiaohao.wzh/rlhf/model/atorch_antnlp_compare_10_nodes

timestamp=$1
output_dir=$output_dir/$timestamp

log_dir=${output_dir}/logs

if [ $NODE_RANK -eq 0 ]; then
    rm -rf ${output_dir}/logs/*
    mkdir -p ${log_dir}
    rm -rf ${output_dir}/runs/
fi

export PYTHONPATH=$PYTHONPATH:"../../../../../"

$torch_cmd \
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
    2>&1 | tee $log_dir/log-${NODE_RANK}.txt