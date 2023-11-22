#! /bin/bash
set -ex

config_script=$1

CONFIG_PATH=./

source ${CONFIG_PATH}/${config_script}


export PYTHONPATH='../../../':${PYTHONPATH}
export PYTHONPATH='../../../../../':${PYTHONPATH}


WORLD_SIZE=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
RANDOM_PORT=$[$RANDOM + 20000]
MASTER_PORT=${MASTER_PORT:-$RANDOM_PORT}

NODE_NAME=`echo $ILOGTAIL_PODNAME | awk -F 'ptjob-' '{print $2}'`
NODE_NAME=${NODE_NAME:-master-0}

NUM_PROCESSES=`echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}'`
if [ $NUM_PROCESSES -eq 0 ]; then
NUM_PROCESSES=`echo $NVIDIA_VISIBLE_DEVICES | awk -F ',' '{print NF}'`
fi


DISTRIBUTED_ARGS="--nnode=$WORLD_SIZE --nproc_per_node=$NUM_PROCESSES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"

python -u -m atorch.distributed.launch $DISTRIBUTED_ARGS pretrain_glm.py ${gpt_options}
