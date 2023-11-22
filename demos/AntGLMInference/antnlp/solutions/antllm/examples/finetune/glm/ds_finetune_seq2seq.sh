set -ex
DATA_ROOT=/mnt_liping/workspace/liping.xj/data/
# directory to save finetune checkpoints
SAVE_PATH=/mnt_liping/workspace/liping.xj/ckps
DATESTR=$(date +"%m-%d-%H-%M")

source $1    # Model
source $2    # Task

#NUM_WORKERS=2
NUM_GPUS_PER_WORKER=$(nvidia-smi -L | wc -l)
MP_SIZE=1

OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
DISTRIBUTED_ARGS="-m torch.distributed.launch --nproc_per_node $NUM_GPUS_PER_WORKER --nnodes 1 --node_rank 0 --master_addr localhost --master_port $RANDOM"
#DISTRIBUTED_ARGS=""

EXPERIMENT_NAME=${EXPERIMENT_NAME}_longer_${lr}
mkdir -p logs
python -u ${DISTRIBUTED_ARGS} finetune_glm.py \
       --deepspeed \
       --deepspeed_config $3 \
       --finetune \
       --experiment-name ${EXPERIMENT_NAME} \
       --task ${TASK_NAME} \
       --data-dir ${DATA_PATH} \
       --save ${SAVE_PATH} \
       --checkpoint-activations \
       --num-workers 1 \
       --epochs ${EPOCH_SINGLE} \
       --batch-size ${BATCH_SINGLE} \
       --no-load-lr-scheduler \
       $MODEL_ARGS \
       $TRAIN_ARGS \
       $COMMON_ARGS \
       $TASK_ARGS \
       --fp16 \
       --model-parallel-size ${MP_SIZE} \
       --overwrite
