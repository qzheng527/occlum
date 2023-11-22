set -ex
# dataset parent directory
DATA_ROOT=/mnt_liping/workspace/liping.xj/data/
# directory to save finetune checkpoints
SAVE_PATH=/mnt_liping/workspace/liping.xj/ckps
DATESTR=$(date +"%m-%d-%H-%M")

source $1    # Model
source $2    # Task

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=$(nvidia-smi -L | wc -l)
MP_SIZE=1
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
DISTRIBUTED_ARGS="-m torch.distributed.launch --nproc_per_node $NUM_GPUS_PER_WORKER --nnodes 1 --node_rank 0 --master_addr localhost --master_port $RANDOM"
#DISTRIBUTED_ARGS=""

EXPERIMENT_NAME=${EXPERIMENT_NAME}_${lr}
mkdir -p logs
python -u ${DISTRIBUTED_ARGS} finetune_glm.py \
       --deepspeed \
       --deepspeed_config $3 \
       --finetune \
       --cloze-eval \
       --experiment-name ${EXPERIMENT_NAME} \
       --task ${TASK_NAME} \
       --data-dir ${DATA_PATH} \
       --save ${SAVE_PATH} \
       --seq-length ${MAX_SEQ_LEN} \
       --checkpoint-activations \
       --eval-batch-size 16 \
       --save-epoch 100000 \
       --num-workers 1 \
       --no-load-optim \
       --no-load-lr-scheduler \
       $MODEL_ARGS \
       $TRAIN_ARGS \
       $COMMON_ARGS \
       --pattern-id 0 \
       --fp16 \
       --model-parallel-size ${MP_SIZE} \
       --epochs ${TRAIN_EPOCH} \
       --overwrite > ${EXPERIMENT_NAME}.log 2>&1
