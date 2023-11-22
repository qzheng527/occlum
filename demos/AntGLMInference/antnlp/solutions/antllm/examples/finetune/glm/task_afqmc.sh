EXPERIMENT_NAME=${MODEL_TYPE}-AFQMC
TASK_NAME=afqmc
DATA_PATH="/mnt_liping/workspace/liping.xj/data/AFQMC"
MAX_SEQ_LEN=140

EVAL_BATCH_SIZE=64
TRAIN_EPOCH=5

TRAIN_ARGS="--lr-decay-style linear \
            --epoch ${TRAIN_EPOCH} \
            --warmup 0.1 \
            --weight-decay 1.0e-1 \
            --pattern-id 0"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 50 \
             --eval-interval 1000 \
             --eval-iters 100 \
             --eval-batch-size ${EVAL_BATCH_SIZE}"

PATTERN_IDS=(0 1)
PROMPT_IDS=(1 2 3)

#BATCH_SIZE=16
