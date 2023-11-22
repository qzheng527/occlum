#! /bin/bash

MP_SIZE=1
script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

config_json="$script_dir/config_block_2b_chinese.json"
gpt_options=" \
       --block-lm \
       --task-mask \
       --bert-prob 0.5 \
       --gap-sentence-prob 0.3 \
       --avg-block-length 3 \
       --gpt-min-ratio 0.25 \
       --block-mask-prob 0.1 \
       --short-seq-prob 0.02 \
       --experiment-name blocklm-2b-mix-chinese \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 36 \
       --hidden-size 2048 \
       --num-attention-heads 32 \
       --seq-length 512 \
       --max-position-embeddings 1024 \
       --save checkpoints \
       --load /data/workspace/kunlong.ckl/ckpt/glm/checkpoints/blocklm-2b-mix-chinese01-05-15-05 \
       --log-interval 50 \
       --eval-interval 1000 \
       --save-interval 2000 \
       --train-iters 300000 \
       --train-data base \
       --resume-dataloader \
       --loader-scatter 32 \
       --no-lazy-loader \
       --num-workers 6 \
       --spm-tokenizer-path zhen_sp5 \
       --fix-command-token \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --lr-decay-ratio 0.1 \
       --lr-decay-iters 270000 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --warmup 0.06 \
       --fp16 \
"
gpt_options="${gpt_options}
               --deepspeed_mpi --deepspeed \
               --deepspeed_config ${config_json} \
"


#       --load /data/workspace/kunlong.ckl/ckpt/glm/checkpoints/blocklm-large-blank-chinese12-21-08-54/ \
