#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

config_json="$script_dir/config_block_10B.json"
gpt_options=" \
       --block-lm \
       --task-mask \
       --bert-prob 0.3 \
       --gap-sentence-prob 0 \
       --avg-block-length 3 \
       --gpt-min-ratio 0.25 \
       --block-mask-prob 0.1 \
       --short-seq-prob 0.5 \
       --experiment-name blocklm-10b-mix-chinese \
       --model-parallel-size 1 \
       --num-layers 48 \
       --hidden-size 4096 \
       --num-attention-heads 64 \
       --seq-length 1024 \
       --max-position-embeddings 4096 \
       --hf-model \
       --atorch-accelerate \
       --save checkpoints \
       --no-load-lr-scheduler \
       --log-interval 50 \
       --eval-interval 1000 \
       --save-interval 2000 \
       --train-iters 50000 \
       --use-prefix-mode \
       --train-data base \
       --resume-dataloader \
       --loader-scatter 8 \
       --no-lazy-loader \
       --spm-tokenizer-path zhen_sp5 \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --lr-decay-ratio 0.1 \
       --lr-decay-iters 30000 \
       --warmup 0.06 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --flash_attn \
       --fp16 \
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"

