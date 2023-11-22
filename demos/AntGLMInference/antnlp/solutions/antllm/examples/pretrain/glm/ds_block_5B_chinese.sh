#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

config_json="$script_dir/config_block_5B.json"
gpt_options=" \
       --block-lm \
       --task-mask \
       --bert-prob 0.3 \
       --gap-sentence-prob 0.0 \
       --avg-block-length 3 \
       --gpt-min-ratio 0.25 \
       --block-mask-prob 0.1 \
       --short-seq-prob 0.5 \
       --experiment-name blocklm-5b-mix-zhencode \
       --model-parallel-size 1 \
       --num-layers 48 \
       --hidden-size 3072 \
       --num-attention-heads 48 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --save checkpoints \
       --load-pretrained /data/workspace/kunlong.ckl/ckpt/glm/checkpoints/blocklm-5b-mix-chinese03-07-00-10-zhenvocab\
       --log-interval 50 \
       --eval-interval 2000 \
       --save-interval 2000 \
       --train-iters 400000 \
       --train-data base \
       --resume-dataloader \
       --loader-scatter 1 \
       --no-lazy-loader \
       --spm-tokenizer-path zhen_sp5 \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --lr-decay-ratio 0.1 \
       --lr-decay-iters 380000 \
       --deepspeed-activation-checkpointing \
       --warmup 0.06 \
       --checkpoint-activations \
       --fp16 \
"
gpt_options="${gpt_options}
               --deepspeed_mpi --deepspeed \
               --deepspeed_config ${config_json} \
"

