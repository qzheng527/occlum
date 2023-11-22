#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

# --load /mnt/models/antllm/glm_large/blocklm-large-mix-chinese07-19-09-57 \

config_json="$script_dir/config_block_large.json"
gpt_options=" \
       --block-lm \
       --task-mask \
       --bert-prob 0 \
       --gap-sentence-prob 0 \
       --avg-block-length 3 \
       --gpt-min-ratio 0.7 \
       --block-mask-prob 0.1 \
       --short-seq-prob 0 \
       --summary-dir [YOUR SUMMARY PATH] \
       --experiment-name glm_large_datasetv2 \
       --new-save-directory \
       --model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --seq-length 2048 \
       --max-position-embeddings 4096 \
       --hf-model \
       --atorch-accelerate \
       --save [YOUR SAVE PATH] \
       --log-interval 5 \
       --eval-interval 50 \
       --save-interval 10000 \
       --train-iters 100 \
       --use-prefix-mode \
       --datasetv2 \
       --train-data pretrain_data_compatible.yaml \
       --resume-dataloader \
       --loader-scatter 32 \
       --no-lazy-loader \
       --spm-tokenizer-path [YOUR TOKENIZER PATH] \
       --split 7,2,1 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --lr-decay-ratio 0.1 \
       --lr-decay-iters 30000 \
       --warmup 0.04 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --flash_attn \
       --fp16 \
       --no-load-optim \
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"
