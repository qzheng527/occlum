# The target directory to save 
dst_dir=$2
# The source data type, support text, json
DATA_TYPE=text

export PYTHONPATH='../../../':${PYTHONPATH}

python -u ../../../antllm/data/tools/preprocess_data.py \
    --input "$1/*" \
    --output_dir $dst_dir \
    --spm-tokenizer-path $tokenizer_dir_path \
    --workers 50 \
    --type $DATA_TYPE \
    --chunk-size 500
