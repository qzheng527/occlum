#! /bin/bash
set -ex

MODEL_SIZE=10B

CONFIG_PATH=./

if [ $MODEL_SIZE = "5B" ]; then
    source ${CONFIG_PATH}/ds_block_5B_chinese.sh
else
    source ${CONFIG_PATH}/ds_block_10B_chinese.sh
fi

export PYTHONPATH='../../../':${PYTHONPATH}
# fix protobuf error
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python -u pretrain_glm.py ${gpt_options}
