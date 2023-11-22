set -ex
export PYTHONPATH='../../..':${PYTHONPATH}

CONFIG_PATH=./
DS_CONFIG=${CONFIG_PATH}/config_blocklm_10B.json
MODEL_CONFIG=./model_blocklm_10B_chinese.sh
TASK_CONFIG=./seq_cnndm.sh
NEW_DS_CONFIG=/tmp/${RANDOM}.json
export lr=3e-6
sed "s/\"lr\": 7e-6/\"lr\": ${lr}/g" $DS_CONFIG > $NEW_DS_CONFIG

sh ds_finetune_seq2seq.sh \
    $MODEL_CONFIG \
    $TASK_CONFIG \
    $NEW_DS_CONFIG
