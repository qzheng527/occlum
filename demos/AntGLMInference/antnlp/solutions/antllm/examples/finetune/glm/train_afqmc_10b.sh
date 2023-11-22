set -ex
export PYTHONPATH='../../..':${PYTHONPATH}

CONFIG_PATH=./
MODEL_CONFIG=${CONFIG_PATH}/model_blocklm_10B_chinese.sh
DS_CONFIG=${CONFIG_PATH}/config_blocklm_10B.json
TASK_CONFIG=./task_afqmc.sh
NEW_DS_CONFIG=/tmp/${RANDOM}.json
export lr=5e-6
sed "s/\"lr\": 5e-6/\"lr\": ${lr}/g" $DS_CONFIG > $NEW_DS_CONFIG


sh ds_finetune_superglue.sh \
     $MODEL_CONFIG \
     $TASK_CONFIG \
     $NEW_DS_CONFIG
