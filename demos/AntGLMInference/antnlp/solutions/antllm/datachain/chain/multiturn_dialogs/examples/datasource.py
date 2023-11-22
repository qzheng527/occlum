'''多轮对话的相关数据获取方法.'''

from pathlib import Path
from typing import Optional, Dict
from solutions.antllm.datachain.utils import download_oss
from solutions.antllm.datachain.utils import load_jsonl


DIALOG_EN_OSS_CONF = {
    'CoQA': [
        'CoQA/coqa-dev-v1.0_processed.jsonl',
        'CoQA/coqa-train-v1.0_processed.jsonl'
    ],
    'Guanaco': [
        'GuanacoDataset/guanaco_chat_all-utf8.en.jsonl',
    ],
    'OIG': [
        'Open_Instruction_Generalist/Open_Instruction_Generalist.jsonl',
    ],
    'UltraChat': [
        'UltraChat/train.jsonl',
    ],
    'baize': [
        'baize-chatbot/medical_chat_data_convert.jsonl',
        'baize-chatbot/quora_chat_data_convert.jsonl',
        'baize-chatbot/stackoverflow_chat_data_convert.jsonl',
    ],
    'oasst1': [
        'oasst1/oasst1_train_processed.jsonl',
        'oasst1/oasst1_vaildation_processed.jsonl',
    ],
    'openchat_sharegpt4': [
        'openchat_sharegpt4_dataset/openchat_8192.eval.text_convert.jsonl',
        'openchat_sharegpt4_dataset/openchat_8192.train.text_convert.jsonl',
        'openchat_sharegpt4_dataset/opencoder.eval.text_convert.jsonl',
        'openchat_sharegpt4_dataset/opencoder.train.text_convert.jsonl',
    ],
    'MOSS': [
        'MOSS_SFT_003/MOSS_SFT_003_others.jsonl',
    ],
    'ShareGPT': [
        'ShareGPT/ShareGPT_unfiltered_cleaned_split_processed_shuf_train.turns.jsonl'
    ],
}
DIALOG_ZH_OSS_CONF = {
    'Guanaco': [
        'GuanacoDataset/guanaco_chat_all-utf8.zh.jsonl',
    ],
    'MOSS': [
        'MOSS_SFT_003/MOSS_SFT_003_zh.jsonl',
    ],
    'RefGPT': [
        'RefGPT/RefGPT-Dataset-V1-CN.jsonl',
    ],
    'MeChatdata': [
        'smile/MeChatdata_smile.jsonl',
    ],
    'ShareGPT': [
        'ShareGPT/ShareGPT_unfiltered_cleaned_split_CH_process_train.turns.jsonl'
    ],
    '多轮身份构造数据': [
        'gpt3.5_构造身份数据/gpt3.5_identify_multurn.jsonl',
    ],
    '长多轮增强数据': [
        '长多轮增强数据/long_dialog_aug.jsonl',
    ],
    '多轮特殊指令数据': [
        '多轮特殊指令数据/dialog_instruction_synthetics.jsonl',
    ],
    '多轮重复指令回复': [
        '多轮重复指令回复/repeat_instruct_dialogs.jsonl',
    ]
}
INSTRUCT_EN_OSS_CONF = {
}
INSTRUCT_ZH_OSS_CONF = {
    '单轮指令采样': [
        '单轮指令采样/train_sft.turns.jsonl'
    ]
}


def download_dialog_data_source(
    to_dir: str,
    oss_conf: Dict[str, list],
):
    '''下载多轮对话 sft 标准格式的数据.'''
    download_data_source(
        to_dir,
        oss_prefix='oss://antsys-adabrain/datasets/llm_multiturn/SFT格式统一数据',
        oss_conf=oss_conf
    )


def download_instruct_data_source(
    to_dir: str,
    oss_conf: Dict[str, list],
):
    '''下载指令 sft 标准格式的数据.'''
    download_data_source(
        to_dir,
        oss_prefix='oss://antsys-adabrain/datasets/llm_multiturn/单轮指令采样',
        oss_conf=oss_conf
    )


def download_data_source(
    to_dir: str,
    oss_prefix: str,
    oss_conf: str,
):
    '''下载 sft 数据.'''
    to_dir = Path(to_dir)
    to_dir.mkdir(exist_ok=True)
    for _, url_list in oss_conf.items():
        for url in url_list:
            local_path = to_dir.joinpath(url.replace('/', '-'))
            if not local_path.exists():
                download_oss(f'{oss_prefix.rstrip("/")}/{url}', local_path)


def load_dialog_data_source(
    lang: str = 'zh',
    to_dir: str = './temp_dir',
    data_source: Optional[str] = None,
) -> list:
    '''获取所有多轮对话 sft 处理后的数据源.

    Params:
        data_source: 数据源名称, 默认为空, 加载所有数据源
    '''
    return load_data_source(
        lang=lang,
        to_dir=to_dir,
        data_type='dialog',
        data_source=data_source,
    )


def load_instruct_data_source(
    lang: str = 'zh',
    to_dir: str = './temp_dir',
    data_source: Optional[str] = None,
) -> list:
    '''获取所有单轮指令 sft 处理后的数据源.

    Params:
        data_source: 数据源名称, 默认为空, 加载所有数据源
    '''
    return load_data_source(
        lang=lang,
        to_dir=to_dir,
        data_type='instruct',
        data_source=data_source,
    )


def load_data_source(
    lang: str = 'zh',
    to_dir: str = './temp_dir',
    data_type: str = 'instruct',
    data_source: Optional[str] = None,
) -> list:
    '''获取所有多轮对话 sft 处理后的数据源.

    Params:
        data_type: 数据类型, instruct-单轮指令, dialog-多轮对话
        data_source: 数据源名称, 默认为空, 加载所有数据源
    '''
    if data_type == 'instruct':
        oss_conf = INSTRUCT_ZH_OSS_CONF if lang == 'zh' else INSTRUCT_EN_OSS_CONF
        download_func = download_instruct_data_source
    elif data_type == 'dialog':
        oss_conf = DIALOG_ZH_OSS_CONF if lang == 'zh' else DIALOG_EN_OSS_CONF
        download_func = download_dialog_data_source
    else:
        raise ValueError(f'not support data_type: {data_type}')

    # download data source
    if data_source:
        oss_conf = {k: v for k, v in oss_conf.items() if k == data_source}
    download_func(to_dir, oss_conf)

    data = []
    for source, url_list in oss_conf.items():
        if data_source and source != data_source:
            continue

        for url in url_list:
            local_path = Path(to_dir).joinpath(url.replace('/', '-'))
            data.extend(load_jsonl(local_path))

    return data
