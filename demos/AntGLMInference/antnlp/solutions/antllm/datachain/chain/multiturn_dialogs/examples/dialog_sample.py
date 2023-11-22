'''多轮对话 SFT 数据采样.'''
import fire
import random
from pathlib import Path
from solutions.antllm.datachain.utils import dump_jsonl
from solutions.antllm.datachain.chain.cleaner.keyword_filter import KeywordsFilterChain
from solutions.antllm.datachain.chain.cleaner.keyword_replace import KeywordsReplaceChain
from solutions.antllm.datachain.chain.multiturn_dialogs.examples.datasource import load_dialog_data_source
from solutions.antllm.datachain.chain.multiturn_dialogs.examples.datasource import DIALOG_EN_OSS_CONF
from solutions.antllm.datachain.chain.multiturn_dialogs.examples.datasource import DIALOG_ZH_OSS_CONF


def sft_dialog_sample(
    to_file: str,
    total_num: int,
    zh_en_ratio: str = '5:1',
    temp_dir: str = './temp_dir',
    max_workers: int = 16,
):
    '''对多轮 SFT 数据进行采样.

        - 采样逻辑: https://yuque.antfin-inc.com/crtpg4/xutwxe/uaifg3ra3g0puqp6
    '''
    filter_chain = KeywordsFilterChain(keys=['turns'], max_workers=max_workers)
    replace_chain = KeywordsReplaceChain(keys=['turns'], max_workers=max_workers)
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(exist_ok=True)

    # 采样比例
    zh_ratio, en_ratio = [float(item) for item in zh_en_ratio.split(':')]
    zh_ratio = zh_ratio / (zh_ratio + en_ratio)
    en_ratio = 1 - zh_ratio
    zh_num = int(total_num * zh_ratio)
    en_num = int(total_num * en_ratio)
    print(f'中文采样比例: {zh_ratio}')
    print(f'英文采样比例: {en_ratio}')
    print(f'中文采样数量: {zh_num}')
    print(f'英文采样数量: {en_num}')

    # 采样权重
    data_weights = {
        'CoQA': 20,
        'Guanaco': 15,
        'OIG': 15,
        'UltraChat': 15,
        'baize': 8,
        'oasst1': 15,
        'openchat_sharegpt4': 10,
        'MOSS': 15,
        'RefGPT': 8,
        'MeChatdata': 6,
        'ShareGPT': 15,
        '多轮身份构造数据': -1,
        '长多轮增强数据': -1,
        '多轮特殊指令数据': -1,
        '多轮重复指令回复': -1,
    }

    en_data = {}
    zh_data = {}
    zh_cnt = 0
    en_cnt = 0

    for idx in range(2):
        if idx == 0:
            print('加载英文数据...')
            lang = 'en'
        else:
            print('加载中文数据...')
            lang = 'zh'

        for source in data_weights.keys():
            data = load_dialog_data_source(lang=lang, to_dir=temp_dir, data_source=source)
            if not data:
                continue
            if idx == 0:
                en_data[source] = data
                en_cnt += len(data)
            else:
                zh_data[source] = data
                zh_cnt += len(data)

    samples_pair = [[], []]
    langs = ['en', 'zh']
    data_pair = [en_data, zh_data]
    oss_config_pair = [DIALOG_EN_OSS_CONF, DIALOG_ZH_OSS_CONF]
    data_num_pair = [en_num, zh_num]
    data_cnt_pair = [en_cnt, zh_cnt]

    for lang, lang_data, oss_config, data_num, data_cnt, samples in zip(
        langs,
        data_pair,
        oss_config_pair,
        data_num_pair,
        data_cnt_pair,
        samples_pair,
    ):
        sample_cnt = {
            name: min(
                len(lang_data[name]),
                int(
                    data_weights[name] / sum([
                        data_weights[_name] for _name in oss_config.keys() if _name in data_weights
                    ]) * data_num
                    if data_weights[name] > 0 else len(lang_data[name])
                ),
            )
            for name in oss_config.keys()
            if name in data_weights
        }
        print(f'各{lang}数据集采样个数: {sample_cnt}')

        if data_num >= data_cnt:
            print(f'采样个数 {data_num} > 数据总量 {data_cnt}, 所有数据集全部采样')
            samples.extend([
                item
                for _, data in lang_data.items()
                for item in data
            ])
        else:
            for name, data in lang_data.items():
                if len(data) < sample_cnt[name]:
                    print(f'数据集 {name} 全部采样')
                    samples.extend(data)
                else:
                    print(f'数据集 {name} 部分采样, 采样个数: {sample_cnt[name]}')
                    samples.extend(random.choices(data, k=sample_cnt[name]))

    for idx in range(len(samples_pair)):
        print(f'数据集过滤...')
        samples_pair[idx] = filter_chain.batch_run(
            samples_pair[idx],
            concurrency='process',
        )
        samples_pair[idx] = replace_chain.batch_run(
            samples_pair[idx],
            concurrency='process',
        )

    all_samples = samples_pair[0] + samples_pair[1]

    print(
        f'总采样个数: {len(all_samples)}\n'
        f'英文采样个数: {len(samples_pair[0])}\n'
        f'中文采样个数: {len(samples_pair[1])}'
    )
    dump_jsonl(all_samples, to_file)


if __name__ == "__main__":
    fire.Fire()
