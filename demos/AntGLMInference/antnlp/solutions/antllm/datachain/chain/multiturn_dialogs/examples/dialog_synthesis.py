'''通过自动合成或 ChatGPT 蒸馏方式, 生成更多有针对性的多轮对话数据.'''
import fire
import random
import time
import logging
from tqdm import tqdm
from typing import Union, List, Optional
from pathlib import Path
from solutions.antllm.datachain.llms.ant_openai import AntOpenAI
from solutions.antllm.datachain.utils import get_uuid
from solutions.antllm.datachain.utils import download_oss
from solutions.antllm.utils.openai_utils import mp_requests_gpt
from solutions.antllm.datachain.utils import dump_jsonl, load_jsonl, load_text
from solutions.antllm.datachain.chain.multiturn_dialogs.examples.datasource import load_dialog_data_source
from solutions.antllm.datachain.chain.multiturn_dialogs.examples.datasource import load_instruct_data_source


logger = logging.getLogger(__file__)


def synthesize_long_dialog(
    to_file: str,
    total_num: int,
    min_turn_num: int = 5,
    max_turn_num: int = 40,
    zh_en_ratio: str = '5:1:1',
    temp_dir: str = './temp_dir',
    data_source: Optional[str] = None,
):
    '''将短对话合并成长对话. 默认使用所有多轮指令库进行合并.'''
    # 采样比例
    zh_w, en_w, mix_w = [float(item) for item in zh_en_ratio.split(':')]
    zh_ratio = zh_w / (zh_w + en_w + mix_w)
    en_ratio = en_w / (zh_w + en_w + mix_w)
    mix_ratio = mix_w / (zh_w + en_w + mix_w)
    zh_num = int(total_num * zh_ratio)
    en_num = int(total_num * en_ratio)
    mix_num = int(total_num * mix_ratio)
    logger.info(f'中文采样比例: {zh_ratio}')
    logger.info(f'英文采样比例: {en_ratio}')
    logger.info(f'中英文采样比例: {mix_ratio}')
    logger.info(f'中文采样数量: {zh_num}')
    logger.info(f'英文采样数量: {en_num}')
    logger.info(f'中英文采样数量: {mix_num}')

    zh_data = load_dialog_data_source(
        lang='zh',
        to_dir=temp_dir,
        data_source=data_source,
    )
    zh_instruct_data = load_instruct_data_source(
        lang='zh',
        to_dir=temp_dir,
        data_source=data_source,
    )
    zh_data.extend(zh_instruct_data)
    en_data = load_dialog_data_source(
        lang='en',
        to_dir=temp_dir,
        data_source=data_source,
    )
    en_instruct_data = load_instruct_data_source(
        lang='en',
        to_dir=temp_dir,
        data_source=data_source,
    )
    en_data.extend(en_instruct_data)

    zh_cnt = len(zh_data)
    en_cnt = len(en_data)
    langs = ['en', 'zh', 'en#zh']
    data_pair = [en_data, zh_data, zh_data]
    en_num = min(en_num, en_cnt)
    zh_num = min(zh_num, zh_cnt)
    num_pair = [en_num, zh_num, mix_num]

    # 合成
    synth_data = []
    for lang, data, data_num in zip(
        langs,
        data_pair,
        num_pair,
    ):
        logger.info(f'处理 {lang} 数据')
        cnt = 0
        random.shuffle(data)
        exclude_idx_set = set()
        while cnt < data_num and len(exclude_idx_set) < len(data):
            cnt += 1
            if cnt % 100:
                logger.info(f'{cnt}/{data_num}')

            synth = merge_long_dialog(
                lang=lang,
                en_data=en_data,
                zh_data=zh_data,
                en_ratio=en_ratio,
                zh_ratio=zh_ratio,
                exclude_idx_set=exclude_idx_set,
                min_turn_num=min_turn_num,
                max_turn_num=max_turn_num,
            )
            synth_data.append(synth)

        logger.info(f'数据集生成 {len(synth_data)}')

    dump_jsonl(synth_data, to_file)


def merge_long_dialog(
    lang: str,
    en_data: list,
    zh_data: list,
    en_ratio: float,
    zh_ratio: float,
    exclude_idx_set: set,
    min_turn_num: int,
    max_turn_num: int,
) -> dict:
    '''将多个对话数据合并成长对话数据.'''
    if lang == 'en' and not en_data:
        logger.warning(
            'merge_long_dialog input en lang, but en_data is empty !!!'
        )
        return {}

    if lang == 'zh' and not zh_data:
        logger.warning(
            'merge_long_dialog input zh lang, but zh_data is empty !!!'
        )
        return {}

    synth = {
        'dialog_id': get_uuid(),
        'source': '多轮扩充: ',
        'lang': lang,
        'topic': '',
        'turns': [],
    }

    turn_num = random.randint(min_turn_num, max_turn_num)
    try_cnt = 0
    stop_cnt = 100
    while len(synth['turns']) < turn_num and try_cnt < stop_cnt:
        try_cnt += 1
        if lang == 'en#zh':
            if random.random() < en_ratio:
                data = en_data
            else:
                data = zh_data
        elif lang == 'en':
            data = en_data
        elif lang == 'zh':
            data = zh_data
        now = time.time()
        idx = random.choice(list(set(range(0, len(data) - 1)) - exclude_idx_set))
        logger.info(time.time() - now)
        item = data[idx]
        if len(synth['turns']) > 0 and item.get('context'):
            continue
        synth['source'] += f'{item["source"]}#'
        synth['turns'].extend(item['turns'])
        exclude_idx_set.add(idx)

    return synth


def dialog_instruction_queries_to_chatgpt(
    to_file: str,
    total_num: int,
    api_key: str,
    aes_key: str,
    model: str = 'gpt-3.5-turbo',
    temperature: float = 1.0,
    max_tokens: int = 2048,
    max_workers: int = 16,
    min_turn_num: int = 3,
    max_turn_num: int = 10,
    zh_en_ratio: str = '10:1',
    instruct_type: Union[list, str] = None,
    temp_dir: str = './temp_dir',
    data_source: Optional[str] = None,
):
    '''用户在多轮对话中常见的指令 query, 调用 ChatGPT 生成多轮数据, 包括:
        - 用户对上一轮的质疑和否定
        - 多轮中用户输入让机器人继续生成
        - 多轮中用户突然和机器人闲聊

    输出数据结构为 ChatGPT chat messages 格式 [{"role": "user", "content": "xxx}]

    Params:
        instruct_type: 指令 query 类型, 默认为空, 表示全部生成
            deny - 用户对上一轮的质疑和否定
            continue - 多轮中用户输入让机器人继续生成
            chitchat - 多轮中用户突然和机器人闲聊
    '''
    default_func_map = {
        'deny': deny_queries,
        'continue': continue_queries,
        'chitchat': chitchat_queries,
    }
    # 采样比例
    zh_ratio, en_ratio = [float(item) for item in zh_en_ratio.split(':')]
    zh_ratio = zh_ratio / (zh_ratio + en_ratio)
    en_ratio = 1 - zh_ratio
    zh_num = int(total_num * zh_ratio)
    en_num = int(total_num * en_ratio)
    logger.info(f'中文采样比例: {zh_ratio}')
    logger.info(f'英文采样比例: {en_ratio}')
    logger.info(f'中文采样数量: {zh_num}')
    logger.info(f'英文采样数量: {en_num}')

    zh_data = load_dialog_data_source(
        lang='zh',
        to_dir=temp_dir,
        data_source=data_source,
    )
    zh_instruct_data = load_instruct_data_source(
        lang='zh',
        to_dir=temp_dir,
        data_source=data_source,
    )
    zh_data.extend(zh_instruct_data)
    en_data = load_dialog_data_source(
        lang='en',
        to_dir=temp_dir,
        data_source=data_source,
    )
    en_instruct_data = load_instruct_data_source(
        lang='en',
        to_dir=temp_dir,
        data_source=data_source,
    )
    en_data.extend(en_instruct_data)

    conversations = []
    exclude_idx_set = set()

    # 测试 queries
    instruct_functions = []
    if not instruct_type:
        instruct_functions = list(default_func_map.values())
    elif isinstance(instruct_type, str):
        instruct_functions = [default_func_map[instruct_type]]
    else:
        instruct_functions = [
            default_func_map[item] for item in instruct_type
        ]

    for query_func in instruct_functions:
        instructs = query_func(zh_data=zh_data, en_data=en_data, temp_dir=temp_dir)
        instructs = random.choices(instructs, k=int(total_num / len(instruct_functions)))
        logger.info(f'指令样本数: {len(instructs)}')

        for query in tqdm(instructs):
            if random.random() < en_ratio:
                lang = 'en'
            else:
                lang = 'zh'
            synth = merge_long_dialog(
                lang=lang,
                en_data=en_data,
                zh_data=zh_data,
                en_ratio=en_ratio,
                zh_ratio=zh_ratio,
                exclude_idx_set=exclude_idx_set,
                min_turn_num=min_turn_num,
                max_turn_num=max_turn_num,
            )
            if not synth:
                continue

            turns = synth['turns']

            # 增加多轮指令 query
            turns.append({'user': query, 'assistant': ''})
            conversations.append(turns)

    # 转成 ChatGPT message 格式
    messages = []
    for turns in conversations:
        message = []
        for idx, turn in enumerate(turns):
            message.append({'role': 'user', 'content': turn['user']})
            if idx < len(turns) - 1:
                message.append({'role': 'assistant', 'content': turn['assistant']})
        messages.append(message)

    # 调用 OpenAI chain
    llm = AntOpenAI(
        model=model,
        api_key=api_key,
        aes_key=aes_key,
        temperature=temperature,
        max_tokens=max_tokens,
        max_workers=max_workers,
    )
    chatgpt_results = llm.batch_chat(messages)

    assert len(messages) == len(chatgpt_results)

    # 合并 chatgpt 结果
    to_data = []
    for message, chatgpt_res in zip(messages, chatgpt_results):
        if not chatgpt_res:
            continue
        if chatgpt_res == 'null':
            continue

        dialog = {
            'dialog_id': get_uuid(),
            'source': 'ChatGPT 蒸馏的多轮特殊指令',
            'lang': 'zh',
            'topic': '多轮特殊指令',
            'turns': [],
        }
        for idx in range(0, len(message), 2):
            if message[idx]['role'] != 'user':
                dialog['turns'] = None
                break
            if idx == len(message) - 1:
                dialog['turns'].append({
                    'user': message[idx]['content'],
                    'reference': '',
                    'assistant': chatgpt_res,
                })
                continue

            if message[idx + 1]['role'] != 'assistant':
                dialog['turns'] = None
                break
            dialog['turns'].append({
                'user': message[idx]['content'],
                'reference': '',
                'assistant': message[idx + 1]['content'],
            })
        if not dialog['turns']:
            continue
        to_data.append(dialog)

    dump_jsonl(to_data, to_file)


def request_chatgpt(
    input_file: str,
    output_file: str,
    api_key: str,
    key: str,
    model: str = 'gpt-3.5-turbo',
    temperature: float = 1.0,
    max_tokens: int = 2048,
    max_workers: int = 16,
):
    '''调用 ChatGPT, 输入数据结构是 ChatGPT chat message 结构.'''
    input_data = load_jsonl(input_file)
    to_data = []
    for idx in tqdm(range(0, len(input_data), max_workers)):
        res_list = mp_requests_gpt(
            api_key,
            key,
            input_data[idx:idx + max_workers],
            max_workers=max_workers,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )
        to_data.extend(res_list)

    dump_jsonl(to_data, output_file)


def deny_queries(**kwargs) -> List[str]:
    '''质疑和否定的 query.'''
    return [
        '你说的不对',
        '你说错了吧',
        '瞎说吧',
        '搞错了吧',
        '你是垃圾吗',
        '太垃圾了吧',
        '不太准确',
        '不对',
        '你确定？',
        '你再想想',
        '你出错了，再试试',
        '你太笨了啊',
        '逻辑太乱了',
        '还是不对',
        '胡说八道啊',
        '重新回答一下',
    ]


def continue_queries(**kwargs) -> List[str]:
    '''让机器人继续说下去的 query.'''
    return [
        '继续说',
        '请继续',
        '继续说下去',
        '然后呢',
        '知道了',
        '结果呢',
        '详细点',
        '展开说说',
        '还有呢',
    ]


def chitchat_queries(
    temp_dir: str = './temp_dir',
    **kwargs,
) -> List[str]:
    '''和机器人闲聊'''
    oss = 'oss://antsys-adabrain/datasets/llm_multiturn/sft/海天瑞声/haitian.dialog.jsonl'
    local_path = Path(temp_dir).joinpath(Path(oss).name)
    if not local_path.exists():
        download_oss(oss, local_path)
    data = load_jsonl(local_path)
    # 只取首轮
    queries = []
    for item in data:
        if not item['conversations']:
            continue
        queries.append(item['conversations'][0]['text'])
    return queries


def merge_chatgpt_results(
    chatgpt_infile: str,
    chatgpt_outfile: str,
    output_file: str,
):
    '''合并 ChatGPT 蒸馏结果成多轮对话格式.

    Params:
        chatgpt_infile: 请求 ChatGPT 的输入文件
        chatgpt_outfile: ChatGPT 请求得到的响应文件, 每一行是一个文本响应结果
        output_file: 合并后的输出文件
    '''
    chatgpt_input = load_jsonl(chatgpt_infile)
    chatgpt_output = load_text(chatgpt_outfile)
    assert len(chatgpt_input) == len(chatgpt_output)

    to_data = []
    for input, output in tqdm(zip(chatgpt_input, chatgpt_output)):
        if output.startswith('"'):
            output = eval(output)
        if not output:
            continue
        if output == 'null':
            continue
        dialog = {
            'dialog_id': get_uuid(),
            'source': 'ChatGPT 蒸馏的多轮特殊指令',
            'lang': 'zh',
            'topic': '多轮特殊指令',
            'turns': [],
        }
        for idx in range(0, len(input), 2):
            if input[idx]['role'] != 'user':
                dialog['turns'] = None
                break
            if idx == len(input) - 1:
                dialog['turns'].append({
                    'user': input[idx]['content'],
                    'reference': '',
                    'assistant': output,
                })
                continue

            if input[idx + 1]['role'] != 'assistant':
                dialog['turns'] = None
                break
            dialog['turns'].append({
                'user': input[idx]['content'],
                'reference': '',
                'assistant': input[idx + 1]['content'],
            })
        if not dialog['turns']:
            continue
        to_data.append(dialog)

    dump_jsonl(to_data, output_file)


if __name__ == "__main__":
    fire.Fire()
