'''将收集到的多轮对话源数据集的不同格式, 转换成统一的多轮对话格式, 格式为:

```
SFT data json
[
    {
        "dialog_id": "对话 id",
        "source": "数据集来源",
        "lang": "语言",
        "topic": "数据类型",
        "context": "整通对话文档内容",
        "turns": [
            {
                "user": "用户输入",
                "reference": "当轮机器人的参考信息，例如搜索引擎召回结果、系统返回结果等，默认为空",
                "assistant": "机器人回复"
            }
        ]
    }
]

Pretrain data json
[
    {"content": "文档内容"}
]


```
'''
import fire
import os
import csv
import re
import string
import json
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
from zhconv import convert
from langdetect import detect
from solutions.antllm.datachain.utils import get_uuid
from solutions.antllm.datachain.utils import load_jsonl, dump_jsonl
from urllib.parse import unquote


def chatalpaca_raw_data_process(in_fname, out_fname):
    '''
    in_fname in ["chatalpaca-10k.json", "chatalpaca-20k.json"]
    
    input:
    {
        "id": "0", 
        "conversations": [
            {
                "from": "human', 
                "value": "Find the product of the numbers: 5 and 8."
            }, 
            {
                "from": "gpt", 
                "value": "The product of 5 and 8 is 40."
            }, 
            {
                "from": "human", 
                "value": "What is the sum of the numbers 6 and 12?"
            }, 
            {
                "from": "gpt", 
                "value": "The sum of the numbers 6 and 12 is 18."
            }
        ]
    }
    output:
    {
        "dialog_id": "0",
        "source": "chatalpaca-10k/chatalpaca-20k",
        "lang": "english",
        "topic": "数据类型，默认为空",
        "context": "整通对话文档内容，默认为空",
        "turns": [
            {
                "user": "Find the product of the numbers: 5 and 8",
                "reference": "当轮机器人的参考信息，例如搜索引擎召回结果、系统返回结果等，默认为空",
                "assistant": "The product of 5 and 8 is 40."
            },
            {
                "user": "What is the sum of the numbers 6 and 12?",
                "reference": "当轮机器人的参考信息，例如搜索引擎召回结果、系统返回结果等，默认为空",
                "assistant": "The sum of the numbers 6 and 12 is 18."
            },
        ]
    }
    '''
    out_file = open(out_fname, 'w')
    for idx, d in enumerate(open(in_fname)):
        d = json.loads(d)
        user_lst = []
        assist_lst = []
        turn_lst = []
        for c in d['conversations']:
            if c['from'] == 'human':
                user_lst.append(c['value'])
            elif c['from'] == 'gpt':
                assist_lst.append(c['value'])
            
        turns = dict(zip(user_lst, assist_lst))
        for k, v in turns.items():
            turn = {
                'user': k,
                'reference': '',
                'assistant': v
            }
            turn_lst.append(turn)
        out_dic = {
            'dialog_id': d['id'],
            'source': in_fname.rstrip('.json'),
            'lang': 'english',
            'topic': '',
            'turns': turn_lst
        }
        out_dic = json.dumps(out_dic, ensure_ascii=False)
        out_file.write(out_dic + '\n')
    out_file.close()


def chatalpaca_translated_data_process(in_fname, out_fname):
    '''
    in_fname in ["chatalpaca-20k-chinese-google-translation.json"]
    
    input:
    [
        {
            "role": "user",
            "content": "找出数字的乘积：5 和 8"
        },
        {
            "role": "assistant",
            "content": "5 和 8 的乘积是 40。"
        },
        {
            "role": "user",
            "content": "数字 6 和 12 的和是多少？"
        },
        {
            "role": "assistant",
            "content": "数字 6 和 12 的和是 18。"
        }
    ]
    output:
    {
        "dialog_id": "0",
        "source": "ChatAlpaca-20k-chinese-google-translation",
        "lang": "chinese",
        "topic": "",
        "turns": [
            {
                "user": "找出数字的乘积：5 和 8",
                "reference": "",
                "assistant": "5 和 8 的乘积是 40。"
            },
            {
                "user": "数字 6 和 12 的和是多少？",
                "reference": "",
                "assistant": "数字 6 和 12 的和是 18。"
            }
            }
        ]
    }
    '''
    out_file = open(out_fname, 'w')
    for _, data in enumerate(json.load(open(in_fname))):
        user_lst = []
        assist_lst = []
        turn_lst = []
        for d in data:
            if d['role'] == 'user':
                user_lst.append(d['content'])
            elif d['role'] == 'assistant':
                assist_lst.append(d['content'])
        turns = dict(zip(user_lst, assist_lst))
        for k, v in turns.items():
            turn = {
                'user': k,
                'reference': '',
                'assistant': v
            }
            turn_lst.append(turn)
        out_dic = {
            'dialog_id': get_uuid(),
            'source': in_fname.rstrip('.json'),
            'lang': 'chinese',
            'topic': '',
            'turns': turn_lst
        }
        out_dic = json.dumps(out_dic, ensure_ascii=False)
        out_file.write(out_dic + '\n')
    out_file.close()


def mechat_data_process(in_fdir, out_fname):
    '''
    in_fdir in ["smile_data/smile/", ""]

    input:
    [
        "求助者：最近我很困扰，我的孩子7岁了，",
        "支持者：你好，关注孩子的学习是每位家长的责任。",
    ]
    output:
    {
        "dialog_id": "0",
        "source": "MeChatdata_smile",
        "lang": "chinese",
        "topic": "mental health",
        "turns": [
            {
                "user": "最近我很困扰，我的孩子7岁了，",
                "reference": "",
                "assistant": "你好，关注孩子的学习是每位家长的责任。"
            },
        ]
    }
    '''
    out_file = open(out_fname, 'w')
    file_list = os.listdir(in_fdir)
    for idx, file in enumerate(file_list):
        file_name = in_fdir + file
        with open(file_name, 'r', encoding='utf-8') as f:
            dialog = json.load(f)
            user_lst = []
            assist_lst = []
            turn_lst = []
            for element in dialog:
                role = element[:3]
                content = element[4:]
                if role == '求助者':
                    user_lst.append(content)
                elif role == '支持者':
                    assist_lst.append(content)
            turns = dict(zip(user_lst, assist_lst))
            for k, v in turns.items():
                turn = {
                    'user': k,
                    'reference': '',
                    'assistant': v
                }
                turn_lst.append(turn)
            out_dic = {
                'dialog_id': str(idx),
                'source': 'MeChatdata_smile',
                'lang': 'chinese',
                'topic': 'mental health',
                'turns': turn_lst}
            out_dic = json.dumps(out_dic, ensure_ascii=False)
            out_file.write(out_dic + '\n')
    out_file.close()


def oasst1_data_process(in_fname, out_fname):
    '''
    in_fname in ["train-00000-of-00001-b42a775f407cee45.parquet", "validation-00000-of-00001-134b8fd0c89408b6.parquet"]
    '''
    out_file = open(out_fname, 'w')
    tree2content = {}
    data = pd.read_parquet(in_fname, engine='pyarrow')
    for d in data.values.tolist():
        text, role, message_tree_id = d[4], d[5], d[-4]
        if message_tree_id not in tree2content:
            tree2content[message_tree_id] = [{'role': role, 'text': text}]
        else:
            tree2content[message_tree_id].append({'role': role, 'text': text})
    for k, v in tree2content.items():
        message_tree_id = k
        user_lst = []
        assist_lst = []
        turn_lst = []
        for idx, data in enumerate(v):
            if data['role'] == 'prompter':
                user_lst.append(data['text'])
            elif data['role'] == 'assistant':
                assist_lst.append(data['text'])
        turns = dict(zip(user_lst, assist_lst))
        for k, v in turns.items():
            turn = {
                'user': k,
                'reference': '',
                'assistant': v
            }
            turn_lst.append(turn)
        out_dic = {
            'dialog_id': message_tree_id,
            'source': 'oasst1',
            'lang': '',
            'topic': '',
            'turns': turn_lst
        }
        out_dic = json.dumps(out_dic, ensure_ascii=False)
        out_file.write(out_dic + '\n')
    out_file.close()


def coqa_data_process(in_fname, out_fname):
    '''
    in_fname in ["coqa-train-v1.0.json", "coqa-dev-v1.0.json"]

    input:
    {
        "source": "mctest",
        "id": "3dr23u6we5exclen4th8uq9rb42tel",
        "filename": "mc160.test.41",
        "story": "Once upon a time, in a barn near a farm house,",
        "questions": [
            {
                "input_text": "What color was Cotton?",
                "turn_id": 1
            },
            {
                "input_text": "Where did she live?",
                "turn_id": 2
            },
            {
                "input_text": "Did she live alone?",
                "turn_id": 3
            }
        ],
        "answers": [
            {
                "span_start": 59,
                "span_end": 93,
                "span_text": "a little white kitten named Cotton",
                "input_text": "white",
                "turn_id": 1
            },
            {
                "span_start": 18,
                "span_end": 80,
                "span_text": "in a barn near a farm house, there lived a little white kitten",
                "input_text": "in a barn",
                "turn_id": 2
            },
            {
                "span_start": 196,
                "span_end": 215,
                "span_text": "Cotton wasn't alone",
                "input_text": "no",
                "turn_id": 3
            }
        ],
        "additional_answers": {
            "0": [
                {
                    "span_start": 68,
                    "span_end": 93,
                    "span_text": "white kitten named Cotton",
                    "input_text": "white",
                    "turn_id": 1
                },
                {
                    "span_start": 17,
                    "span_end": 93,
                    "span_text": " in a barn near a farm house, there lived a little white kitten named Cotton",
                    "input_text": "in a barn",
                    "turn_id": 2
                },
                {
                    "span_start": 192,
                    "span_end": 215,
                    "span_text": "But Cotton wasn't alone",
                    "input_text": "no",
                    "turn_id": 3
                }
            ]
        },
        "name": "mc160.test.41"
    }
    
    output:
    
    {
        "dialog_id": "3dr23u6we5exclen4th8uq9rb42tel",
        "source": "mctest",
        "lang": "english",
        "topic": "",
        "context": "Once upon a time, in a barn near a farm house,",
        "turns": [
            {
                "user": "What color was Cotton?",
                "reference": "a little white kitten named Cotton",
                "assistant": "white"
            },
            {
                "user": "Where did she live?",
                "reference": "in a barn near a farm house, there lived a little white kitten",
                "assistant": "in a barn"
            },
            {
                "user": "Did she live alone?",
                "reference": "Cotton wasn't alone",
                "assistant": "no"
            }
        ]
    }
    '''
    out_file = open(out_fname, 'w')

    data = json.load(open(in_fname))
    for idx, d in enumerate(data['data']):
        if idx == 0:
            print(json.dumps(d, ensure_ascii=False, indent=4))
        user_lst = []
        asstist_lst = []
        span_lst = []
        turn_lst = []
        for q in d['questions']:
            content = q['input_text']
            user_lst.append(content)
        for a in d['answers']:
            content = a['input_text']
            span_text = a['span_text']
            span_lst.append(span_text)
            asstist_lst.append(content)
        turns = dict(zip(user_lst, asstist_lst))
        for k, v in turns.items():
            turn = {
                'user': k,
                'reference': '',
                'assistant': v
            }
            turn_lst.append(turn)
        out_dic = {
            'dialog_id': str(d['id']),
            'source': d['source'],
            'lang': 'english',
            'topic': '',
            'context': d['story'],
            'turns': turn_lst
        }
        out_dic = json.dumps(out_dic, ensure_ascii=False)
        out_file.write(out_dic + '\n')
    out_file.close()


def belle_data_process(fname, tofname):
    '''
    case example:
    {
    "conversations": [
        {
            "from": "human",
            "value": "根据输入的文本，判断它是否属于新闻报道、广告或博客文章类别之一。\n新闻报道：“奥运会在东京隆重开幕，中国代表团获得多项奖牌。”"
        },
        {
            "from": "assistant",
            "value": "属于新闻报道类别。"
        },
        {
            "from": "human",
            "value": "很好，现在请你计算一下这篇文章中出现了多少个“获得”字样。"
        },
        {
            "from": "assistant",
            "value": "经过计数，这篇文章中总共出现了两次“获得”这个词汇。"
        }
    ],
        "id": "66623028"
    }
    '''

    data = load_jsonl(fname)
    convert_data = []
    single_num = 0
    for d in tqdm(data):
        conversations = d['conversations']
        if len(conversations) <= 2:
            single_num += 1
        else:
            dialog_id = d['id']
            item = {
                'dialog_id': dialog_id,
                'source': 'BELLE',
                'lang': 'Chinese',
                'topic': '涵盖各种应用场景，包括日常对话、知识问答、文本生成等'
            }

            turns = []
            for i in range(0, len(conversations), 2):
                per_turn = conversations[i: i + 2]
                assert per_turn[0]['from'] == 'human'
                assert per_turn[1]['from'] == 'assistant'
                turns.append(
                    {
                        'user': per_turn[0]['value'],
                        'reference': '',
                        'assistant': per_turn[1]['value']
                    }
                )
            item['turns'] = turns
            convert_data.append(item)

    dump_jsonl(convert_data, tofname)


def refgpt_data_process(fname, tofname):
    '''
    case example:
    {
        "dialogue": "Human: 你好，请问高小龙导演是做什么工作的？\n\nAssistant: xx",
        "reference": "高小龙中国内地男导演\n高小龙先生自1992年开始从xx"
    }
    '''
    data = load_jsonl(fname)
    convert_data = []
    for d in tqdm(data):
        dialogue = d['dialogue']
        querys = re.split('Human: |Assistant: |Human:|Assistant:', dialogue)
        if '' in querys:
            querys.remove('')
        if len(querys) % 2 != 0:
            continue
        context = d['reference']

        item = {
            'dialog_id': get_uuid(),
            'source': 'refgpt',
            'lang': 'Chinese',
            'context': context,
            'topic': '事实型对话包含人物、科技、医疗、法律、艺术等'
        }

        turns = []
        for i in range(0, len(querys), 2):
            per_turn = querys[i: i + 2]
            turns.append(
                {
                    'user': per_turn[0],
                    'reference': '',
                    'assistant': per_turn[1]
                }
            )

        if len(turns) > 1:
            item['turns'] = turns
            convert_data.append(item)

    dump_jsonl(convert_data, tofname)


def baize_data_process(fname, tofname):
    '''
    case example:
    {
        'topic': 'Are there safety concerns or special precautions xx',
        'input': "The conversation between human and AI assistant.\n[|Human|] xx \n[|AI|] xx \n[|Human|] '
    }
    
    '''
    print(f'process fname {fname}')
    data = load_jsonl(fname)
    convert_data = []
    for d in tqdm(data[0]):
        topic = d['topic']
        dialogue = d['input']
        querys = re.split('\[\|Human\|\] |\[\|AI\|\] ', dialogue)[1:]
        if '' in querys:
            querys.remove('')

        if len(querys) % 2 != 0:
            continue

        item = {
            'dialog_id': get_uuid(),
            'source': 'baize',
            'lang': 'English',
            'topic': topic
        }

        turns = []
        for i in range(0, len(querys), 2):
            per_turn = querys[i: i + 2]
            turns.append(
                {
                    'user': per_turn[0],
                    'reference': '',
                    'assistant': per_turn[1]
                }
            )

        if len(turns) > 1:
            item['turns'] = turns
            convert_data.append(item)

    dump_jsonl(convert_data, tofname)


def openchat_data_process(fname, tofname):
    '''
    case example:
    '<s>Human: xx <|end_of_turn|>Assistant:  xx<|end_of_turn|>'
    '''

    with open(fname, 'r') as f:
        json_str = f.read()

    data = json.loads(json_str)
    convert_data = []
    for d in tqdm(data):
        # 去除 <s>
        if d.startswith('<s>'):
            d = d[3:]
        querys = re.split('\<\|end_of_turn\|\>', d)
        if '' in querys:
            querys.remove('')

        if len(querys) % 2 != 0:
            continue

        item = {
            'dialog_id': get_uuid(),
            'source': 'openchat',
            'lang': 'English',
            'topic': '基于 sharegpt 的精选数据'
        }

        turns = []
        for i in range(0, len(querys), 2):
            per_turn = querys[i: i + 2]

            if 'chat' in os.path.basename(fname):
                assert per_turn[0].startswith('Human:')
                assert per_turn[1].startswith('Assistant:')
                turns.append(
                    {
                        'user': per_turn[0][len('Human: ') + 1:],
                        'reference': '',
                        'assistant': per_turn[1][len('Assistant: ') + 1:]
                    }
                )
            elif 'code' in os.path.basename(fname):
                assert per_turn[0].startswith('User:')
                assert per_turn[1].startswith('Assistant:')
                turns.append(
                    {
                        'user': per_turn[0][len('User:'):],
                        'reference': '',
                        'assistant': per_turn[1][len('Assistant:'):]
                    }
                )

        if len(turns) > 1:
            item['turns'] = turns
            convert_data.append(item)

    tofname = fname.replace('.json', '_convert.json')
    dump_jsonl(convert_data, tofname)


def process_sharegpt(fname, to_file):
    drop_cnt = 0
    to_data = []
    with open(fname, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in tqdm(data):
        turns = []
        raw_turn = item['conversations']
        if len(raw_turn) % 2 != 0:
            drop_cnt += 1
            continue

        for idx in range(0, len(raw_turn), 2):
            if raw_turn[idx]['from'] != 'human' or raw_turn[idx + 1]['from'] != 'gpt':
                turns = []
                break

            turns.append({
                'user': raw_turn[idx]['value'],
                'reference': '',
                'assistant': raw_turn[idx + 1]['value'],
            })

        if not turns:
            drop_cnt += 1
            continue

        to = {
            'dialog_id': item['id'],
            'source': 'sharegpt',
            'lang': 'english',
            'topic': 'NA',
            'turns': turns,
        }
        to_data.append(to)

    print(f'过滤数据个数: {drop_cnt}')
    dump_jsonl(to_data, to_file)


def process_ultrachat(fdir, to_file):
    drop_cnt = 0
    to_data = []
    for fname in Path(fdir).iterdir():
        print(f'processing {fname}')
        if 'jsonl' not in str(fname):
            continue
        data = load_jsonl(fname)
        for item in tqdm(data):
            if len(item['data']) % 2 != 0:
                drop_cnt += 1
                continue

            turns = []
            raw_turn = item['data']
            for idx in range(0, len(raw_turn), 2):
                turns.append({
                    'user': raw_turn[idx],
                    'reference': '',
                    'assistant': raw_turn[idx + 1],
                })

            if not turns:
                drop_cnt += 1
                continue

            to = {
                'dialog_id': get_uuid(),
                'source': 'ultrachat',
                'lang': 'english',
                'topic': 'NA',
                'turns': turns,
            }
            to_data.append(to)

    print(f'过滤数据个数: {drop_cnt}')
    dump_jsonl(to_data, to_file)


def process_guanaco(fname, to_file):
    '''去除 `instruction` 中没有以 User 开头的样本.'''
    with open(fname, 'r', encoding='utf-8') as f:
        data = json.load(f)
    to_data = []
    drop_cnt = 0
    for item in tqdm(data):
        lang = 'NA'
        turns = []
        ins = item['instruction']
        ins = ins.replace('User：', 'User:')
        ins = ins.replace('Associate', 'Assistant')
        ins = ins.replace('Assistant：', 'Assistant:')
        if not ins.startswith('User:'):
            drop_cnt += 1
            continue
        while 'User' in ins:
            if 'Assistant:' not in ins:
                assistant_idx = len(ins)
            else:
                assistant_idx = ins.index('Assistant:')
            user = ins[:assistant_idx].lstrip('User:').strip()
            ins = ins[assistant_idx:]
            if 'User:' not in ins:
                end_idx = len(ins)
            else:
                end_idx = ins.index('User:')
            assistant = ins[:end_idx].lstrip('Assistant:').strip()
            ins = ins[end_idx:]
            if lang == 'NA':
                lang = detect(user)
            turns.append({
                'user': user,
                'reference': '',
                'assistant': assistant,
            })
        input = item['input'].replace('System：', 'System:')
        input = input.replace('User：', 'User:')
        if input.startswith('System:'):
            if 'User:' not in input:
                drop_cnt += 1
                continue
            user_idx = input.index('User:')
            turns[-1]['assistant'] += input[:user_idx].lstrip('System:').strip()
            input = input[user_idx:]
        user = input.lstrip('User:').strip()
        assistant = item['output']
        turns.append({
            'user': user,
            'reference': '',
            'assistant': assistant,
        })

        if not turns:
            drop_cnt += 1
            continue

        turns = [t for t in turns if t['assistant'] and t['user']]
        to = {
            'dialog_id': get_uuid(),
            'source': 'guanaco',
            'lang': lang,
            'topic': 'NA',
            'turns': turns,
        }
        to_data.append(to)

        # 如果是繁体, 复制一份简体数据
        if lang == 'zh-tw':
            add_turns = deepcopy(turns)
            for t in add_turns:
                t['user'] = convert(t['user'], 'zh-tw')
                t['assistant'] = convert(t['assistant'], 'zh-tw')
                t['reference'] = convert(t['reference'], 'zh-tw')
            to_data.append({
                'dialog_id': get_uuid(),
                'source': 'guanaco',
                'lang': 'zh-cn',
                'topic': 'NA',
                'turns': add_turns,
            })

    print(f'过滤数据个数: {drop_cnt}')
    # 全量数据
    dump_jsonl(to_data, to_file)
    # 英文数据
    en_data = [item for item in to_data if item['lang'] == 'en']
    dump_jsonl(en_data, f'{to_file}.en')
    # 英文数据
    zh_data = [item for item in to_data if 'zh' in item['lang']]
    dump_jsonl(zh_data, f'{to_file}.zh')


def process_gene_data(file_folder, save_dir, source):
    all_mul_turn_info = []
    all_file_list = list()
    get_all_subfile(file_folder, all_file_list)
    with open(save_dir, 'w') as f_w:
        for file_dir in all_file_list:
            with open(file_dir, 'r') as f_r:
                for line in f_r:
                    info = json.loads(line)
                    turns = list()
                    text = info.get("text")
                    for turn in text.split("<human>: ")[1:]:
                        if len(turn.split("\n<bot>: ")) < 2:
                            continue
                        human, bot = turn.split("\n<bot>: ")
                        human = human.strip()
                        bot = bot.strip()
                        turns.append(
                            {
                                "user": human,
                                "reference": None,
                                "assistant": bot
                            }
                        )
                    dialog_info = {
                        "dialog_id": get_uuid(),
                        "source": source,
                        "lang": "english",
                        "topic": "dialog",
                        "context": None,
                        "turns": turns
                    }
                    all_mul_turn_info.append(dialog_info)
                    f_w.write(json.dumps(dialog_info, ensure_ascii=False) + '\n')
        # f_w.write(json.dumps(all_mul_turn_info, ensure_ascii=False))


def get_all_subfile(parent_dir, all_file_list):
    if os.path.isdir(parent_dir):
        sub_dirs = os.listdir(parent_dir)
        for sub_name in sub_dirs:
            if '.DS_Store' == sub_name:
                continue
            get_all_subfile(os.path.join(parent_dir, sub_name), all_file_list)
    elif os.path.isfile(parent_dir):
        all_file_list.append(parent_dir)


def proces_moss_plugin_data(file_folder, save_dir, source):
    save_dir = save_dir + source + '.jsonl'
    all_file_list = list()
    get_all_subfile(file_folder, all_file_list)
    with open(save_dir, 'w') as f_w:
        for file_dir in all_file_list:
            print(file_dir)
            with open(file_dir, 'r') as f_r:
                try:
                    info = json.load(f_r)
                except Exception:
                    continue
                turns = list()
                turn_num = info.get("num_turns")
                chat = info.get("chat")
                for turn_id in range(1, turn_num + 1):
                    turn_info = chat.get("turn_%d" % turn_id)
                    ref = turn_info["Inner Thoughts"].replace(
                        "<|Inner Thoughts|>:", "").replace("<eot>\n", "").replace("None", "").strip()
                    turns.append(
                        {
                            "user": turn_info["Human"].replace("<|Human|>: ", "").replace("<eoh>\n", ""),
                            "reference": ref,
                            "assistant": turn_info["MOSS"].replace("<|MOSS|>: ", "").replace("<eom>\n", "")
                        }
                    )
                dialog_info = {
                    "dialog_id": get_uuid(),
                    "source": source,
                    "lang": "english",
                    "topic": "dialog",
                    "context": "",
                    "turns": turns
                }
                f_w.write(json.dumps(dialog_info, ensure_ascii=False) + '\n')


def proces_moss_data(file_dir, save_dir, source):
    def filter_identity(input: str) -> str:
        '''过滤身份认知字符串.'''
        filter_kws = [
            ("I'm MOSS", ""),
            ("MOSS, 你", "你"),
            ("My name is MOSS", ""),
            ("I am MOSS", ""),
            ("MOSS, 能否", "能否"),
            ("Thank you MOSS", "Thank you"),
            ("MOSS, my helpful assistant", "Hi, my helpful assistant"),
            ("hello MOSS", "hello"),
            ("hi MOSS", "hi"),
            ("This is MOSS", ""),
            ("helpful assistant named MOSS", "helpful assistant"),
            ("Dear MOSS", "Dear"),
            ("As MOSS,", ""),
            ("MOSS, 您", "您"),
            ("MOSS, 请", "请"),
            ("MOSS, 如何", "如何"),
        ]

        for _from, _to in filter_kws:
            input = input.replace(_from, _to)

        return input

    idx = 0
    save_zh_dir = save_dir + source + '_zh.jsonl'
    save_others_dir = save_dir + source + '_others.jsonl'
    save_all_dir = save_dir + source + '_all.jsonl'
    with open(save_all_dir, 'w') as f_w_all, open(save_others_dir, 'w') as f_w_others, open(save_zh_dir, 'w') as f_w_zh:
        with open(file_dir, 'r') as f_r:
            for line in f_r:
                if (idx + 1) % 10000 == 0:
                    print("idx: ", idx)
                info = json.loads(line)
                turns = list()
                turn_num = info.get("num_turns")
                chat = info.get("chat")
                for turn_id in range(1, turn_num + 1):
                    turn_info = chat.get("turn_%d" % turn_id)
                    ref = turn_info["Inner Thoughts"].replace(
                        "<|Inner Thoughts|>:", "").replace("<eot>\n", "").replace("None", "").strip()
                    user = turn_info["Human"].replace("<|Human|>: ", "").replace("<eoh>\n", "")
                    user = filter_identity(user)
                    bot = turn_info["MOSS"].replace("<|MOSS|>: ", "").replace("<eom>\n", "")
                    bot = filter_identity(bot)
                    turns.append(
                        {
                            "user": user,
                            "reference": ref,
                            "assistant": bot,
                        }
                    )
                language = "others"
                try:
                    lang = detect(user)
                except Exception:
                    try:
                        lang = detect(bot)
                    except Exception:
                        lang = 'so'
                if 'zh' in lang:
                    language = "chinese"
                dialog_info = {
                    "dialog_id": get_uuid(),
                    "source": source,
                    "lang": language,
                    "topic": "dialog",
                    "context": "",
                    "turns": turns
                }
                f_w_all.write(json.dumps(dialog_info, ensure_ascii=False) + '\n')
                if 'zh' in lang:
                    f_w_zh.write(json.dumps(dialog_info, ensure_ascii=False) + '\n')
                else:
                    f_w_others.write(json.dumps(dialog_info, ensure_ascii=False) + '\n')


def process_gpt4_data(file_dir, save_dir, source):
    all_mul_turn_info = []
    with open(save_dir, 'w') as f_w, open(file_dir, 'r') as f_r:
        info = json.load(f_r)
        for conv_id, sgl_dia in enumerate(info):
            all_mul_turn_info.append(
                {
                    "dialog_id": get_uuid(),
                    "source": source,
                    "lang": "english",
                    "topic": "dialog",
                    "context": None,
                    "turns": [
                        {
                            "user": sgl_dia["instruction"],
                            "reference": None,
                            "assistant": sgl_dia["output"]
                        }
                    ]
                }
            )
        f_w.write(json.dumps(all_mul_turn_info, ensure_ascii=False))


def process_itag_multurn(file_dir, save_dir, source="gpt3.5"):
    with open(file_dir, 'r', encoding="GBK") as f_r, open(save_dir, 'w') as f_w:
        all_lines = csv.reader(f_r)
        for idx, sgl_line in enumerate(all_lines):
            if idx == 0:
                continue
            ori_multurn = json.loads(sgl_line[4])
            change_info = json.loads(sgl_line[6])
            turns = []
            for i in range(0, len(ori_multurn), 2):
                usr_turn = ori_multurn[i]
                assert usr_turn["speaker"] == "用户"
                robot_turn = ori_multurn[i + 1]
                assert robot_turn["speaker"] == "机器人"
                turns.append({
                    "user": usr_turn["speech"],
                    "reference": None,
                    "assistant": robot_turn["speech"]
                })
            for sgl_info in change_info:
                if "result" not in sgl_info.keys() or len(sgl_info["result"]["修改表述，逻辑等有问题的对话"]) == 0:
                    continue
                paragraph_id = sgl_info["paragraphId"]
                sentence_id = sgl_info["sentenceId"]
                if sentence_id % 2 == 1:
                    turns[paragraph_id - 1]["user"] = sgl_info["result"]["修改表述，逻辑等有问题的对话"]
                else:
                    turns[paragraph_id - 1]["assistant"] = sgl_info["result"]["修改表述，逻辑等有问题的对话"]
            dialog_info = {
                "dialog_id": get_uuid(),
                "source": source,
                "lang": "chinese",
                "topic": "dialog",
                "context": None,
                "turns": turns}
            f_w.write(json.dumps(dialog_info, ensure_ascii=False) + '\n')


def alpaca_chat_data(
    input_file: str,
    output_file: str,
):
    '''处理 Alpaca 对话数据.'''
    all_conversations = []
    data = json.load(open(input_file, 'r'))
    for c, conv in enumerate(data):
        conversations = []
        topic = conv['topic']
        chat_str = '\n'.join(conv['input'].split('\n')[1:])
        if not chat_str.startswith('[|Human|] '):
            __import__('pudb').set_trace()
        prev_role = 'human'
        conversations.append({'from': 'human', 'value': ''})
        prev_pos = 10
        for i in range(8, len(chat_str)):
            if chat_str[i : i + 10] == '[|Human|] ':
                role = 'human'
                prev_turn = chat_str[prev_pos: i]
                prev_pos = i + 10
            elif chat_str[i : i + 7] == '[|AI|] ':
                role = 'robot'
                prev_turn = chat_str[prev_pos: i]
                prev_pos = i + 7
            else:
                continue
            if not prev_turn:
                continue
            if conversations[-1]['value']:
                __import__('pudb').set_trace()
                conversations[-1]['value'] += '\n' + prev_turn
            else:
                conversations[-1]['value'] = prev_turn
            if prev_role != role:
                conversations.append({'from': role, 'value': ''})
                prev_role = role
        if not conversations[-1]['value']:
            conversations = conversations[:-1]
        all_conversations.append({
            'dialog_id': get_uuid(),
            'topic': topic,
            'source': 'alpaca',
            'lang': '',
            'turns': conversations,
        })

    all_conversations = conversation_convert(all_conversations)

    json.dump(all_conversations, open(output_file, 'w'), ensure_ascii=False, indent=2)


def belle_multiturn(
    input_file: str,
    output_file: str,
):
    new_data = []
    fin = open(input_file, 'r')
    human_role = 'Human:'
    robot_role = 'Assistant:'
    for i, line in enumerate(fin):
        data = json.loads(line.rstrip('\n\r'))
        input_str = data['input']
        conversation = []
        while True:
            human_pos = input_str.find(human_role)
            robot_pos = input_str.find(robot_role)
            if human_pos == -1 and robot_pos == -1:
                break
            if human_pos == -1:
                human_pos = 1000000000
            if robot_pos == -1:
                robot_pos = 1000000000
            if human_pos < robot_pos:
                role = human_role
            else:
                role = robot_role
            input_str = input_str[len(role):]
            next_human_pos = input_str.find(human_role)
            next_robot_pos = input_str.find(robot_role)
            if next_human_pos == -1 and next_robot_pos == -1:
                text = input_str
                if not text.strip():
                    text = data['output']
                next_pos = -1
            else:
                if next_human_pos == -1:
                    next_human_pos = 1000000000
                if next_robot_pos == -1:
                    next_robot_pos = 1000000000
                if next_human_pos < next_robot_pos:
                    next_pos = next_human_pos
                else:
                    next_pos = next_robot_pos
                text = input_str[:next_pos]
            if not text:
                continue
            if conversation and role == conversation[-1]['from']:
                conversation[-1]['value'] += '\n' + text.strip()
            else:
                conversation.append(
                    {'from': 'human' if role == 'Human:' else 'robot', 'value': text.strip()})
            if next_pos == -1:
                break
            else:
                input_str = input_str[next_pos:]
        new_data.append({
            'dialog_id': get_uuid(),
            'topic': '',
            'source': 'belle',
            'lang': '',
            'turns': conversation,
        })

    new_data = conversation_convert(new_data)
    json.dump(new_data, open(output_file, 'w'), ensure_ascii=False, indent=2)
    fin.close()


def conversation_convert(conversations: list) -> list:
    for item in conversations:
        turns = item['turns']
        to_turns = []
        for idx in range(0, len(turns), 2):
            if turns[idx] != 'human' or turns[idx + 1] != 'robot':
                raise ValueError(f'用户/机器人没有成对出现, turns: {turns}')

            to_turns.append({
                'user': turns[idx],
                'reference': '',
                'assistant': turns[idx + 1],
            })
        item['turns'] = to_turns


def judge_is_dispute(data):
    # 利用 chatgpt 给出 reddit 中比较有争议话题的讨论社区
    dispute_subreddits = [
        'politicaldiscussion', 'conservative', 'socialism', 'feminism',
        'twoxchromosomes', 'relationship_advice', 'atheism', 'BlackLivesMatter',
        'TheRedPill', 'MensRights', 'Bitcoin', 'AmItheAsshole', 'vegan', 'ProLife',
        'ProChoice', 'ClimateSkeptics', 'Christianity', 'Israel', 'Palestine', 'China',
        'The_Donald', 'ChapoTrapHouse', 'KotakuInAction', 'conspiracy', 'TumblrInAction',
        'TheRedPill', 'TwoXChromosomes', 'MensRights', 'politics', 'atheism',
        'SubredditDrama', '4chan', 'NSFW_GIF', 'ShitRedditSays'
    ]

    is_dispute = False
    subreddits = [
        'level_1_subreddit',
        'level_2_subreddit',
        'level_3_subreddit',
        'level_4_subreddit',
        'level_5_subreddit',
        'level_6_subreddit',
        'level_7_subreddit',
        'level_8_subreddit',
        'level_9_subreddit',
        'level_10_subreddit',
        'level_11_subreddit',
        'level_12_subreddit',
        'level_13_subreddit',
        'level_14_subreddit',
        'level_15_subreddit',
    ]

    for sub in subreddits:
        if data[sub] in dispute_subreddits:
            is_dispute = True
            break
        
    return is_dispute


def is_url_encoded(string):
    # 判断是否包含 url encode
    decoded_string = unquote(string)
    return decoded_string != string


remove_nota = u'[’·°–!"#$%&\'()*+,-./:;<=>?@，。?★、…【】（）《》？“”‘’！[\\]^_`{|}~]+'
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)


def filter_str(sentence):
    sentence = re.sub(remove_nota, '', sentence)
    sentence = sentence.translate(remove_punctuation_map)
    return sentence.strip()


def judge_language(s):
    s = filter_str(s)
    result = []
    s = re.sub('[0-9]', '', s).strip()
    # unicode english
    re_words = re.compile(u"[a-zA-Z]")
    res = re.findall(re_words, s)  
    res2 = re.sub('[a-zA-Z]', '', s).strip()
    if len(res) > 0:
        result.append('en')
    if len(res2) <= 0:
        return 'en'

    # unicode chinese
    re_words = re.compile(u"[\u4e00-\u9fa5]+")
    res = re.findall(re_words, s)
    res2 = re.sub(u"[\u4e00-\u9fa5]+", '', s).strip()
    if len(res) > 0:
        result.append('zh')
    if len(res2) <= 0:
        return 'zh'

    # unicode korean
    re_words = re.compile(u"[\uac00-\ud7ff]+")
    res = re.findall(re_words, s)
    res2 = re.sub(u"[\uac00-\ud7ff]+", '', s).strip()
    if len(res) > 0:
        result.append('ko')
    if len(res2) <= 0:
        return 'ko'

    # unicode japanese katakana and unicode japanese hiragana
    re_words = re.compile(u"[\u30a0-\u30ff\u3040-\u309f]+")
    res = re.findall(re_words, s)
    res2 = re.sub(u"[\u30a0-\u30ff\u3040-\u309f]+", '', s).strip()
    if len(res) > 0:
        result.append('ja')
    if len(res2) <= 0:
        return 'ja'
    return ','.join(result)


def reddit_data_process(fname, tofname, chunksize=100000):
    '''
    处理逻辑说明见文档：https://yuque.antfin-inc.com/crtpg4/xutwxe/lldkzpxskl7uu8nx
    '''
    choose_data = []
    lawless_num = 0
    chunk_num = 0
    for chunk in tqdm(pd.read_csv(fname, chunksize=chunksize)):
        chunk_num += 1
        print(f'chunk num = {chunk_num}')
        chunk.fillna('', inplace=True)
        for index, row in chunk.iterrows():
            is_dispute = judge_is_dispute(row)
            assert isinstance(row['raw_over_18'], bool)
            assert isinstance(row['is_match'], bool)

            if row['is_match']:
                lawless_num += 1
                if lawless_num % 10000 == 0:
                    print(f'lawless_num = {lawless_num}')
            
            if not row['raw_over_18'] and (not is_dispute) and (not row['is_match']):
                try:
                    dialog_str_roles = row['dialog_str_roles']
                    dialogs = dialog_str_roles.split('<n>')
                    dialogs_update = []
                    for dialog in dialogs:
                        if dialog.split(': ')[1] and dialog.split(': ')[1] not in ['[removed]', '[deleted]']:
                            dialog = re.sub(r'http\S+', 'url', dialog)
                            if 'en' in judge_language(dialog.strip()) or 'zh' in judge_language(dialog.strip()):
                                lang_flag = True
                            else:
                                lang_flag = False
                            if lang_flag and (not is_url_encoded(dialog.strip())):
                                dialogs_update.append(dialog.strip())
                    
                    if len(dialogs_update) >= 2:
                        dialog_str_roles_update = '<n>'.join(dialogs_update)
                        choose_data.append({'content': dialog_str_roles_update})
                        if len(choose_data) % 10000 == 0:
                            print(f'num = {len(choose_data)}')
                except Exception:
                    continue

        del chunk
    
    dump_jsonl(choose_data, tofname)
    print(f'len = {len(choose_data)}')


if __name__ == "__main__":
    fire.Fire()
