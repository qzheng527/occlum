'''多轮对话接口测试'''
import fire
import re
import string
import zhon.hanzi
import logging
from collections import Counter
from nltk import ngrams
from difflib import SequenceMatcher
from transformers import PreTrainedTokenizer
from solutions.antllm.datachain.chain.multiturn_dialogs.convert import DialogDataConvertChain
from solutions.antllm.antllm.data.data_class import DialogPrompt


logger = logging.getLogger(__file__)


def preprocess_chat_input_truncate(
    input: str,
    tokenizer: PreTrainedTokenizer,
    max_tokens: int,
):
    '''对大模型 chat input 进行按 turn 截断.'''
    try:
        return preprocess_chat_input(
            input=input,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            filter_chat=False,
        )
    except Exception:
        logger.exception(f'preprocess_chat_input_truncate failed, input: {input}')
        return input


def preprocess_chat_input_filter_repeat(
    input: str,
    tokenizer: PreTrainedTokenizer,
    max_tokens: int,
):
    '''对大模型 chat input 进行按重复回复过滤.'''
    try:
        return preprocess_chat_input(
            input=input,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            filter_chat=True,
        )
    except Exception:
        logger.exception(f'preprocess_chat_input_filter_repeat failed, input: {input}')
        return input


def preprocess_chat_input(
    input: str,
    tokenizer: PreTrainedTokenizer,
    max_tokens: int,
    filter_chat: bool = False,
) -> str:
    '''对大模型 chat input 进行前处理.'''
    if not DialogPrompt.turn_pattern.search(input):
        return input

    prompt2turn_chain = DialogDataConvertChain(is_prompt_to_turns=True)
    turn2prompt_chain = DialogDataConvertChain(
        is_turns_to_prompt=True,
        prompt_truncate_type='turn',
        prompt_type='last_turn',
        tokenizer=tokenizer,
        max_tokens=max_tokens,
    )
    dialog = prompt2turn_chain.run({'input': input, 'output': ''})

    if filter_chat and len(dialog['turns']) > 1:
        dialog['turns'] = _filter_turns(dialog['turns'])

    if len(dialog['turns']) == 1:
        return dialog['turns'][0]['user']

    output = turn2prompt_chain.run(dialog)

    if not output:
        print(f'前处理 chat input 失败: {input}')

    return output[0]['input']


def _filter_turns(
    turns: list,
    n: int = 6,
    x: int = 5,
) -> list:
    '''对每一轮对话进行过滤, 规则:
        1. 如果某一轮机器人回复内部有超过 x 次 ngram 重复, 则将机器人回复删除
        2. 如果某几轮机器人回复之间有超过 n 个 gram 重复, 则将机器人回复删除
    '''
    # 规则 1
    for turn in turns:
        if not turn['assistant']:
            continue
        ngram = _get_ngram(turn['assistant'], n=n)
        counter = Counter(ngram)
        if not counter:
            continue
        if max(counter.values()) > x:
            turn['assistant'] = ''

    # 规则 2
    for a_i in range(len(turns)):
        for b_i in range(a_i + 1, len(turns)):
            sm = SequenceMatcher(
                a=turns[a_i]['assistant'],
                b=turns[b_i]['assistant']
            )
            repeats = [item for item in sm.get_matching_blocks() if item.size > n]
            if repeats:
                turns[a_i]['assistant'] = ''
                turns[b_i]['assistant'] = ''

    return turns


def _get_ngram(input: str, n: int = 6) -> list:
    '''提取字符串 min-max gram'''
    ngram = []
    for grams in ngrams(input, n=n):
        ngram.append(''.join(grams))

    return ngram


def _remove_puncs(input: str) -> str:
    '''去除标点.'''
    puncs = [string.punctuation, zhon.hanzi.punctuation]
    punc_str = ''.join(puncs)
    return re.sub('[{}]'.format(re.escape(punc_str)), '', input)


if __name__ == "__main__":
    fire.Fire()
