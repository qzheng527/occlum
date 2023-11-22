'''多轮对话 sft 数据集相关数据处理逻辑.'''
import re
import fire
import random
import logging
from typing import Tuple, List, Optional, Dict, Any, Union
from transformers import PreTrainedTokenizer
from solutions.antllm.datachain.utils import load_jsonl
from solutions.antllm.datachain.utils import dump_jsonl
from solutions.antllm.datachain.utils import get_uuid
from solutions.antllm.antllm.data.data_class import DialogPrompt
from solutions.antllm.datachain.chain.base import DataChain


logger = logging.getLogger(__file__)


class DialogDataConvertChain(DataChain):
    '''多轮对话格式的转换处理器.'''

    def __init__(
        self,
        is_prompt_to_turns: Optional[bool] = None,
        is_turns_to_prompt: Optional[bool] = None,
        prompt_truncate_type: Optional[str] = None,
        prompt_type: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_tokens: Optional[int] = None,
        turn_sep: Optional[str] = None,
        max_workers: int = 16,
        verbose: bool = True,
    ):
        '''关键词过滤处理器, 如果命中了关键词返回 None.

        Params:
            is_prompt_to_turns: `Optional[bool]`, 

            is_turns_to_prompt: `Optional[bool]`, 

        prompt_truncate_type: `Optional[str]`, 轮次截断方式, 默认为空, 使用 turn 方式截断
            token: 按字符串进行截断
            turn: 按轮进行截断
            turn_window: 最大长度当做阶段窗口, 截成多个样本. Examples: abcdefg -> abc, defg
            reserve_user: 按轮截断, 并保留用户问题

        prompt_type: `Optional[str]`, 生成多轮 prompt 类型, 默认为空, 使用 random_turn 方式
            last_turn - 最后一轮 assistant 回复当做 output
            random_turn - 随机选择一轮 assistant (大于 1 轮)回复当做 output, 并丢弃后面轮
            every_turn - 样本有几轮对话, 就生成几个 prompt-output 样本, 生成大于 2 轮数据
            turn_sep - 将每一轮最后用分隔符 `turn_sep` 分隔

        max_tokens: `Optional[int]`, 最大保留 token 长度

        turn_sep: `Optional[str]`, 每轮最后的分隔符

        '''
        super().__init__(max_workers=max_workers, verbose=verbose)
        self.is_prompt_to_turns = is_prompt_to_turns
        self.is_turns_to_prompt = is_turns_to_prompt
        self.prompt_truncate_type = prompt_truncate_type
        self.prompt_type = prompt_type
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.turn_sep = turn_sep

    def run(
        self,
        inputs: Dict[str, Any] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]], None]:
        '''多轮对话数据转换.'''
        if self.is_prompt_to_turns:
            return self._prompt_2_turns(inputs)
        elif self.is_turns_to_prompt:
            return self._turns_2_prompt(inputs)
        else:
            raise ValueError(
                '`DialogDataConvertChain` need at least one input param between'
                '`is_prompt_to_turns/is_turns_to_prompt`'
            )

    def load(self, input_path=None, **kwargs) -> List[Dict[str, Any]]:
        return load_jsonl(input_path)

    def save(self, output_path=None, **kwargs):
        dump_jsonl(self._outputs, output_path)

    def _prompt_2_turns(
        self,
        inputs: Dict[str, Any] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]], None]:
        '''将多轮对话 prompt 格式反向转成 turns 格式.'''
        input: str = inputs['input']
        output: str = inputs['output']

        context = ''
        source = inputs.get('source')
        lang = inputs.get('lang', '')
        topic = inputs.get('topic', '')
        turns = []
        turn_match = DialogPrompt.turn_pattern.search(input)
        if not turn_match:
            logger.warning(
                'input not valid, turn_pattern not match.\n'
                f'turn_pattrn: {DialogPrompt.turn_pattern}, '
                f'input: {input}'
            )
            return
        turn_idx = int(input[turn_match.start():turn_match.end()].replace('第', '').replace('轮', ''))

        if inputs.get('type') != 'multiturn_dialog' and DialogPrompt.bot_sep not in input:
            # 单轮指令
            return {
                'dialog_id': get_uuid(),
                'context': '',
                'source': source if source else '单轮指令',
                'lang': lang,
                'topic': topic,
                'turns': [{
                    'user': input,
                    'ref': '',
                    'assistant': output,
                }],
            }

        if input.startswith(DialogPrompt.bot):
            logger.warning(
                'inputs not valid, inputs startswith '
                f'{DialogPrompt.bot}\ninputs: {inputs}'
            )
            return None

        if input.startswith(DialogPrompt.context_sep):
            user_idx = input.index(DialogPrompt.user_sep)
            if DialogPrompt.ref in input[:user_idx]:
                end_idx = input.index(DialogPrompt.ref_sep)
            else:
                end_idx = user_idx
            context = input[:end_idx].strip().lstrip(
                DialogPrompt.context_sep).replace(f'第{turn_idx}轮', '').strip()
            input = input[end_idx:]

        while DialogPrompt.user_sep in input:
            input = input.replace(DialogPrompt.turn_idx.format(turn_idx), '')
            turn = {}
            user_idx = input.index(DialogPrompt.user_sep)
            turn['reference'] = ''
            if input.startswith(DialogPrompt.ref_sep):
                turn['reference'] = input[:user_idx].strip().lstrip(
                    DialogPrompt.ref_sep).strip()
                input = input[user_idx:]
                user_idx = input.index(DialogPrompt.user_sep)
            if DialogPrompt.bot_sep in input:
                bot_idx = input.index(DialogPrompt.bot_sep)
                next_turn_match = DialogPrompt.turn_pattern.search(input)
                if next_turn_match and bot_idx > next_turn_match.start():
                    sep_idx = next_turn_match.start()
                else:
                    sep_idx = bot_idx
            else:
                sep_idx = len(input)

            turn['user'] = input[user_idx:sep_idx].strip().lstrip(f'{DialogPrompt.user}:').strip()
            input = input[sep_idx:]
            if DialogPrompt.user_sep in input:
                user_idx = input.index(DialogPrompt.user_sep)
                turn['assistant'] = input[:user_idx].strip().lstrip(
                    f'{DialogPrompt.bot}:').replace(
                        DialogPrompt.turn_idx.format(turn_idx + 1), '').strip()
                if DialogPrompt.turn_pattern.search(turn['assistant']):
                    turn['assistant'] = re.sub(DialogPrompt.turn_pattern, '', turn['assistant'])
                input = input[user_idx:]
            elif input.startswith(DialogPrompt.bot_sep):
                turn['assistant'] = input.replace(DialogPrompt.bot_sep, '').strip()

            if not turn['user']:
                turns = []
                logger.warning(f'没有检测到用户 query, 该样本跳过, turn: {turn}')
                break
            if DialogPrompt.user_sep in turn['user']:
                turns = []
                logger.warning(
                    f'用户 prompt `{DialogPrompt.user_sep}` 存在于用户 query, '
                    f'该样本跳过, turn: {turn}'
                )
                break
            if 'assistant' in turn and f'{DialogPrompt.bot_sep}' in turn['assistant']:
                turns = []
                logger.warning(
                    f'机器人 prompt `{DialogPrompt.bot_sep}` 存在于机器人 output, '
                    f'该样本跳过, turn: {turn}'
                )
                break

            if 'assistant' not in turn:
                turn['assistant'] = ''

            turn_idx += 1
            turns.append(turn)

        if not turns:
            logger.warning(
                f'inputs not valid, turns is empty\ninputs: {inputs}'
            )
            return None

        if output:
            turns[-1]['assistant'] = output

        return {
            'dialog_id': get_uuid(),
            'context': context,
            'source': source,
            'lang': lang,
            'topic': topic,
            'turns': turns,
        }

    def _turns_2_prompt(
        self,
        inputs: Dict[str, Any] = None
    ) -> Union[List[Dict[str, Any]], None]:
        '''生成对话格式的 prompt data. 如果超过模型最大输入长度, 按轮进行截断.'''
        # 前处理-去除用户机器人对话前后换行符
        for turn in inputs['turns']:
            for k, v in turn.items():
                if not v:
                    continue
                turn[k] = v.strip()

        input = ''
        if inputs.get('context'):
            input = f'{DialogPrompt.context}: {inputs["context"]}\n'

        inputs['turns'] = [t for t in inputs['turns'] if t['user']]

        if self.prompt_type in ['last_turn', 'random_turn']:
            output = ''
            if self.prompt_type == 'last_turn':
                output_idx = len(inputs['turns']) - 1
            elif self.prompt_type == 'random_turn':
                if len(inputs['turns']) == 1:
                    output_idx = 0
                else:
                    output_idx = random.randint(1, len(inputs['turns']) - 1)

            for idx, turn in enumerate(inputs['turns']):
                if turn.get('reference'):
                    input += f'{DialogPrompt.ref}: {turn["reference"]}\n'
                input += f'第{idx + 1}轮\n{DialogPrompt.user}: {turn["user"]}\n'
                if turn['assistant'] or idx == len(inputs['turns']) - 1:
                    input += f'{DialogPrompt.bot}: '
                if idx == output_idx:
                    output = f'{turn["assistant"]}\n'
                    break
                elif turn['assistant']:
                    input += f'{turn["assistant"]}\n'

            prompts = truncate_to_prompt_data(
                input,
                output,
                self.tokenizer,
                self.max_tokens,
                self.prompt_type,
                trunc_type=self.prompt_truncate_type,
                source=inputs.get('source', '')
            )
            return prompts

        if self.prompt_type == 'every_turn':
            to_data = []
            for idx, turn in enumerate(inputs['turns']):
                if turn.get('reference'):
                    input += f'{DialogPrompt.ref}: {turn["reference"]}\n'
                input += f'第{idx + 1}轮\n{DialogPrompt.user}: {turn["user"]}\n{DialogPrompt.bot}: '
                output = f'{turn["assistant"]}\n'
                if idx == 0:
                    input += output
                    continue
                prompts = truncate_to_prompt_data(
                    input,
                    output,
                    self.tokenizer,
                    self.max_tokens,
                    self.prompt_type,
                    trunc_type=self.prompt_truncate_type,
                    source=inputs.get('source', '')
                )
                to_data.extend(prompts)
                input += output

            return to_data

        if self.prompt_type == 'turn_sep':
            for idx, turn in enumerate(inputs['turns']):
                if turn.get('reference'):
                    input += f'{DialogPrompt.ref}: {turn["reference"]}\n'
                input += (
                    f'第{idx + 1}轮\n{DialogPrompt.user}: {turn["user"]}\n'
                    f'{DialogPrompt.bot}: {turn["assistant"]}\n{self.turn_sep}'
                )

            prompts = truncate_to_prompt_data(
                input,
                '',
                self.tokenizer,
                self.max_tokens,
                self.prompt_type,
                trunc_type=self.prompt_truncate_type,
                source=inputs.get('source', '')
            )
            return prompts


def truncate_to_prompt_data(
    input: str,
    output: str,
    tokenizer,
    max_tokens: int,
    prompt_type: str,
    trunc_type: str = 'turn',
    source: str = None,
) -> List[dict]:
    '''将对话截断成 prompt data.

    Params:
        input: 大模型输入 input
        output: 大模型输出 output
        max_tokens: input 截断 token 数
        prompt_type: 生成多轮 prompt 类型
        trunc_type: 轮次截断方式
        source: 数据源名称
    '''
    trunc_input = trunc_dialog_input(input, tokenizer, max_tokens, trunc_type=trunc_type)

    if isinstance(trunc_input, str):
        return [{
            'input': trunc_input.strip(),
            'output': output.strip(),
            'type': 'multiturn_dialog',
            'source': source,
        }]

    prompts = []
    trunc_data = multi_input_2_data(trunc_input)
    for item in trunc_data:
        item_input = item['input']
        item_output = item['output'] if item['output'] else output
        if prompt_type == 'turn_sep':
            item_input = item_input + item_output
            item_output = ''
        # 确定 input 中有 dialog prompt
        if DialogPrompt.user not in item_input or DialogPrompt.bot not in item_input:
            continue
        prompts.append({
            'input': item_input.strip(),
            'output': item_output.strip(),
            'type': 'multiturn_dialog',
            'source': source,
        })

    return prompts


def multi_input_2_data(str_list: List[str]) -> List[dict]:
    '''将多个对话 input 转成 prompt input-output.

    示例:
        对话 input: "第1轮\n用户: u1\n机器人: a1\n第2轮\n用户: u2\n机器人: a2\n"
        输出 prompt: {"input": "第1轮\n用户: u1\n机器人: a1\n第2轮\n用户: u2\n机器人:", "output": "a2"}

    Params:
        str_list: 对话 input 列表
    '''
    bot_pattern = re.compile(f' :{DialogPrompt.bot[::-1]}\n')
    data = []
    for in_str in str_list:
        span = bot_pattern.search(in_str[::-1])
        last_bot_idx = span.start() if span else 0
        if last_bot_idx == 0:
            input = in_str
            output = ''
        else:
            input = in_str[:-last_bot_idx]
            output = in_str[-last_bot_idx:]
        data.append({
            'input': input,
            'output': output,
        })

    return data


def trunc_dialog_input(
    input: str,
    tokenizer,
    max_tokens: int = None,
    trunc_type: str = 'turn',
) -> Tuple[str, List[str]]:
    '''将 input 按需截断.

    Params:
        input: 需要截断的字符串
        max_tokens: 保留的最大 token 长度
        trunc_type: 轮次截断方式
            token: 按 token 进行截断
            turn: 按轮进行截断
            turn_window: 最大长度当做阶段窗口, 截成多个样本. Examples: abcdefg -> [abc, defg]
            reserve_user: 按轮截断, 并保留用户问题
    '''
    if not max_tokens:
        return input

    input_tokens = tokenizer.tokenize(input)

    if max_tokens >= len(input_tokens):
        return input

    if trunc_type == 'token':
        return tokens_to_string(tokenizer, input_tokens[-max_tokens:])

    turn_pattern = DialogPrompt.turn_pattern
    bot_pattern = re.compile(f'\n{DialogPrompt.bot}: ')

    user_input = ''
    input_windows = []
    win_start = 0
    previous_start = 0

    # TODO 提高长对话截断效率
    tokens = []
    for turn_start, bot_start in zip(
        turn_pattern.finditer(input),
        bot_pattern.finditer(input),
    ):
        if trunc_type == 'turn':
            tokens = tokenizer.tokenize(input[turn_start.start():])
            if max_tokens >= len(tokens):
                return tokens_to_string(tokenizer, tokens)
        elif trunc_type == 'reserve_user':
            tokens = tokenizer.tokenize(user_input + input[turn_start.start():])
            if max_tokens >= len(tokens):
                return tokens_to_string(tokenizer, tokens)
            user_input += input[turn_start.start():bot_start.start()]
        elif trunc_type == 'turn_window':
            tokens = tokenizer.tokenize(input[turn_start.start():])
            if max_tokens >= len(tokens):
                previous_window = input[win_start:turn_start.start()]
                input_windows.append(previous_window)
                input_windows.append(tokens_to_string(tokenizer, tokens))
                return input_windows
            else:
                previous_tokens = tokenizer.tokenize(input[win_start:turn_start.start()])
                if len(previous_tokens) > max_tokens:
                    if win_start == previous_start:
                        window_tokens = tokens_to_string(tokenizer, previous_tokens)
                        win_start = turn_start.start()
                    else:
                        window_tokens = tokenizer.tokenize(input[win_start:previous_start])
                        win_start = previous_start
                    input_windows.append(tokens_to_string(tokenizer, window_tokens))
            previous_start = turn_start.start()
        else:
            raise ValueError(f'not support trunc: {trunc_type}')

    if not tokens:
        return input

    # 无法截断, 只保留最后一轮
    return tokens_to_string(tokenizer, tokens)


def tokens_to_string(tokenizer, tokens: list) -> str:
    '''将 token 转成 string'''
    res = tokenizer.convert_tokens_to_string(tokens)
    res = res.replace('<n>', '\n')
    return res.strip()


if __name__ == "__main__":
    fire.Fire()
