'''约定的一些数据结构定义.'''
import re


class DialogPrompt:
    '''生成多轮 Prompt 所用的模板.'''
    context = '对话背景信息'
    ref = '对话参考信息'
    user = '用户'
    bot = '机器人'
    context_sep = f'{context}:'
    ref_sep = f'{ref}:'
    user_sep = f'\n{user}:'
    bot_sep = f'\n{bot}:'
    turn_idx = '第{}轮'
    turn_sep = f'轮\n{user}:'
    turn_pattern = re.compile('第.*?轮')
