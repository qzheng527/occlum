'''多轮对话数据集转换脚本.'''
import fire
from solutions.antllm.datachain.utils import dump_jsonl
from solutions.antllm.datachain.utils import load_jsonl
from solutions.antllm.datachain.chain.multiturn_dialogs.convert import DialogDataConvertChain
from solutions.antllm.antllm.models.glm.tokenization_glm import GLMTokenizer
from solutions.antllm.datachain.chain.cleaner.keyword_filter import KeywordsFilterChain
from solutions.antllm.datachain.chain.cleaner.keyword_replace import KeywordsReplaceChain


def dialog_prompt_2_turns(
    in_file: str,
    to_file: str,
    max_workers: int = 16,
):
    '''将多轮对话 prompt 格式反向转成 turns 格式.'''
    convert_chain = DialogDataConvertChain(
        is_prompt_to_turns=True,
        max_workers=max_workers,
    )
    data = convert_chain.load(in_file)
    convert_chain.batch_run(data, concurrency='process')
    convert_chain.save(to_file)


def dialog_turns_2_prompt(
    in_file: str,
    to_file: str,
    max_tokens: int,
    tokenizer_path: str,
    trunc_type: str = 'turn',
    prompt_type: str = 'random_turn',
    turn_sep: str = '',
    max_workers: int = 16,
):
    '''将多轮 multiturns 结构转成大模型要求的对话格式 prompt data. 如果超过模型最大输入长度, 按轮进行截断.

    Params:
        trunc_type: 轮次截断方式
            token: 按字符串进行截断
            turn: 按轮进行截断
            turn_window: 最大长度当做阶段窗口, 截成多个样本. Examples: abcdefg -> abc, defg
            reserve_user: 按轮截断, 并保留用户问题

        prompt_type: 生成多轮 prompt 类型
            last_turn - 最后一轮 assistant 回复当做 output
            random_turn - 随机选择一轮 assistant (大于 1 轮)回复当做 output, 并丢弃后面轮
            every_turn - 样本有几轮对话, 就生成几个 prompt-output 样本, 生成大于 2 轮数据
            turn_sep - 将每一轮最后用分隔符 `turn_sep` 分隔
    '''
    # 过滤和替换
    data = load_jsonl(in_file)
    filter_chain = KeywordsFilterChain(keys=['turns'], max_workers=max_workers)
    replace_chain = KeywordsReplaceChain(keys=['turns'], max_workers=max_workers)
    to_data = filter_chain.batch_run(data, concurrency='process')
    to_data = replace_chain.batch_run(to_data, concurrency='process')

    # prompt 数据生成
    convert_chain = DialogDataConvertChain(
        is_turns_to_prompt=True,
        prompt_truncate_type=trunc_type,
        prompt_type=prompt_type,
        tokenizer=GLMTokenizer.from_pretrained(tokenizer_path),
        max_tokens=max_tokens,
        turn_sep=turn_sep,
    )
    to_data = convert_chain.batch_run(to_data, concurrency='process')

    # 对 input output 去重
    unique_set = set()
    excludes = set()
    for idx, item in enumerate(to_data):
        uni_str = item['input'] + item['output']
        if uni_str in unique_set:
            excludes.add(idx)
        else:
            unique_set.add(uni_str)
    print(f'去重个数: {len(excludes)}')

    to_data = [item for idx, item in enumerate(to_data) if idx not in excludes]

    dump_jsonl(to_data, to_file)


if __name__ == "__main__":
    fire.Fire()
