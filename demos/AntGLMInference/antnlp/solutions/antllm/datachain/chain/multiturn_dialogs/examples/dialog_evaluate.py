# coding=utf-8
# @Author: ytt360131
# @Date: 2023-09-08
import re
import csv
import json
import fire
import random
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from solutions.antllm.datachain.chain.multiturn_dialogs.examples.dialog_gen import get_oss_file
from solutions.antllm.datachain.chain.multiturn_dialogs.examples.dialog_gen import get_llm_multurn_res
from solutions.antllm.datachain.llms.ant_openai import AntOpenAI


WINRATE_PROMPT = (
    '已知历史对话1如下:\n{0}\n用户当前输入如下:\n{1}\n模型A根据历史对话1和用户当前输入生成如下答案:\n{2}\n。'
    '同样的，已知历史对话2如下:\n{3}\n用户当前输入如下:\n{4}\n模型B根据历史对话2和用户当前输入生成如下答案:\n{5}\n。'
    '现在你是一名专业的、公正的模型评估师，请比较模型A和模型B模型生成答案的好坏，评估标准：'
    '1.仅对模型当前轮输出答案质量做比较，不考虑历史回答的好坏；'
    '2.当用户问题比较明>确时，模型输出不需要重复提问；'
    '3.模型不能胡编乱造，必须基于事实；'
    '4.当用户问最新科技、新闻、综艺、影视等等实时信息时，因为模型学习的不是最新知识，所以不应该给出具体的回答；'
    '5.希望模型答案具备真实性、帮助性、多样性、直接回答用户问题；'
    '6.不能只根据答案长度判断答案好坏。'
    '输出形式：\n答案：[模型A明显更好，模型B明显更好，两个模型差不多]中选一个，并简要解释说明，解释长度小于40'
)


def write_csv(save_dir, all_info):
    with open(save_dir, 'w', encoding='utf-8-sig') as f_w:
        writer = csv.writer(f_w)
        writer.writerows(all_info)


def jaccard_score_match(src, tgt):
    src_set = set(src)
    tgt_set = set(tgt)
    intersections = len(src_set & tgt_set)
    unions = len(src_set | tgt_set)
    if unions == 0:
        score = 0
    else:
        score = intersections / unions
    return score


def preprocess_text(text):
    # 使用正则表达式去掉所有的标点符号、空格、换行符等非字母、数字和汉字的字符
    text = text.lower()
    text = re.sub(r'[^\w\d一-龥]', '', text)
    return text


def kyw_filter(query, history_query, response):
    """"
    判断对话是否重复时，做一些前置过滤，比如一些拒识类的标准回答
    
    Input
        query: 用户当前轮输入
        history_query: 历史用户的输入
        reponse: 当前轮机器人的回答
    """
    oss_dir = 'oss://antsys-adabrain/datasets/llm_multiturn/eval/自建评估数据/supp_kyw_regex.jsonl'
    regex_file = get_oss_file(oss_dir)
    flag_filter = False
    with open(regex_file, 'r', encoding='utf-8') as f_r:
        all_regex_kyw = json.load(f_r)
        all_query_regex_kyw = all_regex_kyw["query_regex"]
        all_response_regex_kyw = all_regex_kyw["response_regex"]
        for sgl_query_regex in all_query_regex_kyw:
            if all(kyw in query for kyw in sgl_query_regex):
                flag_filter = True
                break
        if not flag_filter:
            for sgl_query_regex in all_query_regex_kyw:
                if all(kyw in history_query for kyw in sgl_query_regex):
                    flag_filter = True
                    break
        if not flag_filter:
            for sgl_response_regex in all_response_regex_kyw:
                if all(kyw in response for kyw in sgl_response_regex):
                    flag_filter = True
                    break
    return flag_filter


def is_wrds_repetition(text, ngram=3, steps=6, sent_length_thr=50, repeat_times_thr=6):
    """
    判断模型生成输出单轮内部是否有很多重复的词语
    1.文本的长度不会太短
    2.为了计算高效，每次从中间取一个子字符串，判断是否重复多次

    Return
        flag_repeat: True/False 分别表示单轮中重复/非重复
    """
    text = preprocess_text(text)
    flag_repeat = False
    if len(text) > sent_length_thr:
        half_idx = int(0.5 * len(text))
        for step in range(steps):
            ref = text[half_idx + step: half_idx + ngram + step]
            if text.count(ref) > repeat_times_thr:
                flag_repeat = True
                break
    return flag_repeat


def check_wrds_repetition(all_info, wrds_repition_save_dir='./wrds_repition_save_dir.txt'):
    """
    对所有生成结果的每一轮生成的回复都进行重复性评估，仅评估生成的句子内部是否有重复
    """
    total_turn = 0
    total_session = 0
    repitition_turn = 0
    repitition_session = 0
    f_w = open(wrds_repition_save_dir, 'w', encoding='utf-8')
    for line_idx, info in enumerate(all_info):
        multurns = json.loads(info[3])
        flag_repition_session = False
        for sentence_id in range(0, len(multurns), 2):
            assert multurns[sentence_id]["speaker"] == "用户"
            assert multurns[sentence_id + 1]['speaker'] == "机器人"
            robot_speech = multurns[sentence_id + 1]["speech"]
            if is_wrds_repetition(robot_speech):
                repitition_turn += 1
                flag_repition_session = True
                multurns[sentence_id + 1]["is_wrds_repeat"] = "1"
            total_turn += 1
        if flag_repition_session:
            repitition_session += 1
            f_w.write(json.dumps(multurns, ensure_ascii=False) + '\n')
        total_session += 1
    f_w.close()
    return repitition_turn / total_turn, repitition_session / total_session


def check_turns_repeat(all_info, 
                       error_save_dir="./error_session_record.txt", 
                       itag_save_dir="./itag_repetiton_record.csv",
                       jaccard_thr=0.5, 
                       len_absdiff_thr=50, 
                       len_ratiodiff_thr=0.7, 
                       res_len_thr=7):
    """
    Description
        在多轮对话中，判断生成的对话和历史对话是否重复：
        1.生成话术和历史生成重复；
        2.用户的问法不一样；
        3.但是这种直接判断比较还是有一些粗糙，比如“您好”和“您好啊”输出是可以相通同的；

    INPUT
        all_info：模型预测结果信息
        [
            [contextId, query_list, cate, result_list]
        ]
    """
    error_turns = 0
    error_session = 0
    total_turn = 0
    total_session = 0
    f_w = open(error_save_dir, 'w', encoding='utf-8')
    all_repetition_list = []
    for line_idx, info in enumerate(all_info):
        multurns = json.loads(info[3])
        flag_error_session = False
        for sentence_id in range(0, len(multurns), 2):
            assert multurns[sentence_id]["speaker"] == "用户"
            assert multurns[sentence_id + 1]['speaker'] == "机器人"
            usr_speech = preprocess_text(multurns[sentence_id]["speech"])
            robot_speech = preprocess_text(multurns[sentence_id + 1]["speech"])
            if len(robot_speech) == 0:  # TODO 兼容模型输出为空的case
                continue
            # 当前生成结果和历史对话中的每一条都会比较
            for history_sentence_id in range(0, sentence_id, 2):
                assert multurns[history_sentence_id]["speaker"] == "用户"
                assert multurns[history_sentence_id + 1]['speaker'] == "机器人"
                history_usr_speech = preprocess_text(multurns[history_sentence_id]["speech"])
                history_robot_speech = preprocess_text(multurns[history_sentence_id + 1]["speech"])
                if len(history_robot_speech) == 0:
                    continue
                # 1.如果输入相似度比较高，回答一致不算重复 TODO：可以替换成更好的相似评估模型
                if jaccard_score_match(usr_speech, history_usr_speech) > jaccard_thr:
                    continue
                # 2.如果两个回复的长度差异比较大，不可能会存在回答重复的问题
                if len(robot_speech) < len(history_robot_speech):
                    short_response = robot_speech
                    long_response = history_robot_speech
                else:
                    short_response = history_robot_speech
                    long_response = robot_speech
                if len(short_response) / len(long_response) < len_ratiodiff_thr or \
                        len(long_response) - len(short_response) > len_absdiff_thr:
                    continue
                # 3.一些比较短的生成话术一般不是重复，比如“嗯嗯”/“好的”等等
                if len(short_response) < res_len_thr:
                    continue
                # 4.一些拒识类的答复，有标准答案的回复不能算重复。TODO 构建数据后可以训练一个分类模型
                if kyw_filter(usr_speech, history_usr_speech, robot_speech):
                    continue
                # 5.不存在上述情况均不存在，如果两个句子有很多重复，则说明回答重复了
                if short_response[int(0.05 * len(short_response)): int(0.95 * len(short_response))] in long_response:
                    flag_error_session = True
                    error_turns += 1
                    multurns[sentence_id + 1]["is_repeat"] = "1"
                    multurns[sentence_id + 1]["history_query"] = history_usr_speech
                    multurns[sentence_id + 1]["history_response"] = history_robot_speech
                    all_repetition_list.append(f"src_query: \n{usr_speech}\nsrc_response: \n{robot_speech}\ntgt_query: \
                                               \n{history_usr_speech}\ntgt_response: \n{history_robot_speech}")
                    break       
            total_turn += 1
        if flag_error_session:
            error_session += 1
            f_w.write(json.dumps(multurns, ensure_ascii=False) + '\n')
        total_session += 1
        write_csv(itag_save_dir, all_repetition_list)
    f_w.close()
    return error_turns / total_turn, error_session / total_session


def get_dialog_info(model_res):
    """
    把模型输出信息保存成字典
    """
    dialog_dict = {}
    for idx, sgl_line in enumerate(model_res):
        contextId, query, description, dialogue = sgl_line
        dialog_dict[str(idx) + str(contextId)] = dialogue
    return dialog_dict


def construct_winrate_multurn(
    model_res_list,
    model_version_list=['模型1', '模型2'],
    save_dir='./winrate_cmp.csv',
):
    """
    Description:
        构造winrate的对话数据
    Input:
        model_res_list: 两个对比模型的输出结果 [model_1_generate_result, model_2_genrate_result]
        model_version_list: 模型版本 [base_model, experiment_model]
        save_dir：结果文件保存路径
    """

    model_1_info_dict = get_dialog_info(model_res_list[0])
    model_2_info_dict = get_dialog_info(model_res_list[1])
    all_info = []
    for context_id in model_1_info_dict.keys():
        ref_dialog_list = json.loads(model_1_info_dict[context_id])
        cmp_dialog_list = json.loads(model_2_info_dict[context_id])
        winrate_dialog_list = []
        paragraphId = 1
        flag_shuffle = random.choice([True, False])
        for turn_idx in range(0, len(ref_dialog_list), 2):
            # TODO 对有些输出为空做兼容
            refer_input_info = ref_dialog_list[turn_idx]
            refer_answer_info = ref_dialog_list[turn_idx + 1]
            cmp_input_info = cmp_dialog_list[turn_idx]
            cmp_answer_info = cmp_dialog_list[turn_idx + 1]
            assert refer_input_info["speech"] == cmp_input_info["speech"]
            if len(refer_answer_info["speech"].strip().replace(" ", "")) == 0 \
               or len(cmp_answer_info["speech"].strip().replace(" ", "")) == 0:
                continue
            refer_input_info["paragraphId"] = paragraphId
            winrate_dialog_list.append(refer_input_info)
            # 模型A的的生成
            refer_answer_info["paragraphId"] = paragraphId
            refer_answer_info["model_version"] = model_version_list[0]
            # 模型B的对话
            cmp_answer_info["paragraphId"] = paragraphId
            cmp_answer_info["model_version"] = model_version_list[1]
            # 是否交换模型输出顺序
            if flag_shuffle:
                cmp_answer_info["sentenceId"] = paragraphId * 3 - 1
                cmp_answer_info["speaker"] = "模型A"
                refer_answer_info["sentenceId"] = paragraphId * 3
                refer_answer_info["speaker"] = "模型B"
                winrate_dialog_list += [cmp_answer_info, refer_answer_info]
            else:
                refer_answer_info["sentenceId"] = paragraphId * 3 - 1
                refer_answer_info["speaker"] = "模型A"
                cmp_answer_info["sentenceId"] = paragraphId * 3
                cmp_answer_info["speaker"] = "模型B"
                winrate_dialog_list += [refer_answer_info, cmp_answer_info]
            paragraphId += 1
        all_info.append([context_id, json.dumps(winrate_dialog_list, ensure_ascii=False) + '\n'])
    with open(save_dir, 'w', encoding='utf-8-sig') as f_w:
        writer = csv.writer(f_w)
        writer.writerows([["contextId", "dialogue"]] + all_info)
    return all_info


def construct_multurn_winrate_cmp(all_winrate_list, winrate_prompt=WINRATE_PROMPT):
    """
    Description:
        构造所有需要比较的query
    Input:
        all_winrate_list: winrate的多轮输入
    Output:
        all_cmp_list: 所有需要比较的历史对话+输出拼接的query
        all_model_list: 所有query对应的模型版本
    """
    all_cmp_list = []
    all_model_list = []

    for idx, sgl_line in enumerate(all_winrate_list):
        dialogue = json.loads(sgl_line[1])
        history_1 = []
        history_2 = []
        for i in range(0, len(dialogue), 3):
            assert dialogue[i]["speaker"] == "用户"
            assert dialogue[i + 1]["speaker"] == "模型A"
            assert dialogue[i + 2]["speaker"] == "模型B"
            usr_speech = dialogue[i]["speech"]
            flag_shuffle = random.choice(["1", "0"])
            model_1_out = dialogue[i + 1]["speech"]
            model_1_version = dialogue[i + 1]["model_version"]
            model_2_out = dialogue[i + 2]["speech"]
            model_2_version = dialogue[i + 2]["model_version"]
            if flag_shuffle:
                cmp_content = winrate_prompt.format("\n".join(history_2),
                                                    usr_speech,
                                                    model_2_out,
                                                    "\n".join(history_1),
                                                    usr_speech,
                                                    model_1_out)
                model_list = [model_2_version, model_1_version]
            else:
                cmp_content = winrate_prompt.format("\n".join(history_1),
                                                    usr_speech,
                                                    model_1_out,
                                                    "\n".join(history_2),
                                                    usr_speech,
                                                    model_2_out)
                model_list = [model_1_version, model_2_version]
            all_cmp_list.append(cmp_content)
            all_model_list.append(model_list)
            history_1 += ["用户: " + usr_speech, "模型A: " + model_1_out]
            history_2 += ["用户: " + usr_speech, "模型B: " + model_2_out]
    return all_cmp_list, all_model_list


def auto_cmp_with_gpt(
    all_winrate_multurn_list,
    api_key,
    aes_key,
    winrate_prompt=WINRATE_PROMPT,
    openai_model="gpt-4",
    batch_gen=False,
    gpt_res_path='./gpt_result.txt',
    visitDomain=None,
    visitBiz=None,
    visitBizLine=None,
):
    """
    Description:
        比较模型输出结果的好坏

    """
    win_dict = {}
    all_cmp_list, all_model_list = construct_multurn_winrate_cmp(
        all_winrate_multurn_list,
        winrate_prompt=winrate_prompt,
    )
    antopenai = AntOpenAI(
        model=openai_model,
        temperature=0,
        api_key=api_key,
        aes_key=aes_key,
        max_workers=8,
        visitDomain=visitDomain,
        visitBiz=visitBiz,
        visitBizLine=visitBizLine,
    )
    all_cmp_res = []
    if batch_gen:
        all_cmp_res = antopenai.batch_generate(all_cmp_list).values()
    else:
        for cmp_prompt in tqdm(all_cmp_list):
            all_cmp_res.append(antopenai.generate(cmp_prompt))

    with open(gpt_res_path, 'w', encoding='utf-8') as outf:
        for cmp_res in all_cmp_res:
            outf.write(cmp_res + '\n')

    for model_version, cmp_res in tqdm(zip(all_model_list, all_cmp_res)):
        win_model = "差不多"
        if '模型A' in cmp_res[:10]:
            win_model = model_version[0]
        elif '模型B' in cmp_res[:10]:
            win_model = model_version[1]
        elif cmp_res == 'failed':
            win_model = 'gpt调用失败'
        win_dict[win_model] = win_dict[win_model] + 1 if win_dict.get(win_model) else 1
    return win_dict


def evaluate_model_response_repition(model_path):
    # 获取模型多轮数据结果
    oss_repetition_multurn = 'oss://antsys-adabrain/datasets/llm_multiturn/eval/自建评估数据/多轮-多轮重复.csv'
    all_multurn_dialog_list = get_llm_multurn_res(model_path, oss_repetition_multurn)

    # 获取多轮之间重复的评估结果
    turn_repiption_raito, session_repiption_raito = check_turns_repeat(all_multurn_dialog_list)
    print(f"sentence, turn_repition_ration: {turn_repiption_raito}, session_repition_ratio: {session_repiption_raito}")

    # 获取模型每一轮生成内部重复的评估结果
    turn_wrd_repition_ratio, session_wrd_repition_ratio = check_wrds_repetition(all_multurn_dialog_list)
    print(f"wrds, turn_repition_ratio: {turn_wrd_repition_ratio}, session_repition_ratio: {session_wrd_repition_ratio}")


def winrate_cmp_model_result(
    multi_query_oss_path,
    model_path_1,
    model_path_2,
    api_key,
    aes_key,
    winrate_prompt=WINRATE_PROMPT,
    model1_version_name=None,
    model2_version_name=None,
    result_dir='./',
    visitDomain=None,
    visitBiz=None,
    visitBizLine=None,
):
    '''比较两个模型的多轮对话 winrate.

    Params:
        multi_query_oss_path: 多轮 query 数据集文件 oss 地址, 格式参考:
            `oss://antsys-adabrain/datasets/llm_multiturn/eval/自建评估数据/normal_multurn_dialog.csv`

        model_path_1: 模型 1 路径
        model_path_2: 模型 2 路径
        api_key: ant_openai api_key
        aes_key: ant_openai aes_key

        winrate_prompt: GPT4 winrate 评估 prompt, 默认使用 WINRATE_PROMPT

        model1_version_name: 模型 1 版本名称
        model2_version_name: 模型 2 版本名称
        result_dir: 结果保存路径
        visitDomain: ant_openai visitDomain 参数
        visitBiz: ant_openai visitBiz 参数
        visitBizLine: ant_openai visitBizLine 参数
    '''

    def read_model_res(path):
        df = pd.read_csv(path)
        return [
            [r['contextId'], r['query'], r['cate'], r['dialogue']]
            for _, r in df.iterrows()
        ]

    # 各保存文件路径
    eval_name = Path(multi_query_oss_path).stem
    result_dir = Path(result_dir)
    model1_res_path = result_dir.joinpath(
        f'model_output.{eval_name}.{Path(model_path_1).name}.csv')
    model2_res_path = result_dir.joinpath(
        f'model_output.{eval_name}.{Path(model_path_2).name}.csv')
    winrate_input_path = result_dir.joinpath(
        f'winrate_input.{eval_name}.{Path(model_path_1).name}.vs.{Path(model_path_2).name}.csv'
    )
    gpt_res_path = result_dir.joinpath(
        f'gpt_result.{eval_name}.{Path(model_path_1).name}.vs.{Path(model_path_2).name}.txt'
    )
    winrate_res_path = result_dir.joinpath(
        f'winrate_result.{eval_name}.{Path(model_path_1).name}.vs.{Path(model_path_2).name}.txt'
    )

    # 获取两个模型的输出结果
    if not model1_res_path.exists():
        model_1_res = get_llm_multurn_res(
            model_path_1,
            multi_query_oss_path,
            save_dir=model1_res_path,
        )
    else:
        print(f'model1 预测结果已存在: {model1_res_path}')
        model_1_res = read_model_res(model1_res_path)
    if not model2_res_path.exists():
        model_2_res = get_llm_multurn_res(
            model_path_2,
            multi_query_oss_path,
            save_dir=model2_res_path,
        )
    else:
        print(f'model2 预测结果已存在: {model2_res_path}')
        model_2_res = read_model_res(model2_res_path)

    # 合并模型输出的结果，输出后的文件可以直接放到itag上，也可以直接用openai自动评估
    if not model1_version_name:
        model1_version_name = model_path_1
    if not model2_version_name:
        model2_version_name = model_path_2

    all_winrate_multurn_list = construct_winrate_multurn(
        [model_1_res, model_2_res],
        winrate_prompt=winrate_prompt,
        model_version_list=[
            model1_version_name,
            model2_version_name,
        ],
        save_dir=winrate_input_path,
    )

    # 自然对话评估
    if not gpt_res_path.exists():
        win_dict = auto_cmp_with_gpt(
            all_winrate_multurn_list, api_key, aes_key,
            batch_gen=True, gpt_res_path=gpt_res_path,
            visitDomain=visitDomain,
            visitBiz=visitBiz,
            visitBizLine=visitBizLine,
        )
        with open(winrate_res_path, 'w', encoding='utf-8') as outf:
            outf.write(json.dumps(win_dict))
        print(win_dict)
    else:
        print('winrate 结果已存在, 不再重新打印.')


if __name__ == "__main__":
    fire.Fire()
