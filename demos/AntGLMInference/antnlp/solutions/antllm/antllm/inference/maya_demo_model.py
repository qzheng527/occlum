import os
import time
import logging
import json
import re
import torch
from aistudio_serving.hanlder.pymps_handler import MayaBaseHandler  # noqa
from solutions.antllm.antllm.inference.glm_predictor import GLMForInference


glogger = logging.getLogger()


class UserHandler(MayaBaseHandler):
    """
     model_dir:模型所在目录,用户在写__init__里的代码时可以认为模型就在model_dir目录下
    """
    def __init__(self, work_dir):
        super(UserHandler, self).__init__(work_dir)
        glogger.info('enter __init__')
        # path是暂时写死一个路径，会在制作镜像时放进去（当前maya上不能挂盘）
        self.bot = GLMForInference(path="/home/admin/model", torch_dtype=torch.bfloat16)

        rule_file_path = os.path.join(self.resource_path, "rule.json")
        if os.path.exists(rule_file_path):
            with open(rule_file_path) as f:
                self.rule = json.load(f)
        else:
            self.rule = {}

        glogger.info(f"rule: {json.dumps(self.rule, ensure_ascii=False)}")
        prefix_file_path = os.path.join(self.resource_path, "prefix_prompt.txt")
        if os.path.exists(prefix_file_path):
            with open(prefix_file_path) as f:
                self.prefix = f.read().strip()
        else:
            self.prefix = ""
        glogger.info(f"prefix prompt: {self.prefix}")

        suffix_file_path = os.path.join(self.resource_path, "suffix_prompt.txt")
        if os.path.exists(suffix_file_path):
            with open(suffix_file_path) as f:
                self.suffix = f.read().strip()
        else:
            self.suffix = ""
        glogger.info(f"suffix prompt: {self.suffix}")

        replace_file_path = os.path.join(self.resource_path, "replace.json")
        if os.path.exists(replace_file_path):
            with open(replace_file_path) as f:
                self.replace_strings = json.load(f)
        else:
            self.replace_strings = {}
        glogger.info(f"replace strings: {json.dumps(self.replace_strings, ensure_ascii=False)}")

    def post_process(self, output, multi_turn=False):
        if multi_turn:
            error_outs = re.findall("第\d+轮", output)
            if len(error_outs) > 0:
                output = output.split(error_outs[0])[0].strip()

        output = re.sub("<((/)|((用户)|(机器人))).*?>|<chat>", "", output)
        for orig_str, repl_str in self.replace_strings.items():
            output = output.replace(orig_str, repl_str)
        return output
        
    def rule_result(self, query):
        if query in self.rule:
            rule_result = self.rule[query]
            return True, rule_result
        else:
            return False, None
    
    def history_concat(self, query, history):
        if len(history) == 0:
            query = f"{query}"
        else:
            history_prompt = ""
            for dialog in history:
                turn = int(dialog["turn"])
                user = dialog["user"].strip()
                bot = dialog["bot"].strip()
                history_prompt += f"第{turn}轮\n用户: {user}\n机器人: {bot}\n"  # 具体拼接 机器人/贞仪 需要根据模型版本确定
            query = history_prompt + f"第{turn+1}轮\n用户: {query}\n机器人:"
        return query

    def predict_np(self, features, trace_id):
        glogger.info("trace_id: {}; features: {}".format(trace_id, list(features.keys())))
        # 解析请求, 字符串类型默认是bytes类型
        resultCode = 0
        errorMessage = "ok"
        try:
            data = json.loads(features.get("data").decode())
            query = data.get("query").strip()
            chat_history = json.loads(data.get("history", "[]"))
            # max_length = data.get("max_length", 2048)
            beam_width = data.get("beam_width", 1)
            temperature = data.get("temperature", 0.4)
            top_k = data.get("top_k", 50)
            top_p = data.get("top_p", 1)
            do_sample = data.get("do_sample", True)
            max_output_length = data.get("max_output_length", -1)
            left_truncate = data.get("left_truncate", True)
            multi_turn = data.get("multi_turn", False) or len(chat_history) > 0

            sync = data.get("sync", True)

            # 命中规则逻辑
            hit_rule, rule_ans = self.rule_result(query)

            # 处理对话历史here
            query = self.history_concat(query, chat_history)

            query = self.prefix + query + self.suffix

            glogger.info("model input query {}".format(repr(query)))

            # 同步or流式调用
            if sync:
                if hit_rule:
                    ans = rule_ans
                    glogger.info("get rule result {}".format(repr(ans)))
                else:
                    ans = self.bot.answer(
                        query,
                        beam_width,
                        temperature=temperature,
                        top_k=top_k, top_p=top_p,
                        max_output_length=max_output_length,
                        left_truncate=left_truncate,
                        do_sample=do_sample
                    )
                    
                glogger.info("predict result {}".format(repr(ans)))  # 可以在平台查看相关日志

                # 后处理
                ans = self.post_process(ans, multi_turn=multi_turn)
                glogger.info("post process result {}".format(repr(ans)))

                # 返回
                resultMap = {"result": ans}
                yield resultCode, errorMessage, resultMap
            else:
                if hit_rule:
                    ans = rule_ans
                    glogger.info("get rule result {}".format(repr(ans)))
                    for char in ans:
                        resultMap = {"result": char}
                        yield resultCode, errorMessage, resultMap
                else:
                    beam_width = 1  # 流式暂时只greedy
                    tmp_ans = ""
                    for ans in self.bot.generate_stream(
                        query,
                        beam_width,
                        temperature,
                        top_k,
                        top_p,
                        max_output_tokens=max_output_length,
                        left_truncate=left_truncate,
                        do_sample=do_sample
                    ):
                        glogger.info("stream predict result {}".format(ans))  # 可以在平台查看相关日志
                        # TODO: 后处理here
                        ans = ans.strip("<|endofpiece|>")

                        # 去除乱码
                        if ans.encode() == b"\xef\xbf\xbd":
                            ans = ""
                        if len(chat_history) > 0:
                            if len(tmp_ans) == 0 and ans == "第":
                                tmp_ans += ans
                                ans = ""
                            elif len(re.findall("第", tmp_ans)) > 0 and len(re.findall("\d+", ans)) > 0:
                                tmp_ans += ans
                                ans = ""
                            elif len(re.findall("第\d+", tmp_ans)) > 0 and len(re.findall("轮", ans)) > 0:
                                break
                            elif len(tmp_ans) > 0:
                                ans = tmp_ans + ans
                                tmp_ans = ""

                        resultMap = {"result": ans}
                        yield resultCode, errorMessage, resultMap
        except Exception as e:
            glogger.error("infer error ", e)
            ans = "infer error {}".format(e)
            resultMap = {"result": ans}
            yield resultCode, errorMessage, resultMap


# 用于调试UserHandler类的功能
if __name__ == '__main__':
    # 获取AISTUDIO workspace的绝对路径，只在AIS容器中生效，请勿在部署服务阶段使用
    print(os.getcwd())
    user_handler = UserHandler(os.getcwd())

    query = "林黛玉倒拔垂杨柳是什么意思"  # noqa

    history = [
        {
            "turn": "1",
            "user": "你是谁",
            "bot": "我是机器人"
        },
        {
            "turn": "2",
            "user": "林黛玉和鲁智深是什么关系",
            "bot": "林黛玉和鲁智深是两个完全不同的人物，没有直接的关系。林黛玉是《红楼梦》中的女主角，鲁智深则是《水浒传》中的一位英雄人物。他们出自不同的文学作品，没有交集。"
        }
    ]

    data = {"data": json.dumps({"query": query, "history": json.dumps(history), "sync": True, "do_sample": False}).encode()}  # noqa
    start_time = time.time()
    for res in user_handler.predict_np(data, 0):
        print(res)
    end_time = time.time()
    print("耗时: {:.2f}秒".format(end_time - start_time))

    data = {"data": json.dumps({"query": query, "history": json.dumps(history), "sync": False, "do_sample": True}).encode()}  # noqa
    start_time = time.time()
    for res in user_handler.predict_np(data, 0):
        print(res)
    end_time = time.time()
    print("耗时: {:.2f}秒".format(end_time - start_time))
