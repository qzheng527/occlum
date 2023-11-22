import re
import logging
glogger = logging.getLogger()


class ChatIO():
    @staticmethod
    def post_process(output, multi_turn=False):
        replace_strings = {'谷歌研发': '蚂蚁集团研发',
                           '谷歌公司研发': '蚂蚁集团研发',
                           '谷歌(Google)研发': '蚂蚁集团研发',
                           '谷歌开发': '蚂蚁集团开发',
                           '谷歌公司开发': '蚂蚁集团开发',
                           '谷歌(Google)开发': '蚂蚁集团开发',
                           '谷歌训练': '蚂蚁集团训练',
                           '谷歌公司训练': '蚂蚁集团训练',
                           '谷歌(Google)训练': '蚂蚁集团训练',
                           '谷歌制造': '蚂蚁集团制造',
                           '谷歌公司制造': '蚂蚁集团制造',
                           '谷歌(Google)制造': '蚂蚁集团制造',
                           '谷歌研究': '蚂蚁集团研究',
                           '谷歌公司研究': '蚂蚁集团研究',
                           '谷歌(Google)研究': '蚂蚁集团研究',
                           }

        if multi_turn:
            error_outs = re.findall("第\d+轮", output)
            if len(error_outs) > 0:
                output = output.split(error_outs[0])[0].strip()

        for orig_str, repl_str in replace_strings.items():
            output = output.replace(orig_str, repl_str)
        return output

    @staticmethod
    def rule_result(query):
        rules = {
            "你好": "你好！有什么可以帮您的吗？"
        }
        if query in rules:
            rule_result = rules[query]
            return True, rule_result
        else:
            return False, None

    @staticmethod
    def history_concat(query, history):
        if len(history) == 0:
            query = query
        else:
            history_prompt = ""
            for dialog in history:
                turn = int(dialog["turn"])
                user = dialog["user"]
                bot = dialog["bot"]
                history_prompt += f"第{turn}轮\n用户: {user}\n机器人: {bot}\n"
            query = history_prompt + f"第{turn+1}轮\n用户: {query}\n机器人:"
        return query
    
    @staticmethod
    def process(query, bot, max_length, beam_width, sync, hit_rule, rule_ans, temperature, top_k, top_p,
                max_output_length, do_sample, chat_history):
        if sync:
            if hit_rule:
                ans = rule_ans
                glogger.info("get rule result {}".format(ans)) 
            else:
                ans = bot.answer(
                    query,
                    beam_width,
                    temperature=temperature,
                    top_k=top_k, top_p=top_p,
                    max_output_length=max_output_length,
                    # max_length=max_length,
                    do_sample=do_sample
                )
            glogger.info("predict result {}".format(ans))  # 可以在平台查看相关日志

            # 后处理
            ans = ChatIO.post_process(ans, multi_turn=len(chat_history) > 0)
            glogger.info("post process result {}".format(ans))

            # 返回
            resultMap = {"result": ans}
            yield (0, 'ok', resultMap)
        else:
            if hit_rule:
                ans = rule_ans
                glogger.info("get rule result {}".format(ans)) 
                for char in ans:
                    resultMap = {"result": char}
                    yield (0, 'ok', resultMap)
            else:
                beam_width = 1  # 流式暂时只greedy
                for ans in bot.generate_stream(
                        query, beam_width, temperature, top_k, top_p,
                        max_output_tokens=max_output_length):
                    glogger.info("stream predict result {}".format(ans))  # 可以在平台查看相关日志
                    # TODO: 后处理here

                    # 返回
                    resultMap = {"result": ans}
                    yield (0, 'ok', resultMap)
