from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys


def bulid_context(context):
    result = ""
    for i, content in enumerate(context):
        user_input, bot_response = content
        result += f"第{i+1}轮\n"
        result += "用户: {}\n".format(user_input)
        result += "机器人: {}\n".format(bot_response)
    return result


if __name__ == "__main__":
    mask = "[gMASK]"
    model_path = sys.argv[1]
    rl_model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
    rl_model = rl_model.half().to("cuda:7")
    rl_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    context = []
    while True:
        msg = input("user: ")
        if msg == "quit":
            result = bulid_context(context)
            with open("temp.txt", "w", encoding="utf8") as fout:
                fout.write(result)
            break
        if msg == "clear":
            result = bulid_context(context)
            with open("temp.txt", "w", encoding="utf8") as fout:
                fout.write(result)
            context = []
            continue
        if len(context) == 0:
            text = msg + " " + mask
        else:
            cur_context = bulid_context(context)
            text = (
                cur_context
                + "第{}轮\n".format(len(context) + 1)
                + "用户: {}\n机器人:".format(msg)
            )
            text = text + " " + mask
        # print(text)
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=1000,
            add_special_tokens=True,
            return_tensors="pt",
        )
        build_inputs = tokenizer.build_inputs_for_generation(
            inputs, max_gen_length=1000
        )
        build_inputs = build_inputs.to("cuda:7")
        outputs = rl_model.generate(
            **build_inputs, max_length=1000, eos_token_id=tokenizer.eop_token_id
        )
        bot_response = str(tokenizer.decode(outputs[0].tolist()))
        # ipdb.set_trace()
        mask_pos = bot_response.index(mask)
        bot_response = bot_response[mask_pos + len(mask) :]
        special_tokens = [tokenizer.eop_token, tokenizer.sop_token, tokenizer.eos_token]
        for token in special_tokens:
            bot_response = bot_response.replace(token, "")
        bot_response = bot_response.lstrip()
        print("bot: " + bot_response)
        context.append((msg, bot_response))
