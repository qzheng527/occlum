"""Processing data for Causal LM SFT."""

import os
import json
import argparse
from tqdm import tqdm
from typing import Union, Optional, List

from solutions.antllm.antllm.utils.version_utils import is_oldest_version
from solutions.antllm.antllm.models.glm.tokenization_glm import GLMTokenizer
from solutions.antllm.antllm.models.glm.tokenization_glm_deprecated import GLMChineseTokenizer


def init_tokenizer(pretrained_model_name_or_path: str) -> Union[GLMTokenizer, GLMChineseTokenizer]:
    if is_oldest_version(pretrained_model_name_or_path):
        auto_tokenization_class = GLMChineseTokenizer
        old_version_tokenizer = True
    else:
        auto_tokenization_class = GLMTokenizer
        old_version_tokenizer = False

    tokenizer = auto_tokenization_class.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
    return tokenizer, old_version_tokenizer


def truncate_sentence(input_ids: List[int], output_ids: List[int], max_length: int, left_truncate: bool = False):
    num_special_tokens = 4
    if len(input_ids) + len(output_ids) > max_length - num_special_tokens:  # 4是需要添加的特殊符号的个数
        if len(input_ids) > (max_length - num_special_tokens) // 2 \
                and len(output_ids) > (max_length - num_special_tokens) // 2:
            # 如果都超过了最大长度的一半,那都截取到最大长度的一半
            half_length = (max_length - num_special_tokens) // 2
            if left_truncate:
                input_ids = input_ids[-half_length:]
            else:
                input_ids = input_ids[:half_length]
            output_ids = output_ids[:half_length]
        else:
            # 从input_ids和output_ids中比较长的那一个截断,input_ids可以选择从左边或右边阶段,output_ids默认从右边截断
            if len(input_ids) >= len(output_ids):
                if left_truncate:
                    input_ids = input_ids[-(max_length -
                                            num_special_tokens - len(output_ids)):]
                else:
                    input_ids = input_ids[
                        :max_length - num_special_tokens - len(output_ids)]
            else:
                output_ids = output_ids[:max_length -
                                        num_special_tokens - len(input_ids)]
    return input_ids, output_ids


def convert_origin_data_format_to_causal(
    data_path: str,
    out_path: Optional[str] = None,
    tokenizer: Union[GLMTokenizer, GLMChineseTokenizer] = None,
    old_version_tokenizer: bool = False,
    max_length: int = 1024,
    left_truncate: bool = False
) -> None:
    if not out_path:
        dir_path = os.path.dirname(data_path)
        file_name, file_ext = os.path.splitext(os.path.basename(data_path))
        out_path = os.path.join(dir_path, file_name + "-packed" + file_ext)

    sop_id = tokenizer.convert_tokens_to_ids(tokenizer.sop_token)
    eop_id = tokenizer.convert_tokens_to_ids(tokenizer.eop_token)
    cls_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    mask_id = tokenizer.convert_tokens_to_ids("[gMASK]")
    num_special_tokens = 4

    cat_input_data = []
    cat_output_data = []
    cat_data_length = 0
    with open(data_path, "r") as rf, open(out_path, "w") as of:
        for line in tqdm(rf):
            data = json.loads(line.strip())

            data["input"] = data["input"].replace("\\n", "\n")
            data["output"] = data["output"].replace("\\n", "\n")
            if old_version_tokenizer:
                data["input"] = data["input"].replace("\n", "<n>")
                data["output"] = data["output"].replace("\n", "<n>")

            input_ids = tokenizer(data["input"])["input_ids"][1:-1]
            output_ids = tokenizer(data["output"])["input_ids"][1:-1]

            if cat_input_data and len(input_ids) + len(output_ids) > (
                    max_length - (num_special_tokens - 1) * len(cat_input_data) - 1 - cat_data_length):
                of.write(json.dumps({
                    "input": cat_input_data,
                    "output": cat_output_data
                }, ensure_ascii=False) + "\n")

                cat_input_data.clear()
                cat_output_data.clear()
                cat_data_length = 0

            if cat_data_length == 0 and len(input_ids) + len(output_ids) > max_length - num_special_tokens:
                input_ids, output_ids = truncate_sentence(
                    input_ids, output_ids, max_length, left_truncate=left_truncate)

                of.write(json.dumps({
                    "input": [data["input"]],
                    "output": [data["output"]]
                }, ensure_ascii=False) + "\n")
                continue

            input_ids = [cls_id] + input_ids + [mask_id]
            output_ids = output_ids + [eop_id]
            tokens = input_ids + [sop_id] + output_ids
            cat_input_data.append(data["input"])
            cat_output_data.append(data["output"])
            cat_data_length += len(tokens)

        if cat_input_data:
            of.write(json.dumps({
                "input": cat_input_data,
                "output": cat_output_data
            }, ensure_ascii=False) + "\n")       


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to input JSON"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The path to the tokenizer file",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="The max length of the concated data",
    )
    parser.add_argument(
        "--left_truncate",
        action="store_true",
        help="Truncate the sentence from left.",
    )

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    if not os.path.exists(args.input):
        return
    tokenizer, old_version_tokenizer = init_tokenizer(args.model)

    if os.path.isfile(args.input):
        convert_origin_data_format_to_causal(
            args.input, args.output, tokenizer, old_version_tokenizer,
            max_length=args.max_length, left_truncate=args.left_truncate
        )
    else:
        for root, ds, fs in os.walk(args.input):
            output_dir = os.path.join(args.output, root[len(args.input) + 1:])
            for name in fs:
                input_file = os.path.join(root, name)
                if input_file.endswith("json") or input_file.endswith("jsonl"):
                    print(output_dir)
                    os.makedirs(
                        output_dir, exist_ok=True)
                    output_file = os.path.join(output_dir, name)
                    convert_origin_data_format_to_causal(
                        input_file, output_file, tokenizer, old_version_tokenizer,
                        max_length=args.max_length, left_truncate=args.left_truncate)


if __name__ == "__main__":
    main()
