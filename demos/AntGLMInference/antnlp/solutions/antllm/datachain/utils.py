#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"

import os
import random
import requests
import yaml
import uuid
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Iterable, List, Mapping, Union, Optional
from adabrain.common.utils.oss_utils import OSSTool


def get_first(dic, keys):
    for key in keys:
        if key in dic:
            return dic[key]
    return None


def contains_any(str, strs):
    for s in strs:
        if s in str:
            return True
    return False


def batch_iter(data: Iterable, size, shuffle=False):
    new_data = list(data)
    if shuffle:
        random.shuffle(new_data)

    for ndx in range(0, len(new_data), size):
        yield new_data[ndx:min(ndx + size, len(new_data))]


def post_with_retry(retry_times=3, *args, **kwargs):
    for _ in range(retry_times):
        try:
            response = requests.post(*args, **kwargs)
            return response
        except Exception:
            continue
    return None


def safe_get(dic, keys):
    if isinstance(keys, List):
        cur_dic = dic
        for k in keys:
            if not isinstance(cur_dic, Mapping) or k not in cur_dic:
                return False, None
            cur_dic = cur_dic[k]
        return True, cur_dic
    else:
        if keys not in dic:
            return False, None
        return True, dic[keys]


def get_all_files(dir):
    paths = [os.path.join(top, name) for top, _,
             names in os.walk(str(dir)) for name in names]
    return paths


def load_yaml(yaml_path: Union[str, Path]) -> Dict:
    conf = yaml.safe_load(Path(yaml_path).read_text())
    return conf


def download_oss(oss_path: str, local_path: str) -> str:
    oss_tool(oss_path, str(local_path), func="download")
    if not Path(local_path).exists():
        raise FileNotFoundError(
            f"Error: oss download failed. oss url: {oss_path}, "
            f"local_path: {local_path}"
        )


def upload_oss(oss_path: str, local_path: str):
    oss_tool(oss_path, str(local_path), func="upload")


def oss_tool(
    oss_path: str,
    local_path: str,
    func: str = "download"
) -> str:
    """oss download / upload"""
    try:
        print(f"try {func} oss file: {oss_path}")
        if not oss_path.startswith("oss://"):
            print(f"invalid oss path: {oss_path}")

        bucket = oss_path.replace("oss://", "").split("/")[0]
        obj_name = "/".join(oss_path.replace("oss://", "").split("/")[1:])
        tool = OSSTool(
            bucket_name=bucket,
            oss_key_prefix=bucket
        )

        if func == "download":
            if "." not in obj_name:
                tool.download_oss_dir(obj_name, local_path)
            else:
                tool.download_oss(obj_name, local_path)
        else:
            if Path(local_path).is_dir():
                tool.upload_oss_dir(obj_name, local_path)
            else:
                tool.upload_oss(obj_name, local_path)
    except Exception:
        print(
            f"oss failed, oss_path: {oss_path}, "
            f"local_path: {local_path}, func: {func}"
        )

    return local_path


def get_uuid():
    return str(uuid.uuid4())


def dump_jsonl(data: List[dict], output_path: str, append: bool = False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = "a+" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for line in tqdm(data):
            try:
                json_record = json.dumps(line, ensure_ascii=False)
            except Exception:
                print(f"dump error: line: {line}")
            f.write(json_record + "\n")
    print("Wrote {} records to {}".format(len(data), output_path))


def load_jsonl(input_path: str, line_num: Optional[int] = None) -> List[dict]:
    """
    Read list of objects from a JSON lines file.
    """
    try:
        input_path = str(input_path)
        data = []
        with open(input_path, "r", encoding="utf-8") as f:
            invalid_cnt = 0
            for idx, line in tqdm(enumerate(f)):
                if line_num and idx >= line_num:
                    break
                try:
                    data.append(json.loads(line.rstrip("\n|\r")))
                except Exception:
                    invalid_cnt += 1
        print("Loaded {} records from {}".format(len(data), input_path))
        print("Invalid json num: {}".format(invalid_cnt))
    except Exception:
        raise Exception(f"load file failed, {input_path}")
    return data


def load_text(fname: str, strip: bool = True) -> List[str]:
    """load text file."""
    texts = []
    with open(fname, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            if strip:
                line = line.strip()
            texts.append(line)
    return texts
