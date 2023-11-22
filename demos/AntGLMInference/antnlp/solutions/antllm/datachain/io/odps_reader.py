#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"
from typing import List

from aistudio_common.utils import env_utils
from pypai.io import TableReader


class ODPSReader:
    def __init__(self,
                 odps_project_table: str,
                 columns=None,
                 access_id=None,
                 access_key=None,
                 project=None,
                 endpoint=None,
                 partition: List[str] = []) -> None:
        self.odps_env = env_utils.get_odps_instance(access_id, access_key, project, endpoint)
        self.odps_project_table = odps_project_table
        self.columns = columns
        self.partition = partition
        print(self.partition)

    def read(self):
        partition = []
        if len(self.partition) > 0:
            # 读取odps分区表时需要指定分区信息，分区信息以[key1, value1, key2, value2, ...]的形式传入，列表的长度需要被2整除
            assert len(partition) % 2 == 0, "The length of partiton should be divided by 2." 
            for i in range(0, len(self.partition), 2):
                partition.append(self.partition[i] + '=' + self.partition[i + 1])  # partition的格式定义
        if len(partition) == 0:  # 如果没有传入partition信息，默认读全量表
            reader = TableReader.from_ODPS_type(self.odps_env, self.odps_project_table)
        else:
            reader = TableReader.from_ODPS_type(self.odps_env, self.odps_project_table, 
                                                partition=','.join(partition))
        df = reader.to_pandas()
        res = []
        if self.columns and len(self.columns) > 0:
            cols = self.columns
        else:
            cols = df.columns

        for _, row in df.iterrows():
            item = {}
            for col in cols:
                item[col] = row[col]
            res.append(item)
        return res
