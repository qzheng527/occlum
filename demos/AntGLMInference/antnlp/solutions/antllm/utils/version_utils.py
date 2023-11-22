#!/usr/bin/env python
# coding=utf-8
# @Author: tianxuan.jl
# @Date: Fri 05 May 2023 04:09:36 PM CST
import os


def is_old_version(path):
    new_vocab_files = ['merge.model']
    new_vocab_file_exists = []
    for filename in new_vocab_files:
        if not os.path.exists(os.path.join(path, filename)):
            new_vocab_file_exists.append(False)
        else:
            new_vocab_file_exists.append(True)
    if all(new_vocab_file_exists):
        return False
    if any(new_vocab_file_exists):
        return 'new_version_file_absent'
    else:
        return True
