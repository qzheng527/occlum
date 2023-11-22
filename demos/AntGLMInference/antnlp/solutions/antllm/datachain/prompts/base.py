#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"

import json


class Instruct:
    def __init__(self, instruct: str = "", **kwargs) -> None:
        self.instruct = instruct
        self.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self) -> str:
        return json.dumps({"instruct": self.instruct, **self.kwargs}, ensure_ascii=False)
