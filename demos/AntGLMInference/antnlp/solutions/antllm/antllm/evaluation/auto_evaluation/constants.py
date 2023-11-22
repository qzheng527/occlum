#!/usr/bin/env python
# coding=utf-8
# @Author: xinyu.kxy
# @Date: Fri 17 July 2023 09:22:56 PM CST

from pathlib import Path

DEFAULT_CACHE_DIR = None
CURRENT_DIR = Path(__file__).parent
EVALUATORS_CONFIG_DIR = CURRENT_DIR / "evaluators_configs"
MODELS_CONFIG_DIR = CURRENT_DIR / "models_configs"
BASE_DIR = Path(__file__).parents[2]

# ANOTATORS = {"single":annotators.pairsewise}
