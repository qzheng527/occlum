#! /bin/bash

# Exit when error
set -xe

export PATH=/opt/python-occlum/bin:$PATH
cd /Langchain-Chatchat-ipex-llm
python3 startup.py -a
