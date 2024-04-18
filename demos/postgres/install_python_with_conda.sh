#!/bin/bash
set -e
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" >/dev/null 2>&1 && pwd )"
pip_mirror="https://mirrors.aliyun.com/pypi/simple"

# Install python and dependencies to specified position
[ -f Miniconda3-latest-Linux-x86_64.sh ] || wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
[ -d miniconda ] || bash ./Miniconda3-latest-Linux-x86_64.sh -b -p $script_dir/miniconda

# Create conda env
$script_dir/miniconda/bin/conda create \
    --prefix $script_dir/python-occlum -y \
    python=3.10.0

# Install python packages
$script_dir/python-occlum/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
$script_dir/python-occlum/bin/pip3 install mercantile xyconvert xgboost scikit-learn lightgbm -i $pip_mirror
