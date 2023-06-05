#!/bin/bash
set -e
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" >/dev/null 2>&1 && pwd )"
pip_mirror="https://mirrors.aliyun.com/pypi/simple"

# Install python and dependencies to specified position
[ -f Miniconda3-latest-Linux-x86_64.sh ] || wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
[ -d miniconda ] || bash ./Miniconda3-latest-Linux-x86_64.sh -b -p $script_dir/miniconda

# Create conda env
$script_dir/miniconda/bin/conda create --prefix $script_dir/python-occlum -y python=3.8.10 \
    gdal Pillow numpy shapely pyproj rasterio pandas pathlib numba pysftp
$script_dir/python-occlum/bin/pip3 install geopandas==0.13.0 mercantile xyconvert -i $pip_mirror
