#!/bin/bash
set -e

BLUE='\033[1;34m'
NC='\033[0m'

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" >/dev/null 2>&1 && pwd )"

function build_instance()
{
    rm -rf occlum_instance && occlum new occlum_instance
    pushd occlum_instance
    rm -rf image
    copy_bom -f ../faiss.yaml --root image --include-dir /opt/occlum/etc/template

    new_json="$(jq '.resource_limits.user_space_size = "32MB" |
                    .resource_limits.user_space_max_size = "8GB" |
                    .resource_limits.kernel_space_heap_size = "16MB" |
                    .resource_limits.kernel_space_heap_max_size="128MB" |
                    .resource_limits.max_num_of_threads = 128 |
                    .env.default += ["PYTHONHOME=/opt/python-occlum"] |
                    .env.default += ["PATH=/bin"] |
                    .env.default += ["HOME=/root"] |
                    .env.untrusted += ["OMP_NUM_THREADS"]' Occlum.json)" && \
    echo "${new_json}" > Occlum.json

    occlum build
    popd
}

build_instance

