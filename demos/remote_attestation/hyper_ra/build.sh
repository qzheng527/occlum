#!/bin/bash
set -e

INSTALL_PREFIX="/usr/local"
CJSON_VER=1.7.15

# Download, build and install cJSON
function build_cjson() {
    rm -rf cJSON* v${CJSON_VER}.tar.gz
    wget https://github.com/DaveGamble/cJSON/archive/refs/tags/v${CJSON_VER}.tar.gz
    tar zxvf v${CJSON_VER}.tar.gz

    pushd cJSON-${CJSON_VER}
    rm -rf build && mkdir build && cd build
    cmake -DENABLE_CJSON_UTILS=On -DENABLE_CJSON_TEST=Off -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
        -DCMAKE_C_COMPILER=gcc ..
    make install
    popd
}

function build_lib_and_app() {
    # Build lib
    make -C src clean
    make -C src

    # Build test
    make -C test clean
    make -C test
}

function build_occlum_instance()
{
    rm -rf occlum_instance && occlum new occlum_instance
    pushd occlum_instance

    rm -rf image
    copy_bom -f ../bom.yaml --root image --include-dir /opt/occlum/etc/template
    occlum build
    popd
}

build_cjson
build_lib_and_app
build_occlum_instance

