#!/bin/bash
set -e

function build_occlum_instance()
{
    rm -rf occlum_instance && occlum new occlum_instance
    pushd occlum_instance

    rm -rf image
    copy_bom -f ../bom.yaml --root image --include-dir /opt/occlum/etc/template
    occlum build
    popd
}

# cargo clean
cargo build --release  --examples

build_occlum_instance
