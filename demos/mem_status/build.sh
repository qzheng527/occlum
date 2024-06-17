#!/bin/bash
set -e

# compile mem_status
pushd mem_status
cargo clean
cargo build --release
popd

# reduce binary size
strip mem_status/target/release/mem_status

# initialize occlum workspace
rm -rf occlum_instance
occlum new occlum_instance
cd occlum_instance

rm -rf image
copy_bom -f ../bom.yaml --root image --include-dir /opt/occlum/etc/template
occlum build
