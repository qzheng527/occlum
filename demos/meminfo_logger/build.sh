#!/bin/bash
set -e

# compile meminfo logger
make clean
make

# initialize occlum workspace
rm -rf occlum_instance
occlum new occlum_instance
cd occlum_instance

rm -rf image
copy_bom -f ../bom.yaml --root image --include-dir /opt/occlum/etc/template
occlum build
