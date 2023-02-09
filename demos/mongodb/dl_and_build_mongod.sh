#! /bin/bash
set -e

# Install dependencies
# apt update && apt install -y libcurl4-openssl-dev liblzma-dev

rm -rf mongo
git clone --branch=r6.0.0 --depth=1 https://github.com/mongodb/mongo
pushd mongo
python3 -m pip install -r etc/pip/compile-requirements.txt
python3 buildscripts/scons.py \
    DESTDIR=../install/mongo install-mongod \
    --ssl --disable-warnings-as-errors
popd

strip ./install/mongo/bin/mongod
