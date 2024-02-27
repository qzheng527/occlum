#!/bin/bash
set -e

BLUE='\033[1;34m'
NC='\033[0m'



build_instance() {
    # Init Occlum instance
    rm -rf occlum_instance && occlum new occlum_instance
    cd occlum_instance

    new_json="$(jq '.resource_limits.user_space_size = "2GB" |
                .resource_limits.kernel_space_heap_size="256MB" |
                .resource_limits.max_num_of_threads = 64 |
                .metadata.debuggable = true |
                .metadata.disable_log = false |
                .entry_points += [ "/opt/taobao/java/bin" ] |
                .env.default += [ "HOME=/home/admin" ] |
                .env.default += [ "MALLOC_ARENA_MAX=1" ] |
                .env.default += [ "LD_LIBRARY_PATH=/opt/taobao/java/lib:/opt/taobao/java/jre/lib" ] ' Occlum.json)" && \
    echo "${new_json}" > Occlum.json

    rm -rf image
    copy_bom -f ../bom_ajdk.yaml --root image --include-dir /opt/occlum/etc/template
    mkdir image/app
    cp -f ../*.class image/
    # rm -rf app && mkdir app
    # cp -f ../*.class app/

    # # mount app as hostfs
    # new_json="$(cat Occlum.json | jq '.mount+=[{"target": "/app", "type": "hostfs","source": "./app"}]')" && \
    # echo "${new_json}" > Occlum.json

    occlum build
}

run() {
    occlum run /opt/taobao/java/bin/java \
        -XX:ActiveProcessorCount=4 \
        -Dos.name=Linux \
        InMemoryJavaCompilerExample
}

rm -f *.class
/opt/taobao/java/bin/javac InMemoryJavaCompilerExample.java
build_instance
run
