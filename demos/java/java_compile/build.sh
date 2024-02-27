#!/bin/bash
set -e

BLUE='\033[1;34m'
NC='\033[0m'



build_instance() {
    # Init Occlum instance
    rm -rf occlum_instance && occlum new occlum_instance
    cd occlum_instance

    new_json="$(jq '.resource_limits.user_space_size = "1MB" |
                .resource_limits.user_space_max_size = "4GB" |
                .resource_limits.kernel_space_heap_size="1MB" |
                .resource_limits.kernel_space_heap_max_size="256MB" |
                .resource_limits.max_num_of_threads = 64 |
                .metadata.debuggable = true |
                .metadata.disable_log = false |
                .entry_points = [ "/usr/lib/jvm/java-8-openjdk-amd64/jre/bin/" ] |
                .env.untrusted +=[ "MALLOC_ARENA_MAX" ] |
                .env.default = [ "LD_LIBRARY_PATH=/usr/lib/jvm/java-8-openjdk-amd64/jre/lib:/usr/lib/jvm/java-8-openjdk-amd64/lib" ]' Occlum.json)" && \
    echo "${new_json}" > Occlum.json

    rm -rf image
    copy_bom -f ../bom.yaml --root image --include-dir /opt/occlum/etc/template
    mkdir image/app
    cp -f ../*.class image/
    cp ../FilePrinter.java image/
    cp ../file.txt image/
    # rm -rf app && mkdir app
    # cp -f ../*.class app/

    # # mount app as hostfs
    # new_json="$(cat Occlum.json | jq '.mount+=[{"target": "/app", "type": "hostfs","source": "./app"}]')" && \
    # echo "${new_json}" > Occlum.json

    occlum build
}

run() {
    occlum run /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java \
        -XX:ActiveProcessorCount=4 \
        -Dos.name=Linux \
        InMemoryJavaCompilerExample
}

rm -f *.class
javac InMemoryJavaCompilerExample.java
javac DynamicCompiler.java
build_instance
run
