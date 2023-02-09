#! /bin/bash
set -e

rm -rf occlum_instance
occlum new occlum_instance

pushd occlum_instance
rm -rf image
copy_bom -f ../mongodb.yaml --root image --include-dir /opt/occlum/etc/template

mkdir -p image/var/lib/mongodb

new_json="$(jq '.resource_limits.user_space_size = "5000MB" |
                .resource_limits.kernel_space_heap_size = "300MB" |
                .resource_limits.max_num_of_threads = 128 ' Occlum.json)" && \
    echo "${new_json}" > Occlum.json

occlum build

popd
