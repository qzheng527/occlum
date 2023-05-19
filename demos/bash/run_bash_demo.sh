#! /bin/bash
set -e

if [[ $1 == "musl" ]]; then
    echo "*** Run musl-libc bash demo ***"
    bomfile="../bash-musl.yaml"
else
    echo "*** Run glibc bash demo ***"
    bomfile="../bash.yaml"
fi

rm -rf occlum_instance
occlum new occlum_instance

pushd occlum_instance
rm -rf image
copy_bom -f $bomfile --root image --include-dir /opt/occlum/etc/template

new_json="$(jq '.resource_limits.user_space_size = "600MB" |
                .resource_limits.kernel_space_stack_size ="2MB"	' Occlum.json)" && \
    echo "${new_json}" > Occlum.json

# Create external mount point
rm -rf /occlum_mnt && mkdir -p /occlum_mnt
mkdir image/mnt

new_json="$(cat Occlum.json | jq '.mount+=[{"target": "/mnt","type": "sefs", "source": "/occlum_mnt"}]')" && \
echo "${new_json}" > Occlum.json

occlum build
occlum run /bin/occlum_bash_test.sh

popd
