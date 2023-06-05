#! /bin/bash
set -e

rm -rf occlum_instance
occlum new occlum_instance

pushd occlum_instance
rm -rf image
copy_bom -f ../pg.yaml --root image --include-dir /opt/occlum/etc/template

# Fix " version `XCRYPT_2.0' not found " issue
cp /lib/x86_64-linux-gnu/libcrypt.so.1 image/opt/occlum/glibc/lib/

new_json="$(jq '.resource_limits.user_space_size = "8000MB" |
                .resource_limits.kernel_space_heap_size ="1000MB" |
                .resource_limits.max_num_of_threads = 96 |
                .env.default = [ "PATH=/usr/local/pgsql/bin" ] |
                .env.default += ["PYTHONHOME=/opt/python-occlum"] |
                .env.default += [ "LD_LIBRARY_PATH=/usr/local/pgsql/lib/:$LD_LIBRARY_PATH" ] |
                .entry_points += [ "/usr/local/pgsql/bin" ]' Occlum.json)" && \
echo "${new_json}" > Occlum.json

# Create external mount point
rm -rf ../pg_data && mkdir -p ../pg_data
mkdir image/pg_data

# Put PG data in external mount
new_json="$(cat Occlum.json | jq '.mount+=[{"target": "/pg_data","type": "sefs", "source": "../pg_data"}]')" && \
echo "${new_json}" > Occlum.json

# A root passwd is required for initdb
echo "root:x:0:0:root:/root:/bin/bash" > image/etc/passwd

occlum build

popd
