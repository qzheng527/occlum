#! /bin/bash
set -e

rm -rf occlum_instance
occlum new occlum_instance

pushd occlum_instance
rm -rf image
copy_bom -f ../pg.yaml --root image --include-dir /opt/occlum/etc/template

# Fix " version `XCRYPT_2.0' not found " issue
cp /lib/x86_64-linux-gnu/libcrypt.so.1 image/opt/occlum/glibc/lib/

new_json="$(jq '.resource_limits.user_space_size = "10000MB" |
                .resource_limits.kernel_space_heap_size = "2000MB" |
                .resource_limits.kernel_space_heap_max_size = "2000MB" |
                .resource_limits.kernel_space_stack_size ="4MB" |
                .resource_limits.max_num_of_threads = 192 |
                .env.default = [ "PATH=/usr/local/pgsql/bin" ] |
                .env.default += ["PYTHONHOME=/opt/python-occlum"] |
                .env.default += [ "LD_LIBRARY_PATH=/usr/local/pgsql/lib" ] |
                .entry_points += [ "/usr/local/pgsql/bin" ]' Occlum.json)" && \
echo "${new_json}" > Occlum.json

# Create external mount point
rm -rf ../pg_data
mkdir -p ../pg_data/upper && mkdir -p ../pg_data/lower
mkdir image/pg_data

# Put PG data in external mount
new_json="$(cat Occlum.json | jq '.mount+=[{"target": "/pg_data","type": "unionfs", "options": {"layers":[{"target": "/pg_data", "type": "sefs", "source": "../pg_data/lower"},{"target": "/pg_data", "type": "sefs", "source": "../pg_data/upper"}]}}]')" && \
echo "${new_json}" > Occlum.json

# Create external mount point
rm -rf ../raw
mkdir -p ../raw/upper && mkdir -p ../raw/lower
mkdir image/raw

# Add raw directory in external mount for other data
new_json="$(cat Occlum.json | jq '.mount+=[{"target": "/raw","type": "unionfs", "options": {"layers":[{"target": "/raw", "type": "sefs", "source": "../raw/lower"},{"target": "/raw", "type": "sefs", "source": "../raw/upper"}]}}]')" && \
echo "${new_json}" > Occlum.json

# Set /tmp directory as ramfs
new_json="$(cat Occlum.json | jq '.mount+=[{"target": "/tmp", "type": "ramfs"}]')" && \
echo "${new_json}" > Occlum.json

# A root passwd is required for initdb
echo "root:x:0:0:root:/root:/bin/bash" > image/etc/passwd

occlum build

popd
