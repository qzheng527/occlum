#!/bin/bash
set -e

BLUE='\033[1;34m'
NC='\033[0m'

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" >/dev/null 2>&1 && pwd )"

MINIO_VER=RELEASE.2022-12-12T19-27-27Z
export GOBIN=${script_dir}/bin

# rm -rf bin

# # Install Minio with occlum-go
# CC=gcc occlum-go install github.com/minio/minio@${MINIO_VER}

# Init Occlum Workspace
rm -rf occlum_instance && occlum new occlum_instance
cd occlum_instance
new_json="$(jq '.resource_limits.user_space_size = "1MB" |
	.resource_limits.user_space_max_size = "4GB" |
	.resource_limits.kernel_space_heap_size="1MB" |
	.resource_limits.kernel_space_heap_max_size="500MB" |
	.resource_limits.max_num_of_threads = 256 |
	.env.default += [ "HOME=/minio", "PATH=/bin" ] |
	.env.default += [ "MINIO_ROOT_USER=occlum_minio", "MINIO_ROOT_PASSWORD=occlum_minio" ] |
	.env.untrusted += [ "MINIO_ROOT_USER", "MINIO_ROOT_PASSWORD" ] ' Occlum.json)" && \
echo "${new_json}" > Occlum.json

# Copy program into Occlum Workspace and build
rm -rf image
copy_bom -f ../minio.yaml --root image --include-dir /opt/occlum/etc/template
mkdir image/data
mkdir image/minio
occlum build

# Run the Golang Minio demo
echo -e "${BLUE}occlum run /bin/minio server /data${NC}"
occlum run /bin/minio server /data
