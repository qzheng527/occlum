#!/bin/bash
THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" >/dev/null 2>&1 && pwd )"
BUILD_DIR=/tmp/occlum_golang_toolchain
INSTALL_DIR=/opt/occlum/toolchains/golang

function go_build()
{
    local go_branch=$1
    local go_version=$2
    local build_dir=${BUILD_DIR}_$go_version
    local install_dir=${3:-${INSTALL_DIR}_$go_version}

    # Clean previous build and installation if any
    rm -rf ${build_dir}
    rm -rf ${install_dir}

    # Create the build directory
    mkdir -p ${build_dir}
    cd ${build_dir}

    # Download Golang
    git clone -b ${go_branch} https://github.com/occlum/go.git .

    # Build Golang
    cd src
    ./make.bash
    mv ${build_dir} ${install_dir}
}

# Exit if any command fails
set -e

# Install golang 1.18 first for later occlum golang build
wget https://go.dev/dl/go1.18.10.linux-amd64.tar.gz
rm -rf /usr/local/go
tar -C /usr/local -xzf go1.18.10.linux-amd64.tar.gz
rm -f go1.18.10.linux-amd64.tar.gz

PATH=/usr/local/go/bin:$PATH

echo "Building Go 1.20 for Occlum"
# 1.20 is default version for Occlum go
go_build "ago1.20.12_for_occlum" "1.20" ${INSTALL_DIR}

echo "Building Go 1.18 for Occlum"
go_build "go1.18.4_for_occlum" "1.18"

# Generate the wrappers for Go
mkdir -p ${INSTALL_DIR}/bin
cat > ${INSTALL_DIR}/bin/occlum-go <<EOF
#!/bin/bash
OCCLUM_GCC="\${CC:-\$(which occlum-gcc)}"
OCCLUM_GO_VER="\${OCCLUM_GO_VER:-1.20}"
OCCLUM_GOFLAGS="-buildmode=pie \$GOFLAGS"
if [ \$OCCLUM_GO_VER = "1.18" ]; then
    OCCLUM_GO_DIR=${INSTALL_DIR}_\$OCCLUM_GO_VER
else
    OCCLUM_GO_DIR=${INSTALL_DIR}
fi
CC=\$OCCLUM_GCC GOFLAGS=\$OCCLUM_GOFLAGS \${OCCLUM_GO_DIR}/bin/go "\$@"
EOF

chmod +x ${INSTALL_DIR}/bin/occlum-go
rm -rf /usr/local/go
