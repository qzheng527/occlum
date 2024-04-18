#!/bin/bash
set -e

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" >/dev/null 2>&1 && pwd )"

function dl_build_install_pg()
{
    rm -rf pg15
    git clone -b REL_15_STABLE --depth=1 https://github.com/postgres/postgres.git pg15

    pushd pg15
    git apply ../0001-Make-pg15-running-on-Occlum.patch
    git apply ../0002-Occlum-do-not-support-setitimer-with-warning-message.patch
    git apply ../0001-Hack-lwlock-to-support-postgresml-extension.patch
    ./configure --without-readline --prefix=/usr/local/pgsql \
        --with-python --with-openssl PYTHON=../python-occlum/bin/python
        # --enable-debug CFLAGS="-ggdb -Og -g3 -fno-omit-frame-pointer"
    make -j$(nproc)
    make install

    # Build and install some plugins
    pushd contrib

    pushd postgres_fdw
    make
    make install
    popd

    pushd pgcrypto
    make
    make install
    popd

    pushd pg_stat_statements
    make
    make install
    popd

    popd
}

function dl_build_install_postgis()
{
    # build and install postgis
    rm -rf postgis-3.3.3dev*
    wget http://postgis.net/stuff/postgis-3.3.3dev.tar.gz
    tar -xvzf postgis-3.3.3dev.tar.gz
    pushd postgis-3.3.3dev
    ./configure --with-pgconfig=/usr/local/pgsql/bin/pg_config --without-protobuf
    make -j$(nproc)
    make install
    popd
}

function dl_build_install_citus()
{
    rm -rf citus
    git clone -b v11.2.0 https://github.com/citusdata/citus.git
    pushd citus
    PG_CONFIG=/usr/local/pgsql/bin/pg_config ./configure
    make -j$(nproc)
    make install
    popd
}

function dl_install_pgml()
{
    # update rust toolchain for posrgresml build
    # OCCLUM_RUST_VERSION=nightly-2023-11-17
    # curl https://sh.rustup.rs -sSf | \
    #     sh -s -- --default-toolchain ${OCCLUM_RUST_VERSION} -y

    cd ${script_dir}
    rm -rf postgresml
    git clone -b v2.8.2 https://github.com/postgresml/postgresml
    cd postgresml
    git submodule update --init --recursive
    cd pgml-extension
    cargo install cargo-pgrx --version 0.11.2
    cargo pgrx init --pg15=/usr/local/pgsql/bin/pg_config
    cargo pgrx package --pg-config /usr/local/pgsql/bin/pg_config

    pgml_path="./target/release/pgml-pg15/usr/local/pgsql"
    install -m 755 ${pgml_path}/lib/* `pg_config --libdir`
    install -m 755 ${pgml_path}/share/extension/* `pg_config --sharedir`/extension
}


echo "Download, build and install Postgresql ..."
dl_build_install_pg

export PATH=/usr/local/pgsql/bin:$PATH

# echo "Download, build and install PostGIS ..."
# Please refer to https://postgis.net/docs/postgis_installation.html#make_install_postgis_extensions
dl_build_install_postgis
dl_build_install_citus
dl_install_pgml
