# Demos

This directory contains sample projects that demonstrate how Occlum can be used to build and run user applications.

## Toolchain demos

This set of demos shows how the Occlum toolchain can be used with different build tools.

* [hello_c](hello_c/): A sample C project built with Makefile/CMake.
* [hello_cc](hello_cc/): A sample C++ project built with Makefile/CMake.
* [hello_bazel](hello_bazel/): A sample C++ project built with [Bazel](https://bazel.build).

## Application demos

This set of demos shows how real-world apps can be easily run inside SGX enclaves with Occlum.

* [bash](bash/): A demo of [Bash](https://www.gnu.org/software/bash/) shell script.
* [cluster_serving](cluster_serving/): A demo of [Analytics Zoo Cluster Serving](https://analytics-zoo.github.io/master/#ClusterServingGuide/ProgrammingGuide/) inference solution.
* [fish](fish/): A demo of [FISH](https://fishshell.com) shell script.
* [flink](flink/): A demo of [Apache Flink](https://flink.apache.org).
* [font](font/font_support_for_java): A demo of supporting font with Java.
* [grpc](grpc/): A client and server communicating through [gRPC](https://grpc.io), containing [glibc-supported demo](grpc/grpc_glibc) and [musl-supported demo](grpc/grpc_musl).
* [https_server](https_server/): A HTTPS file server based on [Mongoose Embedded Web Server Library](https://github.com/cesanta/mongoose).
* [mysql](mysql/): A demo of [MySQL](https://www.mysql.com/).
* [openvino](openvino/) A benchmark of [OpenVINO Inference Engine](https://docs.openvinotoolkit.org/2019_R3/_docs_IE_DG_inference_engine_intro.html).
* [pytorch](pytorch/): Demos of standalone and distributed [PyTorch](https://pytorch.org/).
* [paddlepaddle](paddlepaddle/): A demo of [PaddlePaddle](https://www.paddlepaddle.org.cn/).
* [redis](redis/): A demo of [Redis](https://redis.io).
* [sofaboot](sofaboot/): A demo of [SOFABoot](https://github.com/sofastack/sofa-boot), an open source Java development framework based on Spring Boot.
* [sqlite](sqlite/) A demo of [SQLite](https://www.sqlite.org) SQL database engine.
* [swtpm](swtpm/) A demo of [SWTPM](https://github.com/stefanberger/swtpm) Software Trusted Platform Module (TPM) Emulator.
* [tensorflow](tensorflow/tensorflow_training): A demo of [TensorFlow](https://www.tensorflow.org/) MNIST classification training.
* [tensorflow_lite](tensorflow_lite/): A demo and benchmark of [TensorFlow Lite](https://www.tensorflow.org/lite) inference engine.
* [tensorflow_serving](tensorflow/tensorflow_serving_2.5.x): A demo of [TensorFlow Serving](https://github.com/tensorflow/serving)
* [vault](golang/vault/): A demo of [HashiCorp Vault](https://github.com/hashicorp/vault).
* [xgboost](xgboost/): A demo of [XGBoost](https://xgboost.readthedocs.io/en/latest).

## Benchmark demos

This set of demos shows how commonly used benchmarking tools can be run inside SGX enclaves with Occlum.

* [fio](benchmarks/fio/): A demo of [Flexible I/O Tester](https://github.com/axboe/fio).
* [iperf2](benchmarks/iperf2/): A demo of [Iperf2](https://sourceforge.net/projects/iperf2/), a tool for measuring Internet bandwidth performance.
* [iperf3](benchmarks/iperf3/): A demo of [Iperf3](https://github.com/esnet/iperf), a tool for measuring Internet bandwidth performance.
* [sysbench](benchmarks/sysbench/): A demo of [Sysbench](https://github.com/akopytov/sysbench), a scriptable multi-threaded benchmark tool for Linux.


## Programming language demos

This set of demos shows how apps written with popular programming languages can be run inside SGX enclaves with Occlum.

* [golang](golang/): A collection of [Golang](https://golang.org) program demos.
* [java](java/): A demo of [Java](https://openjdk.java.net) program.
* [python](python/) A collection of [Python](https://www.python.org) program demos, contain [glibc-supported python](python/python_glibc) demo and [musl-supported python](python/python_musl) demo.
* [rust](rust/) A demo of [Rust](https://www.rust-lang.org) program.

## Other demos

* [embedded_mode](embedded_mode/): A cross-enclave memory throughput benchmark enabled by the embedded mode of Occlum.
* [enclave_tls](enclave_tls/): Running TLS server inside Occlum. Client connects with server associate with Enclave-RA information.  
* [gdb_support](gdb_support/): This demo explains the technical detail of GDB support and demonstrates how to debug an app running upon Occlum with GDB.
* [local_attestation](local_attestation/): This project demonstrates how an app running upon Occlum can perform SGX local attestation.
* [remote_attestation](remote_attestation/): This project demonstrates how an app running upon Occlum can perform SGX remote attestation.
