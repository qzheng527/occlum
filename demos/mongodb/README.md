# MongoDB on Occlum

[`MongoDB`](https://www.mongodb.com/) is a widely used MongoDB is a NoSQL distributed database program.

### Download and Build
```
./dl_and_build_mongod.sh
```
This command downloads Mongodb r6.0.0 source code and builds from it.
When completed, mongod binary is installed in path `./install/mongo/bin`.

### Build Occlum instance
```
./build_occlum_instance.sh
```
When completed, it generates one occlum instance to start MongoDB server in TEE.

### Run MongoDB server
```
cd occlum_instance
occlum run /bin/mongod --config /etc/mongod.conf
```
The template [`mongod.conf](./mongod.conf) is provided as a reference. Uses can do customization per their scenarios.

### Run Mongo shell

### Benchmark

We use [`mongodb-performance-test`](https://github.com/idealo/mongodb-performance-test) to do the benchmark test.

Uses can just run the script [`benchmark.sh`](benchmark.sh) or modify the script to do more customized benchmarks.