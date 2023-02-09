#! /bin/bash
set -e

# Install dependencies
# apt update && apt install -y openjdk-8-jdk

# Download perf tools
rm -rf mongodb-performance-test
git clone https://github.com/idealo/mongodb-performance-test.git

pushd mongodb-performance-test
jarfile=./latest-version/mongodb-performance-test.jar

#### Insert test
java -jar $jarfile -m insert -o 50000 -t 10 -db test -c perf

#### Balanced update/find test (1/1)
java -jar $jarfile -m update_one iterate_many -d 120 -t 10 10 20 20 30 30 -db test -c perf

popd