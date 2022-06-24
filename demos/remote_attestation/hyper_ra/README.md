# HyperEnclave Remote Attestation Demo

This demo demonstrates how to do HyperEnclave remote attestation quote generation and verification on Occlum Enclave.

The demos utilizes the ioctl for quote generation and API `sgx_verify_quote` for quote verification. The `sgx_verify_quote` is provided in Hyper mode SGX SDK library `libsgx_uquote_verify_hyper.so` which is prebuilt and located in HyperEnclave docker image path `/opt/intel/sgxsdk/lib64/`.

Two APIs are wrapped to support HyperEnclave remote attestation in this demo.

* `hyper_ra_gen` to generate HyperEnclave remote attestation quote.
* `hyper_ra_verify` to verify HyperEnclave remote attestation quote.

Development certificates are provided in json file which is provided to `hyper_ra_verify` for verification.

Two example certificates json files are provided for reference.
* [`cfca.json`](./cfca.json)
* [`nocert.json`](nocert.json)

Because general certificate has multiple lines with newline characters which json can't support, thus to put the certificate to json string a extra operation needs to be done. For example, user has a certificate file cert.pem, convert it to json recognized string with below command.
```
# jq -sR . cert.pem
```

Users could modify the certificates per their usage.

## Hyper RA library

The library source files are in the [`src`](./src).
Once built is done, the library `libhyper_ra.so` will be in the `src` directory as well.

## Hyper RA Test App

The test application source file is in the [`test`](./test).
Once built is done, there is a executable binary `ra_test` in the `test` directory.

## How to build

Just run the script `build.sh` to do the build.

## How to run

Once build is done, go to `occlum_instance` then run
```
# occlum run /bin/ra_test <jsonfile> // cfca.json or nocert.json
```

It does below two things.
1. Generate HyperEnclave RA quote.
2. Verify HyperEnclave RA quote with CFCA_CERT mode or NO_CERT mode.

