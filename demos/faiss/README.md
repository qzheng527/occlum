# FAISS Demo

[Faiss](https://github.com/facebookresearch/faiss) is a library for efficient similarity search and clustering of dense vectors.

## Prepare

* Install faiss by conda
```
./install_python_with_conda.sh
```

* download the ANN_SIFT1M dataset from http://corpus-texmex.irisa.fr/ and unzip it to the subdirectory **sift1M**.

## Build Occlum Instance

```
./build_occlum_instance.sh
```

## Run demo

The demo is from the [official faiss demo](https://github.com/facebookresearch/faiss/blob/master/demos/demo_auto_tune.py).

```
cd occlum_instance
OMP_NUM_THREADS=16 occlum run /bin/python3 demo_auto_tune.py
```

If you met **mprotect** error, you can try to add **max_map_count** then try again.
```
sysctl -w vm.max_map_count=655300
```
