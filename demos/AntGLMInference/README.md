# Demos

This directory contains sample projects that demonstrate how to build and run inference on Occlum with AntGLM model.

## Build

* Install python packages with Conda
```
./install_python_with_conda.sh
```

* Build the Occlum instance
```
./build_occlum_instance.sh
```

## Run

It assumed that the AntGLM model **AntGLM-10B-RLHF-20230930** has been downloaded into directory **/work/models**.

1. Run inference on non-TEE env
```
./python-occlum/bin/python3 antglm_evaluator.py \
    --model-path /work/models/AntGLM-10B-RLHF-20230930/
```

2. Run inference on TEE env
```
cd occlum_instance
occlum run /bin/python3 /antglm/antglm_evaluator.py
```
