# Build

## Install python packages by conda

```
./install_python_with_conda.sh
```

## Build Occlum instance

```
./build_occlum_instance.sh
```

# Run

Assume the model is already downloaded into the `/work/models/` directory.

## Non-TEE mode

### Simple case test

```
# No BigDL LLM optimization
./python-occlum/bin/python ./eval/qwen_evaluator.py \
    --model-path /work/models/Qwen-14B-Chat/

# With BigDL LLM optimization
./python-occlum/bin/python ./eval/bigdl_qwen_evaluator.py \
    --model-path /work/models/Qwen-14B-Chat/
```

### Test with datasets

```
mkdir results

# No BigDL LLM optimization
./python-occlum/bin/python ./eval/model_exec_fineva_main.py \
    --model-name qwen \
    --model-path /work/models/Qwen-14B-Chat/ \
    --datasets-path ./datasets/fineval_sample.json \
    --save-path ./results/ \
    --eval qwen

# With BigDL LLM optimization
./python-occlum/bin/python ./eval/model_exec_fineva_main.py \
    --model-name bigdl-qwen \
    --model-path /work/models/Qwen-14B-Chat/ \
    --datasets-path ./datasets/fineval_sample.json \
    --save-path ./results/ \
    --eval bigdl-qwen
```

## TEE mode

```
cd occlum_instance
mkdir results
cp ../datasets .
```

### Simple case test

```
# No BigDL LLM optimization
occlum run /bin/python3 /eval/qwen_evaluator.py \
    --model-path /models/Qwen-14B-Chat/

# With BigDL LLM optimization
occlum run /bin/python3 /eval/bigdl_qwen_evaluator.py \
    --model-path /models/Qwen-14B-Chat/
```

### Test with datasets

```
mkdir results

# No BigDL LLM optimization
occlum run /bin/python3 /eval/model_exec_fineva_main.py \
    --model-name qwen \
    --model-path /models/Qwen-14B-Chat/ \
    --datasets-path ./host/datasets/fineval_sample.json \
    --save-path ./host/results/ \
    --eval qwen

# With BigDL LLM optimization
occlum run /bin/python3 /eval/model_exec_fineva_main.py \
    --model-name bigdl-qwen \
    --model-path /models/Qwen-14B-Chat/ \
    --datasets-path ./host/datasets/fineval_sample.json \
    --save-path ./host/results/ \
    --eval bigdl-qwen
```

# Get Score

```
./python-occlum/bin/python ./eval/get_score.py \
    --model_name qwen \
    --result_path ./results/ \
    --score_save_path ./results
```