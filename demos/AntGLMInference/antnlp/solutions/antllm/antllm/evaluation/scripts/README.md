# 下载benchmark数据集

你可以使用 `fetch_benchmark_datasets.py`脚本下的FetchBenchmarkDataset函数来下载所有评测数据集或者指定的数据集，在下载数据集后，会自动做一个格式的检查（需要提供一个dataset_des.json文件，而这个文件放在了antnlp下的 `solutions/antllm/evaluate/`路径下）。

我们基于原来的 `DownloadUtil`，重写了 `DownloadUtilNew `类，具体的代码运行方式如下：

例子

`cd ~/antnlp`

`python solutions/antllm/evaluate/scripts/fetch_benchmark_datasets.py IFLYTEK ./antllm_test/ solutions/antllm/antllm/evaluation/configs/datasets_des.json`

当你需要下载所有数据集时，可以把第一个参数改为all（`IFLYTEK` --> all)。

# 检查数据集格式

我们提供了 `check_dataset_format.py`脚本来帮助检查下载的数据格式是否符合我们的规范。你可以通过python命令调用，也可以用命令行的方式执行。

`python solutions/antllm/evaluate/scripts/check_dataset_format.py antllm_test/AFQMC solutions/antllm/antllm/evaluation/configs/datasets_des.json AFQM`
