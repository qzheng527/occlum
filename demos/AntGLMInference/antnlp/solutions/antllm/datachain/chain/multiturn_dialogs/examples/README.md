多轮 SFT 相关的数据处理代码.

以 `dialog_` 开头的是多轮对话数据处理方法，包括:
  - `dialog_convert.py`: 多轮对话统一结构和对话 prompt(input-output) 结构相互转换，以及一些对话截断等处理方法
  - `dialog_sample.py`: 采样脚本
  - `dialog_synthesis.py`: 多轮对话合成相关脚本
  - `datasource.py`: 多轮 SFT 数据源
  - `source_process.py`: 从各种渠道收集到的多轮对话原始数据集，每个原始数据的处理脚本，处理成多轮统一格式
