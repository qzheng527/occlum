# 针对开发者的用户使用说明文档

## 环境准备
开发者通过 pip 安装 antllm库
```pip install antllm -i https://artifacts.antgroup-inc.cn/simple/```
也可以直接在 antnlp 代码库中，通过 bazel 的方式，在 BUILD 中在依赖最新的 antllm库
```deps = ["//solutions/antllm"]```

机器环境可以直接使用 AIStudio 打好的“大模型官方镜像”。
![aistudio镜像](https://intranetproxy.alipay.com/skylark/lark/0/2023/png/3456424/1688961638610-c4dca791-97e9-4a96-b860-df67c3a359e6.png)

## FineTune
### 数据准备
数据文件使用jsonl格式，每行需包含input和output两个字段，input作为模型输入，output作为模型输出，示例如下：
```json
{
  "input": "小明今天带了20元钱去买文具。他先在一家店买了两支笔，花了8元钱。然后他又去了另一家店，花了5元钱买了一本笔记本>和3支笔。请问小明剩下多少元钱？\n",
  "output": "1. 计算小明在第一家店买笔后还剩下的钱数：20元 - 8元 = 12元\n2. 计算小明在第二家店购买后还剩下的钱数：12元 - 5元 - (3*1元) = 4元\n3. 因此，小明还剩下4元钱。"
}
```
数据上传：
```python
from antllm.api.data_utils import easy_upload


# 上传数据集，默认上传的数据集权限是public公开的，若不想公开可以加privilege=private参数
dataset_id = easy_upload(train_fpath, validation_fpath)
print(dataset_id)  # 记住该dataset_id，后续在远程训练时会用到
```
### 本地SFT训练
通过deepspeed在本地执行单机多卡训练
- 用法示例
```python
from antllm.api.fine_tune import FineTune


# 支持模型：AntGLM系列模型
tuner = FineTune(model="llm_model_name")

# 全量微调
tuner.train_local(
    train_fpath = "train_data_path",
    output_dir = "output_dir",
    validation_fpath="valid_data_path"
)

# 部分参数微调
# 支持的微调方式PEFT："lora", "adalora", "bitfit", "roem", "unipelt", "prompt", "ptuning", "prefix", "qlora"。如果为None，则表示全量参数微调。
# 使用lora的方式进行部分参数微调
tuner.train_local(
    train_fpath = "train_data_path",
    output_dir = "output_dir",
    validation_fpath="valid_data_path",
    peft="lora"
)

# 长度扩展微调
# 0.0.5版本AntLLM API支持长度扩展微调，添加 use_long_glm = True 参数即可进行长度扩展微调；
# 默认长度扩展为原始两倍，即 max_lenght * 2；
# 如需调整扩展倍数，请调整 fine_tune.json 配置文件中 long_glm_factor 参数调整长度放缩倍率；
tuner.train_local(
    train_fpath = "train_data_path",
    output_dir = "output_dir",
    validation_fpath="valid_data_path",
    peft="lora",
    use_long_glm=True
)

# 数据packed微调
# 0.0.6版本AntLLM API支持长度扩展微调，添加 use_packed_training = True 参数即可进行数据packed微调；
# 数据packed微调经验证可以在保持效果一致情况下训练速度提升7-8倍
tuner.train_local(
    train_fpath = "train_data_path",
    output_dir = "output_dir",
    validation_fpath="valid_data_path",
    use_packed_training=True
)
```

训练完成后可以在`output_dir`查看模型产出
```
output_dir/
├── runs # tensorboard 日志
│   ├── Jun26_17-20-41_h07b11256.sqa.eu95
│   │   ├── 1687771248.2987487
│   │   │   └── events.out.tfevents.1687771248.h07b11256.sqa.eu95.81153.1
│   │   ├── events.out.tfevents.1687771248.h07b11256.sqa.eu95.81153.0
│   │   └── events.out.tfevents.1687771260.h07b11256.sqa.eu95.81153.2
├── checkpoint-250  # 第250step或第N轮epoch的训练产出
│   ├── adapter_config.json     # peft方法的配置文件 (在调用API使用peft参数时产出)
│   ├── adapter_model.bin       # 训练后的peft方法参数文件 (在调用API使用peft参数时产出)
│   ├── pytorch_model.bin       # 训练后的基座模型参数 (在调用API设置peft=None时产出)
│   ├── config.json             # 模型配置文件
│   ├── configuration_glm.py    # glm配置文件
│   ├── hyper_parameters.json   # 训练中保存的超参数，用于识别是否使用peft方法
│   ├── tokenizer_config.json   # tokenizer的配置文件
│   ├── ...... # 一些训练的中间参数、模型代码或者文件
│   ├── cog-pretrain.model      # 旧版wordpiece文件
│   └── zhen_sp # 新版wordpiece文件
│       ├── added_tokens.json
│       ├── merge.model
│       ├── merge.txt
│       ├── merge.vocab
│       └── tokenizer_config.json
├── adapter_config.json
├── adapter_model.bin
├── pytorch_model.bin
├── cog-pretrain.model
├── deepspeed.json              # deepspeed参数
├── fine_tune.json              # 微调参数
├── log.txt                     # 训练日志
├── special_tokens_map.json
└── tokenizer_config.json
```

### 远程SFT训练（推荐）
通过发起aistudio任务执行远程训练
- 用法示例
```python
from antllm.api.data_utils import easy_upload
from antllm.api.fine_tune import FineTune
from antllm.api.object_classes import AntLLMk8sConf


# 支持已发布的模型名：AntGLM-10B-RLHF-20230602、AntGLM-10B-SFT-20230602、AntGLM-5B-20230407、AntGLM-CS-5B-20230525
# 训练参数配置有默认的可以不用指定，demo体验用的 V100 显卡上，需要使用一份小一点的配置
# 参考 https://code.alipay.com/ai-dls/antnlp/blob/master/solutions/antllm/antllm/api/configs/fine_tune_mini.json
tuner = FineTune(model="llm_model_name")
# 上传数据集
dataset_id = easy_upload(train_fpath, validation_fpath)
# k8s资源配置，gpu_num大于8使用分布式模式，默认使用公共资源池(gpudefault)的低保 A100 资源
k8s_conf = AntLLMk8sConf(app_name='gpudefault', gpu_num=2, priority='low', gpu_type='a100')

# 使用lora的方式进行微调
# 支持的微调方式PEFT："lora", "adalora", "bitfit", "roem", "unipelt", "prompt", "ptuning", "prefix", "qlora"。
# 如果peft=None，则表示全量参数微调。
task_id = tuner.train_remote(
    dataset_id,
    k8s_conf,
    peft="lora"
)

# 长度扩展微调，具体配置与本地训练相同
task_id = tuner.train_remote(
    dataset_id,
    k8s_conf,
    peft="lora",
    use_long_glm=True
)

# 数据packed微调，具体配置与本地训练相同
task_id = tuner.train_remote(
    dataset_id,
    k8s_conf,
    peft="lora",
    use_packed_training=True
)

# 断点续跑
task_id = tuner.train_remote(
    dataset_id,
    k8s_conf,
    peft="lora",
    resume_from_run='last_task_id'
)
```
task_id可以用来获取训练产出物，执行以下命令

```shell
adabench_cli run-download --run_id ${task_id} --download_path ${download_path}
```

任务提交以后会立即返回，日志会输出aistudio训练任务链接和task_id，当训练完成后，task_id可以用来获取训练产出物，也可以用于模型部署。
![](https://intranetproxy.alipay.com/skylark/lark/0/2023/png/3456424/1687944675104-b7973b81-32e0-4351-a968-6e086cc491bd.png)

### Packed数据格式
`0.0.6`版本AntLLM提供了数据Packed能力，允许用户通过数据Packed形式进行训练，
通过Packed训练可以在效果保持一致的情况下，使得训练速度提升7-8倍。
数据packed需要对数据进行预处理，当然我们提供两种处理方法，首先是在线预处理方法，用户无需关心具体处理流程和实现（**默认推荐**）；
其次，提供了离线处理脚本，用户可以自定义处理自己的数据构造为Packed形式，数据格式如下：
```json
{
    "input": ["1 + 2", "1 - 3", "1 + 2"], 
    "output": ["1 + 2 = 3, 答案是 3", "1 - 3 = -2, 答案是 -2", "1 + 2 = 3, 答案是 3"]
}
{
    "input": ["以下是一道小学数学题：小明手里有三个糖果，小红给了他两个糖果，现在他一共有几个糖果？", "小明家有5只小鸡和3只小鸭，它们一起在花园里玩耍。问花园里有几只鸟？", "以下是一道小学数学题：小明手中有 5 个糖果，他吃了 2 个，现在手中还有几个糖果？"],
    "output": ["小明一开始手里有三个糖果，小红又给了他两个糖果，所以现在他手里的糖果数量会增加两个。因此，小明现在手里有五个糖果。<n>答案：小明现在有五个糖果。", "<n>小鸡和小鸭都是鸟类，所以小明家里的小鸡和小鸭都属于鸟类。<n>小明家里有小鸡5只，小鸭3只，所以花园里有的鸟的总数是5+3=8只。所以答案是8。", " <n>首先，读懂题目，明确小明开始有 5 个糖果。接着，小明吃掉了 2 个糖果。我们可以用减法算出小明现在手中剩余了多少糖果。<n>5 - 2 = 3<n>因此，小明现在手中还有 3 个糖果。"]}
```
目前我们在代码库中准备了对应的离线脚本给大家使用，具体为：`antllm.data.tools.preprocess_packed_data.py`。

注意：如果需要使用离线的packed数据，请在训练参数中将`online_packed`参数设置为`False`。


### 动态Batch Size计算
SFT API中提供了动态batch size计算功能，帮助用户最大化利用显存，可以通过在`train_local`中设置`dynamic_batch=True`开启该功能。

**计算原理**:

通常一个GPU模型训练程序的显存占用分为四个部分：模型自身大小、前向传播、后向传播以及优化器显存占用。
在Deepspeed场景下优化器会将梯度卸载至CPU进行计算，因此只需计算前三个部分即可。
需要注意，在CUDA启动和计算后向传播过程中会有不同的CUDA CONTEXT，即torch所必须的CUDA环境占用内存，
占用内存大小和GPU型号、CUDA版本以及torch版本有关，这里统一默认设置为2048 MiB。

**参数计算**：
```latex
- 模型自身大小：根据模型`hidden_size`、`num_layers`、`num_heads`等参数进行计算，再加上额外的CUDA CONTEXT。对于AntGLM，其计算公式为:

    $$vocab\_size \times hidden\_size + num\_layers \times hidden\_size \times hidden\_size \times (4 + 4 + 3 + 1)$$

    其中`4+4`代表`MLP`中的`FFN`网络，`3`代表`attention`中的`query_key_value`计算，`1`代表`attention`中的全连接。注意，这个计算公式中忽略了`LayerNorm`中的少量参数。

- 前向传播：针对AntGLM，将前向激活值计算分为以下部分
    - MLP层中的激活值：两次线性计算（`4 + 1`）、一次激活计算（`4`）和一次dropout（`0.5`）:
        $$mlp\_forward\_size = max\_length \times hidden\_size \times (4 + 1 + 4 + 0.5)$$

    - 注意力权重激活值：`attention`计算过程中的注意力图计算，和序列长度高度相关，包含两次权重计算，一次全链接和一次`dropout`:
        $$
        \begin{split}
        attention\_score &= head\_size \times max\_length^2 \times 2 \\
        & + max\_length \times hidden\_size \\
        & + 1.5 \times max\_length^2 \times head\_size
        \end{split}
        $$
    
    - `qkv`映射函数激活值，会受到`peft`方法的影响（公式中的后半部分，包含`dropout`）：
        $$
        \begin{split}
        qkv &= max\_length \times hidden\_size \times 3 \\ 
        & + (max\_length \times hidden\_size \times 1.5 + max\_length \times rank)_{lora}
        \end{split}
        $$
    
    - `attention`中所有的激活值（`qkv`映射函数激活值和注意力权重激活值）以及dropout和一个全联接层的计算：
        $$
        \begin{split}
        attention &= attention\_score + qkv \\
        & + max\_length \times hidden\_size \times 2 + hidden\_size \times max\_length \times 0.5
        \end{split}
        $$
    
    - 残差激活值，每个`GLMLayer`会进行两次残差计算：
        $$ res\_connect = max\_length \times hidden\_size \times 3 + hidden\_size $$
    
    - 最终计算公式：
        $$
        \begin{split}
        &forward\_memory\_per\_batch = \\
        &(attention * 0.75 + mlp\_forward\_size * 0.65 + res\_connect) \\
        & * num\_layers \div 1024^2 * tensor\_byte\_size
        \end{split}
        $$
        其中`0.75`和`0.65`是放缩系数，由实际运算估计得出，这是torch在计算图构建过程中会做一定的优化，释放一些不需要保存的激活值以节约显存，因此该值会略微大于实际的显存占用值，最终的`batch size`可能略小于真实的最大值。

- 后向传播：后向梯度计算部分需要有以下显存占用：
    - 类似于CUDA CONTEXT的显存占用，设计为固定值2048 MiB，P100上实际值为1900 MiB
    - 需要训练的模型参数
    - 最后一层输出用于回传的前向激活值
```

**参考资料**

- [Pytorch动态计算图规则](https://www.pytorchmaster.com/2-3%2C%E5%8A%A8%E6%80%81%E8%AE%A1%E7%AE%97%E5%9B%BE/)
- [PyTorch显存机制分析](https://zhuanlan.zhihu.com/p/424512257)
- [deepspeed原理](https://www.deepspeed.ai/training/)
- Torch官方对于这个问题的讨论：[GPU memory estimation given a network](https://discuss.pytorch.org/t/gpu-memory-estimation-given-a-network/1713)、[How to calculate the GPU memory that a model uses?](https://discuss.pytorch.org/t/how-to-calculate-the-gpu-memory-that-a-model-uses/157486)、[GPU memory that model uses](https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822)

## Completion
本地模型推理补全，支持单条和batch的方式
```python
from antllm.api.completion import Completion


completer = Completion("local_model_path")
text = "请问北京在哪里？"
# 单条补全
outs = completer.generate(text)
print(outs.texts[0])

# batch补全
texts = ["请问北京在哪里？", "中国的首都在哪里？"]
outs = completer.generate_batch(texts)
print(outs[0].texts[0])
```

`0.0.5`版本AntLLM API支持32K上下文窗口推理
```python
from antllm.api.completion import Completion


completer = Completion("local_model_path", use_long_glm=True)
text = "请问北京在哪里？"

# 目前上下文扩展仅支持单条补全
outs = completer.generate(text)
print(outs.texts[0])

```

## RemoteCompletion
通过指定部署服务的英文名称scene_name以及对应的版本chain_name来进行远程预测
支持单条预测以及batch预测
```python
from antllm.api.completion import RemoteCompletion

scene_name = "xuesun"
chain_name = "v1"
remote_completer = RemoteCompletion(
    scene_name=scene_name, chain_name=chain_name)
# 直接使用
query="今天天气不错"
output = remote_completer.generate(query)
# {'texts': [',适合出门散步。我打算去公园，呼吸新鲜空气，放松身心。'], 'finish_reasons': ['EOS']}
# 直接使用
batch_query=["今天天气不错", "中国在哪里？"]
batch_output = remote_completer.generate_batch(query)
# [{'texts': [',适合出门散步。我打算去公园，呼吸新鲜空气，放松身心。'], 'finish_reasons': ['EOS']}, {'texts': [' 很抱歉，这个问题可能会涉及到个人政治观点和立场，回答可能会引起不必要的争议或者歧视。'], 'finish_reasons': ['EOS']}]


# 指定 adapter_name
remote_completer.generate(query, adapter_name="test")
remote_completer.generate_batch(query, adapter_name="test")
```

## 大模型蒸馏

将大模型的能力蒸馏到规模较小的模型上，小模型学习到大模型的能力后可以以低计算资源成本、低服务反馈时延的形式提供模型预测能力

hard target 形式的模型蒸馏：小模型直接学习大模型在无标签数据上的输出，来进行学习
soft target 形式的模型蒸馏：小模型通过学习大模型的在每个token上的prob分布来学习大模型的能力，同时也会学习训练数据标签

```python
import antllm

distiller = antllm.api.Distill(model="your student model", teacher_model="your llm teacher model",
                               distill_config="distill config in type str or dict")
distiller.train_local(
    data_folder="train_data_folder",
    output_dir="output_dir"
)

# 注：model 的例子，如：glm-300m、bart-base-chinese等，
#    如果使用train_local形式训练模型，则需要是模型所在的路径
#
#    teacher_model 的例子，如：AntGLM-10B-SFT-20230602、AntGLM-CS-5B-20230525等，
#    同样的，如果使用train_local形式训练模型，则需要是模型所在的路径
# 
#    在antllm/api/define.py下可以在ALLOWED_MODEL_NAMES、STUDENT_MODEL_NAMES可以看到所有远端运行允许的teacher模型和student模型
```

更加推荐使用远端形式运行的模型蒸馏训练
```python
from antllm.api.data_utils import distill_data_upload
from antllm.api.object_classes import AntLLMk8sConf
import antllm

distiller = antllm.api.Distill(model="your student model", teacher_model="your llm teacher model",
                               distill_config="distill config in type str or dict")

# 上传数据集
dataset_id = distill_data_upload(teacher_finetune_train_path="", teacher_finetune_validation_path="",
                                 distill_train_path="", distill_validation_path="",
                                 student_finetune_train_path="", student_finetune_validation_path="",
                                 dataset_id="")
# 其中 distill_train_path、distill_validation_path为必选参数，其他按需配置

# k8s资源配置，gpu_num大于8使用分布式模式
k8s_conf = AntLLMk8sConf(app_name='gpudefault', gpu_num=4)
# 执行训练
distiller.train_remote(dataset_id, k8s_conf)



```

在初始化阶段需要配置需要用到的student模型路径和teacher模型路径，distill_config配置distill用到的
配置信息，可以是文件路径或者是dict形式的配置信息。

distill_config，其参数样例如下：

```jsonnet
{
  "distill_method": "",  # 使用的蒸馏方案，可选项为 ["hard_target", "soft_target"]
  "trainer_type": "Seq2SeqTrainer", # 在当前distill_method下使用的trainer_type
  ....            缺省的配置为与trainer_type适配的细化配置，可以直接参考solutions/antllm/antllm/api/configs/distill下的配置示例
}
```

对于进行本地运行的蒸馏训练数据，需要满足如下结构
```text
distill_data
    # 大模型精调部分
    ｜--llm_finetune_train.jsonl #可选
    ｜--llm_finetune_eval.jsonl  #可选
    # 模型蒸馏部分
    ｜--distill_train.jsonl  #必选
    ｜--distill_eval.jsonl   #必选
    # 小模型精调部分
    ｜--student_train.jsonl  #可选
    ｜--student_eval.jsonl   #可选
```

每个文件中的每一行的形式为{"input": text, "output": text} 

### hard_target形式蒸馏
hard_target形式蒸馏包含三个步骤，可通过配置来确定是否执行此步骤
- 1 大模型精调
此步骤会使用大模型精调数据对大模型本身进行精调操作。
（如果config["teacher_fine_tune"]["do_fine_tune"]为true，则执行此步骤）
- 2 模型蒸馏
本步骤实现小模型对大模型的能力蒸馏：首先使用teacher的form_mimic_data方法产生伪标签数据，再使用蒸馏trainer来训练小模型拟合大模型输出

- 3 小模型精调
本步骤是否执行可通过config["student_fine_tune"]["do_fine_tune"]字段进行配置。
小模型在完成蒸馏学习后继续使用标注数据进行模型精调


### soft_target形式蒸馏
soft_target只包含了小模型精调，如需训练teacher大模型可以使用api中FineTune的方法
soft_target提供了3种蒸馏的loss：

- 对预测的每个token位置的logit，做teacher的与student的 KLDivLoss
通过配置 config["logit_weight"] 的大于0的参数，实现该loss是否使用的开关以及使用时的权重
注意：使用该方法，会要求teacher的与student的tokenizer是一样的。

- 对预测的每个token位置的last_hidden_states，做teacher的与student的 cosine_similarity
通过配置 config["hidden_state_cos_weight"] 的大于0的参数，实现该loss是否使用的开关以及使用时的权重
注意：使用该方法，会要求teacher的与student的hidden_size是一样的。

- 对预测的每个token位置的last_hidden_states，做teacher的与student的 MSELoss
通过配置 config["hidden_state_mse_weight"] 的大于0的参数，实现该loss是否使用的开关以及使用时的权重
注意：使用该方法，会要求teacher的与student的hidden_size是一样的。

- 不使用hard_target的loss
通过配置 config["hard_target_weight"] 的大于0的参数，实现该loss是否使用的开关以及使用时的权重


### 方案特异性

- chain of thought distillation

antllm/api/configs/distill/hard_target_cot.json 中有一个示例的思维链蒸馏的示例配置
思维链蒸馏主要的差异是会将大模型的推理过程同样给到小模型进行学习。
运用思维链蒸馏需要配置在cot_distill键下配置
few_shot_templates（cot的示例）
student_reason_input_template（解释的模版）
student_reason_output_template （模型输出需要满足的格式）

方案会首先在训练数据上精调大模型，然后让大模型给无标签数据给出label预测，再让大模型给出对当前样本做出这个预测label的原因。
最后student模型学习大模型的预测结果与预测原因。

## 关于模型部署
* 模型部署的相关文档请参考子文档中的模型部署文档

# 交流答疑群
大家使用过程中有问题或需求欢迎加群反馈。
钉钉群号：28335022111
<img src="https://intranetproxy.alipay.com/skylark/lark/0/2023/png/923/1694604482188-35dcee83-7b71-4d28-9979-fa41d6e09b3e.png">

[大模型API手册](https://yuque.antfin.com/crtpg4/xutwxe/gkllgd9gil4wwokm#J1dPt)
