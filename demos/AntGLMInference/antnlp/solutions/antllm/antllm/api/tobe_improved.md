一些待建设完善的能力，正式发布之后，则将相关内容移到`README.md`文档中

## Embedding

通用Embedding的获取，一般是不用训练，直接调用训练好的模型得到。
专为Embedding服务的模型还在训练中，下面展示的是基础大模型的embedding获取。
- 用法示例
```python
from antllm.api.embedding import Embedding


embedder = Embedding('model_path')

# 单条计算
# 目前支持三种embedding计算规则，分别是：last、sum、mean
# 请通过reduction参数指定，对于AntGLM推荐last，默认reduction=last
text = "支付宝应该如何提现？"
embedding = embedder.get_embedding(
    text,
    reduction="last"
)
print(embedding)
print(embedding.size())


# 批量计算
texts = [
    "我觉得这只猫长得可爱",
    "这条狗好丑",
    "这只猫长得真可爱",
    "请问我的支付宝账号如何注销",
    "请问我如何注销我的支付宝，我不想用了",
    "支付宝应该如何提现？"
]
embeddings = embedder.get_embedding(texts)
print(embeddings.size())
similarity = torch.cosine_similarity(embeddings[None, ...], embeddings[:, None], dim=-1)
print(similarity)

```

## Tasks
常见的一些任务，如对话、分类、匹配
### 对话 chat
```python

import antllm

bot = antllm.api.Chat(model="antglm-10b-0407")
# 多轮对话，需要先初始化，用于清空对话历史
bot.init_chat()
output: CompletionOutput = bot.chat(prompt='你是谁研发的', max_tokens=32, stream=False, num_beams=5, temperature=1, topk=10, topp=0.9)
# 获取对话历史
print(bot.history_messages())
# 列表结构，每个item为一个json，包括role（角色），角色包括：user、assistant、system，content（内容）
[..., {"role": "user", "content": "你是谁研发的"}]
```

### 分类
```python

import antllm


cls = antllm.api.Classification(model="antglm-10b-0408")

# 分类的数据检查作为推荐环节，并产出详细的数据分析报告。
# 可以把一些异常数据（比如超长）过滤出来，并给出一些针对性的建议。
cls.analyse_data(fpath, report_path=data_report_dir)

# 设置antllm包装的peft相关配置
peft_type = antllm.trainer.peft.LORA
# peft_type = antllm.trainer.peft.PTUING
# peft_type = antllm.trainer.peft.PREFIX
# peft_type = antllm.trainer.peft.PROMPT
# peft_type = antllm.trainer.peft.ADALORA

# 执行训练，会先检查类别数、
# 数据格式jsonl
# output_dir目录下文件和内容说明见Finetue
sft.train_local(train_fpath, output_dir, validation_fpath=None, peft=None, epoch=2)

# 发起k8s任务，数据支持本地或者nas路径，nas路径需要包含nas盘名的完整路径，比如：
# train_fpath: ***.nas.aliyuncs.com:/train_path
# output_dir: ***.nas.aliyuncs.com:/output_dir
# validation_fpath: ***.nas.aliyuncs.com:/validation_path
# 根据gpu_num计算需要多少cpu，多少memory，多少disk，是否需要分布式以及worker数
k8s_conf = AntLLMk8sConf(app_name='your_app', gpu_num=32)
sft.train_remote(train_fpath, output_dir, validation_fpath=None, peft=None, epoch=2, k8s_conf=k8s_conf)

# 分类后处理，预测的时候会调用，可选后处理器
generation_results = cls.generate(prompt='', normalizer=None)

generation_results = cls.generate_batch(prompts=[], normalizer=None)
```
