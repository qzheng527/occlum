## 项目介绍
antllm是蚂蚁对话大模型项目代码库，包括基础底座预训练，instruct-tuning，RLHF等，期望最终达到chatGPT的能力，详细介绍参考[文档](https://yuque.antfin.com/crtpg4/xutwxe/xgxg4gbt5w4p71s6)

## 代码库结构
```
antllm
├── antllm # antllm核心打码，包括模型、dataset、optimizer等
│   ├── data
│   ├── model
│   └── optimizer
├── BUILD
├── config # 配置文件
│   ├── codex
│   ├── glm
│   ├── gpt
│   ├── rlhf
│   └── sft
├── data # 数据处理相关代码
├── evaluate # 评测相关代码
├── examples # 使用示例
├── OWNERS
├── README.md
├── tasks # 任务代码，包括：基座GLM、GPT，代码预训练(codex)、instruct tuing(sft)和RLHF等
│   ├── codex
│   ├── glm
│   ├── gpt
│   ├── rlhf
│   └── sft
├── tests # 测试用例
└── utils # 工具类代码
```