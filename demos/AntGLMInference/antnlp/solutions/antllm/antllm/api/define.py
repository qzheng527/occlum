VERSION = '0.0.7'


# 支持的基础模型，模型名到模型文件名的映射
ALLOWED_MODEL_NAMES = {
    'AntGLM-10B-RLHF-20230602': 'glm10b_rlhf_20230602',
    'AntGLM-10B-SFT-20230602': 'glm10b_sft_20230602',
    'AntGLM-5B-20230407': 'glm5b_sft',
    'AntGLM-5B-SFT-20230930': 'glm5b_sft_20230930',
    'AntGLM-CS-5B-20230525': 'antglm_cs_5b_20230525',
    'glm-300m': 'glm_300m',
    'AntGLM-10B-RLHF-20231031': 'antglm_10b_rlhf_20231031',
    'AntGLM-5B-PRETRAINED-20230930': 'antglm_5b_pretrained_20230930',
    'AntGLM-10B-RLHF-20230930': 'glm10b_rlhf_20230930',
    'AntGLM-10B-RLHF-20230831': 'antglm_10b_rlhf_20230831',
    'AntGLM-10B-RLHF-20230831-Identity': 'antglm_10b_rlhf_20230831_identity',
    'AntGLM-10B-SFT-20230831': 'antglm_10b_sft_20230831'
}

STUDENT_MODEL_NAMES = {
    'bart-base-chinese': 'bart-base-chinese',
    'bart-large-chinese': 'bart-large-chinese',
    'glm-300m': 'glm_300m'
}

# 系统命令：下载代码库、安装必要依赖以及挂载模型盘
AISTUDIO_SYSTEM_CMD = 'pip install antllm=={} -i https://artifacts.antgroup-inc.cn/simple/' \
                      ' --extra-index-url https://artifacts.antgroup-inc.cn/artifact/repositories/simple-dev/' \
                      ' && '.format(VERSION) + 'pip install http://alps-common.oss-cn-hangzhou-zmf.aliyuncs.com/' + \
                      'users%2Flizhi%2Fatorch-0.18.0.dev0%2Bfix.fsdp.prefix-py3-none-any.whl && ' + \
                      'mkdir /adabench_mnt && ' + \
                      'mount -t nfs -o vers=3,timeo=600,nolock,rsize=1048576,wsize=1048576,hard,retrans=3,noresvport' \
                      ' alipayshnas-026-jxd68.cn-shanghai-eu13-a01.nas.aliyuncs.com:/ /adabench_mnt'

# 数据schema
DATA_SCHEMA = {
    "type": "object",
    "properties":
        {
            "id": {},
            "input": {"type": "string"},
            "output": {"type": "string"}
        },
    "required": ["input", "output"]
}


# Packed数据schema
PACKED_DATA_SCHEMA = {
    "type": "object",
    "properties":
        {
            "id": {},
            "input": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            },
            "output": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            }
        },
    "required": ["input", "output"]    
}


# AISTUDIO URI
AISTUDIO_JOB_DETAIL = 'https://aistudio.alipay.com/project/job/detail'


# 模型和显存大小对应的 batch size 映射表
# 设计为模型、显存大小、最大训练文本长度以及是否使用FlashAttention于batch size对应关系，
# -1 表示该配置无法进行训练，会在训练代码中抛出异常
MODEL_AND_GPU_TO_BATCH_MAP = {
    "5B_MODEL_80G_MEMORY_1024_LENGTH_ATORCH": 8,
    "5B_MODEL_40G_MEMORY_1024_LENGTH_ATORCH": -1,
    "5B_MODEL_32G_MEMORY_1024_LENGTH_ATORCH": -1,
    "5B_MODEL_16G_MEMORY_1024_LENGTH_ATORCH": -1,
    "5B_MODEL_80G_MEMORY_1024_LENGTH": 3,
    "5B_MODEL_40G_MEMORY_1024_LENGTH": -1,
    "5B_MODEL_32G_MEMORY_1024_LENGTH": -1,
    "5B_MODEL_16G_MEMORY_1024_LENGTH": -1,
    "10B_MODEL_80G_MEMORY_1024_LENGTH_ATORCH": 6,
    "10B_MODEL_40G_MEMORY_1024_LENGTH_ATORCH": -1,
    "10B_MODEL_32G_MEMORY_1024_LENGTH_ATORCH": -1,
    "10B_MODEL_16G_MEMORY_1024_LENGTH_ATORCH": -1,
    "10B_MODEL_80G_MEMORY_1024_LENGTH": 2,
    "10B_MODEL_40G_MEMORY_1024_LENGTH": -1,
    "10B_MODEL_32G_MEMORY_1024_LENGTH": -1,
    "10B_MODEL_16G_MEMORY_1024_LENGTH": -1,
}


# 模型和显存大小对应的 batch size 映射表（使用Peft情况）
MODEL_AND_GPU_WITH_PEFT_TO_BATCH_MAP = {
    "5B_MODEL_80G_MEMORY_1024_LENGTH_ATORCH": 14,
    "5B_MODEL_40G_MEMORY_1024_LENGTH_ATORCH": 5,
    "5B_MODEL_32G_MEMORY_1024_LENGTH_ATORCH": 3,
    "5B_MODEL_16G_MEMORY_1024_LENGTH_ATORCH": -1,
    "5B_MODEL_80G_MEMORY_1024_LENGTH": 4,
    "5B_MODEL_40G_MEMORY_1024_LENGTH": 1,
    "5B_MODEL_32G_MEMORY_1024_LENGTH": 1,
    "5B_MODEL_16G_MEMORY_1024_LENGTH": -1,
    "10B_MODEL_80G_MEMORY_1024_LENGTH_ATORCH": 10,
    "10B_MODEL_40G_MEMORY_1024_LENGTH_ATORCH": 3,
    "10B_MODEL_32G_MEMORY_1024_LENGTH_ATORCH": 1,
    "10B_MODEL_16G_MEMORY_1024_LENGTH_ATORCH": -1,
    "10B_MODEL_80G_MEMORY_1024_LENGTH": 2,
    "10B_MODEL_40G_MEMORY_1024_LENGTH": -1,
    "10B_MODEL_32G_MEMORY_1024_LENGTH": -1,
    "10B_MODEL_16G_MEMORY_1024_LENGTH": -1,
}
