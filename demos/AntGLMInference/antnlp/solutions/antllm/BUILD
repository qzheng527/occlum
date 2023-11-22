package(default_visibility = ["//visibility:public"])

load("@antllm//:requirements.bzl", "all_requirements")
load("@rules_python//python:defs.bzl", "py_library")
load("//tools/build_rules:pytest_suite.bzl", "pytest_suite")
load("//tools/build_rules:oss_data.bzl", "oss_data")

py_library(
    name = "antllm",
    srcs = glob([
        "*.py",
        "**/*.py",
    ]),
    data = [
        "//solutions/antllm/tests/data:rlhf_data",
    ],
    deps = ["//solutions/antllm/antllm"] + all_requirements,
)

py_library(
    name = "antllm_code",
    srcs = glob([
        "*.py",
        "**/*.py",
    ]),
    data = [],
    deps = ["//solutions/antllm/antllm"],
)

pytest_suite(
    name = "tests",
    size = "medium",
    tests = glob([
        "tests/**/test_*.py",
    ]),
    data = glob([
        "**/*",
    ]) + [":zhen_sp5"] + [":glm-super-mini-model"] + [":chatglm2-6b"] + [":super-mini-bart"],
    deps = [
        ":antllm",
    ],
)

oss_data(
    name = "zhen_sp5",
    outs = [
        "zhen_sp5/merge.model",
        "zhen_sp5/merge.vocab",
        "zhen_sp5/merge.txt",
        "zhen_sp5/tokenizer_config.json",
    ],
    oss_key_prefix = "adabrain_oss",
    sha256 = "5832f0093539a8b219c18e2f23f75b3baba5988b799ad5adbbfe2b43002d9831",
    url = "oss://antsys-adabrain/solutions/antllm/models/zhen_sp5.tar.gz",
)

oss_data(
    name = "glm-super-mini-model",
    outs = [
        "glm-super-mini-model/cog-pretrain.model",
        "glm-super-mini-model/config.json",
        "glm-super-mini-model/config.json.bak",
        "glm-super-mini-model/configuration_glm.py",
        "glm-super-mini-model/generation_config.json",
        "glm-super-mini-model/modeling_glm.py",
        "glm-super-mini-model/pytorch_model.bin",
        "glm-super-mini-model/tokenization_glm.py",
        "glm-super-mini-model/tokenizer_config.json",
        "glm-super-mini-model/zhen_sp/merge.model",
        "glm-super-mini-model/zhen_sp/merge.vocab",
        "glm-super-mini-model/zhen_sp/merge.txt",
        "glm-super-mini-model/zhen_sp/added_tokens.json",
        "glm-super-mini-model/zhen_sp/tokenizer_config.json",
    ],
    oss_key_prefix = "adabrain_oss",
    sha256 = "88a2412481ab52714f29d94dae2a1ae54dbf67b0ddbfda39fcf9581075d5a0ef",
    url = "oss://antsys-adabrain/solutions/antllm/models/glm-super-mini-model.tar.gz",
)

oss_data(
    name = "chatglm2-6b",
    outs = [
        "chatglm2-6b/tokenization_chatglm.py",
        "chatglm2-6b/tokenizer.model",
        "chatglm2-6b/tokenizer_config.json",
    ],
    oss_key_prefix = "adabrain_oss",
    sha256 = "0710fc992dfe27439e10bb2d36c7d236916f7c2d38a67d9ca8920f472c355de8",
    url = "oss://antsys-adabrain/solutions/antllm/models/chatglm2-6b.tar.gz",
)

oss_data(
    name = "super-mini-bart",
    outs = [
        "super_mini_bart/added_tokens.json",
        "super_mini_bart/config.json",
        "super_mini_bart/generation_config.json",
        "super_mini_bart/special_tokens_map.json",
        "super_mini_bart/tokenizer_config.json",
        "super_mini_bart/vocab.txt",
        "super_mini_bart/pytorch_model.bin"
    ],
    oss_key_prefix = "adabrain_oss",
    sha256 = "0ddee41c89ecb09ce299fedf4517e70229abcc08a1009bdf85427220972dacb2",
    url = "oss://antsys-adabrain/adabrain/pretrained/bart/super_mini_bart.tar.gz",
)
