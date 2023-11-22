package(default_visibility = ["//visibility:public"])

load("@rules_python//python:defs.bzl", "py_library")
load("//tools/build_rules:adabrain_packaging.bzl", "adabrain_wheel")


py_library(
    name = "antllm",
    srcs = glob([
        "*.py",
        "**/*.py",
    ]),
    data = glob([
        "api/configs/*.json",
        "api/configs/**/*.json"
    ])
)


adabrain_wheel(
    name = "antllm_whl",
    author = "Ant Group Large Language Modeling Team",
    description_file = ":api/README.md",
    homepage = "https://yuque.antfin-inc.com/innersource/antnlp/solutions-antllm-antllm-api-readme",
    license = "Copyright 2023 Ant Group Co., Ltd.",
    python_requires = ">=3.7",
    python_tag = "py3",
    requirements_file = "//:requirements/requirements_antllm.txt",
    strip_path_prefixes = [
        # "solutions/antllm/antllm/wheel_package_",
        "solutions/antllm/antllm/",
        "requirements/"
    ],
    version = "0.0.7",
    wheel_path_prefix = "antllm",
    entry_points = {
        "console_scripts": [
            "antllm_train_sft_aistudio = antllm.commands.sft.train_sft_aistudio:main",
            "antllm_train_model_distill = antllm.commands.distill.train_distill_aistudio:main"
        ]
    },
    deps = [":antllm"],
)
