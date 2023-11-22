load("//tools/build_rules:oss_data.bzl", "oss_data")

package(default_visibility = ["//visibility:public"])

oss_data(
    name = "rlhf_data",
    outs = [
        "glm-test-model/cog-pretrain.model",
        "glm-test-model/config.json",
        "glm-test-model/configuration_glm.py",
        "glm-test-model/generation_config.json",
        "glm-test-model/modeling_glm.py",
        "glm-test-model/tokenization_glm.py",
        "glm-test-model/tokenizer_config.json",
    ],
    oss_key_prefix = "adabrain_oss",
    sha256 = "314d7328ec8105b956d33aaf50f503f20124cae88d36be41743853c8a386d42f",
    url = "oss://antsys-adabrain/adabrain/test_data/glm-test-model.tar.gz",
)
