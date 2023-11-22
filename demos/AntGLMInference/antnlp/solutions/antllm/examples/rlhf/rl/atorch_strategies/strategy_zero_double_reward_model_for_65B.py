import atorch
import torch
from solutions.antllm.antllm.models.glm.modeling_glm import GLMBlock, VocabEmbedding
from atorch.auto.opt_lib.zero_optimization import (
    get_skip_match_module_child_wrap_policy,
)

res_wrap_cls = get_skip_match_module_child_wrap_policy(
    (GLMBlock, torch.nn.Linear, torch.nn.LayerNorm, torch.nn.Embedding, VocabEmbedding)
)

actor_critic_ref_train_strategy = [
    ("half", "bf16"),
    (
        "fsdp",
        {
            "auto_wrap_policy": res_wrap_cls,
            "limit_all_gathers": True,
            "use_orig_params": True,
        },
    ),
]
# train_strategy: data_parallel + hybrid float precision
world_size = atorch.world_size()
nproc_per_node = atorch.distributed.nproc_per_node()  # type:ignore
node_size = atorch.distributed.node_size()  # type:ignore

actor_critic_ref_inference_strategy = [("parallel_mode", ([("tensor", 4)], None, True))]


nproc_per_node = atorch.distributed.nproc_per_node()  # type:ignore
word_size = atorch.world_size()
node_size = atorch.distributed.node_size()  # type:ignore
reward_model_strategy = [
    ("half", "fp16"),
    ("fsdp", {"atorch_wrap_cls": {"GLMBlock"}}),
]  # half float precision for inference

cost_model_strategy = [
    ("half", "fp16"),
    ("fsdp", {"atorch_wrap_cls": {"GLMBlock"}}),
]  # half float precision for inference
