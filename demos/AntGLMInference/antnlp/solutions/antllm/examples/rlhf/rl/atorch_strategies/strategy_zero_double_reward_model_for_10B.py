import atorch

# actor_critic_ref_strategy = [
# ("half", "bf16"),
# ("zero2", {"not_use_fsdp": True, "sync_models_at_startup":False}),
# ]
actor_critic_ref_strategy = "configs/ds_config_trlx_bf16.json"

nproc_per_node = atorch.distributed.nproc_per_node()  # type:ignore
word_size = atorch.world_size()
node_size = atorch.distributed.node_size()  # type:ignore
reward_model_strategy = [
    ("half", "fp16"),
    ("fsdp", {"atorch_wrap_cls": {"GLMBlock"}}),
]

cost_model_strategy = [
    ("half", "fp16"),
    ("fsdp", {"atorch_wrap_cls": {"GLMBlock"}}),
]

if node_size is not None and node_size > 1:
    cost_model_strategy.append(
        ("parallel_mode", ([("data", nproc_per_node)], None, True))
    )

    reward_model_strategy.append(
        ("parallel_mode", ([("data", nproc_per_node)], None, True))
    )
