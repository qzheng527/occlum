from typing import List
import pandas as pd
import json
from atorch.auto import auto_accelerate
import argparse
import atorch
import torch
from solutions.antllm.antllm.training.trainer.atorch_rl_ppo_trainer import (
    AtorchRLTrainer,
)
from atorch.rl.config import AtorchRLConfig
from solutions.antllm.examples.rlhf.rl.utils import (
    process_raw_sample,
    get_glm_prompt_dataset,
    get_scores_glm,
    get_actor_tokenizer,
)
import deepspeed
from atorch.rl.model_engine import ModelEngine
from atorch.common.log_utils import default_logger as logger
from solutions.antllm.antllm.models.glm.tokenization_glm import GLMTokenizer
from atorch.rl.model_utils.model_util import set_seed
from solutions.antllm.antllm.data.dataset.rl_dataset.offline_pipeline import GlmPipeline


class NewModelEngine(ModelEngine):
    def unwarp_inference_model(self, model, model_type):
        """
        unwarp model for inferencing
        """
        return model

    def apply_strategy_to_child_model(
        self,
        model_type,
        dataset=None,
        dataloader_args=None,
        loss_func=None,
        model_input_format="unpack_sequence",
    ):
        if self.models_strategies[model_type] != "torch_native":
            if isinstance(self.models_strategies[model_type], str):
                ds_config = json.load(open(self.models_strategies[model_type]))
                model = self.models[model_type]

                # initialize torch.optim.AdamW optimizer in trainer
                optimizer_class = self.optimizer_cls[model_type]
                optimizer_params = self.config.model.actor_critic_ref.optimizer.kwargs

                optimizer = optimizer_class(
                    model.parameters(),
                    **optimizer_params,
                )

                # create scheduler
                kwargs = self.config.train.scheduler["kwargs"]

                # scheduler_class = get_scheduler_class("cosine_warmup")
                scheduler = self.scheduler_class(optimizer, **kwargs)

                # create cpu adam optimizer and replace torch.optim.AadmW.optimizer in accelerator.py
                from deepspeed.ops.adam import DeepSpeedCPUAdam

                if ds_config.get("zero_optimization", None) is not None:
                    zero_optimization = ds_config["zero_optimization"]
                    if zero_optimization.get("offload_optimizer", None) is not None:
                        defaults = {
                            k: v
                            for k, v in optimizer.defaults.items()
                            if k in ["lr", "weight_decay"]
                        }
                        optimizer = DeepSpeedCPUAdam(optimizer.param_groups, **defaults)
                kwargs = {}
                if (
                    ds_config.get("gradient_accumulation_steps", None)
                    != self.config.train.gradient_accumulation_steps
                ):
                    ds_config[
                        "gradient_accumulation_steps"
                    ] = self.config.train.gradient_accumulation_steps
                if ds_config.get("gradient_clipping", None):
                    ds_config["gradient_clipping"] = (
                        self.config.train.max_grad_norm * 1.0
                    )

                if ds_config.get("bf16", None) is not None:
                    if ds_config.get("bf16").get("enabled", False):
                        ds_config.update(
                            {"fp16": {"enabled": False, "auto_cast": False}}
                        )
                kwargs["config_params"] = ds_config
                kwargs["model"] = model
                kwargs["optimizer"] = optimizer
                engine, optimizer, _, _ = deepspeed.initialize(**kwargs)
                self.auto_accelerated_models[model_type] = engine
                self.auto_accelerated_optimizer[model_type] = optimizer
                self.scheduler[model_type] = scheduler

            else:
                status, result, best_strategy = auto_accelerate(
                    self.models[model_type],
                    self.optimizer_cls[model_type],
                    dataset=dataset,
                    loss_func=loss_func,
                    prepare_input=None,
                    model_input_format=model_input_format,
                    optim_args=self.optimizer_cls_kwargs[model_type],
                    optim_param_func=None,
                    dataloader_args=dataloader_args,
                    ignore_dryrun_on_load_strategy=True,
                    load_strategy=self.models_strategies[model_type],
                    find_unused_parameters=True,
                )
                assert status, "Failed to apply atorch strategy"
                logger.info(
                    "best strategy for {} is {}".format(model_type, best_strategy)
                )
                self.auto_accelerated_models[model_type] = result.model
                self.auto_accelerated_optimizer[model_type] = result.optim
                self.loss_func[model_type] = result.loss_func
        else:
            self.auto_accelerated_models[model_type] = self.models[model_type]
            optimizer_cls_kwargs = (
                {}
                if self.optimizer_cls_kwargs[model_type] is None
                else self.optimizer_cls_kwargs[model_type]
            )
            self.auto_accelerated_optimizer[model_type] = None
            if self.optimizer_cls[model_type] is not None:
                self.auto_accelerated_optimizer[model_type] = self.optimizer_cls[
                    model_type
                ](self.models[model_type].parameters(), **optimizer_cls_kwargs)
        opt = self.auto_accelerated_optimizer[model_type]
        if opt is not None and self.scheduler.get(model_type, None) is None:
            self.scheduler[model_type] = self.scheduler_class(
                opt, **self.config.train.scheduler["kwargs"]
            )

        # model/optimizer class type is printed to help double check that strategy is applied
        logger.info(
            "after atorch applying optimizing strategy for {},  "
            "the type of model is {} and the type of optimizer is {}".format(
                model_type,
                self.auto_accelerated_models[model_type],
                self.auto_accelerated_optimizer[model_type],
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    atorch.init_distributed("nccl")
    print("initialize", atorch.rank())
    # model path
    parser.add_argument("--actor_path", type=str, required=False, default=None)
    parser.add_argument("--critic_path", type=str, required=False, default=None)
    parser.add_argument("--cost_model_path", type=str, required=False, default=None)
    parser.add_argument("--reward_model_path", type=str, required=False, default=None)

    # reward function related
    parser.add_argument("--norm_reward", action="store_true", default=False)
    parser.add_argument("--lambda_value", type=float, default=-0.6)
    # reward_model related paramters
    parser.add_argument("--rm_use_position_id", action="store_true", default=False)
    parser.add_argument("--rm_num_head", type=int, default=1)
    parser.add_argument("--rm_mean_value", action="store_true", default=False)
    parser.add_argument(
        "--rm_use_normalized_reward", action="store_true", default=False
    )
    # cost_model related paramters
    parser.add_argument("--cost_use_position_id", action="store_true", default=False)
    parser.add_argument("--cost_num_head", type=int, default=1)
    parser.add_argument("--cost_mean_value", action="store_true", default=False)
    parser.add_argument(
        "--cost_use_normalized_reward", action="store_true", default=False
    )

    # train data path
    parser.add_argument("--prompt_path", type=str, default=None)
    # random seed
    parser.add_argument("--seed", type=int, default=1000)

    parser.add_argument("--mask_type", type=str, default="[gMASK]")
    parser.add_argument("--val_size", type=int, default=200)
    parser.add_argument("--train_size", type=int, default=0)

    # atorch related params
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--exp_cfg_path", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)

    parser.add_argument("--tokenizer_path", type=str, default=None)

    args = parser.parse_args()
    # set seed before start training
    set_seed(args.seed)
    # get config
    config_file = (
        "./atorch_exps/config_actor_0.1b_critic_0.1b_sep_double_reward_model_10b_frozen_block_debug.yaml"
        if args.exp_cfg_path is None
        else args.exp_cfg_path
    )
    config = AtorchRLConfig.load_yaml(config_file)
    assert "log" in args.logdir
    config.train.logdir = args.logdir
    # config.train.checkpoint_dir = os.path.dirname(args.logdir)
    if args.checkpoint_dir is not None:
        config.train.checkpoint_dir = args.checkpoint_dir
    if args.strategy is not None:
        # config.model.actor_critic_ref.train_strategy = args.strategy
        # config.model.reward_model.train_strategy = args.strategy
        config.model.cost_model.train_strategy = args.strategy

    # overriding model path in config file
    if args.actor_path is not None:
        config.model.actor_critic_ref.model_path = args.actor_path
        # config.model.actor_critic_ref.model_params.update({"actor_path": args.actor_path})

    if args.critic_path is not None:
        config.model.actor_critic_ref.model_params.update(
            {"pretrained_critic_model_path": args.critic_path}
        )

    if args.cost_model_path is not None:
        config.model.cost_model.model_path = args.cost_model_path

    if args.reward_model_path is not None:
        config.model.reward_model.model_path = args.reward_model_path

    if args.strategy is not None:
        config.model.actor_critic_ref.train_strategy = args.strategy
        config.model.reward_model.train_strategy = args.strategy
        config.model.cost_model.train_strategy = args.strategy

    if args.tokenizer_path is None:
        config.tokenizer.tokenizer_path = args.actor_path
    else:
        config.tokenizer.tokenizer_path = args.tokenizer_path

    # update reward model params
    reward_model_params = {}
    reward_model_params = {
        "use_position_id": args.rm_use_position_id,
        "num_head": args.rm_num_head,
        "use_mean_value": args.rm_mean_value,
        "use_normalized_reward": args.rm_use_normalized_reward,
    }
    config.model.model["reward_model"].model_params.update(reward_model_params)

    cost_model_params = {}

    cost_model_params = {
        "use_position_id": args.cost_use_position_id,
        "num_head": args.cost_num_head,
        "use_mean_value": args.cost_mean_value,
        "use_normalized_reward": args.cost_use_normalized_reward,
    }

    config.model.model["cost_model"].model_params.update(cost_model_params)
    logger.info("args is {}".format(args))
    logger.info("config is {}".format(config))

    tokenizer = get_actor_tokenizer(
        config.tokenizer.tokenizer_path,
        config.tokenizer.params["padding_side"],
        config.tokenizer.params["truncation_side"],
    )
    truncation_side = tokenizer.truncation_side
    assert truncation_side in ["left", "right"]
    max_length_input = (
        config.train.seq_length - config.generation.gen_kwargs["max_new_tokens"]
    )

    model_engine = NewModelEngine(config)
    model_engine.tokenizer = tokenizer
    model_engine.apply_strategy()

    score_device = torch.device(
        "cuda:{}".format(atorch.local_rank())
    )  # set reward model device

    def reward_fn(samples: List[str], **kwargs):
        setattr(kwargs["reward_model"], "device", atorch.local_rank())
        setattr(kwargs["cost_model"], "device", atorch.local_rank())

        rewards = get_scores_glm(
            samples,
            tokenizer=rw_tokenizer,
            mask_type=args.mask_type,
            max_input_length=max_length_input,
            max_length=config.train.seq_length,
            score_device=score_device,
            score_model=kwargs["reward_model"],
            truncation_side=truncation_side,
        )

        costs = get_scores_glm(
            samples,
            tokenizer=cost_tokenizer,
            mask_type=args.mask_type,
            max_input_length=max_length_input,
            max_length=config.train.seq_length,
            score_device=score_device,
            score_model=kwargs["cost_model"],
            truncation_side=truncation_side,
        )
        scores = rewards - args.lambda_value * costs
        return scores

    def reward_fn_norm(samples: List[str], **kwargs):
        setattr(kwargs["reward_model"], "device", atorch.local_rank())
        setattr(kwargs["cost_model"], "device", atorch.local_rank())
        rewards = get_scores_glm(
            samples,
            tokenizer=rw_tokenizer,
            mask_type=args.mask_type,
            max_input_length=max_length_input,
            max_length=config.train.seq_length,
            score_device=score_device,
            score_model=kwargs["reward_model"],
            truncation_side=truncation_side,
        )
        costs = get_scores_glm(
            samples,
            tokenizer=cost_tokenizer,
            mask_type=args.mask_type,
            max_input_length=max_length_input,
            max_length=config.train.seq_length,
            score_device=score_device,
            score_model=kwargs["cost_model"],
            truncation_side=truncation_side,
        )
        scores = rewards - args.lambda_value * costs

        origin_idx = kwargs.get("prompts_idx", None)
        is_train = kwargs.get("is_train", None)

        if is_train:
            origin_samples = [
                process_raw_sample(
                    train_posts[idx],
                    train_outputs[idx],
                    tokenizer=tokenizer,
                    max_prompt_length=max_length_input,
                    mask_type=args.mask_type,
                    max_output_length=config.method.gen_kwargs["max_new_tokens"],
                )
                for idx in origin_idx
            ]
        else:
            origin_samples = [
                process_raw_sample(
                    val_posts[idx],
                    val_outputs[idx],
                    tokenizer=tokenizer,
                    max_prompt_length=max_length_input,
                    mask_type=args.mask_type,
                    max_output_length=config.method.gen_kwargs["max_new_tokens"],
                )
                for idx in origin_idx
            ]

            origin_rewards = get_scores_glm(
                origin_samples,
                tokenizer=rw_tokenizer,
                mask_type=args.mask_type,
                max_input_length=max_length_input,
                max_length=config.train.seq_length,
                score_device=score_device,
                score_model=kwargs["reward_model"],
                truncation_side=truncation_side,
            )

            origin_costs = get_scores_glm(
                origin_samples,
                tokenizer=cost_tokenizer,
                mask_type=args.mask_type,
                max_input_length=max_length_input,
                max_length=config.train.seq_length,
                score_device=score_device,
                score_model=kwargs["cost_model"],
                truncation_side=truncation_side,
            )
            origin_scores = origin_rewards - args.lambda_value * origin_costs
            scores = scores - origin_scores
        return scores

    def reward_samples(samples, **kwargs):
        reward_scores = None
        if args.norm_reward:
            reward_scores = reward_fn_norm(samples, **kwargs)
        else:
            reward_scores = reward_fn(samples, **kwargs)
        return reward_scores

    rw_tokenizer = GLMTokenizer.from_pretrained(
        config.model.reward_model.model_path, trust_remote_code=True
    )
    cost_tokenizer = GLMTokenizer.from_pretrained(
        config.model.cost_model.model_path, trust_remote_code=True
    )

    train_val_data = pd.read_csv(args.prompt_path)
    train_val_data = train_val_data.dropna().reset_index(drop=True)
    prompt_dataset = train_val_data["prompt"]
    output_dataset = train_val_data["output"]

    val_size = args.val_size
    train_size = val_size

    if args.train_size != 0:
        train_size = args.train_size
        train_posts = [d for d in prompt_dataset[:train_size]]
        train_outputs = [d for d in output_dataset[:train_size]]
    else:
        train_posts = [d for d in prompt_dataset[:-val_size]]
        train_outputs = [d for d in output_dataset[:-val_size]]

    val_posts = [d for d in prompt_dataset[-val_size:]]
    val_outputs = [d for d in output_dataset[-val_size:]]

    train_prompts = get_glm_prompt_dataset(
        train_posts, max_length_input, tokenizer=tokenizer, mask_type=args.mask_type
    )
    val_prompts = get_glm_prompt_dataset(
        val_posts, max_length_input, tokenizer=tokenizer, mask_type=args.mask_type
    )

    train_prompt_dataset = GlmPipeline(train_prompts, max_length_input, tokenizer)
    eval_prompt_dataset = GlmPipeline(val_prompts, max_length_input, tokenizer)
    ppo_trainer = AtorchRLTrainer(
        model_engine,
        {"train": train_prompt_dataset, "eval": eval_prompt_dataset},
        config,
        reward_fn=reward_samples,
    )
    ppo_trainer.learn()
