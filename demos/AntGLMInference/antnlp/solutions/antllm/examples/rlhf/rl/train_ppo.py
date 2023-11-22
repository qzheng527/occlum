from typing import List
import pandas as pd
import argparse
import json

import os
import torch

from transformers import AutoTokenizer

import trlx
import torch.distributed as dist
from solutions.antllm.antllm.training.arguments.rl_arguments import TRLConfig
from solutions.antllm.antllm.training.trainer.accelerate_ppo_mix_trainer import AcceleratePPOMixTrainer # NOQA
from solutions.antllm.antllm.training.trainer.accelerate_ppo_separate_trainer import AcceleratePPOSeparateTrainer # NOQA
from solutions.antllm.antllm.data.dataset.rl_dataset.offline_pipeline import GlmPipeline # NOQA
from solutions.antllm.antllm.models.glm.modeling_glm_ppo import PPOMixConfig, PPOSeparateConfig # NOQA
from solutions.antllm.antllm.models.glm.tokenization_glm import GLMTokenizer
from solutions.antllm.antllm.models.glm.configuration_glm import GLMConfig

from solutions.antllm.examples.rlhf.rl.utils import (
    process_raw_sample,
    get_glm_prompt_dataset,
    load_pretrained_models,
    get_scores_glm,
    init_eval_engine,
    get_deepspeed_eval_config,
    token_repetition_rate
)


if __name__ == "__main__":
    try:
        from alps.pytorch.components.transformers import patch_get_class_in_module

        patch_get_class_in_module()
    except ModuleNotFoundError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--rm_model_path", type=str, default=None)
    parser.add_argument("--ppo_model_path", type=str, default=None)
    parser.add_argument("--prompt_path", type=str, default=None)
    parser.add_argument("--data_type", type=str, default="csv")
    parser.add_argument("--exp_cfg_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--mask_type", type=str, default="[gMASK]")
    parser.add_argument("--rl_norm_reward", action="store_true", default=False)
    parser.add_argument("--rm_mean_value", action="store_true", default=False)
    parser.add_argument("--rm_use_position_id", action="store_true", default=False)
    parser.add_argument(
        "--rm_use_normalized_reward", action="store_true", default=False
    )
    parser.add_argument("--val_size", type=int, default=500)
    parser.add_argument("--num_head", type=int, default=1)
    parser.add_argument("--rw_device", type=int, default=7)
    parser.add_argument("--repetition_penalty", type=float, default=0.)
    parser.add_argument("--repetition_ngrams", type=str, default="1,2")
    parser.add_argument("--sigmoid_reward", action="store_true", default=False)

    args = parser.parse_args()
    config = TRLConfig.load_yaml(args.exp_cfg_path)

    if args.save_dir is not None:
        config.train.checkpoint_dir = args.save_dir
    if args.log_dir is not None:
        config.train.logging_dir = args.log_dir
    if args.ppo_model_path is not None:
        config.model.model_path = args.ppo_model_path
        config.tokenizer.tokenizer_path = args.ppo_model_path
    config.model.critic_model_path = args.rm_model_path

    eos_token = "<|endofpiece|>"
    if os.path.exists(os.path.join(config.model.model_path, "hyper_parameters.json")):
        with open(os.path.join(config.model.model_path, "hyper_parameters.json")) as f:
            hyper_parameters = json.load(f)
            eos_token = hyper_parameters.get("eos_token", eos_token)
    config.tokenizer.eos_token = eos_token

    model_config = GLMConfig.from_pretrained(config.model.model_path)
    rotary_type = model_config.to_dict().get("rotary_type", "none")
    config.model.rotary_type = rotary_type

    if int(os.environ.get("LOCAL_RANK")) == 0:
        print("eos token:", eos_token)
        print(config)
        print(args)

    # actor tokenizer
    if config.model.model_arch_type == "glm":
        tokenizer = GLMTokenizer.from_pretrained(config.tokenizer.tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer.tokenizer_path, trust_remote_code=True
        )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = config.tokenizer.padding_side
    tokenizer.truncation_side = config.tokenizer.truncation_side
    truncation_side = tokenizer.truncation_side
    assert truncation_side in ["left", "right"]
    max_length_input = (
        config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]
    )

    ds_eval_config = get_deepspeed_eval_config(
        stage=3,
        fp16=True,
        bf16=False,
    )

    rw_device = torch.device(
        "cuda:{}".format(args.rw_device)
    )  # set reward model device
    # if int(os.environ.get("LOCAL_RANK")) == 0:
    rw_model, rw_tokenizer = load_pretrained_models(
        args.rm_model_path,
        num_head=args.num_head,
        rm_mean_value=args.rm_mean_value,
        use_position_id=args.rm_use_position_id,
        use_normalized_reward=False  # rm 默认为pairwise
    )
    rw_model = init_eval_engine(
        model=rw_model,
        ds_config=ds_eval_config,
    )
    rw_model.eval()
    dist.barrier()

    repetition_ngrams = [int(ngram) for ngram in args.repetition_ngrams.split(",")]

    def reward_fn(samples: List[str], **kwargs):
        scores = get_scores_glm(
            samples,
            tokenizer=rw_tokenizer,
            mask_type=args.mask_type,
            max_input_length=max_length_input,
            max_length=config.train.seq_length,
            score_device=rw_device,
            score_model=rw_model,
            truncation_side=truncation_side,
            eos_token=eos_token,
            rotary_type=rw_model.model.config.to_dict().get("rotary_type", "none"),
            sigmoid_reward=args.sigmoid_reward
        )
        if args.repetition_penalty > 0.:
            repetition_rates = []
            for sample in samples:
                prompt, response = sample.split(args.mask_type, 1)
                repetition_rate = token_repetition_rate(rw_tokenizer, response, ngrams=repetition_ngrams)
                repetition_rates.append(repetition_rate)
            repetition_rates = torch.tensor(repetition_rates, device=scores.device)
            scores -= args.repetition_penalty * repetition_rates
        return scores

    def reward_fn_norm(samples: List[str], **kwargs):
        scores = get_scores_glm(
            samples,
            tokenizer=rw_tokenizer,
            mask_type=args.mask_type,
            max_input_length=max_length_input,
            max_length=config.train.seq_length,
            score_device=rw_device,
            score_model=rw_model,
            truncation_side=truncation_side,
            eos_token=eos_token,
            rotary_type=rw_model.model.config.to_dict().get("rotary_type", "none"),
            sigmoid_reward=args.sigmoid_reward
        )
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

        origin_scores = get_scores_glm(
            origin_samples,
            tokenizer=rw_tokenizer,
            mask_type=args.mask_type,
            max_input_length=max_length_input,
            max_length=config.train.seq_length,
            score_device=rw_device,
            score_model=rw_model,
            truncation_side=truncation_side,
            eos_token=eos_token,
            rotary_type=rw_model.model.config.to_dict().get("rotary_type", "none"),
            sigmoid_reward=args.sigmoid_reward
        )
        scores = scores - origin_scores
        if args.repetition_penalty > 0.:
            repetition_rates = []
            for sample in samples:
                prompt, response = sample.split(args.mask_type, 1)
                repetition_rate = token_repetition_rate(rw_tokenizer, response, ngrams=repetition_ngrams)
                repetition_rates.append(repetition_rate)
            repetition_rates = torch.tensor(repetition_rates, device=scores.device)
            scores -= args.repetition_penalty * repetition_rates
        return scores

    if args.data_type == "csv":
        train_val_data = pd.read_csv(args.prompt_path)
    else:
        train_val_data = []
        with open(args.prompt_path) as f:
            for line in f:
                train_val_data.append(json.loads(line))
        train_val_data = pd.DataFrame(train_val_data)
    train_val_data = train_val_data.dropna().reset_index(drop=True)
    prompt_dataset = train_val_data["prompt"]
    output_dataset = train_val_data["output"]

    val_size = args.val_size

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

    if args.rl_norm_reward:
        trainer = trlx.train(
            reward_fn=reward_fn_norm,
            prompts=train_prompts,
            eval_prompts=val_prompts,
            config=config,
        )
    else:
        trainer = trlx.train(
            reward_fn=reward_fn,
            prompts=train_prompts,
            eval_prompts=val_prompts,
            config=config,
        )
