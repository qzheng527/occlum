import json
import os
import sys
import uuid
from time import time
from typing import List, Optional, Tuple

import ray
import torch
import transformers
import accelerate
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from accelerate import Accelerator
from ray.air import session
from ray.air.checkpoint import Checkpoint
from rich.console import Console
from rich.table import Table

import trlx.utils.logging as logging
from packaging import version
from trlx.utils import Clock
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.configs import TRLConfig
from trlx.data.ppo_types import PPORLBatch, PPORLElement
from trlx.trainer import BaseRLTrainer, register_trainer
from trlx.utils import (
    filter_non_scalars,
    get_distributed_config,
    get_git_tag,
    get_optimizer_class,
    significant,
)
from ..scheduler.ppo_scheduler import get_scheduler_class
from trlx.pipeline.ppo_pipeline import PPORolloutStorage
from trlx.utils.modeling import RunningMoments, logprobs_of_labels

from solutions.antllm.antllm.models.glm.modeling_ppo import (
    AdaptiveKLController,
    AutoModelForCausalLMWithHydraValueHead,
    AutoModelForSeq2SeqLMWithHydraValueHead,
    FixedKLController,
)
from trlx.pipeline.offline_pipeline import PromptPipeline

from trlx.utils.modeling import flatten_dict
from solutions.antllm.antllm.models.peft.modeling_peft import AntPeftForCausalLM  # NOQA
from solutions.antllm.antllm.models.glm.modeling_glm_ppo import (
    AutoModelForGLMSeparate,
    AutoModelForGLMWithHydraValueHead,
)
from solutions.antllm.antllm.data.dataset.rl_dataset.glm_ppo_pipeline import (
    GLMPPORolloutStorage,
)
from solutions.antllm.antllm.utils.generation_utils import (
    prepare_inputs_for_generation_glm,
)
from solutions.antllm.antllm.utils.modeling_glm_ppo_utils import (
    freeze_bottom_causal_layers,
    freeze_bottom_seq2seq_layers,
    get_mixed_precision_dtype,
)
from solutions.antllm.antllm.models.glm.tokenization_glm import GLMTokenizer

try:
    from bigmodelvis import Visualization

    HAS_OPENDELTA = True
except ModuleNotFoundError:
    HAS_OPENDELTA = False


logger = logging.get_logger(__name__)


@register_trainer
class AcceleratePPOSeparateTrainer(BaseRLTrainer):
    """PPO Accelerate Separate Trainer"""

    def __init__(self, config: TRLConfig, **kwargs):
        """PPO Accelerate Separate Trainer initialization

        Args:
            config: Config
        """
        super().__init__(config, **kwargs)
        self.max_length = config.train.seq_length
        if version.parse(accelerate.__version__) >= version.parse("0.20.0"):
            self.accelerator = Accelerator(
                log_with=config.train.tracker,
                project_dir=config.train.logging_dir,
                gradient_accumulation_steps=config.train.gradient_accumulation_steps,
            )
        else:
            self.accelerator = Accelerator(
                log_with=config.train.tracker,
                logging_dir=config.train.logging_dir,
                gradient_accumulation_steps=config.train.gradient_accumulation_steps,
            )

        if self.accelerator.state.deepspeed_plugin is not None:
            # by accelerate's default, arguments in `model.forward` would be casted to half
            if "fp16" in self.accelerator.state.deepspeed_plugin.deepspeed_config:
                self.accelerator.state.deepspeed_plugin.deepspeed_config["fp16"][
                    "auto_cast"
                ] = False

        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            torch.distributed.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", 0))])

        self.model = self.setup_model()
        self.opt = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()

        script_name = os.path.basename(sys.argv[0]).rsplit(".", 1)[0]
        if not isinstance(config.model.model_path, str):
            model_name = str(config.model.model_path).split()[0]
        else:
            model_name = config.model.model_path.split("/")[-1]

        if self.accelerator.num_processes == 1:
            num_gpus = "1gpu"
        else:
            num_gpus = f"{self.accelerator.num_processes}gpus"
        branch = get_git_tag()[0]

        run_name = "/".join([script_name, model_name, num_gpus]) + f":{branch}"

        if self.accelerator.is_main_process and not ray.is_initialized():
            config_dict = self.config.to_dict()
            dist_config = get_distributed_config(self.accelerator)
            config_dict["distributed"] = dist_config
            init_trackers_kwargs = {}

            if config.train.tracker == "wandb":
                init_trackers_kwargs["wandb"] = {
                    "name": run_name,
                    "entity": self.config.train.entity_name,
                    "group": self.config.train.group_name,
                    "tags": ["/".join(get_git_tag())],
                    "mode": "disabled" if os.environ.get("debug", False) else "online",
                }

                self.accelerator.init_trackers(
                    project_name=self.config.train.project_name,
                    config=config_dict,
                    init_kwargs=init_trackers_kwargs,
                )
            elif config.train.tracker == "tensorboard":
                # flatten config for tensorboard, split list in hparams into flatten config
                if config_dict["model"].get(
                    "peft_config", None
                ):  # tensorboard does not support peft config type
                    config_dict["model"]["peft_config"] = str(
                        config_dict["model"]["peft_config"]
                    )
                config_dict_flat = flatten_dict(config_dict)
                config_dict_flat["optimizer/kwargs/beta_1"] = config_dict_flat[
                    "optimizer/kwargs/betas"
                ][0]
                config_dict_flat["optimizer/kwargs/beta_2"] = config_dict_flat[
                    "optimizer/kwargs/betas"
                ][1]
                config_dict_flat.pop("optimizer/kwargs/betas", None)
                config_dict_flat.pop("model/delta_kwargs/modified_modules", None)
                self.accelerator.init_trackers(
                    project_name=self.config.train.project_name,
                    config=config_dict_flat,
                )
            elif config.train.tracker is None:
                self.accelerator.init_trackers(
                    project_name=self.config.train.project_name
                )
            else:
                raise ValueError(
                    f"Only supported trackers are `wandb` and `tensorboard`. Got: `{config.train.tracker}`. "
                    "Set `tracker` to `None` to disable tracking."
                )
        # init tokenier
        if config.model.model_arch_type == "glm":
            self.tokenizer = GLMTokenizer.from_pretrained(
                config.tokenizer.tokenizer_path
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer.tokenizer_path
            )
        self.tokenizer.padding_side = config.tokenizer.padding_side
        self.tokenizer.truncation_side = config.tokenizer.truncation_side
        self.tokenizer.sep_token = "<sep>"
        if config.model.model_arch_type != "seq2seq":
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Setup rollout logging
        if config.train.rollout_logging_dir is not None:
            self.log_rollouts = True
            self.setup_rollout_logging(config)
        else:
            self.log_rollouts = False

        # Setup the rollout store
        # Rollouts contain the prompt & response, log probs, values and rewards - from each rollout
        if config.model.model_arch_type == "glm":
            self.store = GLMPPORolloutStorage(
                self.tokenizer.pad_token_id, self.tokenizer.sop_token_id
            )
        else:
            self.store = PPORolloutStorage(self.tokenizer.pad_token_id)

        # Create the rollout store dataloader (for batching up rollouts)
        # TODO (jon-tow): This is only used to satisfy to `accelerator.prepare` call constraint below - remove in future
        rollout_loader: DataLoader = self.store.create_loader(
            self.config.train.batch_size, shuffle=True
        )

        # Prepare multi-GPU acceleration
        (
            self.model,
            self.opt,
            self.scheduler,
            rollout_loader,
        ) = self.accelerator.prepare(
            self.model, self.opt, self.scheduler, rollout_loader
        )

        self.store.clear_history()  # Clear the rollout store

        # Setup a reference model when hydra heads are not used
        if not hasattr(self.model.base_model, "frozen_head") and not self.model.base_model.peft_type:
            self.ref_model = self.get_ref_model(self.config)
            dtype = get_mixed_precision_dtype(self.accelerator)
            self.ref_model.to(self.accelerator.device, dtype=dtype)
            self.ref_model.eval()
            logger.info("Reference model is sattled")

        # Setup the KL controller
        # This helps prevent large divergences in the controller (policy)
        if config.method.target is not None:
            self.kl_ctl = AdaptiveKLController(
                config.method.init_kl_coef, config.method.target, config.method.horizon
            )
        else:
            self.kl_ctl = FixedKLController(config.method.init_kl_coef)

        # Create the parameters for the Hugging Face language model's generator
        # method (that generates new tokens from a prompt).
        # https://huggingface.co/docs/transformers/v4.25.1/en/main_classes/text_generation#transformers.GenerationMixin.generate
        if config.model.model_arch_type == "seq2seq":
            self.generate_kwargs = dict(
                config.method.gen_kwargs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if config.method.gen_experience_kwargs is not None:
                self.generate_experience_kwargs = dict(
                    config.method.gen_experience_kwargs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            else:
                self.generate_experience_kwargs = None
        elif config.model.model_arch_type == "glm":
            self.generate_kwargs = dict(
                config.method.gen_kwargs,
                eos_token_id=self.tokenizer.convert_tokens_to_ids(config.tokenizer.eos_token),
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if config.method.gen_experience_kwargs is not None:
                self.generate_experience_kwargs = dict(
                    config.method.gen_experience_kwargs,
                    eos_token_id=self.tokenizer.convert_tokens_to_ids(config.tokenizer.eos_token),
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            else:
                self.generate_experience_kwargs = None
        else:
            self.generate_kwargs = dict(
                config.method.gen_kwargs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            if config.method.gen_experience_kwargs is not None:
                self.generate_experience_kwargs = dict(
                    config.method.gen_experience_kwargs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            else:
                self.generate_experience_kwargs = None

        # Setup stats tracker
        self.running_moments = RunningMoments()
        self.ref_mean = self.config.method.ref_mean
        self.ref_std = self.config.method.ref_std

    def get_ref_model(self, config: TRLConfig):
        """Get the model"""
        model_class = AutoModelForCausalLMWithHydraValueHead
        if config.model.model_arch_type == "seq2seq":
            model_class = AutoModelForSeq2SeqLMWithHydraValueHead
        elif config.model.model_arch_type == "glm":
            model_class = AutoModelForGLMWithHydraValueHead

        from_fn = model_class.from_pretrained
        # backward-compat: Try to create a randomly initialized architecture from a config
        if issubclass(type(config.model.model_path), transformers.PretrainedConfig):
            from_fn = model_class.from_config
        
        if isinstance(self.config.model.peft_config, str):
            peft_config = eval(self.config.model.peft_config)
        else:
            peft_config = self.config.model.peft_config

        return from_fn(
            config.model.model_path,
            num_layers_unfrozen=config.model.num_layers_unfrozen,
            peft_config=peft_config,
        )

    def get_arch(self, config: TRLConfig):
        """Get the model"""
        model_class = AutoModelForCausalLMWithHydraValueHead
        if config.model.model_arch_type == "seq2seq":
            model_class = AutoModelForSeq2SeqLMWithHydraValueHead
        elif config.model.model_arch_type == "glm":
            model_class = AutoModelForGLMSeparate

        from_fn = model_class.from_pretrained
        # backward-compat: Try to create a randomly initialized architecture from a config
        if issubclass(type(config.model.model_path), transformers.PretrainedConfig):
            from_fn = model_class.from_config
        
        if isinstance(self.config.model.peft_config, str):
            peft_config = eval(self.config.model.peft_config)
        else:
            peft_config = self.config.model.peft_config

        return from_fn(
            config.model.model_path,
            pretrained_critic_model_path=config.model.critic_model_path,
            num_layers_unfrozen=config.model.num_layers_unfrozen,
            peft_config=peft_config,
        )

    def setup_model(self):
        """
        Returns a model derived from an instance's TRLConfig
        """
        logger.info(f"Initializing model: {self.config.model.model_path}")

        # Retrieves model equipped for ppo, ilql, etc
        model = self.get_arch(self.config)
        if not model.base_model.peft_type:
            if self.config.model.model_arch_type == "seq2seq":
                freeze_bottom_seq2seq_layers(
                    model.base_model, self.config.model.num_layers_unfrozen
                )
            else:
                # 目前对glm生效separate的功能，
                freeze_bottom_causal_layers(
                    model.base_model.base_model, self.config.model.num_layers_unfrozen
                )
        else:
            if self.accelerator.is_main_process and hasattr(
                model.base_model.base_model, "print_trainable_parameters"
            ):
                model.base_model.base_model.print_trainable_parameters()
            if self.config.model.num_layers_unfrozen >= 0:
                logger.warning(
                    "The argument num_layers_unfrozen for actor model is ignored when using peft, to prevent "
                    "unexpected behaviour. For Lora, use the `LoraConfig` argument `modules_to_save` instead."
                )
        # 目前对glm生效separate的功能，这里的critic需要freeze层数
        if self.config.model.model_arch_type == "glm":
            freeze_bottom_causal_layers(
                model.critic_model.base_model, self.config.model.num_layers_unfrozen
            )
        
        if HAS_OPENDELTA and self.accelerator.is_main_process:
            model_vis = Visualization(model)
            model_vis.structure_graph()
        
        return model

    def setup_optimizer(self):
        """
        Returns an optimizer derived from an instance's TRLConfig
        """
        optimizer_class = get_optimizer_class(self.config.optimizer.name)
        optimizer = optimizer_class(
            self.model.parameters(),
            **self.config.optimizer.kwargs,
        )

        if "bitsandbytes" in optimizer.__class__.__module__:
            # Force 32-bit `nn.Embedding` weights for stability. See discussion:
            # https://github.com/huggingface/transformers/issues/14819#issuecomment-1016017746
            from bitsandbytes.optim import GlobalOptimManager

            manager = GlobalOptimManager.get_instance()
            for module in self.model.modules():
                if isinstance(module, torch.nn.Embedding):
                    manager.register_module_override(
                        module, "weight", {"optim_bits": 32}
                    )

        return optimizer

    def setup_scheduler(self):
        """
        Returns a learning rate scheduler derived from an instance's TRLConfig
        """
        scheduler_class = get_scheduler_class(self.config.scheduler.name)
        scheduler = scheduler_class(self.opt, **self.config.scheduler.kwargs)
        return scheduler

    def decode(
        self,
        prompts: List[torch.LongTensor],
        samples: List[torch.LongTensor],
        prompt_sizes: torch.LongTensor = None,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Decode tensor generations into lists of strings (`samples`: List[str],
        `prompts`: List[str], `outputs`: List[str])
        """
        if prompt_sizes is None:
            # Assuming prompts were left-padded
            prompt_sizes = [prompts.shape[1]] * len(prompts)

        str_samples, str_prompts, str_outputs = [], [], []
        for prompt, sample, prompt_size in zip(prompts, samples, prompt_sizes):
            if self.config.model.model_arch_type == "seq2seq":
                output_start_ix = 0
            else:
                output_start_ix = prompt_size
            if self.config.model.model_arch_type == "glm":
                str_prompt = self.tokenizer.decode(
                    prompt[:prompt_size], skip_special_tokens=False
                )
                str_output = self.tokenizer.decode(
                    sample[output_start_ix:], skip_special_tokens=False
                )
            else:
                str_prompt = self.tokenizer.decode(
                    prompt[:prompt_size], skip_special_tokens=True
                )
                str_output = self.tokenizer.decode(
                    sample[output_start_ix:], skip_special_tokens=True
                )

            # Trim outputs up to `self.stop_sequences` if any are present
            if self.stop_sequences:
                for stop in self.stop_sequences:
                    stop_ix = str_output.find(stop)
                    if stop_ix >= 0:
                        str_output = str_output[:stop_ix].rstrip()

            str_prompts.append(str_prompt)
            str_outputs.append(str_output)

            if self.config.model.model_arch_type == "seq2seq":
                sample = str_prompt + self.tokenizer.sep_token + str_output
            else:
                sample = str_prompt + str_output

            str_samples.append(sample)

        return str_samples, str_prompts, str_outputs

    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Wraps hf's `generate` adding some specific method's defaults"""
        input_ids = input_ids.to(self.accelerator.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.accelerator.device)
        if self.generate_experience_kwargs is not None:
            kwargs = dict(self.generate_experience_kwargs, **kwargs)
        else:
            kwargs = dict(self.generate_kwargs, **kwargs)

        with torch.no_grad():
            return self.accelerator.unwrap_model(self.model).base_model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )

    def save_pretrained(self, directory: Optional[str] = None, **kwargs):
        """Save the underlying Hugging Face model, tokenizer, and configuration files to a directory for
        later use.

        Args:
            directory (str, *optional*): The directory to save the trainer files to.
                NOTE: If not specified, the model will be saved to a directory named `hf_model` in the
                checkpoint directory as specified by the Trainer's config.
            **kwargs: Additional keyword arguments passed to the underlying Hugging Face model's
                `save_pretrained` method.
        """
        if directory is None:
            directory = os.path.join(self.config.train.checkpoint_dir, "hf_model")
        self.accelerator.wait_for_everyone()
        self.accelerator.unwrap_model(self.model).base_model.save_pretrained(
            directory, is_main_process=self.accelerator.is_main_process, **kwargs
        )
        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(directory)
            exp_config_path = os.path.join(directory, "exp_config.json")
            with open(exp_config_path, "w", encoding="utf8") as fout:
                json.dump(self.config.to_dict(), fout, indent=4)
            
            if isinstance(self.config.model.peft_config, str):
                peft_config = eval(self.config.model.peft_config)
            hyper_parameters = {
                "eos_token": self.config.tokenizer.eos_token,
                "max_length": self.max_length,
                "peft_type": peft_config["peft_type"]
                if self.config.model.peft_config
                else None,
            }
            hyper_parameter_path = os.path.join(directory, "hyper_parameters.json")
            with open(hyper_parameter_path, "w", encoding="utf8") as fout:
                json.dump(hyper_parameters, fout, indent=4)

    def save(self, directory: Optional[str] = None, **kwargs):
        """Creates a checkpoint of the optimizer, scheduler and model"""
        self.accelerator.save_state(
            directory or self.config.train.checkpoint_dir, **kwargs
        )

    def load(self, directory: Optional[str] = None, **kwargs):
        """Load checkpoint of optimizer, scheduler and a model"""
        if self.config.model.peft_config is not None:

            def load_state_hook(models: List[torch.nn.Module], input_dir: str):
                with self.accelerator.main_process_first():
                    for model in models:
                        model.from_pretrained(input_dir)

            self.accelerator.register_load_state_pre_hook(load_state_hook)

            strict = False
        else:
            strict = True

        self.accelerator.load_state(
            directory or self.config.train.checkpoint_dir, strict=strict, **kwargs
        )

    def add_eval_pipeline(self, eval_pipeline):
        """Adds pipeline from with validation prompts"""
        self.eval_pipeline = eval_pipeline

    def evaluate(self):  # noqa: C901
        """Samples model on `eval_prompts`, logs stats with `reward_fn` or `metric_fn` if provided"""
        logger.info("Evaluating model")

        # Do multiple evaluations over a single list in `gen_kwargs` if present
        if self.generate_sweep_kwarg is not None:
            gen_sweep_arg, gen_sweep_values = self.generate_sweep_kwarg
        else:
            gen_sweep_values = [None]

        desc = [
            f"generation sweep 0/{len(gen_sweep_values)}",
            f"eval batch 0/{len(self.eval_dataloader)}",
        ]
        tbar = logging.tqdm(
            total=len(self.eval_dataloader) * len(gen_sweep_values),
            desc=f"[{' | '.join(desc)}]",
            disable=not self.accelerator.is_main_process,
            position=0,
            leave=True,
        )

        stats = {}
        table = []

        for i_sweep, gen_sweep_value in enumerate(gen_sweep_values):
            # A dedicated suffix for wandb logging
            if gen_sweep_value is not None:
                sweep_suffix = f"@{gen_sweep_arg}={gen_sweep_value}"
            else:
                sweep_suffix = ""

            all_samples = []
            all_prompts = []
            all_prompt_sizes = []
            all_idx = []
            all_scores = []

            generate_time = time()
            for i_prompt, prompts in enumerate(self.eval_dataloader):
                idx = prompts.pop("idx", None)
                if self.generate_sweep_kwarg:
                    samples = self.generate_eval(
                        **prompts, **{gen_sweep_arg: gen_sweep_value}
                    )
                else:
                    samples = self.generate_eval(**prompts)

                # TODO(reciprocated): this should be moved into `decode`
                # but that needs to be synced with indexing in `make_experience`
                if self.config.model.model_arch_type == "seq2seq":
                    samples = samples[:, 1:].contiguous()

                prompt_sizes = torch.tensor(prompts.input_ids.shape[1]).repeat(
                    len(prompts.input_ids)
                )

                str_samples, str_prompts, str_outputs = self.decode(
                    prompts.input_ids, samples, prompt_sizes
                )

                rewards = torch.tensor(
                    self.reward_fn(
                        samples=str_samples,
                        prompts=str_prompts,
                        outputs=str_outputs,
                        prompts_idx=idx,
                        is_train=False,
                    ),
                    dtype=float,
                )

                rewards = self.accelerator.gather_for_metrics(rewards)
                prompts, samples, prompt_sizes = self.accelerator.gather_for_metrics(
                    self.accelerator.pad_across_processes(
                        [prompts.input_ids, samples, prompt_sizes.to(samples.device)],
                        dim=1,
                        pad_index=self.tokenizer.pad_token_id,
                    )
                )
                idx = self.accelerator.gather_for_metrics(idx)
                all_idx.extend(idx.tolist())

                all_samples.extend(samples.tolist())
                all_prompts.extend(prompts.tolist())
                all_prompt_sizes.extend(prompt_sizes.tolist())
                all_scores.extend(rewards.tolist())

                desc = [
                    f"generation sweep {i_sweep + 1}/{len(gen_sweep_values)}",
                    f"eval batch {i_prompt + 1}/{len(self.eval_dataloader)}",
                ]
                tbar.set_description(f"[{' | '.join(desc)}]")
                tbar.update()
            tbar.close()

            stats["time/generate"] = time() - generate_time

            if self.accelerator.is_main_process:
                str_samples, str_prompts, str_outputs = self.decode(
                    all_prompts, all_samples, all_prompt_sizes
                )

                columns = ["prompt", "output"]
                columns_data = [str_prompts, str_outputs]

                # in online setting, compute the reward for validation
                logger.info("Computing rewards")
                all_scores = torch.tensor(all_scores)
                mean_reward = all_scores.mean().item()
                columns.append("reward")
                if not isinstance(rewards, list):
                    all_scores = all_scores.tolist()
                columns_data.append(all_scores)
                stats[f"reward/mean{sweep_suffix}"] = mean_reward

                # Prepend the sweep argument along with samples
                if self.generate_sweep_kwarg:
                    columns.insert(0, gen_sweep_arg)
                    columns_data.insert(0, [gen_sweep_value] * len(samples))

                table.append(list(zip(*columns_data)))

        # Log and display evaluation metrics
        logger.info("Summarizing evaluation")
        if self.accelerator.is_main_process:
            rows = sum(list(map(list, zip(*table))), [])

            # Add metrics/rewards to the table's title
            table_title = f"Evaluation #{self.nth_evaluation}"
            for k, x in stats.items():
                if k.startswith("reward") or k.startswith("metrics"):
                    table_title += f" {k}: {significant(x)}"

            rich_table = Table(*columns, title=table_title, show_lines=True)
            for ix in range(max(min(8, len(rows)), len(gen_sweep_values))):
                rich_table.add_row(*[str(significant(x)) for x in rows[ix]])
            Console().print(rich_table)

            if not ray.is_initialized():
                if self.config.train.tracker == "wandb":
                    import wandb

                    stats["samples"] = wandb.Table(columns, rows)

        self.nth_evaluation += 1
        return stats

    def learn(self):  # noqa: C901
        """
        Samples batches from `self.store`, updates model and periodically evaluates it on `self.eval_dataloader`
        """
        logger.info("Starting training")

        self.generate_sweep_kwarg = None
        for k, v in self.config.method.gen_kwargs.items():
            if isinstance(v, list):
                if self.generate_sweep_kwarg is not None:
                    logger.info(
                        "Only a single sweep is allowed, {k} is going to be set to {v[0]}"
                    )
                    self.generate_kwargs[k] = v[0]
                else:
                    self.generate_sweep_kwarg = (k, v)

        self.prepare_learning()
        self.iter_count = 0
        self.nth_evaluation = 0

        if ray.is_initialized():
            checkpoint = session.get_checkpoint()
            if checkpoint:
                with checkpoint.as_directory() as dir:
                    self.accelerator.load_state(dir)

                    with open(os.path.join(dir, "state.json")) as f:
                        state = json.load(f)
                        self.iter_count = state["iter_count"]
        else:
            results = self.evaluate()
            self.accelerator.log(results, step=self.iter_count)

        tbar = logging.tqdm(
            initial=self.iter_count,
            total=self.total_steps,
            disable=not self.accelerator.is_local_main_process,
            position=0,
            leave=True,
        )

        best_reward = -float("inf")
        # kl_early_stop = False

        # For each epoch
        for _ in range(self.config.train.epochs):
            # For each batch
            for batch in self.train_dataloader:
                # For each update per batch
                for _ in range(self.n_updates_per_batch):
                    # Note that whereas standard policy gradient methods perform one
                    # gradient update per batch, PPO for example commonly performs
                    # multiple gradient updates on the same batch of data.
                    # https://arxiv.org/pdf/1707.06347.pdf
                    with self.accelerator.accumulate(self.model):
                        forward_time = time()
                        loss, stats = self.loss(batch)
                        forward_time = time() - forward_time
                        backward_time = time()
                        # approx_policy_kl = stats["policy/approx_kl"]
                        # if kl_early_stop or (self.config.method.kl_early_stop is not None and \
                        #     approx_policy_kl > self.config.method.kl_early_stop):
                        #     kl_early_stop = True
                        #     logger.info("kl early stop")
                        #     continue
                        self.accelerator.backward(loss)
                        backward_time = time() - backward_time

                        if self.config.train.max_norm is not None:
                            if self.accelerator.sync_gradients:
                                self.accelerator.clip_grad_norm_(
                                    self.model.parameters(),
                                    max_norm=self.config.train.max_norm,
                                )

                        self.opt.step()
                        self.opt.zero_grad()
                        self.scheduler.step()
                        self.iter_count += 1

                        if self.iter_count % self.config.train.checkpoint_interval == 0:
                            subfolder = f"checkpoint_{self.iter_count:0{len(str(self.total_steps))}d}"
                            directory = os.path.join(
                                self.config.train.checkpoint_dir, subfolder
                            )
                            # self.save(directory)
                            self.save_pretrained(directory, max_shard_size="30GB")

                        stats["time/forward"] = forward_time
                        stats["time/backward"] = backward_time
                        for group_number, lr in enumerate(self.scheduler.get_last_lr()):
                            stats[f"learning_rate_group_{group_number}"] = lr

                        if self.iter_count % self.config.train.eval_interval == 0:
                            results = self.evaluate()
                            stats.update(results)

                            # always save checkpoint with the greatest mean reward
                            if self.config.train.save_best:
                                if (
                                    stats.get("reward/mean", -float("inf"))
                                    > best_reward
                                ):
                                    best_reward = stats.get("reward/mean")
                                    do_save = True
                                # in case ILQL reports reward estimate as one of its metrics
                                elif (
                                    stats.get("metrics/reward", -float("inf"))
                                    > best_reward
                                ):
                                    best_reward = stats.get("metrics/reward")
                                    do_save = True
                                else:
                                    do_save = False
                                do_save = torch.tensor(
                                    do_save, device=self.accelerator.device
                                )
                                if torch.distributed.is_initialized():
                                    torch.distributed.all_reduce(
                                        do_save, torch.distributed.ReduceOp.MAX
                                    )
                                if do_save:
                                    best_path = f"{self.config.train.checkpoint_dir}/best_checkpoint"
                                    logger.info(
                                        f"Saving the best state so far into {best_path}"
                                    )
                                    # self.save(best_path)
                                    # self.save_pretrained(best_path, max_shard_size="30GB")

                            # Report the metrics to Ray Tune.
                            if ray.is_initialized():
                                self.save("state")
                                with open("state/state.json", "w") as f:
                                    json.dump(dict(iter_count=self.iter_count), f)
                                checkpoint = Checkpoint.from_directory("state")
                                session.report(
                                    filter_non_scalars(stats), checkpoint=checkpoint
                                )

                        if not ray.is_initialized():
                            self.accelerator.log(stats, step=self.iter_count)

                        desc = " | ".join(
                            f"{k}: {v:.2f}"
                            for k, v in stats.items()
                            if k.startswith("loss")
                        )
                        tbar.set_description(f"[{desc}]")
                        tbar.update()

                        if self.iter_count >= self.total_steps:
                            subfolder = f"checkpoint_{self.iter_count:0{len(str(self.total_steps))}d}"
                            directory = os.path.join(
                                self.config.train.checkpoint_dir, subfolder
                            )
                            # self.save(directory)
                            self.save_pretrained(directory, max_shard_size="30GB")
                            return self.evaluate()

                self.post_backward_callback()

            self.post_epoch_callback()
            # kl_early_stop = False
        tbar.close()

    def loss(self, batch: PPORLBatch):
        """Forward pass & loss

        Args:
            batch: Previous batch of episodes
        """
        # Move `batch` data to `accelerator` device
        query_tensors = batch.query_tensors.to(self.accelerator.device)
        response_tensors = batch.response_tensors.to(self.accelerator.device)
        old_logprobs = batch.logprobs.to(self.accelerator.device)
        old_values = batch.values.to(self.accelerator.device)
        old_rewards = batch.rewards.to(self.accelerator.device)
        response_length = old_rewards.shape[1]

        advantages, returns = self.config.method.get_advantages_and_returns(
            old_values, old_rewards, response_length, use_whitening=False
        )

        if self.config.model.model_arch_type == "seq2seq":
            input_ids = query_tensors
            decoder_input_ids = response_tensors
            attention_mask = (
                input_ids.ne(self.tokenizer.pad_token_id)
                .long()
                .to(self.accelerator.device)
            )
            decoder_attention_mask = (
                decoder_input_ids.ne(self.tokenizer.pad_token_id)
                .long()
                .to(self.accelerator.device)
            )
            decoder_attention_mask[:, 0] = 1

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )

            logits = outputs.logits
            values_pred = outputs.value
            logprobs = logprobs_of_labels(logits[:, :-1, :], decoder_input_ids[:, 1:])
            mask = (
                decoder_input_ids.ne(self.tokenizer.pad_token_id)
                .long()
                .to(self.accelerator.device)
            )
            start = 0
            end = start + response_length
            logprobs, values_pred, mask = (
                logprobs[:, start:end],
                values_pred[:, start:end],
                mask[:, start:end],
            )
        elif self.config.model.model_arch_type == "glm":
            tokens = torch.cat((query_tensors, response_tensors), dim=1)
            # glm的多了'<|startofpiece|>'特殊token，经过PPORolloutStorage后，query_tensor是左pad，response_tensor是右pad
            query_tensors_input_ids = query_tensors[:, :-1]
            query_tensors_attention_mask = (
                query_tensors_input_ids.not_equal(self.tokenizer.pad_token_id)
                .long()
                .to(tokens.device)
            )
            query_input = {
                "input_ids": query_tensors_input_ids,
                "attention_mask": query_tensors_attention_mask,
            }
            query_input = BatchEncoding(data=query_input)

            glm_inputs = self.tokenizer.build_inputs_for_generation(
                query_input,
                max_gen_length=self.config.method.gen_experience_kwargs[
                    "max_new_tokens"
                ]
                + 2,
            )

            # TODO 确认后续 modeling_glm 改动方式
            # 1d rope 可以直接相加 pos_id 和 block_pos_id
            if "1d" in self.config.model.rotary_type:
                glm_inputs.position_ids[:, 0, :] = glm_inputs.position_ids[:, 0, :] + glm_inputs.position_ids[:, 1, :]

            generation_attention_mask = glm_inputs.generation_attention_mask
            position_ids = glm_inputs.position_ids
            model_inputs = prepare_inputs_for_generation_glm(
                tokens,
                position_ids=position_ids,
                generation_attention_mask=generation_attention_mask,
            )
            glm_attention_mask = model_inputs["attention_mask"]
            position_ids = model_inputs["position_ids"]
            attention_mask = (
                tokens.not_equal(self.tokenizer.pad_token_id).long().to(tokens.device)
            )
            outputs = self.model(
                tokens,
                position_ids=position_ids,
                attention_mask=glm_attention_mask,
                return_dict=True,
            )
            logits = outputs.logits
            # values_pred = outputs.value
            critic_outputs = self.model.forward_critic(
                tokens,
                position_ids=position_ids,
                attention_mask=glm_attention_mask,
                return_dict=True,
            )
            values_pred = critic_outputs.value
            values_pred = values_pred[:, :-1]
            logprobs = logprobs_of_labels(logits[:, :-1, :], tokens[:, 1:])
            logits_pred = logits[:, :-1, :]

            start = query_tensors.shape[1] - 1
            end = start + response_length
            logprobs, values_pred, mask, logits_pred = (
                logprobs[:, start:end],
                values_pred[:, start:end],
                attention_mask[:, start:end],
                logits_pred[:, start:end, :],
            )
        else:
            tokens = torch.cat((query_tensors, response_tensors), dim=1)
            attention_mask = (
                tokens.not_equal(self.tokenizer.pad_token_id).long().to(tokens.device)
            )
            outputs = self.model(tokens, attention_mask, return_dict=True)
            logits = outputs.logits
            values_pred = outputs.value
            values_pred = values_pred[:, :-1]
            logprobs = logprobs_of_labels(logits[:, :-1, :], tokens[:, 1:])

            start = query_tensors.shape[1] - 1
            end = start + response_length
            logprobs, values_pred, mask = (
                logprobs[:, start:end],
                values_pred[:, start:end],
                attention_mask[:, start:end],
            )

        if hasattr(self.config.method, "ent_coef"):
            loss, stats = self.config.method.loss(
                logprobs=logprobs,
                values=values_pred,
                old_logprobs=old_logprobs,
                old_values=old_values,
                advantages=advantages,
                returns=returns,
                mask=mask,
                logits=logits_pred,
            )
        else:
            loss, stats = self.config.method.loss(
                logprobs=logprobs,
                values=values_pred,
                old_logprobs=old_logprobs,
                old_values=old_values,
                advantages=advantages,
                returns=returns,
                mask=mask,
            )

        return loss, stats

    def setup_rollout_logging(self, config):
        # Make rollout logging dir for this run and store config
        exists = os.path.exists(config.train.rollout_logging_dir)
        isdir = os.path.isdir(config.train.rollout_logging_dir)
        assert exists and isdir

        self.run_id = f"run-{uuid.uuid4()}"
        self.rollout_logging_dir = os.path.join(
            config.train.rollout_logging_dir, self.run_id
        )
        os.mkdir(self.rollout_logging_dir)

        with open(os.path.join(self.rollout_logging_dir, "config.json"), "w") as f:
            f.write(json.dumps(config.to_dict(), indent=2))

    def post_epoch_callback(self):
        """Post epoch callback

        Clears the store and creates `num_rollouts` new episodes.
        """
        if self.log_rollouts:
            self.store.export_history(location=self.rollout_logging_dir)
        self.store.clear_history()
        # Collect more rollouts for training
        self.make_experience(self.config.method.num_rollouts, self.iter_count)

    def post_backward_callback(self):
        self.kl_ctl.update(self.mean_kl.item(), n_steps=self.config.train.batch_size)

    def prepare_learning(self):
        eval_dataloader = self.eval_pipeline.create_loader(
            self.config.train.batch_size * 4
        )
        self.eval_dataloader = self.accelerator.prepare_data_loader(eval_dataloader)
        self.train_dataloader = self.store.create_loader(
            self.config.train.batch_size, shuffle=True
        )

        self.n_updates_per_batch = self.config.method.ppo_epochs
        self.total_steps = (
            self.config.train.epochs
            * self.n_updates_per_batch
            * len(self.train_dataloader)
        )
        self.total_steps = min(self.total_steps, self.config.train.total_steps)

    def add_prompt_pipeline(self, pipeline: PromptPipeline):
        """Add a prompt pipeline dataloader to a trainer instance for the `make_experience` stage"""
        prompt_dataloader = pipeline.create_loader(
            self.config.method.chunk_size, shuffle=True
        )
        self.prompt_dataloader = self.accelerator.prepare_data_loader(prompt_dataloader)
        self.prompt_iterator = iter(self.prompt_dataloader)

    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):  # noqa:
        """Make experiences

        Takes `chunk_size` number of prompts from `prompt_iterator`, samples
        from the model and then computes the KL against a reference model. Finally it
        then appends PPOElements to trainer's `store`.

        Args:
            num_rollouts: Number of rollouts to generate
            iter_count: Total number of updates run (i.e. number of updates run for all batches & epochs)
        """
        logger.info("Collecting rollouts")
        tbar = logging.tqdm(
            total=num_rollouts,
            disable=os.environ.get("RANK", 0) != "0",
            desc=f"[rollout 0 / {num_rollouts}]",
            # Lower progress bar by 1 if we're in WARNING mode or above to avoid hiding high priority progress
            # bars (e.g. loss progress in trainers)
            position=logging.get_verbosity() >= logging.WARNING,
            # Leave progress bar if we're in INFO mode or lower to avoid spamming in suppressed verbosity levels
            leave=logging.get_verbosity() < logging.WARNING,
        )
        ppo_rl_elements = []
        stats = {}
        clock = Clock()

        # self.model.eval()
        while len(ppo_rl_elements) < num_rollouts:
            # Get next batch in prompt dataset and refresh if exhausted
            # TOOD (jon-tow): Make `prompt_dataloader` a cyclic/infinite DataLoader to not require manually
            # "refreshing" the contents of the `prompt_iterator`
            try:
                batch: PromptBatch = next(self.prompt_iterator)
            except StopIteration:
                self.prompt_iterator = iter(self.prompt_dataloader)
                batch = next(self.prompt_iterator)

            exp_generate_time = time()

            # Generate samples from the language model (similar to using HuggingFace `generate` method)
            if self.config.model.model_arch_type == "glm":
                model_input = {
                    "input_ids": batch.input_ids,
                    "attention_mask": batch.attention_mask,
                }
                model_input = BatchEncoding(data=model_input)
                glm_inputs = self.tokenizer.build_inputs_for_generation(
                    model_input,
                    max_gen_length=self.config.method.gen_experience_kwargs[
                        "max_new_tokens"
                    ]
                    + 2,
                )

                # TODO 确认后续 modeling_glm 改动方式
                # 1d rope 可以直接相加 pos_id 和 block_pos_id
                # token: a  b  c  mask  pad  pad  pad  sop      
                # pos  : 0  1  2   3     4    5    6    4   5  6  ...
                if "1d" in self.config.model.rotary_type:
                    glm_inputs.position_ids[:, 0, :] = glm_inputs.position_ids[:, 0, :] + \
                        glm_inputs.position_ids[:, 1, :]
                
                samples = self.generate(**glm_inputs)
                # glm模型调用build_inputs_for_generation后会增加'<|startofpiece|>'特殊token
                batch.input_ids = glm_inputs.input_ids
            else:
                samples = self.generate(**batch)
            prompt_length = batch.input_ids.shape[1]
            generate_length = samples.shape[1] - batch.input_ids.shape[1]
            sequence_length = samples.shape[1]

            stats["length/prompt_length"] = prompt_length
            stats["length/generate_length"] = generate_length
            stats["length/sequence_length"] = sequence_length
            stats["time/exp_generate"] = time() - exp_generate_time

            prompt_tensors = batch.input_ids
            device = samples.device

            prompt_sizes = torch.tensor(
                [prompt_tensors.shape[1]] * len(prompt_tensors), device=device
            )
            # padded_samples = self.accelerator.pad_across_processes(
            #     samples, dim=1, pad_index=self.tokenizer.eos_token_id, pad_first=False
            # )
            # padded_prompts = self.accelerator.pad_across_processes(
            #     prompt_tensors,
            #     dim=1,
            #     pad_index=self.tokenizer.eos_token_id,
            #     pad_first=False,
            # )
            # gathered_samples = self.accelerator.gather(padded_samples)
            # gathered_prompts = self.accelerator.gather(padded_prompts)
            # gathered_prompt_sizes = self.accelerator.gather(prompt_sizes)
            # all_idx = self.accelerator.gather(batch.idx)

            gathered_samples = samples
            gathered_prompts = prompt_tensors
            gathered_prompt_sizes = prompt_sizes
            all_idx = batch.idx

            # if self.accelerator.is_main_process:
            all_str_samples, all_str_prompts, all_str_outputs = self.decode(
                gathered_prompts, gathered_samples, gathered_prompt_sizes
            )

            exp_score_time = time()
            all_scores = torch.tensor(
                self.reward_fn(
                    samples=all_str_samples,
                    prompts=all_str_prompts,
                    outputs=all_str_outputs,
                    prompts_idx=all_idx.tolist(),
                    is_train=True,
                ),
                dtype=torch.float,
                device=device,
            )
            stats["time/exp_score"] = time() - exp_score_time

            scores = all_scores
            # all_scores = list(
            #     all_scores.reshape(self.accelerator.num_processes, -1).unbind()
            # )
            # else:
            #     all_scores = None

            # if torch.distributed.is_initialized():
            #     scores = torch.empty(len(samples), device=device)
            #     torch.distributed.scatter(scores, all_scores)
            # else:
            #     scores = all_scores[0].clone().detach()

            str_samples, str_prompts, str_outputs = self.decode(prompt_tensors, samples)

            # Pad the sample outputs
            outputs = self.tokenizer(str_outputs).input_ids
            if self.config.model.model_arch_type == "seq2seq":
                # add <pad> to the start of the output
                for i in range(len(outputs)):
                    outputs[i] = [self.tokenizer.pad_token_id] + outputs[i]

            start_output = prompt_tensors.shape[1]
            sample_outputs = samples[..., start_output:]

            # store statistics of the initial rollout as reference
            if self.ref_mean is None:
                self.ref_mean, self.ref_std = scores.mean(), scores.std()
            all_scores_mean, all_scores_std = self.running_moments.update(scores)
            stats["exp_scores/mean"] = all_scores_mean.item()
            stats["exp_scores/std"] = all_scores_std.item()
            stats["exp_scores/running_mean"] = self.running_moments.mean.item()
            stats["exp_scores/running_std"] = self.running_moments.std.item()

            if self.config.method.scale_reward == "running":
                scores /= self.running_moments.std
            elif self.config.method.scale_reward == "ref":
                scores /= self.ref_std

            clip_reward = self.config.method.cliprange_reward
            if clip_reward:
                scores = torch.clip(scores, -clip_reward, clip_reward)

            # Precompute logprobs, values
            if self.config.model.model_arch_type == "seq2seq":
                attention_mask = batch.attention_mask.to(device)
                prompt_tensors = batch.input_ids.to(device)
                decoder_attention_mask = sample_outputs.not_equal(
                    self.tokenizer.pad_token_id
                )
                decoder_attention_mask[:, 0] = 1
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=prompt_tensors,
                        attention_mask=attention_mask,
                        decoder_input_ids=sample_outputs,
                        decoder_attention_mask=decoder_attention_mask,
                    )
                    logits = outputs.logits
                    values = outputs.value
                    if hasattr(self.model.base_model, "frozen_head") or self.model.base_model.peft_type:
                        ref_logits = self.model.forward_hydra(
                            input_ids=prompt_tensors,
                            attention_mask=attention_mask,
                            decoder_input_ids=sample_outputs,
                            decoder_attention_mask=decoder_attention_mask,
                            return_dict=True,
                        ).logits
                    else:
                        ref_logits = self.ref_model(
                            input_ids=prompt_tensors,
                            attention_mask=attention_mask,
                            decoder_input_ids=sample_outputs,
                            decoder_attention_mask=decoder_attention_mask,
                            return_dict=True,
                        ).logits
            elif self.config.model.model_arch_type == "glm":
                # all_tokens = torch.cat((prompt_tensors.to(device), sample_outputs), dim=1)
                all_tokens = samples
                generation_attention_mask = glm_inputs.generation_attention_mask
                position_ids = glm_inputs.position_ids
                model_inputs = prepare_inputs_for_generation_glm(
                    all_tokens,
                    position_ids=position_ids,
                    generation_attention_mask=generation_attention_mask,
                )
                glm_attention_mask = model_inputs["attention_mask"]
                position_ids = model_inputs["position_ids"]
                attention_mask = (
                    all_tokens.not_equal(self.tokenizer.pad_token_id).long().to(device)
                )
                with torch.no_grad():
                    logits, *_, _ = self.model(
                        all_tokens,
                        attention_mask=glm_attention_mask,
                        position_ids=position_ids,
                    )
                    *_, values = self.model.forward_critic(
                        all_tokens,
                        attention_mask=glm_attention_mask,
                        position_ids=position_ids,
                    )
                    # TODO(dahoas): When hydra model works need to also support generation on hydra head
                    if hasattr(self.model.base_model, "frozen_head") or self.model.base_model.peft_type:
                        ref_logits = self.model.forward_hydra(
                            all_tokens,
                            attention_mask=glm_attention_mask,
                            position_ids=position_ids,
                            return_dict=True,
                        ).logits
                    else:
                        ref_logits = self.ref_model(
                            all_tokens,
                            attention_mask=glm_attention_mask,
                            position_ids=position_ids,
                            return_dict=True,
                        ).logits
            else:
                all_tokens = torch.cat(
                    (prompt_tensors.to(device), sample_outputs), dim=1
                )
                attention_mask = (
                    all_tokens.not_equal(self.tokenizer.pad_token_id).long().to(device)
                )
                with torch.no_grad():
                    logits, *_, values = self.model(
                        all_tokens,
                        attention_mask=attention_mask,
                    )
                    # TODO(dahoas): When hydra model works need to also support generation on hydra head
                    if hasattr(self.model.base_model, "frozen_head") or self.model.base_model.peft_type:
                        ref_logits = self.model.forward_hydra(
                            all_tokens,
                            attention_mask=attention_mask,
                            return_dict=True,
                        ).logits
                    else:
                        ref_logits = self.ref_model(
                            all_tokens,
                            attention_mask=attention_mask,
                            return_dict=True,
                        ).logits

            # torch.exp 不支持 fp16 在 cpu上计算：https://github.com/pytorch/pytorch/issues/54774
            if ref_logits.dtype == torch.float16:
                ref_logits = ref_logits.float()

            if iter_count == 0 and not self.model.base_model.peft_type:
                # peft微调的话因为加了参数，两者不一致，跳过检查，直接从peft pretrained model加载除外
                assert torch.all(ref_logits == logits).cpu().item() is True

            if self.config.model.model_arch_type == "seq2seq":
                logprobs = logprobs_of_labels(logits[:, :-1, :], sample_outputs[:, 1:])
                ref_logprobs = logprobs_of_labels(
                    ref_logits[:, :-1, :], sample_outputs[:, 1:]
                )
            else:
                logprobs = logprobs_of_labels(logits[:, :-1, :], all_tokens[:, 1:])
                ref_logprobs = logprobs_of_labels(
                    ref_logits[:, :-1, :], all_tokens[:, 1:]
                )

            n_samples: int = samples.shape[0]
            logprobs = logprobs.detach().cpu()
            ref_logprobs = ref_logprobs.detach().cpu()
            prompt_tensors = prompt_tensors.cpu()
            sample_outputs = sample_outputs.cpu()
            values = values.detach().cpu()[:, :-1]

            # Estimate the KL divergence between the model and reference model
            if self.config.model.model_arch_type == "seq2seq":
                attention_mask = sample_outputs != self.tokenizer.pad_token_id
                start = 0
            else:
                start = prompt_tensors.shape[1] - 1

            # glm中[start: end]位置包括 [sop] + response + [eop]
            ends = start + attention_mask[:, start:].sum(1)

            # Get the logprobs and values, for tokens that are not padding
            # or beginning of sequences tokens. These are from the model (not the reference model)
            all_values = [values[ix, start : ends[ix]] for ix in range(n_samples)]
            all_logprobs = [logprobs[ix, start : ends[ix]] for ix in range(n_samples)]

            log_ratio = (logprobs - ref_logprobs) * attention_mask[:, :-1].cpu()
            self.mean_kl = (log_ratio.exp() - 1 - log_ratio).mean().to(device)
            kl_penalty = self.kl_ctl.value * -log_ratio
            # 增加 log_ratio 和 kl_penalty的统计
            stats["policy/mean_log_ratio"] = log_ratio.mean().item()
            stats["policy/mean_kl_penalty"] = kl_penalty.mean().item()
            kl_penalty = [xs[start : ends[ix]] for ix, xs in enumerate(kl_penalty)]

            rollout_count = 0

            for sample_idx in range(n_samples):
                if (
                    len(kl_penalty[sample_idx]) == 0
                    or len(all_logprobs[sample_idx]) == 0
                ):
                    continue

                rewards = kl_penalty[sample_idx]
                rewards[-1] += scores[sample_idx].cpu()

                ppo_rl_elements.append(
                    PPORLElement(
                        query_tensor=prompt_tensors[sample_idx],
                        response_tensor=sample_outputs[sample_idx],
                        logprobs=all_logprobs[sample_idx],
                        values=all_values[sample_idx],
                        rewards=rewards,
                    )
                )

                rollout_count += 1
            exp_time = clock.tick()
            tbar.set_description(f"[rollout {len(ppo_rl_elements)} / {num_rollouts}]")
            tbar.update(min(rollout_count, num_rollouts))
        tbar.close()

        # self.model.train()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.mean_kl, torch.distributed.ReduceOp.AVG)

        stats["policy/sqrt_kl"] = torch.sqrt(self.mean_kl).item()
        stats["kl_ctl_value"] = self.kl_ctl.value
        stats["time/exp"] = exp_time

        if not ray.is_initialized():
            self.accelerator.log(stats, step=iter_count)

        # Push samples and rewards to trainer's rollout storage
        self.push_to_store(ppo_rl_elements)

    def generate_eval(self, input_ids, attention_mask=None, **kwargs):
        """Wraps hf's `generate` adding some specific method's defaults"""
        input_ids = input_ids.to(self.accelerator.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.accelerator.device)

        if self.config.model.model_arch_type == "glm":
            model_input = {"input_ids": input_ids, "attention_mask": attention_mask}
            model_input = BatchEncoding(data=model_input)
            glm_inputs = self.tokenizer.build_inputs_for_generation(
                model_input,
                max_gen_length=self.config.method.gen_experience_kwargs[
                    "max_new_tokens"
                ]
                + 2,
            )

            # TODO 确认后续 modeling_glm 改动方式
            # 1d rope 可以直接相加 pos_id 和 block_pos_id
            # token: a  b  c  mask  pad  pad  pad  sop      
            # pos  : 0  1  2   3     4    5    6    4   5  6  ...
            if "1d" in self.config.model.rotary_type:
                glm_inputs.position_ids[:, 0, :] = glm_inputs.position_ids[:, 0, :] + glm_inputs.position_ids[:, 1, :]

        kwargs = dict(self.generate_kwargs, **kwargs)

        with torch.no_grad():
            if self.config.model.model_arch_type == "glm":
                return self.accelerator.unwrap_model(self.model).base_model.generate(
                    **glm_inputs, **kwargs
                )
            else:
                return self.accelerator.unwrap_model(self.model).generate(
                    input_ids=input_ids, attention_mask=attention_mask, **kwargs
                )
