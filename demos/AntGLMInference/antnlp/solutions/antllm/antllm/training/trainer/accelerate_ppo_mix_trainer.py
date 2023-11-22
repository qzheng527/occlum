import json
import os
import uuid
from time import time
from typing import Callable, List

import ray
import torch

# import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import trlx.utils.logging as logging
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.configs import TRLConfig
from trlx.data.ppo_types import PPORLBatch, PPORLElement
from solutions.antllm.antllm.models.glm.modeling_ppo import (
    AdaptiveKLController,
    AutoModelForCausalLMWithHydraValueHead,
    AutoModelForSeq2SeqLMWithHydraValueHead,
    FixedKLController,
)
from trlx.pipeline.offline_pipeline import PromptPipeline
from trlx.pipeline.ppo_pipeline import PPORolloutStorage
from trlx.trainer import register_trainer
from trlx.utils import Clock
from trlx.utils.modeling import RunningMoments, logprobs_of_labels
from transformers.tokenization_utils_base import BatchEncoding

# from ..utils.generation_utils import prepare_inputs_for_generation_glm
# from ..models.modeling_ppo import AutoModelForGLMWithHydraValueHead
# from ..pipeline.glm_ppo_pipeline import GLMPPORolloutStorage

from solutions.antllm.antllm.models.peft.modeling_peft import AntPeftForCausalLM # NOQA
from .accelerate_base_trainer import AccelerateRLTrainer
from solutions.antllm.antllm.models.glm.modeling_glm_ppo import (
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


logger = logging.get_logger(__name__)


@register_trainer
class AcceleratePPOMixTrainer(AccelerateRLTrainer):
    """PPO Accelerate Trainer"""

    reward_fn: Callable[[List[str], List[str], List[str]], List[float]]
    tokenizer: AutoTokenizer

    def __init__(self, config: TRLConfig, **kwargs):
        """PPO Accelerate Trainer initialization

        Args:
            config: Config
        """
        super().__init__(config, **kwargs)

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
        self.model, self.opt, self.scheduler, rollout_loader = self.accelerator.prepare(
            self.model, self.opt, self.scheduler, rollout_loader
        )

        self.store.clear_history()  # Clear the rollout store

        # Setup a reference model when hydra heads are not used
        # 如果是peft model是已加载adapter_config训练后的版本，则需要重新起个ref model
        if not hasattr(self.model, "frozen_head") and not self.model.peft_type:
            self.ref_model = self.get_arch(self.config)
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

    def get_arch(self, config: TRLConfig):
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
            old_values, old_rewards, response_length
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
            glm_input = self.tokenizer.build_inputs_for_generation(
                query_input, max_gen_length=self.config.train.seq_length
            )
            generation_attention_mask = glm_input.generation_attention_mask
            position_ids = glm_input.position_ids
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
            values_pred = outputs.value
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
                    model_input, max_gen_length=self.config.train.seq_length
                )
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

            #     all_scores = list(
            #         all_scores.reshape(self.accelerator.num_processes, -1).unbind()
            #     )
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
            # elif self.config.model.model_arch_type == "glm":
            #     # remove [CLS] & <|endoftext|> after glm tokenier
            #     for i in range(len(outputs)):
            #         outputs[i] = outputs[i][1:-1]

            # outputs = list(map(torch.LongTensor, outputs))
            # maxsize = max(map(len, outputs))
            # outputs = [
            #     F.pad(
            #         output,
            #         (0, maxsize - len(output)),
            #         value=self.tokenizer.pad_token_id,
            #     )
            #     for output in outputs
            # ]
            # sample_outputs = torch.vstack(outputs).to(device)
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
                    if hasattr(self.model, "frozen_head") or self.model.peft_type:
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
                    logits, *_, values = self.model(
                        all_tokens,
                        attention_mask=glm_attention_mask,
                        position_ids=position_ids,
                    )
                    # TODO(dahoas): When hydra model works need to also support generation on hydra head
                    if hasattr(self.model, "frozen_head") or self.model.peft_type:
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
                    if hasattr(self.model, "frozen_head") or self.model.peft_type:
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

            if iter_count == 0 and not self.model.peft_type:  
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

    def setup_model(self):
        """
        Returns a model derived from an instance's TRLConfig
        """
        logger.info(f"Initializing model: {self.config.model.model_path}")

        # Retrieves model equipped for ppo, ilql, etc
        model = self.get_arch(self.config)
        if not model.peft_type:
            if self.config.model.model_arch_type == "seq2seq":
                freeze_bottom_seq2seq_layers(
                    model.base_model, self.config.model.num_layers_unfrozen
                )
            # elif self.config.model.model_arch_type == "glm":
            #     freeze_bottom_glm_layers(model.base_model, self.config.model.num_layers_unfrozen)
            else:
                freeze_bottom_causal_layers(
                    model.base_model, self.config.model.num_layers_unfrozen
                )
        # Set the delta tuning strategies
        else:
            if self.accelerator.is_main_process and hasattr(
                model.base_model, "print_trainable_parameters"
            ):
                model.base_model.print_trainable_parameters()
            if self.config.model.num_layers_unfrozen >= 0:
                logger.warning(
                    "The argument num_layers_unfrozen is ignored when using peft, to prevent unexpected behaviour."
                    "For Lora, use the `LoraConfig` argument `modules_to_save` instead."
                )
        return model

    def generate_eval(self, input_ids, attention_mask=None, **kwargs):
        """Wraps hf's `generate` adding some specific method's defaults"""
        input_ids = input_ids.to(self.accelerator.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.accelerator.device)

        if self.config.model.model_arch_type == "glm":
            model_input = {"input_ids": input_ids, "attention_mask": attention_mask}
            model_input = BatchEncoding(data=model_input)
            glm_inputs = self.tokenizer.build_inputs_for_generation(
                model_input, max_gen_length=self.config.train.seq_length
            )

        kwargs = dict(self.generate_kwargs, **kwargs)

        with torch.no_grad():
            if self.config.model.model_arch_type == "glm":
                return self.accelerator.unwrap_model(self.model).generate(
                    **glm_inputs, **kwargs
                )
            else:
                return self.accelerator.unwrap_model(self.model).generate(
                    input_ids=input_ids, attention_mask=attention_mask, **kwargs
                )
