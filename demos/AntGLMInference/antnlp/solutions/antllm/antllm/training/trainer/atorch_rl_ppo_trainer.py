import atorch
import os
import math
import torch
import torch.distributed as dist
from trlx.utils.modeling import RunningMoments, logprobs_of_labels
from atorch.common.log_utils import DashBoardWriter, Timer, TimeStats
from transformers.tokenization_utils_base import BatchEncoding
from atorch.distributed.distributed import ParallelGroupContextManager
from solutions.antllm.antllm.utils.generation_utils import (
    prepare_inputs_for_generation_glm,
)

from atorch.data.unshuffled_batch_dataloader import DistributedUnshuffledBatchSampler
import json

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from atorch.rl.model_engine.model_engine import ModelEngineState
from rich.console import Console
from rich.table import Table
from solutions.antllm.antllm.models.glm.modeling_glm_ppo import PPOMixConfig

from solutions.antllm.antllm.models.glm.modeling_ppo import (
    FixedKLController,
)
from trlx.data.ppo_types import PPORLElement
from solutions.antllm.antllm.data.dataset.rl_dataset.glm_ppo_pipeline import (
    GLMPPORolloutStorage,
)
from typing import Optional
import trlx.utils.logging as logging

logger = logging.get_logger(__name__)


class AtorchRLTrainer:
    def __init__(self, model_engine, dataset, config, reward_fn=None):
        self.config = config
        self.dataset = dataset
        self.train_dataset_length = len(self.dataset["train"])
        self.evaluation_dataset_length = len(self.dataset["eval"])
        self.train_epoch = config.train.epoch
        self.num_rollouts = config.train.num_rollouts
        self.n_updates_per_batch = config.ppo_config.ppo_epoch
        self.model_engine = model_engine
        self.create_replay_buffer()
        self.rl_dataloader = None
        self.initial_start = True
        self.stop_sequences = False
        self.reward_fn = reward_fn
        self.rank = atorch.rank()
        tensorboard_log_dir = os.path.join(config.train.logdir, str(self.rank))
        self.dashboard_writer = DashBoardWriter(logdir=tensorboard_log_dir)
        self.make_experience_n_iter = 0
        self.train_n_iter = 0
        self.tokenizer = model_engine.tokenizer
        self.running_moments = RunningMoments()
        self.ref_mean = self.config.ppo_config.ref_mean
        self.ref_std = self.config.ppo_config.ref_std
        self.kl_ctl = FixedKLController(self.config.ppo_config.init_kl_coef)
        self.device = atorch.local_rank()

        self.trlx_ppo_config = PPOMixConfig(
            name="PPOSeparateConfig",
            ppo_epochs=config.ppo_config.ppo_epoch,
            num_rollouts=config.train.num_rollouts,
            chunk_size=config.generation.batch_size,
            init_kl_coef=config.ppo_config.init_kl_coef,
            target=None,
            horizon=config.ppo_config.horizon,
            gamma=config.ppo_config.gamma,
            lam=config.ppo_config.lam,
            cliprange=config.ppo_config.cliprange,
            cliprange_value=config.ppo_config.cliprange_value,
            gen_experience_kwargs=config.generation.gen_experience_kwargs,
            gen_kwargs=config.generation.gen_kwargs,
            ent_coef=config.ppo_config.ent_coef,
            vf_coef=config.ppo_config.vf_coef,
            ref_mean=config.ppo_config.ref_mean,
            ref_std=config.ppo_config.ref_std,
            cliprange_reward=config.ppo_config.cliprange_reward,
            scale_reward=config.ppo_config.scale_reward,
            kl_early_stop=False,
            clip_ratio=config.ppo_config.clip_ratio,
        )

        self.generate_kwargs = dict(
            config.generation.gen_kwargs,
            eos_token_id=self.tokenizer.eop_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if config.generation.gen_experience_kwargs is not None:
            self.generate_experience_kwargs = dict(
                config.generation.gen_experience_kwargs,
                eos_token_id=self.tokenizer.eop_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        else:
            self.generate_experience_kwargs = None
        self.running_moments = RunningMoments()

    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):
        # 抽象出业务逻辑
        # from solutions.xxx.xxx import set_gen_experience_kwargs
        self.gen_experience_kwargs = dict(
            self.config.generation.gen_experience_kwargs,
            eos_token_id=self.tokenizer.eop_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        self.model_engine.eval()

        self._make_experience_sequentially(num_rollouts, iter_count)

    def _make_experience_by_phase(self):
        # 65B方案
        # 先generate 本次训练所有prompts的response 主要原因是模型转换存在成本
        # 再分别计算logits/ref_logits/values
        pass

    def create_replay_buffer(self):
        # todo 复用 GLMElement？
        self.replay_buffer = GLMPPORolloutStorage(
            self.model_engine.tokenizer.pad_token_id,
            self.model_engine.tokenizer.sop_token_id,
        )
        self.replay_buffer.clear_history()

    # 抽象出业务逻辑
    # from solutions.antllm.xxxx.xxx import decode
    def decode(
        self,
        prompts,
        samples,
        prompt_sizes=None,
    ):
        """
        Decode tensor generations into lists of strings (`samples`: List[str],
        `prompts`: List[str], `outputs`: List[str])
        """
        if prompt_sizes is None:
            # Assuming prompts were left-padded
            prompt_sizes = [prompts.shape[1]] * len(prompts)

        str_samples, str_prompts, str_outputs = [], [], []
        for prompt, sample, prompt_size in zip(prompts, samples, prompt_sizes):
            output_start_ix = prompt_size
            str_prompt = self.tokenizer.decode(
                prompt[:prompt_size], skip_special_tokens=False
            )
            str_output = self.tokenizer.decode(
                sample[output_start_ix:], skip_special_tokens=False
            )
            # Trim outputs up to `self.stop_sequences` if any are present
            if self.stop_sequences:
                for stop in self.stop_sequences:
                    stop_ix = str_output.find(stop)
                    if stop_ix >= 0:
                        str_output = str_output[:stop_ix].rstrip()

            str_prompts.append(str_prompt)
            str_outputs.append(str_output)

            sample = str_prompt + str_output

            str_samples.append(sample)

        return str_samples, str_prompts, str_outputs

    def _make_experience_sequentially(
        self, num_rollouts: int = 1024, iter_count: int = 0
    ):
        # 10B 方案
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

        stats = {}
        experience_state = TimeStats("make_experience")
        ppo_rl_elements = []

        while len(ppo_rl_elements) < num_rollouts:
            try:
                batch = next(self.prompt_iterator)
            except StopIteration:
                self.prompt_iterator = iter(self.train_prompt_dataloader)
                batch = next(self.prompt_iterator)

            # print(
            #     "rank {} consumes {} for making experience".format(
            #         atorch.rank(), batch.idx
            #     )
            # )

            with Timer("time/exp_generate", experience_state):
                with ParallelGroupContextManager("actor_critic_ref_inference"):
                    model = self.model_engine.actor_critic_ref.module.base_model
                    model.eval()
                    model_input = {
                        "input_ids": batch.input_ids.to(self.device),
                        "attention_mask": batch.attention_mask.to(self.device),
                    }
                    model_input = BatchEncoding(data=model_input)
                    glm_inputs = self.tokenizer.build_inputs_for_generation(
                        model_input,
                        max_gen_length=self.config.generation.gen_experience_kwargs[
                            "max_new_tokens"
                        ]
                        + 2,
                    )

                    # 65B
                    # samples = model.generate(**glm_inputs,**self.gen_experience_kwargs)
                    # lm_data = torch.clone(lm_head.weight.data)
                    # 10B
                    with torch.no_grad():
                        samples = model.generate(
                            **glm_inputs, **self.gen_experience_kwargs
                        )
                    # glm模型调用build_inputs_for_generation后会增加'<|startofpiece|>'特殊token
                    batch.input_ids = glm_inputs.input_ids

                    prompt_tensors = batch.input_ids
                    device = samples.device

                    prompts = prompt_tensors
                    prompt_sizes = torch.tensor(
                        [prompt_tensors.shape[1]] * len(prompt_tensors), device=device
                    )

                    str_samples, str_prompts, str_outputs = self.decode(
                        prompts, samples, prompt_sizes
                    )

                    start_output = prompt_tensors.shape[1]
                    sample_outputs = samples[..., start_output:]

            idx = batch.idx
            scores = torch.tensor(
                self.reward_fn(
                    reward_model=self.model_engine.get_model(
                        model_type="reward_model", mode="inference"
                    ),
                    cost_model=self.model_engine.get_model(
                        model_type="cost_model", mode="inference"
                    ),
                    samples=str_samples,
                    prompts=str_prompts,
                    outputs=str_outputs,
                    prompts_idx=idx.tolist(),
                    is_train=True,
                ),
                dtype=torch.float,
                device=device,
            )

            if self.ref_mean is None:
                self.ref_mean, self.ref_std = scores.mean(), scores.std()
            all_scores_mean, all_scores_std = self.running_moments.update(scores)
            stats["exp_scores/mean"] = all_scores_mean.item()
            stats["exp_scores/std"] = all_scores_std.item()
            stats["exp_scores/running_mean"] = self.running_moments.mean.item()
            stats["exp_scores/running_std"] = self.running_moments.std.item()

            if self.trlx_ppo_config.scale_reward == "running":
                scores /= self.running_moments.std
            elif self.trlx_ppo_config.scale_reward == "ref":
                scores /= self.ref_std

            clip_reward = self.trlx_ppo_config.cliprange_reward
            if clip_reward:
                scores = torch.clip(scores, -clip_reward, clip_reward)

            # Precompute logprobs, values
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

            # 10B
            # model = self.model_engine.get_model(model_type="actor_critic_ref", mode="inference")
            model = self.model_engine.actor_critic_ref

            with torch.no_grad():
                logits, *_, _ = model(
                    all_tokens,
                    attention_mask=glm_attention_mask,
                    position_ids=position_ids,
                )

                *_, values = model.forward_critic(
                    all_tokens,
                    attention_mask=glm_attention_mask,
                    position_ids=position_ids,
                )

                # TODO(dahoas): When hydra model works need to also support generation on hydra head
                if (
                    hasattr(model.base_model, "frozen_head")
                    or model.base_model.peft_type
                ):
                    ref_logits = model.forward_hydra(
                        all_tokens,
                        attention_mask=glm_attention_mask,
                        position_ids=position_ids,
                        return_dict=True,
                    ).logits
                else:
                    # TODO 这里无ref model
                    ref_logits = self.ref_model(
                        all_tokens,
                        attention_mask=glm_attention_mask,
                        position_ids=position_ids,
                        return_dict=True,
                    ).logits

            # if ref_logits.dtype == torch.float16:
            #     ref_logits = ref_logits.float()

            # torch版本在bf16下存在差异，2.1.0环境下有些问题，这里检查先跳过
            # if iter_count == 0 and not model.base_model.peft_type:
            #     # peft微调的话因为加了参数，两者不一致，跳过检查，直接从peft pretrained model加载除外
            #     assert torch.all(ref_logits == logits).cpu().item() is True

            logprobs = logprobs_of_labels(logits[:, :-1, :], all_tokens[:, 1:])
            ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], all_tokens[:, 1:])

            n_samples: int = samples.shape[0]
            logprobs = logprobs.detach().cpu()
            ref_logprobs = ref_logprobs.detach().cpu()
            prompt_tensors = prompt_tensors.cpu()
            sample_outputs = sample_outputs.cpu()
            values = values.detach().cpu()[:, :-1]

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
            tbar.set_description(f"[rollout {len(ppo_rl_elements)} / {num_rollouts}]")
            tbar.update(min(rollout_count, num_rollouts))
        tbar.close()

        self.replay_buffer.push(ppo_rl_elements)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.mean_kl, torch.distributed.ReduceOp.AVG)

        stats["policy/sqrt_kl"] = torch.sqrt(self.mean_kl).item()
        stats["kl_ctl_value"] = self.kl_ctl.value
        self.dashboard_writer.add_scalars(stats, iter_count)
        experience_state.to_dashboard(self.dashboard_writer, iter_count)
        self.make_experience_n_iter += 1

    def generate_eval(self, input_ids, attention_mask=None, **kwargs):
        """Wraps hf's `generate` adding some specific method's defaults"""
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        model_input = {"input_ids": input_ids, "attention_mask": attention_mask}
        model_input = BatchEncoding(data=model_input)

        glm_inputs = self.tokenizer.build_inputs_for_generation(
            model_input,
            max_gen_length=self.config.generation.gen_experience_kwargs[
                "max_new_tokens"
            ]
            + 2,
        )

        kwargs = dict(self.generate_kwargs, **kwargs)
        model = self.model_engine.actor_critic_ref.module.base_model

        with torch.no_grad():
            samples = model.generate(**glm_inputs, **kwargs)
        prompt_sizes = torch.tensor(input_ids.shape[1]).repeat(len(input_ids))

        str_samples, str_prompts, str_outputs = self.decode(
            input_ids, samples, prompt_sizes
        )
        return str_samples, str_prompts, str_outputs

    def actor_critic_ref_has_inference_strategy(self):
        return False

    def evaluate(self):
        # Todo: add evaulation prompts and dataloader
        # Use table to show results in logs or console
        logger.info("Evaluating model")

        desc = [
            f"eval batch 0/{len(self.eval_prompt_dataloder)}",
        ]
        tbar = logging.tqdm(
            total=len(self.eval_prompt_dataloder),
            desc=f"[{' | '.join(desc)}]",
            disable=os.environ.get("RANK", 0) != "0",
            position=0,
            leave=True,
        )

        scores = []
        all_idx = []
        all_str_samples = []
        all_str_prompts = []
        all_str_outputs = []

        for i_prompt, prompts in enumerate(self.eval_prompt_dataloder):
            idx = prompts.pop("idx", None)
            # print("rank {} consumes {} data for evaluation".format(self.rank, idx))
            str_samples, str_prompts, str_outputs = self.generate_eval(
                prompts["input_ids"], attention_mask=prompts["attention_mask"]
            )
            all_str_samples.extend(str_samples)
            all_idx.extend(idx.tolist())
            all_str_prompts.extend(str_prompts)
            all_str_outputs.extend(str_outputs)
            desc = [
                f"eval batch {i_prompt + 1}/{len(self.eval_prompt_dataloder)}",
            ]
            tbar.set_description(f"[{' | '.join(desc)}]")
            tbar.update()
        tbar.close()

        scores = self.reward_fn(
            all_str_samples,
            reward_model=self.model_engine.get_model(
                model_type="reward_model", mode="inference"
            ),
            cost_model=self.model_engine.get_model(
                model_type="cost_model", mode="inference"
            ),
            is_train=False,
            idx=all_idx,
            batch_size=2,  # hard code
        )

        batch_size = len(idx)
        padding_length = (
            len(self.eval_prompt_dataloder.dataset) - self.evaluation_dataset_length
        )
        padding_rank = math.ceil(padding_length / batch_size)
        remove_size = (
            padding_length - math.floor(padding_length / batch_size) * batch_size
        )

        if self.rank >= atorch.world_size() - padding_rank:
            if self.rank == atorch.world_size() - padding_rank:
                scores = scores[:-remove_size]
                # print(
                #     "valid sample idx for rank {} are {}".format(
                #         self.rank, all_idx[:-remove_size]
                #     )
                # )
            else:
                scores = scores[:-batch_size]
                # print(
                #     "valid sample idx for rank {} are {}".format(
                #         self.rank, all_idx[:-batch_size]
                #     )
                # )
        else:
            # print("valid sample idx for rank {} are {}".format(self.rank, all_idx))
            pass

        effective_sum_reward = scores.sum()
        dist.all_reduce(effective_sum_reward, op=dist.ReduceOp.SUM)
        mean_reward = effective_sum_reward * 1.0 / self.evaluation_dataset_length

        table_title = (
            f"Evaluation #{self.nth_evaluation}" + f"reward/mean: {mean_reward}"
        )
        columns = ["prompt", "output", "scores"]
        csv_rows = []
        rich_table = Table(*columns, title=table_title, show_lines=True)

        # write all evaluation details to csv file
        num_show = 0
        for i, j, k in zip(all_str_prompts, all_str_outputs, scores):
            csv_rows.append([i, j, str(k.item())])
            if num_show >= 8:
                continue
            # Add metrics/rewards to the table's title
            # 每个样本是一个row 包括prompts，outputs and scores
            rich_table.add_row(i, j, str(k.item()))
            num_show += 1
        if atorch.local_rank() == 0:
            try:
                Console().print(rich_table)
            except Exception as e:
                logger.warning(e)

        # filename = os.path.join(
        #     self.config.train.logdir,
        #     "eval_rewards",
        #     str(self.rank),
        #     str(self.train_n_iter) + "_eval_records.csv",
        # )
        # if not os.path.exists(os.path.dirname(filename)):
        #     os.makedirs(os.path.dirname(filename), exist_ok=False)

        # with open(filename, "w") as csvfile:
        #     # creating a csv writer object
        #     csvwriter = csv.writer(csvfile)
        #     # writing the fields
        #     csvwriter.writerow(columns)
        #     # writing the data rows
        #     csvwriter.writerows(csv_rows)

        self.nth_evaluation += 1

        return mean_reward

    def loss(self, batch):
        """
        do forward pass
        """
        # 继承即可以通过.get方法获取，也可以通过.获取
        query_tensors = batch.query_tensors.to(self.device)
        response_tensors = batch.response_tensors.to(self.device)
        old_logprobs = batch.logprobs.to(self.device)
        old_values = batch.values.to(self.device)
        old_rewards = batch.rewards.to(self.device)
        response_length = old_rewards.shape[1]

        advantages, returns = self.trlx_ppo_config.get_advantages_and_returns(
            old_values, old_rewards, response_length, use_whitening=False
        )

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
            query_input,
            max_gen_length=self.config.generation.gen_experience_kwargs[
                "max_new_tokens"
            ]
            + 2,
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
        # model.train()
        self.model_engine.actor_critic_ref.eval()
        # diff model 获取logits
        outputs = self.model_engine.actor_critic_ref(
            tokens,
            position_ids=position_ids,
            attention_mask=glm_attention_mask,
            return_dict=True,
        )
        logits = outputs.logits
        # values_pred = outputs.value
        critic_outputs = self.model_engine.actor_critic_ref.forward_critic(
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

        return self.trlx_ppo_config.loss(
            logprobs=logprobs,
            values=values_pred,
            old_logprobs=old_logprobs,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            mask=mask,
            logits=logits_pred,
        )

    def create_dataloader(self, dataset, state=None):
        # 需要兼容10B和65B两套方案
        """
        create dataloader for make experience and rl training
        Args:
            dataset: For making experience, all process reads the same csv file and
                    get prompts. Each process needs to use seperate prompts to generate
                    experience. For training, to be decided.
            state: for debug. Since we can switch model_engine.stat in pre_rl_training_hook
                    and pre_experience_generation_hook.
        Returns:
            dataloader: pytorch dataloader
        """

        if state is not None:
            self.stat = state
        if self.stat == ModelEngineState.ExperienceGenerationState:
            if self.actor_critic_ref_has_inference_strategy():
                with ParallelGroupContextManager(
                    "{}_inference".format("actor_critic_ref")
                ):
                    ddp_size = 1
                    rank = 0
                    pass
            else:
                with ParallelGroupContextManager("{}".format("actor_critic_ref")):
                    # for 10B
                    ddp_size = atorch.world_size()
                    rank = atorch.rank()
            # if actor train_strategy is different from it's inference strategy, we use it's inference strategy
            # Todo: dataset sampler should take hybrid tensor/model/pipeline parallilism in to consideration
            # for debug
            # dataset_sampler = DistributedUnshuffledBatchSampler(
            #     dataset, num_replicas=ddp_size, batch_size=4, rank=rank
            # )
            dataset_sampler = DistributedSampler(dataset, shuffle=True, num_replicas=ddp_size, rank=rank)
            dataloader = DataLoader(
                dataset,
                sampler=dataset_sampler,
                collate_fn=dataset.collate_fn(),
                batch_size=self.config.generation.batch_size,
                pin_memory=True,
                shuffle=False,
                drop_last=True,
            )

        elif self.stat == ModelEngineState.EvaluationState:
            # 为了并行评估，需要将数据不齐
            # 例如80卡，总共200条数据
            # 每个卡batch=2
            ddp_size = atorch.world_size()
            rank = atorch.rank()
            # make sure that every rank has evaluation data
            # and drop padding dataset
            dataset_length = len(dataset)
            i = self.config.generation.batch_size * 4
            while i * ddp_size < dataset_length:
                i = i + self.config.generation.batch_size * 4
            padding_length = i * ddp_size - dataset_length

            dataset.prompts = dataset.prompts + [dataset.prompts[0]] * padding_length

            dataset_sampler = DistributedUnshuffledBatchSampler(
                dataset,
                num_replicas=ddp_size,
                rank=rank,
                batch_size=self.config.generation.batch_size * 4,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                collate_fn=dataset.collate_fn(),
                sampler=dataset_sampler,
                batch_size=self.config.generation.batch_size * 4,
                pin_memory=True,
                drop_last=True,
            )

        elif self.stat == ModelEngineState.RLTrainingState:
            dataloader = DataLoader(
                dataset,
                shuffle=True,
                collate_fn=dataset.collate_fn(),
                batch_size=self.config.train.batch_size,
                pin_memory=True,
                drop_last=True,
            )
        return dataloader

    def get_train_dataset(self):
        return self.dataset["train"]

    def get_eval_dataset(self):
        return self.dataset["eval"]

    def prepare_training(self):
        self.model_engine.set_state(ModelEngineState.ExperienceGenerationState)
        self.train_prompt_dataloader = self.create_dataloader(
            self.get_train_dataset(), state=ModelEngineState.ExperienceGenerationState
        )
        self.prompt_iterator = iter(self.train_prompt_dataloader)

        self.eval_prompt_dataloder = self.create_dataloader(
            self.get_eval_dataset(), state=ModelEngineState.EvaluationState
        )
        self.total_steps = (
            self.config.train.epoch
            * self.n_updates_per_batch
            * len(self.train_prompt_dataloader)
        )

    def get_train_rl_dataset(self):
        self.train_dataloader = self.create_dataloader(
            self.replay_buffer, state=ModelEngineState.RLTrainingState
        )

    def post_backward_callback(self):
        self.kl_ctl.update(self.mean_kl.item(), n_steps=self.config.train.batch_size)

    def post_epoch_callback(self):
        """Post epoch callback
        Clears the store and creates `num_rollouts` new episodes.
        """
        self.replay_buffer.clear_history()
        logger.info("rank {} clears replay buffer".format(atorch.rank()))
        assert len(self.replay_buffer) == 0
        # Collect more rollouts for training
        self.make_experience(self.num_rollouts, self.train_n_iter)
        self.get_train_rl_dataset()
        torch.cuda.empty_cache()
 
    def learn(self):
        """
        train on ppo data
        params:
            train_epoch:
            rl_dataloader:
            n_updates_per_batch:
        """

        self.nth_evaluation = 0
        self.prepare_training()

        results = self.evaluate()
        self.dashboard_writer.add_scalars({"reward/mean": results}, self.train_n_iter)
        self.make_experience(self.num_rollouts, self.train_n_iter)
        self.get_train_rl_dataset()

        tbar = logging.tqdm(
            initial=self.train_n_iter,
            total=self.total_steps,
            disable=os.environ.get("RANK", 0) != "0",
            position=0,
            leave=True,
        )

        for _ in range(self.config.train.epoch):
            for data in self.train_dataloader:
                for _ in range(self.n_updates_per_batch):
                    train_states = TimeStats("train")
                    self.train_n_iter += 1
                    with Timer("train_step", train_states):
                        with ParallelGroupContextManager("actor_critic_ref"):
                            with Timer("time/forward"):
                                ppo_loss, stats = self.loss(data)
                                # rank = os.environ["LOCAL_RANK"]
                                # print(
                                #     "rank {} self.train_n_iter step {} ppo_loss is {}".format(
                                #         rank, self.train_n_iter - 1, ppo_loss
                                #     )
                                # )
                            with Timer("time/backward", train_states):
                                # clip gradients 和 gradient accumulation steps 是由deepspeed自己维护
                                self.model_engine.actor_critic_ref.backward(ppo_loss)
                                self.model_engine.actor_critic_ref.step()

                    self.dashboard_writer.add_scalars(stats, self.train_n_iter)
                    desc = " | ".join(
                        f"{k}: {v:.2f}"
                        for k, v in stats.items()
                        if k.startswith("loss")
                    )
                    tbar.set_description(f"[{desc}]")
                    tbar.update()
                    if (
                        self.train_n_iter
                        % self.config.train.gradient_accumulation_steps
                        == 0
                    ):
                        for i in range(atorch.world_size()):
                            self.model_engine.actor_critic_ref_scheduler.step()
                    learning_stats = {}
                    for group_number, lr in enumerate(
                        self.model_engine.actor_critic_ref_scheduler.get_last_lr()
                    ):
                        learning_stats[f"learning_rate_group_{group_number}"] = lr
                    self.dashboard_writer.add_scalars(learning_stats, self.train_n_iter)
                    # logger.info("train step cost {} s".format(t2 - t1))
                    train_states.to_dashboard(self.dashboard_writer, self.train_n_iter)

                    # pbar.set_description(f"[{desc}]")
                    # pbar.update()
                    with Timer("time/evaluting", train_states):
                        # to do add eval_interval
                        if self.train_n_iter % self.config.train.eval_interval == 0:
                            results = self.evaluate()
                            torch.cuda.empty_cache()
                            self.dashboard_writer.add_scalars(
                                {"reward/mean": results}, self.train_n_iter
                            )
                    with Timer("time/saving_checkpoint", train_states):
                        if (
                            self.train_n_iter % self.config.train.checkpoint_interval
                            == 0
                        ):
                            directory = os.path.join(
                                self.config.train.checkpoint_dir,
                                "ckpt" + str(self.train_n_iter),
                            )
                            self.save_pretrained(directory)
                            logger.info("skip save pretrained")

                    self.dashboard_writer.flush()
                self.post_backward_callback()
            self.post_epoch_callback()

    def save_pretrained(self, directory: Optional[str] = None, **kwargs):
        if self.rank == 0:
            self.model_engine.actor_critic_ref.base_model.base_model.save_pretrained(
                directory
            )
            if self.model_engine.actor_critic_ref.base_model.peft_config:
                # also save the base model itself
                state_dict = (
                    self.model_engine.actor_critic_ref.base_model.base_model.base_model.model.state_dict()
                )
                # Filter the peft params ...
                param_keys = list(state_dict.keys())
                peft_param_prefix = "base_model.model."
                for key in param_keys:
                    if "lora" in key:
                        state_dict.pop(key)
                    elif peft_param_prefix in key:
                        value = state_dict.pop(key)
                        new_key = key.replace(peft_param_prefix, "")
                        state_dict[new_key] = value
                torch.save(state_dict, os.path.join(directory, "pytorch_model.bin"))
            # save tokenizer
            self.model_engine.tokenizer.save_pretrained(directory)
            # save
            exp_config_path = os.path.join(directory, "exp_config.json")
            with open(exp_config_path, "w", encoding="utf8") as fout:
                json.dump(self.config.to_dict(), fout, indent=4)
            # write hyper_parameters.json
            peft_type = None
            # peft_config = self.config.model.actor_critic_ref.peft_config
            peft_config = None
            if peft_config is not None:
                peft_type = peft_config["peft_type"]
            hyper_parameters = {
                "max_length": self.config.train.seq_length,
                "peft_type": peft_type,
            }
            hyper_parameter_path = os.path.join(directory, "hyper_parameters.json")
            with open(hyper_parameter_path, "w", encoding="utf8") as fout:
                json.dump(hyper_parameters, fout, indent=4)
