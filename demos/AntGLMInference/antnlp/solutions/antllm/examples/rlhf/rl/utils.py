from tqdm import tqdm
from solutions.antllm.antllm.models.peft.modeling_peft import AntPeftForCausalLM # NOQA
from solutions.antllm.antllm.models.glm.modeling_glm_rm import RewardModel
import torch
import json
import numpy as np
import torch.nn as nn
import deepspeed
import pathlib
from typing import List, Dict, Any
from solutions.antllm.antllm.models.glm.tokenization_glm import GLMTokenizer
from solutions.antllm.antllm.models.glm.tokenization_glm import GLMTokenizerMixin
from solutions.antllm.antllm.utils.modeling_glm_rm_utils import build_glm_inputs_from_sample

TEMPLATE_DIR = pathlib.Path(__file__).absolute().parent
EVAL_TEMPLATE_FILE = TEMPLATE_DIR / "configs/ds_eval_config_template.json"


def get_glm_prompt_dataset(
    prompts: str, max_length: int, tokenizer: GLMTokenizerMixin, mask_type: str
):
    """
    Get the prompt after T5 decoding to make sure dictionary
    of prompts and summaries is consistent decode prompt from trlX pipeline
    """
    formatted_prompts = []
    for i in tqdm(range(len(prompts))):
        tmp = tokenizer.decode(
            tokenizer(
                prompts[i],
                truncation=True,
                max_length=max_length - 2,
                add_special_tokens=False,
            )["input_ids"],
        ).strip()
        # 直接拼接 prompt + mask
        formatted_prompts.append(tmp + mask_type)
    return formatted_prompts


def get_ngrams(tokens, n):
    tokens = [str(token) for token in tokens]
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append("_".join(tokens[i: i + n]))
    return ngrams
    

def token_repetition_rate(tokenizer, response, ngrams=[1, 2, 3]):
    tokens = tokenizer(response, add_special_tokens=False)["input_ids"]
    repetition_rates = []
    for ngram in ngrams:
        ngram_tokens = get_ngrams(tokens, ngram)
        if ngram_tokens == []:
            continue
        else:
            repetition_rate = 1 - len(set(ngram_tokens)) / len(ngram_tokens)
            repetition_rates.append(repetition_rate)
    return np.mean(repetition_rates)


def process_raw_sample(
    prompt: str,
    response: str,
    tokenizer: GLMTokenizerMixin,
    max_prompt_length: int,
    mask_type: str,
    max_output_length: int,
):
    prompt = tokenizer.decode(
        tokenizer(
            prompt,
            truncation=True,
            max_length=max_prompt_length - 2,
            add_special_tokens=False,
        )["input_ids"],
    ).strip()
    prompt = tokenizer.cls_token + prompt + mask_type

    response = tokenizer.decode(
        tokenizer(
            response,
            truncation=True,
            max_length=max_output_length - 2,
            add_special_tokens=False,
        )["input_ids"],
    ).strip()
    response = tokenizer.sop_token + response + tokenizer.eop_token
    return prompt + response


def load_pretrained_models(
    model_path: str,
    rm_mean_value: bool,
    use_position_id: bool,
    use_normalized_reward: bool = False,
    num_head: int = 1
):
    tokenizer = GLMTokenizer.from_pretrained(model_path)
    model = RewardModel.from_pretrained(
        model_path,
        num_head=num_head,
        use_mean_value=rm_mean_value,
        use_position_id=use_position_id,
        use_normalized_reward=use_normalized_reward,
    )
    return model, tokenizer


def get_scores_glm(
    samples: List[str],
    tokenizer: GLMTokenizerMixin,
    mask_type: str,
    max_input_length: int,
    max_length: int,
    score_device: torch.device,
    score_model: RewardModel,
    truncation_side: str = "left",
    eos_token: str = "<|endoftext|>",
    rotary_type: str = "none",
    sigmoid_reward: bool = False
):
    scores_list = []
    batch_size = 2

    cls_token = tokenizer.cls_token
    eop_token = tokenizer.eop_token
    sop_token = tokenizer.sop_token
    eos_token = eos_token

    samples = [
        sample.replace(cls_token, "")
        .replace(sop_token, "")
        .replace(eop_token, "")
        .replace(tokenizer.eos_token, "")
        .rstrip()
        for sample in samples
    ]

    for i in range(0, len(samples), batch_size):
        sub_samples = samples[i: i + batch_size]

        batch_input_ids = []
        batch_attention_mask = []
        batch_position_ids = []

        for sample in sub_samples:
            prompt, response = sample.split(mask_type, 1)
            processed_feature = build_glm_inputs_from_sample(
                prompt,
                response,
                tokenizer=tokenizer,
                mask=mask_type,
                max_input_length=max_input_length,
                max_length=max_length,
                truncation_side=truncation_side,
                eos_token=eos_token,
                rotary_type=rotary_type
            )
            input_ids = processed_feature["input_ids"]
            attention_mask = processed_feature["attention_mask"]
            position_ids = processed_feature["position_ids"]

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_position_ids.append(position_ids)

        batch_input_ids = torch.stack(batch_input_ids, dim=0).to(score_model.device)
        batch_attention_mask = torch.stack(batch_attention_mask, dim=0).to(score_model.device)
        batch_position_ids = torch.stack(batch_position_ids, dim=0).to(score_model.device)

        with torch.no_grad():
            sub_scores = score_model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                position_ids=batch_position_ids,
            )
        scores_list.append(sub_scores)
    scores = torch.cat(scores_list, dim=0)

    if sigmoid_reward:
        scores = 2 * torch.sigmoid(scores) - 1

    assert scores.shape[0] == len(samples), scores.shape

    return scores


def init_eval_engine(
    model: nn.Module,
    ds_config: Dict[str, Any],
) -> deepspeed.DeepSpeedEngine:
    engine, *_ = deepspeed.initialize(
        model=model,
        config=ds_config,
    )
    return engine


def get_deepspeed_eval_config(
    *,
    stage: int = 3,
    fp16: bool = False,
    bf16: bool = False,
) -> Dict[str, Any]:
    """Get the DeepSpeed config for evaluation.

    Args:
        stage (int, optional): The stage of ZeRO. Defaults to 3.
        fp16 (bool, optional): Whether to use FP16 precision. Defaults to False.
        bf16 (bool, optional): Whether to use BF16 precision. Defaults to False.

    Returns:
        The DeepSpeed config for evaluation.
    """
    with EVAL_TEMPLATE_FILE.open(mode="rt", encoding="utf-8") as f:
        eval_config = json.load(f)

    if stage in {1, 2}:
        # The evaluation config only works for ZeRO stage 0 and ZeRO stage 3
        stage = 0

    eval_config["train_batch_size"] = None
    eval_config["train_micro_batch_size_per_gpu"] = 1
    eval_config["gradient_accumulation_steps"] = 1
    eval_config["zero_optimization"]["stage"] = stage
    if fp16 or "fp16" in eval_config:
        eval_config.setdefault("fp16", {})
        eval_config["fp16"]["enabled"] = fp16
    if bf16 or "bf16" in eval_config:
        eval_config.setdefault("bf16", {})
        eval_config["bf16"]["enabled"] = bf16
    return eval_config


def get_actor_tokenizer(tokenizer_path, padding_side, truncation_side):
    tokenizer = GLMTokenizer.from_pretrained(tokenizer_path)
    tokenizer.padding_side = padding_side
    tokenizer.truncation_side = truncation_side
    tokenizer.sep_token = "<sep>"
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer