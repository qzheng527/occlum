from typing import List
import pandas as pd
import argparse

import os
import torch

from transformers import AutoTokenizer

from solutions.antllm.antllm.utils.modeling_glm_rm_utils import build_glm_inputs_from_sample
from solutions.antllm.antllm.models.glm.modeling_glm_rm import RewardModel
from solutions.antllm.antllm.inference.glm_predictor import GLMForInference
from solutions.antllm.antllm.training.arguments.rl_arguments import TRLConfig

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
    parser.add_argument("--exp_cfg_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--infer_dir", type=str, default=None)
    parser.add_argument("--mask_type", type=str, default="[gMASK]")
    parser.add_argument("--infer_size", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--num_head", type=int, default=None)
    parser.add_argument("--rm_mean_value", action="store_true", default=False)

    args = parser.parse_args()
    config = TRLConfig.load_yaml(args.exp_cfg_path)

    if args.save_dir is not None:
        config.train.checkpoint_dir = args.save_dir
    if args.log_dir is not None:
        config.train.logging_dir = args.log_dir
    if args.ppo_model_path is not None:
        config.model.model_path = args.ppo_model_path
        config.tokenizer.tokenizer_path = args.ppo_model_path

    if int(os.environ.get("LOCAL_RANK")) == 0:
        print(config)

    # Load the pre-trained reward model
    rw_tokenizer = AutoTokenizer.from_pretrained(
        args.rm_model_path, trust_remote_code=True
    )

    rw_model = RewardModel.from_pretrained(
        args.rm_model_path,
        num_head=args.num_head,
        use_mean_value=args.rm_mean_value,
        use_position_id=args.use_position_id,
    )
    rw_model.half()
    rw_model.eval()

    if int(os.environ.get("LOCAL_RANK")) == 0:
        rw_device = torch.device("cuda:{}".format(0))  # set reward model device
        rw_model.to(rw_device)

    def get_scores_glm(samples: List[str]):
        scores_list = []
        batch_size = 2

        cls_token = rw_tokenizer.cls_token
        eop_token = rw_tokenizer.eop_token
        sop_token = rw_tokenizer.sop_token
        eos_token = rw_tokenizer.eos_token

        samples = [
            sample.replace(cls_token, "")
            .replace(sop_token, "")
            .replace(eop_token, "")
            .replace(eos_token, "")
            .rstrip()
            for sample in samples
        ]

        for i in range(0, len(samples), batch_size):
            # sample 的 pad 有的在左边有的在右边？
            sub_samples = samples[i : i + batch_size]

            batch_input_ids = []
            batch_attention_mask = []
            batch_position_ids = []

            for sample in sub_samples:
                prompt, response = sample.split(args.mask_type)
                input_ids, attention_mask, position_ids = build_glm_inputs_from_sample(
                    prompt,
                    response,
                    tokenizer=rw_tokenizer,
                    mask=args.mask_type,
                    max_input_length=config.train.seq_length
                    - config.method.gen_kwargs["max_new_tokens"],
                    max_length=config.train.seq_length,
                )
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_position_ids.append(position_ids)

            batch_input_ids = torch.stack(batch_input_ids, dim=0).to(rw_device)
            batch_attention_mask = torch.stack(batch_attention_mask, dim=0).to(
                rw_device
            )
            batch_position_ids = torch.stack(batch_position_ids, dim=0).to(rw_device)

            with torch.no_grad():
                if args.num_head > 1:
                    sub_scores = torch.sum(
                        rw_model(
                            input_ids=batch_input_ids,
                            attention_mask=batch_attention_mask,
                            position_ids=batch_position_ids,
                        ),
                        dim=1
                    )
                else:
                    sub_scores = rw_model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        position_ids=batch_position_ids,
                    )
            scores_list.append(sub_scores)
        scores = torch.cat(scores_list, dim=0)
        assert scores.shape[0] == len(samples), scores.shape

        return scores

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer.tokenizer_path, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = config.tokenizer.padding_side
    max_length_input = (
        config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]
    )

    kwargs = dict(
        config.method.gen_kwargs,
        eos_token_id=tokenizer.eop_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    device = torch.device("cuda:{}".format(0))

    model = GLMForInference(config.model.model_path)
    model.to(device) 
    model.eval()

    infer_data = pd.read_csv(args.prompt_path).dropna().reset_index(drop=True)
    prompts = infer_data["prompt"]

    data_dict = {'prompt': [], 'out': []}
    cnt = 0
    for i in range(0, len(prompts)):
        row = prompts[i]
        cnt += 1

        with torch.no_grad():
            outputs = model.answer(row)
            sample = tokenizer.sop_token + outputs + tokenizer.eop_token
            max_reward = get_scores_glm(sample, batch_size=1)
            for j in range(args.runs):
                temp_outputs = model.answer(row)
                temp_sample = tokenizer.sop_token + temp_outputs + tokenizer.eop_token
                temp_reward = get_scores_glm(temp_sample, batch_size=1)
                if temp_reward > max_reward:
                    max_reward = temp_reward
                    outputs = temp_outputs

        print(f"/n response : {outputs} /n")

        data_dict['prompt'].append(row)
        data_dict['out'].append(outputs)
        print(cnt)

        if cnt % args.infer_size == 0:
            out_pd = pd.DataFrame(data_dict)
            out_pd.to_csv(args.infer_dir, encoding='utf-8', index=False)
            data_dict = {'prompt': [], 'out': []}
