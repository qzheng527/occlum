from solutions.antllm.antllm.models.glm.modeling_glm_rm import RewardModel
from transformers import AutoTokenizer
import os
import argparse
import torch
import json
import copy
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rm_model_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_output_length", type=int, default=512)
    parser.add_argument("--use_mean_value", action="store_true", default=False)
    parser.add_argument("--use_position_id", action="store_true", default=False)
    parser.add_argument("--mask", type=str, default="[gMASK]")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.rm_model_path, trust_remote_code=True
    )
    rw_model = RewardModel.from_pretrained(
        args.rm_model_path, use_mean_value=args.use_mean_value, use_position_id=args.use_position_id
    )
    rw_model.half()
    rw_model.eval()

    rw_device = torch.device("cuda:{}".format(0))
    rw_model.to(rw_device)

    sop_id = tokenizer.sop_token_id
    eop_id = tokenizer.eop_token_id
    cls_id = tokenizer.cls_token_id
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.convert_tokens_to_ids(args.mask)

    max_input_length = args.max_input_length
    max_output_length = args.max_output_length
    max_length = args.max_len

    def get_scores_glm(data):
        prompt = data["prompt"].replace("\\n", "\n").rstrip()
        chosen = data["chosen"].replace("\\n", "\n").rstrip()
        rejected = data["rejected"].replace("\\n", "\n").rstrip()

        tokenizer_outs = tokenizer(
            prompt,
            padding=False,
            add_special_tokens=False,
            return_attention_mask=True,
        )
        input_ids = tokenizer_outs["input_ids"]

        # 截断 prompt
        if len(input_ids) > max_input_length - 2:
            input_ids = input_ids[: max_input_length - 2]

        # 添加 cls 和 mask
        input_ids = [cls_id] + input_ids + [mask_id]

        sep = len(input_ids)
        mask_pos = input_ids.index(mask_id)
        position_ids = list(range(len(input_ids)))
        block_position_ids = [0] * len(input_ids)

        # sop & eop token
        max_output_length = max_length - max_input_length - 2

        # 处理 chosen response
        chosen_ids = tokenizer(chosen, add_special_tokens=False)["input_ids"]

        if len(chosen_ids) > max_output_length:
            chosen_ids = chosen_ids[:max_output_length]
        chosen_ids = input_ids + [sop_id] + chosen_ids + [eop_id]

        if len(chosen_ids) < max_length:
            chosen_pad_length = max_length - len(chosen_ids)
            chosen_ids += [pad_id] * chosen_pad_length

        # position_ids在mask之后全部补mask_pos
        chosen_position_ids = position_ids + [mask_pos] * (
            len(chosen_ids) - len(position_ids)
        )
        chosen_position_ids = torch.tensor(chosen_position_ids, dtype=torch.long)

        # block_position_ids在mask之后补1 2 3 4 5..
        chosen_block_position_ids = block_position_ids + list(
            range(1, len(chosen_ids) - len(block_position_ids) + 1)
        )
        chosen_block_position_ids = torch.tensor(
            chosen_block_position_ids, dtype=torch.long
        )

        chosen_ids = torch.tensor([chosen_ids]).long()
        # attn_masks = chosen_ids.eq(mask_id).nonzero(as_tuple=True)[1]
        attn_masks = torch.tensor([[sep]]).long()
        chosen_position_ids_all = torch.stack(
            (chosen_position_ids, chosen_block_position_ids), dim=0
        ).reshape(1, 2, -1)

        chosen_ids = chosen_ids.to(rw_device)
        attn_masks = attn_masks.to(rw_device)
        chosen_position_ids_all = chosen_position_ids_all.to(rw_device)

        with torch.no_grad():
            chosen_score = rw_model(
                input_ids=chosen_ids,
                attention_mask=attn_masks,
                position_ids=chosen_position_ids_all,
            )[0]

        # 处理 rejected response
        rejected_ids = tokenizer(rejected, add_special_tokens=False)["input_ids"]

        if len(rejected_ids) > max_output_length:
            rejected_ids = rejected_ids[:max_output_length]

        rejected_ids = input_ids + [sop_id] + rejected_ids + [eop_id]
        if len(rejected_ids) < max_length:
            rejected_pad_length = max_length - len(rejected_ids)
            rejected_ids += [pad_id] * rejected_pad_length

        # position_ids在mask之后全部补mask_pos
        rejected_position_ids = position_ids + [mask_pos] * (
            len(rejected_ids) - len(position_ids)
        )
        rejected_position_ids = torch.tensor(rejected_position_ids, dtype=torch.long)

        # block_position_ids在mask之后补1 2 3 4 5..
        rejected_block_position_ids = block_position_ids + list(
            range(1, len(rejected_ids) - len(block_position_ids) + 1)
        )
        rejected_block_position_ids = torch.tensor(
            rejected_block_position_ids, dtype=torch.long
        )

        # attention_mask = build_mask(max_length, sep)
        assert (
            len(rejected_ids)
            == len(rejected_position_ids)
            == len(rejected_block_position_ids)
            == max_length
        )

        rejected_ids = torch.Tensor([rejected_ids]).long()
        rejected_position_ids_all = torch.stack(
            (rejected_position_ids, rejected_block_position_ids), dim=0
        ).reshape(1, 2, -1)

        # attn_masks = rejected_ids.eq(mask_id).nonzero(as_tuple=True)[1]
        rejected_ids = rejected_ids.to(rw_device)
        rejected_position_ids_all = rejected_position_ids_all.to(rw_device)

        with torch.no_grad():
            rejected_score = rw_model(
                input_ids=rejected_ids,
                attention_mask=attn_masks,
                position_ids=rejected_position_ids_all,
            )[0]

        return chosen_score.item(), rejected_score.item()

    os.makedirs(args.out_path, exist_ok=True)
    with open(args.test_path, "r") as fin, open(
        os.path.join(args.out_path, "calibration_score.jsonl"), "w"
    ) as fout:
        for line in tqdm(fin):
            line_dict = json.loads(line)
            chosen_score, rejected_score = get_scores_glm(line_dict)
            out_dict = copy.deepcopy(line_dict)
            out_dict["chosen_score"] = chosen_score
            out_dict["rejected_score"] = rejected_score
            fout.write(json.dumps(out_dict, ensure_ascii=False) + "\n")
