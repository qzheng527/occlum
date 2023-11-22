import logging
import json
import os  # noqa
import pandas as pd
from transformers import HfArgumentParser, AutoTokenizer, AutoConfig
from transformers.data.data_collator import default_data_collator

from solutions.antllm.antllm.training.arguments.rm_arguments import \
    ModelArguments, DataArguments, RMTrainingArguments as TrainingArguments
from solutions.antllm.antllm.models.glm.tokenization_glm import GLMTokenizer
from solutions.antllm.antllm.models.glm.configuration_glm import GLMConfig
from solutions.antllm.antllm.data.dataset.rm_dataset.reward_dataset import GLMDynamicPaddingCollator
from solutions.antllm.antllm.models.glm.modeling_glm_rm import RM_HYPER_PARAMETERS_SAVE_FILE


def load_jsonl_file(file_path, as_whole=False):
    try:
        logging.info(f"File loading: {file_path}")
        if as_whole:
            with open(file_path, "r") as ifile:
                data = json.loads(ifile.read())
        else:
            with open(file_path, "r", encoding="utf-8") as ifile:
                lines = ifile.readlines()
                data = [json.loads(line) for line in lines]
            logging.info("Loaded {} records from {}".format(len(data), file_path))
        return data
    except Exception:
        raise ValueError(f"load_json_file failed: {file_path}")


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.data_format == "jsonl":
        eval_data = load_jsonl_file(data_args.eval_data_path)
    else:
        eval_data = pd.read_csv(data_args.eval_data_path).dropna().reset_index(drop=True)

    eos_token = "<|endofpiece|>"
    hyper_parameters = {}
    if os.path.exists(os.path.join(model_args.model_name_or_path, RM_HYPER_PARAMETERS_SAVE_FILE)):
        with open(os.path.join(model_args.model_name_or_path, RM_HYPER_PARAMETERS_SAVE_FILE)) as f:
            hyper_parameters = json.load(f)
            if "eos_token" in hyper_parameters:
                eos_token = hyper_parameters["eos_token"]
    print("eos token:", eos_token)

    data_collator = default_data_collator
    rotary_type = "none"
    if model_args.model_type == "glm":
        tokenizer = GLMTokenizer.from_pretrained(model_args.model_name_or_path)
        config = GLMConfig.from_pretrained(model_args.model_name_or_path)
        rotary_type = config.to_dict().get("rotary_type", "none")
        if data_args.dynamic_padding:
            data_collator = GLMDynamicPaddingCollator(
                pad_id=tokenizer.convert_tokens_to_ids(tokenizer.pad_token),
                data_type=data_args.data_type,
                mask_id=tokenizer.convert_tokens_to_ids(data_args.mask_type),
                rotary_type=rotary_type
            )
        
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    print("rotary type:", rotary_type)

    if data_args.data_type == "pairwise":
        from solutions.antllm.antllm.models.glm.modeling_glm_rm import RewardModelForPairWise
        from solutions.antllm.antllm.data.dataset.rm_dataset.reward_dataset import RewardDataset, GLMRewardDataset
        from solutions.antllm.antllm.data.dataset.rm_dataset.chatglm2_reward_dataset import ChatGLM2RewardDataset
        from solutions.antllm.antllm.training.trainer.rm_trainer import RMTrainer

        if model_args.model_type == "glm":
            eval_dataset = GLMRewardDataset(
                dataset=eval_data,
                tokenizer=tokenizer,
                max_length=data_args.max_len,
                max_input_length=data_args.max_input_len,
                mask=data_args.mask_type,
                return_dict=True,
                truncation_side=data_args.truncation_side,
                dynamic_padding=data_args.dynamic_padding,
                eos_token=eos_token,
                rotary_type=rotary_type
            )
        elif model_args.model_type == "chatglm2":
            eval_dataset = ChatGLM2RewardDataset(
                dataset=eval_data,
                tokenizer=tokenizer,
                max_length=data_args.max_len,
                max_input_length=data_args.max_input_len,
                return_dict=True,
                truncation_side=data_args.truncation_side,
            )
        else:
            eval_dataset = RewardDataset(eval_data, tokenizer, data_args.max_len, return_dict=True)
        model = RewardModelForPairWise(
            model_args.model_name_or_path,
            model_type=model_args.model_type,
            use_mean_value=model_args.use_mean_value,
            use_position_id=model_args.use_position_id,
            use_normalized_reward=model_args.use_normalized_reward
        )
        trainer = RMTrainer(
            model=model,
            args=training_args,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )
        predictions = trainer.predict(eval_dataset).predictions

        chosens = predictions[1]
        rejecteds = predictions[2]

        acc = 0
        data_with_score = []
        for data, chosen_score, rejected_score in zip(eval_data, chosens, rejecteds):
            data.update({
                "chosen_score": float(chosen_score),
                "rejected_score": float(rejected_score),
            })
            data_with_score.append(data)
            if chosen_score > rejected_score:
                acc += 1
        accuracy = acc / len(chosens)
        print("accuracy:", accuracy)
    else:
        from solutions.antllm.antllm.models.glm.modeling_glm_rm import RewardModelForPointWise
        from solutions.antllm.antllm.data.dataset.rm_dataset.reward_dataset import \
            RewardDatasetForPointWise as RewardDataset
        from solutions.antllm.antllm.data.dataset.rm_dataset.reward_dataset import \
            GLMRewardDatasetForPointWise as GLMRewardDataset
        from solutions.antllm.antllm.training.trainer.rm_trainer import \
            RMTrainerForPointWise as RMTrainer

        weights = data_args.weights

        if model_args.model_type == "glm":
            eval_dataset = GLMRewardDataset(
                dataset=eval_data,
                weights=weights,
                tokenizer=tokenizer,
                max_length=data_args.max_len,
                max_input_length=data_args.max_input_len,
                num_head=model_args.num_head,
                mask=data_args.mask_type,
                return_dict=True,
                truncation_side=data_args.truncation_side,
                dynamic_padding=data_args.dynamic_padding,
                eos_token=eos_token,
                rotary_type=rotary_type
            )
        elif model_args.model_type == "chatglm2":
            raise NotImplementedError
        else:
            eval_dataset = RewardDataset(eval_data, tokenizer, data_args.max_len, return_dict=True)
        model = RewardModelForPointWise(
            model_args.model_name_or_path,
            num_head=model_args.num_head,
            model_type=model_args.model_type,
            use_mean_value=model_args.use_mean_value,
            use_normalized_reward=model_args.use_normalized_reward
        )

        trainer = RMTrainer(
            model=model,
            num_head=model_args.num_head,
            args=training_args,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )
        predictions = trainer.predict(eval_dataset).predictions
        reward_engaged, labels_engaged = predictions[:2]

        data_with_score = []
        for data, reward_engaged, labels_engaged in zip(eval_data, reward_engaged, labels_engaged):
            data.update({
                "reward": float(reward_engaged),
                "labels": float(labels_engaged),
            })
            data_with_score.append(data)

    if data_args.predict_output_path is not None:
        with open(os.path.join(data_args.predict_output_path, "pred_scores.jsonl"), 'w') as fout:
            for data in data_with_score:
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")

        if data_args.data_type == "pairwise":
            with open(os.path.join(data_args.predict_output_path, "eval_result.json"), 'w') as fout:
                acc = {"acc": accuracy}
                fout.write(json.dumps(acc, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()