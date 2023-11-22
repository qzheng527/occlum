import os
from pathlib import Path
import json
import shutil
from transformers.utils import logging
from transformers import (
    HfArgumentParser,
    set_seed,
    AutoConfig
)

from datasets import load_dataset
from transformers.data.data_collator import default_data_collator
from transformers.trainer_callback import TrainerCallback
from transformers import AutoTokenizer
from solutions.antllm.antllm.training.arguments.rm_arguments import \
    ModelArguments, DataArguments, RMTrainingArguments as TrainingArguments
from peft import LoraConfig, TaskType
from solutions.antllm.antllm.models.glm.tokenization_glm import GLMTokenizer
from solutions.antllm.antllm.models.glm.modeling_glm_rm import AntPeftForCausalLM  # noqa
from solutions.antllm.antllm.data.dataset.rm_dataset.reward_dataset import GLMDynamicPaddingCollator
from solutions.antllm.antllm.models.glm.configuration_glm import GLMConfig
from solutions.antllm.antllm.models.glm.modeling_glm_rm import RM_HYPER_PARAMETERS_SAVE_FILE
from glob import glob

logger = logging.get_logger(__name__)


class SaveCallback(TrainerCallback):
    def __init__(self, files_to_save, args_to_save=None):
        self.files_to_save = files_to_save
        self.args_to_save = args_to_save

    def on_save(self, args, state, control, **kwargs):
        if args.process_index == 0:
            ckpt_dir = f"checkpoint-{state.global_step}"
            artifact_path = os.path.join(args.output_dir, ckpt_dir)
            for name in self.files_to_save:
                if not os.path.exists(name):
                    continue
                try:
                    if os.path.isfile(name):
                        shutil.copy(name, artifact_path)
                    elif os.path.isdir(name):
                        shutil.copytree(name, os.path.join(
                            artifact_path, os.path.basename(name)))
                except Exception:
                    continue

            if self.args_to_save is not None:
                with open(os.path.join(artifact_path, RM_HYPER_PARAMETERS_SAVE_FILE), "w") as f:
                    json.dump(self.args_to_save, f, indent=2, ensure_ascii=False)


def main():
    try:
        from alps.pytorch.components.transformers import patch_get_class_in_module
        patch_get_class_in_module()
    except ModuleNotFoundError:
        pass

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger.setLevel(training_args.get_process_log_level())
    logger.warning(
        "Global rank: %s, local rank: %s, device: %s, \
            n_gpu: %s, distributed training: %s, fp16 training: %s, bf16 training: %s",
        training_args.process_index,
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
        training_args.bf16
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)
    
    if model_args.model_type == "chatglm2":
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    else:
        tokenizer = GLMTokenizer.from_pretrained(model_args.model_name_or_path)
        config = GLMConfig.from_pretrained(model_args.model_name_or_path)

    rotary_type = config.to_dict().get("rotary_type", "none")
    print("rotary type:", rotary_type)

    lora_config = None
    files_to_save = []
    if model_args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.ANT_CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )
        for filename in Path(model_args.model_name_or_path).iterdir():
            filename = str(filename)
            files_to_save.append(filename)

    eos_token = "<|endofpiece|>"
    hyper_parameters = {}
    if os.path.exists(os.path.join(model_args.model_name_or_path, RM_HYPER_PARAMETERS_SAVE_FILE)):
        with open(os.path.join(model_args.model_name_or_path, RM_HYPER_PARAMETERS_SAVE_FILE)) as f:
            hyper_parameters = json.load(f)
            if "eos_token" in hyper_parameters:
                eos_token = hyper_parameters["eos_token"]
    print("eos token:", eos_token)
    hyper_parameters.update({"eos_token": eos_token})

    args_to_save = {
        "num_head": model_args.num_head,
        "use_mean_value": model_args.use_mean_value,
        "use_position_id": model_args.use_position_id,
        "use_normalized_reward": model_args.use_normalized_reward
    }
    args_to_save.update(hyper_parameters)

    # Create the comparisons datasets
    train_files = glob(os.path.join(data_args.dataset_dir, "*train.jsonl"))
    eval_files = glob(os.path.join(data_args.dataset_dir, "*test.jsonl"))

    logger.info(f"train files: {train_files}")
    logger.info(f"test files: {eval_files}")

    train_data = load_dataset("json", data_files=train_files, split="train", streaming=True)
    eval_data = load_dataset("json", data_files=eval_files, split="train", streaming=True)

    data_collator = default_data_collator
    if data_args.dynamic_padding:
        data_collator = GLMDynamicPaddingCollator(
            pad_id=tokenizer.convert_tokens_to_ids(tokenizer.pad_token),
            data_type=data_args.data_type,
            mask_id=tokenizer.convert_tokens_to_ids(data_args.mask_type),
            rotary_type=rotary_type
        )

    if data_args.data_type == "pointwise":
        from solutions.antllm.antllm.data.dataset.rm_dataset.reward_dataset import \
            RewardDatasetForPointWise
        from solutions.antllm.antllm.data.dataset.rm_dataset.reward_dataset import \
            GLMRewardDatasetForPointWise, GLMIterableRewardDatasetForPointWise
        from solutions.antllm.antllm.models.glm.modeling_glm_rm import RewardModelForPointWise
        from solutions.antllm.antllm.training.trainer.rm_trainer import RMTrainerForPointWise
        from solutions.antllm.antllm.data.dataset.rm_dataset.chatglm2_reward_dataset import ChatGLM2RewardDataset

        model = RewardModelForPointWise(
            model_args.model_name_or_path,
            num_head=model_args.num_head,
            model_type=model_args.model_type,
            use_mean_value=model_args.use_mean_value,
            lora_config=lora_config,
            use_position_id=model_args.use_position_id,
            num_layers_unfrozen=model_args.num_layers_unfrozen,
            use_normalized_reward=model_args.use_normalized_reward
        )

        weights = data_args.weights
        if model_args.model_type == "glm":
            if data_args.lazy_load:
                train_dataset = GLMIterableRewardDatasetForPointWise(
                    dataset=train_data,
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
                eval_dataset = GLMIterableRewardDatasetForPointWise(
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
            else:
                train_dataset = GLMRewardDatasetForPointWise(
                    dataset=train_data,
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
                eval_dataset = GLMRewardDatasetForPointWise(
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
        else:
            train_dataset = RewardDatasetForPointWise(train_data, tokenizer, data_args.max_len, return_dict=True)
            eval_dataset = RewardDatasetForPointWise(eval_data, tokenizer, data_args.max_len, return_dict=True)

        trainer = RMTrainerForPointWise(
            model=model,
            num_head=model_args.num_head,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[SaveCallback(files_to_save, args_to_save)]
        )
    else:
        from solutions.antllm.antllm.models.glm.modeling_glm_rm import RewardModelForPairWise
        from solutions.antllm.antllm.training.trainer.rm_trainer import RMTrainer
        from solutions.antllm.antllm.data.dataset.rm_dataset.reward_dataset import RewardDataset
        from solutions.antllm.antllm.data.dataset.rm_dataset.reward_dataset import \
            GLMRewardDataset, GLMIterableRewardDataset
        from solutions.antllm.antllm.data.dataset.rm_dataset.chatglm2_reward_dataset import \
            ChatGLM2RewardDataset

        model = RewardModelForPairWise(
            model_args.model_name_or_path,
            model_type=model_args.model_type,
            use_mean_value=model_args.use_mean_value,
            lora_config=lora_config,
            use_position_id=model_args.use_position_id,
            num_layers_unfrozen=model_args.num_layers_unfrozen,
            use_normalized_reward=model_args.use_normalized_reward
        )

        if model_args.model_type == "glm":
            if data_args.lazy_load:
                train_dataset = GLMIterableRewardDataset(
                    dataset=train_data,
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
                eval_dataset = GLMIterableRewardDataset(
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
            else:
                train_dataset = GLMRewardDataset(
                    dataset=train_data,
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
            train_dataset = ChatGLM2RewardDataset(
                dataset=train_data,
                tokenizer=tokenizer,
                max_length=data_args.max_len,
                max_input_length=data_args.max_input_len,
                return_dict=True,
                truncation_side=data_args.truncation_side
            )
            eval_dataset = ChatGLM2RewardDataset(
                dataset=eval_data,
                tokenizer=tokenizer,
                max_length=data_args.max_len,
                max_input_length=data_args.max_input_len,
                return_dict=True,
                truncation_side=data_args.truncation_side
            )
        else:
            train_dataset = RewardDataset(train_data, tokenizer, data_args.max_len, return_dict=True)
            eval_dataset = RewardDataset(eval_data, tokenizer, data_args.max_len, return_dict=True)

        trainer = RMTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[SaveCallback(files_to_save, args_to_save)]
        )

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_training)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # if training_args.do_eval:
    #     trainer.evaluate()


if __name__ == "__main__":
    main()
