import json
import os
import torch
from pathlib import Path

from dataclasses import dataclass, field
from solutions.antllm.antllm.data.dataset.rl_dataset.dpo_dataset import GLMDPODataset
from solutions.antllm.antllm.models.glm.modeling_glm import GLMForConditionalGeneration
from solutions.antllm.antllm.models.glm.tokenization_glm import GLMTokenizer
from solutions.antllm.antllm.training.trainer.dpo_trainer import DPOTrainer
from solutions.antllm.antllm.utils.modeling_glm_rm_utils import freeze_model
from solutions.antllm.antllm.data.dataset.glm_instruction_dataset import (
    GLMInstructionDataset,
)
from transformers import HfArgumentParser, TrainingArguments  # noqa
from transformers.trainer_callback import TrainerCallback
from transformers.data.data_collator import DataCollatorMixin
from transformers.utils import logging
from transformers.modeling_utils import unwrap_model

try:
    from bigmodelvis import Visualization

    HAS_OPENDELTA = True
except ModuleNotFoundError:
    HAS_OPENDELTA = False

logger = logging.get_logger(__name__)


class DynamicPaddingCollator(DataCollatorMixin):
    def __init__(self, pad_id=None):
        super().__init__()
        self.pad_id = pad_id

    def __call__(self, features):
        if self.pad_id is not None:
            max_length = -1
            for row in range(len(features)):
                input_ids_chosen = features[row]["input_ids_chosen"]
                for col in range(input_ids_chosen.shape[0]):
                    if input_ids_chosen[col].item() == self.pad_id:
                        break
                max_length = col if col > max_length else max_length

                input_ids_rejected = features[row]["input_ids_rejected"]
                for col in range(input_ids_rejected.shape[0]):
                    if input_ids_rejected[col].item() == self.pad_id:
                        break
                max_length = col if col > max_length else max_length

            for row in range(len(features)):
                features[row]["input_ids_chosen"] = features[row]["input_ids_chosen"][
                    :max_length
                ]
                features[row]["labels_chosen"] = features[row]["labels_chosen"][
                    :max_length
                ]
                features[row]["position_ids_chosen"] = features[row][
                    "position_ids_chosen"
                ][:, :max_length]

                features[row]["input_ids_rejected"] = features[row][
                    "input_ids_rejected"
                ][:max_length]
                features[row]["labels_rejected"] = features[row]["labels_rejected"][
                    :max_length
                ]
                features[row]["position_ids_rejected"] = features[row][
                    "position_ids_rejected"
                ][:, :max_length]
        else:
            logger.info("Ignore dynamic_padding, while dynamic_padding, pad_id muast be set")
        batch = {}
        for feature in features:
            for key in feature:
                if key not in batch:
                    batch[key] = []
                batch[key].append(feature[key].unsqueeze(0))
        batch = {key: torch.cat(value) for key, value in batch.items()}
        batch["return_loss"] = True
        return batch


class SaveCallback(TrainerCallback):
    def __init__(self, files_to_save, args_to_save={}):
        self.files_to_save = files_to_save
        self.args_to_save = args_to_save

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = f"checkpoint-{state.global_step}"
        artifact_path = os.path.join(args.output_dir, ckpt_dir)
        if os.environ.get("RANK", "0") == "0":
            json.dump(
                self.args_to_save,
                open(os.path.join(artifact_path, "hyper_parameters.json"), "w"),
                ensure_ascii=False,
                indent=2,
            )


class SampleGenerateCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        if os.environ.get("RANK", "0") == "0":
            logger.info("on_evaluate in SampleGenerateCallback...")
        sample_inputs = ["马化腾是马云的儿子，你知道吗", "蚂蚁集团主要有哪些产品呢", "什么是钝角，什么是锐角"]
        if "model" in kwargs:
            for sample_input in sample_inputs:
                tokenizer = kwargs["tokenizer"]
                model = kwargs["model"]
                unwrap_base_model = unwrap_model(model)
                data = {"input": sample_input}
                _, inputs = GLMInstructionDataset.build_feature_from_sample(
                    data, tokenizer, for_generation=True
                )
                inputs = inputs.to(torch.device("cuda"))
                generation_output = unwrap_base_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    eos_token_id=tokenizer.eop_token_id,
                    pad_token_id=tokenizer.eop_token_id,
                )
                if os.environ.get("RANK", "0") == "0":
                    logger.info(
                        f"sample output: {tokenizer.decode(generation_output[0])}"
                    )

        else:
            if os.environ.get("RANK", "0") == "0":
                logger.info(f"model not found in kwargs, skipping")


@dataclass
class ModelArguments:
    pretrained_model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    reference_model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained reference model or model identifier from huggingface.co/models"
        }
    )
    beta_coef: float = field(metadata={"help": "beta coefficient for reference model"})
    num_layers_unfrozen: int = field(default=2, metadata={"help": "number of layers unftrozen of the model"})
    reference_free: bool = field(
        default=False,
        metadata={
            "help": "If True, we ignore the _provided_ reference model and implicitly use a "
                    "reference model that assigns equal probability to all responses."
        },
    )
    no_save_deepspeed_checkpoint: bool = field(
        default=False, metadata={"help": "Whether to save deepspeed checkpoint."}
    )
    no_save_base_model: bool = field(
        default=False, metadata={"help": "Whether to save base model."}
    )


@dataclass
class PeftArguments:
    peft_type: str = field(default=None, metadata={"help": "Whether use peft"})
    prompt_init_text: str = field(
        default=None, metadata={"help": "The init text of prompt learning"}
    )


@dataclass
class DataArguments:
    max_length: int = field(
        metadata={"help": "The maximum total input sequence length"}
    )
    max_input_length: int = field(metadata={"help": "The maximum total prompt length"})
    max_output_length: int = field(metadata={"help": "The maximum total output length"})
    train_data: str = field(default=None, metadata={"help": "Path to train data"})
    test_data: str = field(default=None, metadata={"help": "Path to test data"})
    glm_mask: str = field(default="[gMASK]", metadata={"help": "Mask to use in glm"})
    left_truncate: bool = field(
        default=False, metadata={"help": "Whether truncate at the left side"}
    )
    dynamic_padding: bool = field(
        default=True, metadata={"help": "Whether dynamically padding in each batch"}
    )


@dataclass
class TrainingArguments(TrainingArguments):
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Removed unused columns. Needed to make this codebase work."},
    )


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, PeftArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        peft_args,
    ) = parser.parse_args_into_dataclasses()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_rank = int(os.environ.get("RANK", 0))
    podname = os.environ.get("ILOGTAIL_PODNAME", "master")
    if "master" in podname:
        save_checkpoint = True
    elif "worker" in podname:
        save_checkpoint = False
    else:
        save_checkpoint = True
    logger.info(f"world_size: {world_size}, global_rank: {global_rank}")

    train_data_path = data_args.train_data
    test_data_path = data_args.test_data

    auto_model_class = GLMForConditionalGeneration  # noqa
    dataset_class = GLMDPODataset
    auto_tokenization_class = GLMTokenizer

    tokenizer = auto_tokenization_class.from_pretrained(  # noqa
        model_args.pretrained_model_name_or_path, trust_remote_code=True
    )

    model = auto_model_class.from_pretrained(
        model_args.pretrained_model_name_or_path,
    )
    # frozen model layers
    freeze_model(model, model_args.num_layers_unfrozen)

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    reference_model = auto_model_class.from_pretrained(
        model_args.reference_model_name_or_path,
        torch_dtype=compute_dtype,
    )

    if HAS_OPENDELTA and global_rank == 0:
        model_vis = Visualization(model)
        model_vis.structure_graph()

    files_to_save = []
    for filename in Path(model_args.pretrained_model_name_or_path).iterdir():
        filename = str(filename)
        if filename.endswith("pytorch_model.bin") or filename.endswith(
            "hyper_parameters.json"
        ):
            continue
        files_to_save.append(filename)

    train_dataset = dataset_class(
        data_path=train_data_path,
        tokenizer=tokenizer,
        max_length=data_args.max_length,
        max_input_length=data_args.max_input_length,
        max_output_length=data_args.max_output_length,
        left_truncate=data_args.left_truncate,
        glm_mask=data_args.glm_mask,
    )
    test_dataset = dataset_class(
        data_path=test_data_path,
        tokenizer=tokenizer,
        max_length=data_args.max_length,
        max_input_length=data_args.max_input_length,
        max_output_length=data_args.max_output_length,
        left_truncate=data_args.left_truncate,
        glm_mask=data_args.glm_mask,
    )

    data_collator = None
    if data_args.dynamic_padding:
        data_collator = DynamicPaddingCollator(
            pad_id=tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        )

    trainer = DPOTrainer(
        reference_model=reference_model,
        beta=model_args.beta_coef,
        reference_free=model_args.reference_free,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        no_save_deepspeed_checkpoint=model_args.no_save_deepspeed_checkpoint,
        save_pytorch_model_bin_checkpoint=save_checkpoint,
        rank=global_rank,
        data_collator=data_collator,
        no_save_base_model=model_args.no_save_base_model,
    )
    trainer.add_callback(SampleGenerateCallback)
    trainer.add_callback(
        SaveCallback(
            files_to_save,
            args_to_save={
                "max_length": data_args.max_length,
                "peft_type": peft_args.peft_type,
            },
        )
    )

    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_eval:
        trainer.evaluate()


if __name__ == "__main__":
    main()
