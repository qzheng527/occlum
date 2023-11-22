from dataclasses import dataclass, field
from transformers import TrainingArguments
from typing import List


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    num_head: int = field(default=1, metadata={"help": "number of labels"})
    model_type: str = field(default="glm", metadata={"help": "model type"})
    use_mean_value: bool = field(
        default=False, metadata={"help": "use mean tokens value or last token value as reward"}
    )
    use_lora: bool = field(default=False, metadata={"help": "use mean tokens value or last token value as reward"})
    use_position_id: bool = field(default=True, metadata={"help": "whether use position id"})
    num_layers_unfrozen: int = field(default=2, metadata={"help": "number layers unfrozen for reward model"})
    use_normalized_reward: bool = field(default=False, metadata={"help": "whether to normalize reward value"})


@dataclass
class DataArguments:
    max_len: int = field(metadata={"help": "The maximum total input sequence length"})
    max_input_len: int = field(default=251, metadata={"help": "max prompt length"})
    dataset_dir: str = field(
        default=None, metadata={"help": "Path to dataset directory"}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to eval data file"}
    )
    data_format: str = field(default="jsonl", metadata={"help": "data format"})
    data_type: str = field(default="pairwise", metadata={"help": "data type"})
    mask_type: str = field(default="[gMASK]", metadata={"help": "MASK type"})
    truncation_side: str = field(default="left", metadata={"help": "truncation side"})   
    predict_output_path: str = field(
        default=None, metadata={"help": "predict output path"}
    )
    weights: List[float] = field(default=None, metadata={"help": "weights for different labels"})
    dynamic_padding: bool = field(default=False, metadata={"help": "whether to use dynamic padding"})
    lazy_load: bool = field(default=False, metadata={"help": "whether to use lazy load"})


@dataclass
class RMTrainingArguments(TrainingArguments):
    no_save_deepspeed_checkpoint: bool = field(
        default=False, metadata={"help": "whether to save deepspeed checkpoint"}
    )
    no_shuffle_dataloader: bool = field(
        default=False, metadata={"help": "where to shuffle dataloader"}
    )
    resume_training: bool = field(
        default=False, metadata={"help": "resume training from saved checkpoint"}
    )


@dataclass
class RMInferArguments:
    infer_size: int = field(default=1000, metadata={"help": "size of infer dataset"})
    infer_dir: str = field(
        default=None, metadata={"help": "Path to infer results"}
    )