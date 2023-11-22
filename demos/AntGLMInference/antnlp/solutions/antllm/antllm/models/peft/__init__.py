"""peft models interface."""

from . import utils, tuner
from peft.mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING
from peft.utils import TaskType
from .modeling_peft import AntPeftForCausalLM, AntPeftForEmbedding


# PEFT的部分介绍如 https://ata.atatech.org/articles/12020034409
SUPPORTED_PEFT_TYPES = ["lora", "adalora", "bitfit", "roem", "unipelt", "prompt", "ptuning", "prefix", "qlora"]

# Register the Ant Causal Language Model
MODEL_TYPE_TO_PEFT_MODEL_MAPPING["ANT_CAUSAL_LM"] = AntPeftForCausalLM
TaskType.ANT_CAUSAL_LM = "ANT_CAUSAL_LM"

MODEL_TYPE_TO_PEFT_MODEL_MAPPING["ANT_EMBEDDING"] = AntPeftForEmbedding
TaskType.ANT_EMBEDDING = "ANT_EMBEDDING"
