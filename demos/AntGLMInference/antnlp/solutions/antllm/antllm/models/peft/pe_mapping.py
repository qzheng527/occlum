from .pe_model import (
    MultiAdapterModelForCausalLM,
    MultiAdapterModel,
    MultiAdapterModelForSeq2SeqLM,
)
from peft import PEFT_TYPE_TO_CONFIG_MAPPING  # noqa F401
from .utils.aggregator import (
    StaticAggregatorConfig,
    MoEAggregatorConfig,
    DirectAggregatorConfig,
    MoEV2AggregatorConfig,
)

MODEL_TYPE_TO_MULTI_ADAPTER_MODEL_MAPPING = {
    "CAUSAL_LM": MultiAdapterModelForCausalLM,
    "SEQ_2_SEQ_LM": MultiAdapterModelForSeq2SeqLM,
    "ORIGINAL": MultiAdapterModel,
}

AGGREGAOTR_TYPE_TO_CONFIG_MAPPING = {
    "STATIC": StaticAggregatorConfig,
    "MOE": MoEAggregatorConfig,
    "DIRECT": DirectAggregatorConfig,
    "MOE2": MoEV2AggregatorConfig,
}
