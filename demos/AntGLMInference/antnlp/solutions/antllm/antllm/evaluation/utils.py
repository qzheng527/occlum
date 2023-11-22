import torch
from solutions.antllm.antllm.data import dataset
from solutions.antllm.antllm.models.llama2.modeling_llama import LlamaForCausalLM
from solutions.antllm.antllm.models.llama2.tokenization_llama import LlamaTokenizer
from solutions.antllm.antllm.inference.glm_predictor import GLMForInference
from solutions.antllm.antllm.models.glm.modeling_glm import GLMForConditionalGeneration
from solutions.antllm.antllm.models.glm.configuration_glm import GLMConfig
from solutions.antllm.antllm.models.glm.tokenization_glm import GLMTokenizer
from accelerate import load_checkpoint_and_dispatch
from accelerate import init_empty_weights
from transformers.generation import GenerationConfig


def get_dataset_by_name(name: str):
    """获取dataset"""
    return getattr(dataset, name)


def load_llama(model_path: str, device_id: int):
    model = LlamaForCausalLM.from_pretrained(model_path)
    model.half()
    with init_empty_weights():
        model = LlamaForCausalLM.from_pretrained(model_path)
    model.tie_weights()

    model = load_checkpoint_and_dispatch(
        model, model_path, device_map={"": device_id}, no_split_module_classes=["LlamaDecoderLayer"], 
        dtype=torch.float16
    )
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model.eval()
    max_length = 4090
    return model, tokenizer, max_length


def load_glm(model_path, model_size: str, device_id: int):
    if model_size == "10b":
        glm_inference_util = GLMForInference(model_path, gpu_index=device_id)
        model = glm_inference_util.model
        tokenizer = glm_inference_util.tokenizer
        max_length = glm_inference_util.training_args.get("max_length", 2048)
    elif model_size == "65b":
        config = GLMConfig.from_pretrained(model_path)
        tokenizer = GLMTokenizer.from_pretrained(model_path)  # noqa
        with init_empty_weights():
            model = GLMForConditionalGeneration(config)
        model.tie_weights()
        # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in device_ids])
        model = load_checkpoint_and_dispatch(
            model, model_path, device_map="auto", no_split_module_classes=["GLMBlock"], dtype=torch.float16
        )
        max_length = 2048
    else:
        raise Exception()
    model.eval()
    return model, tokenizer, max_length
