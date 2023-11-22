import sys
import os
import torch
import shutil
import json
from solutions.antllm.antllm.models.glm.modeling_glm import GLMForConditionalGeneration
from solutions.antllm.antllm.models.peft.modeling_peft import PeftModel


def export_hf_checkpoint(model_dir: str, output_dir: str, torch_dtype=torch.bfloat16):
    os.makedirs(output_dir, exist_ok=True)
    files_to_save = ["hyper_parameters.json", "merge.model", "merge.txt", "merge.vocab", "tokenizer_config.json"]
    base_model = GLMForConditionalGeneration.from_pretrained(model_dir, device_map={"": "cpu"}, torch_dtype=torch_dtype)
    lora_model = PeftModel.from_pretrained(base_model, model_dir, device_map={"": "cpu"},)

    # merge weights - new merging method from peft
    lora_model = lora_model.merge_and_unload()
    lora_model_sd = lora_model.state_dict()
    deloreanized_sd = {
        k.replace("base_model.model.", ""): v
        for k, v in lora_model_sd.items()
        if "lora" not in k
    }
    GLMForConditionalGeneration.save_pretrained(
        base_model, output_dir, state_dict=deloreanized_sd, max_shard_size="30GB"
    )
    for file in files_to_save:
        shutil.copy(os.path.join(model_dir, file), os.path.join(output_dir, file))
    
    with open(os.path.join(output_dir, "hyper_parameters.json"), "r") as fin:
        json_rest = json.load(fin)
        del json_rest["peft_type"]
    
    with open(os.path.join(output_dir, "hyper_parameters.json"), "w") as fout:
        json.dump(json_rest, fout, indent=2)


if __name__ == "__main__":
    model_dir = sys.argv[1]
    output_dir = sys.argv[2]
    export_hf_checkpoint(model_dir, output_dir)