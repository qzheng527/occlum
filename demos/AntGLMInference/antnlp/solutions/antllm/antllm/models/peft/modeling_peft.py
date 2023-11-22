#!/usr/bin/env python
# coding=utf-8
# @Author: tianxuan.jl
# @Date: Wed 10 May 2023 05:35:48 PM CST

import os
import torch
import inspect
import warnings
from accelerate.utils import get_balanced_memory
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules

from peft.peft_model import (
    PeftModel,
    PeftModelForCausalLM,
)
from peft.utils import (
    PeftConfig,
    PromptLearningConfig,
    _set_trainable,
)
from solutions.antllm.antllm.models.peft.utils import (
    WEIGHTS_NAME,
    get_peft_model_state_dict,
    set_peft_model_state_dict
)
from solutions.antllm.antllm.models.peft.tuner import ( # noqa
    AdaLoraModel,
    RouteLoraModel,
    PeftType,
    PEFT_TYPE_TO_MODEL_MAPPING
)


class AntPeftForCausalLM(PeftModelForCausalLM):
    def __init__(self, model, peft_config: PeftConfig, adapter_name: str = "default"):
        super(PeftModel, self).__init__()
        self.base_model = model
        self.config = self.base_model.config
        self.modules_to_save = None
        self.peft_config = {}
        self.active_adapter = adapter_name
        self.peft_type = peft_config.peft_type
        self.base_model_torch_dtype = getattr(model, "dtype", None)
        if not isinstance(peft_config, PromptLearningConfig):
            self.peft_config[adapter_name] = peft_config
            self.base_model = PEFT_TYPE_TO_MODEL_MAPPING[peft_config.peft_type](
                self.base_model, self.peft_config, adapter_name
            )
            self.set_additional_trainable_modules(peft_config, adapter_name)
        else:
            self.add_adapter(adapter_name, peft_config)

        if getattr(self.peft_config[adapter_name], "modules_to_save", None) is not None:
            self.modules_to_save = self.peft_config[adapter_name].modules_to_save
            _set_trainable(self, adapter_name)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation

    def set_route_id(self, route_id: int):
        peft_config = self.active_peft_config
        if peft_config.peft_type == PeftType.ROUTELORA:
            self.base_model.activate_route_lora(route_id)
        else:
            warnings.warn("The route setting only support for Route Lora method,"
                          f"but the current method is {peft_config.peft_type}")

    def expand_external_router(self, path: str):
        peft_config = self.active_peft_config
        if peft_config.peft_type == PeftType.ROUTELORA:
            self.base_model.expand_external_router(path)
        else:
            warnings.warn("The route setting only support for Route Lora method,"
                          f"but the current method is {peft_config.peft_type}")        

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        route_id: int = 0,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config
        if not isinstance(peft_config, PromptLearningConfig):
            if peft_config.peft_type == PeftType.ROUTELORA:
                self.base_model.activate_route_lora(route_id)

            return self.base_model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            # concat prompt attention mask
            if len(attention_mask.size()) == 2:
                prefix_attention_mask = torch.ones(
                    batch_size, peft_config.num_virtual_tokens).to(self.device)
                attention_mask = torch.cat(
                    (prefix_attention_mask, attention_mask), dim=1)
            elif len(attention_mask.size()) == 1:
                pass
            else:
                assert ValueError(
                    f"The size of attention mask must in 1 or 2 dim, but get {len(attention_mask.size())}")

        # if kwargs.get("position_ids", None) is not None:
        #     warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
        #     kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn(
                "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "position_ids": position_ids,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        model_config = self.base_model.config.to_dict()
        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size)
            if model_config["model_type"] == "glm":
                batch_size = past_key_values[0].size(1)
                past_key_values = [feat.sum(dim=0).permute(
                    0, 2, 1, 3).contiguous() for feat in past_key_values]
                past_key_values = [
                    feat.view(
                        batch_size, peft_config.num_virtual_tokens, -1)
                    for feat in past_key_values
                ]
                if model_config["block_position_encoding"] is True:
                    position_ids[:, 0] += peft_config.num_virtual_tokens
                else:
                    position_ids += peft_config.num_virtual_tokens
                kwargs["position_ids"] = position_ids.contiguous()

                return self.base_model(input_ids=input_ids, mems=past_key_values, **kwargs)

            return self.base_model(input_ids=input_ids, past_key_values=past_key_values, **kwargs)
        else:
            # TODO: support p-tuning and prompt tuning for GLM
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            # concat prompt labels
            if labels is not None:
                prefix_labels = torch.full(
                    (batch_size, peft_config.num_virtual_tokens), -100).to(self.device)
                kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)

            attention_mask = attention_mask + peft_config.num_virtual_tokens

            virtual_token_position_ids = torch.arange(
                peft_config.num_virtual_tokens, device=self.device, dtype=torch.long).unsqueeze(0)
            virtual_token_position_ids = virtual_token_position_ids.expand(batch_size, -1)
            if model_config["block_position_encoding"] is True:
                position_ids[:, 0] += peft_config.num_virtual_tokens
                virtual_token_position_ids = virtual_token_position_ids.unsqueeze(1)
                block_virtual_token_position_ids = position_ids.new_zeros((
                    batch_size, 1, peft_config.num_virtual_tokens))
                position_ids = torch.cat([
                    torch.cat([virtual_token_position_ids, block_virtual_token_position_ids], dim=1),
                    position_ids], dim=-1
                )

            else:
                position_ids += peft_config.num_virtual_tokens
                position_ids = torch.cat([virtual_token_position_ids, position_ids], dim=-1)
            
            kwargs["attention_mask"] = attention_mask.contiguous()
            kwargs["position_ids"] = position_ids.contiguous()

            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def save_pretrained(self, save_directory, **kwargs):
        r"""
        This function saves the adapter model and the adapter configuration files to a directory, so that it can be
        reloaded using the [`LoraModel.from_pretrained`] class method, and also used by the [`LoraModel.push_to_hub`]
        method.

        Args:
            save_directory (`str`):
                Directory where the adapter model and configuration files will be saved (will be created if it does not
                exist).
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the `push_to_hub` method.
        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)

        for adapter_name, peft_config in self.peft_config.items():
            # save only the trainable weights
            output_state_dict = get_peft_model_state_dict(
                self, state_dict=kwargs.get("state_dict", None), adapter_name=adapter_name
            )
            output_dir = os.path.join(save_directory, adapter_name) if adapter_name != "default" else save_directory
            os.makedirs(output_dir, exist_ok=True)
            torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            # save the config and change the inference mode to `True`
            if peft_config.base_model_name_or_path is None:
                peft_config.base_model_name_or_path = (
                    self.base_model.__dict__.get("name_or_path", None)
                    if isinstance(peft_config, PromptLearningConfig)
                    else self.base_model.model.__dict__.get("name_or_path", None)
                )
            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True
            peft_config.save_pretrained(output_dir)
            peft_config.inference_mode = inference_mode

    def load_adapter(self, model_id, adapter_name, is_trainable=False, **kwargs):
        """
        Rewrite the load_adapter fuc from the Peft repo for AntGLM,
        which support the [`xxx.from_pretrained()`] method for petuing,
        routelora, and adalora.

        Args:
            model_id (`str`):
                Directory where the model saved.
            adapter_name (`str`):
                The adapter name use for identify the peft params.
            is_trainable (`bool`):
                Whether the model is used for training.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the `hf_hub_download` method.            
        """
        from .tuner import PEFT_TYPE_TO_CONFIG_MAPPING
        from huggingface_hub import hf_hub_download

        if adapter_name not in self.peft_config:
            # load the config
            peft_config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig.from_pretrained(model_id, subfolder=kwargs.get("subfolder", None)).peft_type
            ].from_pretrained(model_id, subfolder=kwargs.get("subfolder", None))
            if isinstance(peft_config, PromptLearningConfig) and is_trainable:
                raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
            else:
                peft_config.inference_mode = not is_trainable
            self.add_adapter(adapter_name, peft_config)

        # load weights if any
        path = os.path.join(model_id, kwargs["subfolder"]) if kwargs.get("subfolder", None) is not None else model_id

        if os.path.exists(os.path.join(path, WEIGHTS_NAME)):
            filename = os.path.join(path, WEIGHTS_NAME)
        else:
            try:
                filename = hf_hub_download(model_id, WEIGHTS_NAME, subfolder=kwargs.get("subfolder", None))
            except:  # noqa
                raise ValueError(
                    f"Can't find weights for {model_id} in {model_id} or in the Hugging Face Hub. "
                    f"Please check that the file {WEIGHTS_NAME} is present at {model_id}."
                )

        adapters_weights = torch.load(
            filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # load the weights into the model
        set_peft_model_state_dict(self, adapters_weights, adapter_name=adapter_name)
        if (
            (getattr(self, "hf_device_map", None) is not None)
            and (len(set(self.hf_device_map.values()).intersection({"cpu", "disk"})) > 0)
            and len(self.peft_config) == 1
        ):
            device_map = kwargs.get("device_map", "auto")
            max_memory = kwargs.get("max_memory", None)
            offload_dir = kwargs.get("offload_folder", None)
            offload_index = kwargs.get("offload_index", None)

            dispatch_model_kwargs = {}
            # Safety checker for previous `accelerate` versions
            # `offload_index` was introduced in https://github.com/huggingface/accelerate/pull/873/
            if "offload_index" in inspect.signature(dispatch_model).parameters:
                dispatch_model_kwargs["offload_index"] = offload_index

            no_split_module_classes = self._no_split_modules

            if device_map != "sequential":
                max_memory = get_balanced_memory(
                    self,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                    low_zero=(device_map == "balanced_low_0"),
                )
            if isinstance(device_map, str):
                device_map = infer_auto_device_map(
                    self, max_memory=max_memory, no_split_module_classes=no_split_module_classes
                )
            dispatch_model(
                self,
                device_map=device_map,
                offload_dir=offload_dir,
                **dispatch_model_kwargs,
            )
            hook = AlignDevicesHook(io_same_device=True)
            if isinstance(self.peft_config[adapter_name], PromptLearningConfig):
                remove_hook_from_submodules(self.prompt_encoder)
            add_hook_to_module(self.get_base_model(), hook)

        # Set model in evaluation mode to deactivate Dropout modules by default
        self.eval()

    @classmethod
    def from_pretrained(
        cls,
        model,
        model_id: str,
        adapter_name: str = "default",
        is_trainable: bool = False,
        resume_from_checkpoint: bool = False,
        **kwargs
    ):
        r"""
        Instantiate a [`LoraModel`] from a pretrained Lora configuration and weights.

        Args:
            model ([`~transformers.PreTrainedModel`]):
                The model to be adapted. The model should be initialized with the
                [`~transformers.PreTrainedModel.from_pretrained`] method from the 🤗 Transformers library.
            model_id (`str` or `os.PathLike`):
                The name of the Lora configuration to use. Can be either:
                    - A string, the `model id` of a Lora configuration hosted inside a model repo on the Hugging Face
                      Hub.
                    - A path to a directory containing a Lora configuration file saved using the `save_pretrained`
                      method (`./my_lora_config_directory/`).
        """
        from .tuner import PEFT_TYPE_TO_CONFIG_MAPPING
        from peft import MODEL_TYPE_TO_PEFT_MODEL_MAPPING
        from transformers.trainer_utils import get_last_checkpoint

        if resume_from_checkpoint is True:
            last_ckpt_id = None
            last_ckpt_id = get_last_checkpoint(model_id)
            if last_ckpt_id is None:
                last_ckpt_id = get_last_checkpoint(os.path.join(model_id, "epochs"))
            if last_ckpt_id is not None:
                model_id = last_ckpt_id
            model.resume_ckpt_dir = model_id
            print(f"The last checkpoint path is: {model_id}")

        # load the config
        config = PEFT_TYPE_TO_CONFIG_MAPPING[
            PeftConfig.from_pretrained(model_id, subfolder=kwargs.get("subfolder", None)).peft_type
        ].from_pretrained(model_id, subfolder=kwargs.get("subfolder", None))

        if (getattr(model, "hf_device_map", None) is not None) and len(
            set(model.hf_device_map.values()).intersection({"cpu", "disk"})
        ) > 0:
            remove_hook_from_submodules(model)

        if isinstance(config, PromptLearningConfig) and is_trainable:
            raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
        else:
            config.inference_mode = not is_trainable

        if config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys():
            model = cls(model, config, adapter_name)
        else:
            model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type](model, config, adapter_name)
        model.load_adapter(model_id, adapter_name, **kwargs)
        return model

    def generate(self, **kwargs):
        """
        Override the `Generate` fuction for AntGLM from `PeftModel`,
        which support the position ids for prepare generation inputs.
        """
        peft_config = self.active_peft_config
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        try:
            if not isinstance(peft_config, PromptLearningConfig):
                outputs = self.base_model.generate(**kwargs)
            else:
                model_config = self.base_model.config.to_dict()
                if "input_ids" not in kwargs:
                    raise ValueError("input_ids must be provided for Peft model generation")
                # For gpt2 models, we construct postion_ids on the fly by using attention mask,
                # and position ids need to match input_shape.
                # for prefix tuning, input shape is determined using `input_ids`.
                # Thus we should not expand 'attention_mask' here
                # for prompt tuning input_ids is not passed but a concatenated input_embeds is passed.
                # Thus attention_mask needs to be of same size of num_virtual_tokens + input_ids
                if kwargs.get("attention_mask", None) is not None and model_config["model_type"] != "glm" \
                        and peft_config.peft_type in [PeftType.PROMPT_TUNING, PeftType.P_TUNING]:
                    # concat prompt attention mask
                    prefix_attention_mask = torch.ones(
                        kwargs["input_ids"].shape[0], peft_config.num_virtual_tokens
                    ).to(kwargs["input_ids"].device)
                    kwargs["attention_mask"] = torch.cat((prefix_attention_mask, kwargs["attention_mask"]), dim=1)

                if kwargs.get("token_type_ids", None) is not None:
                    warnings.warn(
                        "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                    )
                    kwargs["token_type_ids"] = None

                outputs = self.base_model.generate(**kwargs)
        except Exception:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        generation_attention_mask: torch.Tensor = None,
        *args,
        **kwargs
    ):
        """
        The generation inputs preprocess for AntGLM.
        """

        kwargs.update(
            {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "generation_attention_mask": generation_attention_mask,
            }
        )

        batch_size = kwargs["input_ids"].size(0)
        seq_length = kwargs["input_ids"].shape[1]
        peft_config = self.active_peft_config
        model_config = self.base_model.config.to_dict()
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        if isinstance(peft_config, PromptLearningConfig):
            # handle the attention mask for generation in PromptLearning method
            attention_mask = model_kwargs["attention_mask"]
            # concat prompt attention mask
            if len(attention_mask.size()) == 4:
                prefix_attention_mask = torch.ones(
                    batch_size, 1, attention_mask.size(2), peft_config.num_virtual_tokens).to(self.device)
                attention_mask = torch.cat(
                    (prefix_attention_mask, attention_mask), dim=-1)
            else:
                raise ValueError(
                    f"The size of attention mask must in 4 dim, but get {len(attention_mask.size())}")
            model_kwargs["attention_mask"] = attention_mask

            if (model_kwargs.get("mems", None) is None and model_kwargs.get("past_key_values", None) is None) \
                    and peft_config.peft_type == PeftType.PREFIX_TUNING:
                past_key_values = self.get_prompt(batch_size=batch_size)

                if model_config["model_type"] == "glm":
                    # handle the past_key_values for AntGLM prefix-tuning
                    past_key_values = [feat.sum(dim=0).permute(
                        0, 2, 1, 3).contiguous() for feat in past_key_values]
                    past_key_values = [
                        feat.view(
                            batch_size, peft_config.num_virtual_tokens, -1)
                        for feat in past_key_values
                    ]
                    if model_config["block_position_encoding"] is True:
                        position_ids[:, 0] += peft_config.num_virtual_tokens
                    else:
                        position_ids += peft_config.num_virtual_tokens
                    model_kwargs["position_ids"] = position_ids[..., :seq_length].contiguous()
                    model_kwargs["mems"] = past_key_values
                    
                else:
                    if self.base_model_torch_dtype is not None:
                        # handle the case for Bloom where it outputs tuple of tuples
                        if isinstance(past_key_values[0], tuple):
                            past_key_values = tuple(
                                tuple(
                                    past_key_value.to(self.base_model_torch_dtype)
                                    for past_key_value in past_key_value_tuple
                                )
                                for past_key_value_tuple in past_key_values
                            )
                        else:
                            past_key_values = tuple(
                                past_key_value.to(self.base_model_torch_dtype) for past_key_value in past_key_values
                            )

                    model_kwargs["past_key_values"] = past_key_values
            else:
                if (model_kwargs.get("mems", None) is None and model_kwargs.get("past_key_values", None) is None):
                    if model_config["model_type"] == "glm":
                        # handle the attention mask for generation in PromptLearning method
                        attention_mask = model_kwargs["attention_mask"]
                        # concat prompt attention mask
                        prefix_attention_mask = torch.ones(
                            batch_size, 1, peft_config.num_virtual_tokens, attention_mask.size(3)).to(self.device)
                        attention_mask = torch.cat(
                            (prefix_attention_mask, attention_mask), dim=-2)
                        model_kwargs["attention_mask"] = attention_mask

                        # handle the case for AntGLM where it use the block-position-ids and generation attention mask
                        # handle the block position ids
                        position_ids = model_kwargs["position_ids"]
                        position_ids = position_ids[..., :seq_length].contiguous()
                        virtual_token_position_ids = torch.arange(
                            peft_config.num_virtual_tokens, device=self.device, dtype=torch.long).unsqueeze(0)
                        virtual_token_position_ids = virtual_token_position_ids.expand(batch_size, -1)
                        if model_config["block_position_encoding"] is True:
                            position_ids[:, 0] += peft_config.num_virtual_tokens
                            virtual_token_position_ids = virtual_token_position_ids.unsqueeze(1)
                            block_virtual_token_position_ids = position_ids.new_zeros((
                                batch_size, 1, peft_config.num_virtual_tokens))
                            position_ids = torch.cat([
                                torch.cat([virtual_token_position_ids, block_virtual_token_position_ids], dim=1),
                                position_ids], dim=-1
                            )
                        else:
                            position_ids += peft_config.num_virtual_tokens
                            position_ids = torch.cat([virtual_token_position_ids, position_ids], dim=-1)
                        model_kwargs["position_ids"] = position_ids

                    inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
                    prompts = self.get_prompt(batch_size=batch_size)
                    prompts = prompts.to(inputs_embeds.dtype)
                    model_kwargs["inputs_embeds"] = torch.cat((prompts, inputs_embeds), dim=1)
                    model_kwargs["input_ids"] = None
                else:
                    position_ids = model_kwargs["position_ids"]
                    position_ids = position_ids[..., 0] + peft_config.num_virtual_tokens

        return model_kwargs


class AntPeftForEmbedding(PeftModel):
    def __init__(self, model, peft_config: PeftConfig, adapter_name: str = "default"):
        super(PeftModel, self).__init__()
        self.base_model = model
        self.config = self.base_model.config
        self.modules_to_save = None
        self.peft_config = {}
        self.active_adapter = adapter_name
        self.peft_type = peft_config.peft_type
        self.base_model_torch_dtype = getattr(model, "dtype", None)
        if not isinstance(peft_config, PromptLearningConfig):
            self.peft_config[adapter_name] = peft_config
            self.base_model = PEFT_TYPE_TO_MODEL_MAPPING[peft_config.peft_type](
                self.base_model, self.peft_config, adapter_name
            )
            self.set_additional_trainable_modules(peft_config, adapter_name)
        else:
            self.add_adapter(adapter_name, peft_config)

        if getattr(self.peft_config[adapter_name], "modules_to_save", None) is not None:
            self.modules_to_save = self.peft_config[adapter_name].modules_to_save
            _set_trainable(self, adapter_name)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation

    def set_route_id(self, route_id: int):
        peft_config = self.active_peft_config
        if peft_config.peft_type == PeftType.ROUTELORA:
            self.base_model.activate_route_lora(route_id)
        else:
            warnings.warn("The route setting only support for Route Lora method,"
                          f"but the current method is {peft_config.peft_type}")

    def expand_external_router(self, path: str):
        peft_config = self.active_peft_config
        if peft_config.peft_type == PeftType.ROUTELORA:
            self.base_model.expand_external_router(path)
        else:
            warnings.warn("The route setting only support for Route Lora method,"
                          f"but the current method is {peft_config.peft_type}")        

    def forward(
        self,
        query_ids: torch.Tensor,
        query_position_ids: torch.Tensor = None,
        query_attention_mask: torch.Tensor = None,
        query_mask: torch.Tensor = None,
        passage_ids: torch.Tensor = None,
        passage_position_ids: torch.Tensor = None,
        passage_attention_mask: torch.Tensor = None,
        passage_mask: torch.Tensor = None,
        route_id: int = 0,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        reduction: str = "mean",
        return_dict=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config
        if not isinstance(peft_config, PromptLearningConfig):
            if peft_config.peft_type == PeftType.ROUTELORA:
                self.base_model.activate_route_lora(route_id)

            return self.base_model(
                query_ids=query_ids,
                query_position_ids=query_position_ids,
                query_attention_mask=query_attention_mask,
                query_mask=query_mask,
                passage_ids=passage_ids,
                passage_position_ids=passage_position_ids,
                passage_attention_mask=passage_attention_mask,
                passage_mask=passage_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                reduction=reduction,
                return_dict=return_dict,
                **kwargs,
            )

        batch_size = query_ids.shape[0]
        if query_attention_mask is not None and passage_attention_mask is not None:
            # concat prompt attention mask
            if len(query_attention_mask.size()) == 2:
                prefix_attention_mask = torch.ones(
                    batch_size, peft_config.num_virtual_tokens).to(self.device)
                query_attention_mask = torch.cat(
                    (prefix_attention_mask, query_attention_mask), dim=1)
                passage_attention_mask = torch.cat(
                    (prefix_attention_mask, passage_attention_mask), dim=1)
            elif len(query_attention_mask.size()) == 1:
                for i in range(batch_size):
                    query_attention_mask[i] += peft_config.num_virtual_tokens
                    passage_attention_mask[i] += passage_attention_mask.num_virtual_tokens
                query_attention_mask = query_attention_mask.contiguous()
                passage_attention_mask = passage_attention_mask.contiguous()
            else:
                assert ValueError(
                    f"The size of attention mask must in 1 or 2 dim, "
                    f"but get {len(query_attention_mask.size())} and {len(passage_attention_mask.size())}"
                )

        # if kwargs.get("position_ids", None) is not None:
        #     warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
        #     kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn(
                "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "query_position_ids": query_position_ids,
                "query_attention_mask": query_attention_mask,
                "query_mask": query_mask,
                "passage_ids": passage_ids,
                "passage_position_ids": passage_position_ids,
                "passage_attention_mask": passage_attention_mask,
                "passage_mask": passage_mask,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size)
            model_config = self.base_model.config.to_dict()
            if model_config["model_type"] == "glm":
                batch_size = past_key_values[0].size(1)
                past_key_values = [feat[0].permute(
                    0, 2, 1, 3).contiguous() for feat in past_key_values]
                past_key_values = [
                    feat.view(
                        batch_size, peft_config.num_virtual_tokens, -1)
                    for feat in past_key_values
                ]
                if model_config["block_position_encoding"] is True:
                    query_position_ids[:, 0] += peft_config.num_virtual_tokens
                    passage_position_ids[:, 0] += peft_config.num_virtual_tokens
                else:
                    query_position_ids += peft_config.num_virtual_tokens
                    passage_position_ids += peft_config.num_virtual_tokens
                kwargs["query_position_ids"] = query_position_ids.contiguous()
                kwargs["passage_position_ids"] = passage_position_ids.contiguous()
                return self.base_model(
                    query_ids=query_ids, query_mems=past_key_values, passage_mems=past_key_values, **kwargs)

            return self.base_model(query_ids=query_ids, past_key_values=past_key_values, **kwargs)
        else:
            # TODO: support p-tuning and prompt tuning for GLM
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(query_ids)
            # concat prompt labels
            # if labels is not None:
            #     prefix_labels = torch.full(
            #         (batch_size, peft_config.num_virtual_tokens), -100).to(self.device)
            #     kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def save_pretrained(self, save_directory, **kwargs):
        r"""
        This function saves the adapter model and the adapter configuration files to a directory, so that it can be
        reloaded using the [`LoraModel.from_pretrained`] class method, and also used by the [`LoraModel.push_to_hub`]
        method.

        Args:
            save_directory (`str`):
                Directory where the adapter model and configuration files will be saved (will be created if it does not
                exist).
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the `push_to_hub` method.
        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)

        for adapter_name, peft_config in self.peft_config.items():
            # save only the trainable weights
            output_state_dict = get_peft_model_state_dict(
                self, state_dict=kwargs.get("state_dict", None), adapter_name=adapter_name
            )
            output_dir = os.path.join(save_directory, adapter_name) if adapter_name != "default" else save_directory
            os.makedirs(output_dir, exist_ok=True)
            torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            # save the config and change the inference mode to `True`
            if peft_config.base_model_name_or_path is None:
                peft_config.base_model_name_or_path = (
                    self.base_model.__dict__.get("name_or_path", None)
                    if isinstance(peft_config, PromptLearningConfig)
                    else self.base_model.model.__dict__.get("name_or_path", None)
                )
            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True
            peft_config.save_pretrained(output_dir)
            peft_config.inference_mode = inference_mode

    def load_adapter(self, model_id, adapter_name, is_trainable=False, **kwargs):
        """
        Rewrite the load_adapter fuc from the Peft repo for AntGLM,
        which support the [`xxx.from_pretrained()`] method for petuing,
        routelora, and adalora.

        Args:
            model_id (`str`):
                Directory where the model saved.
            adapter_name (`str`):
                The adapter name use for identify the peft params.
            is_trainable (`bool`):
                Whether the model is used for training.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the `hf_hub_download` method.            
        """
        from .tuner import PEFT_TYPE_TO_CONFIG_MAPPING
        from huggingface_hub import hf_hub_download

        if adapter_name not in self.peft_config:
            # load the config
            peft_config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig.from_pretrained(model_id, subfolder=kwargs.get("subfolder", None)).peft_type
            ].from_pretrained(model_id, subfolder=kwargs.get("subfolder", None))
            if isinstance(peft_config, PromptLearningConfig) and is_trainable:
                raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
            else:
                peft_config.inference_mode = not is_trainable
            self.add_adapter(adapter_name, peft_config)

        # load weights if any
        path = os.path.join(model_id, kwargs["subfolder"]) if kwargs.get("subfolder", None) is not None else model_id

        if os.path.exists(os.path.join(path, WEIGHTS_NAME)):
            filename = os.path.join(path, WEIGHTS_NAME)
        else:
            try:
                filename = hf_hub_download(model_id, WEIGHTS_NAME, subfolder=kwargs.get("subfolder", None))
            except:  # noqa
                raise ValueError(
                    f"Can't find weights for {model_id} in {model_id} or in the Hugging Face Hub. "
                    f"Please check that the file {WEIGHTS_NAME} is present at {model_id}."
                )

        adapters_weights = torch.load(
            filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # load the weights into the model
        set_peft_model_state_dict(self, adapters_weights, adapter_name=adapter_name)
        if (
            (getattr(self, "hf_device_map", None) is not None)
            and (len(set(self.hf_device_map.values()).intersection({"cpu", "disk"})) > 0)
            and len(self.peft_config) == 1
        ):
            device_map = kwargs.get("device_map", "auto")
            max_memory = kwargs.get("max_memory", None)
            offload_dir = kwargs.get("offload_folder", None)
            offload_index = kwargs.get("offload_index", None)

            dispatch_model_kwargs = {}
            # Safety checker for previous `accelerate` versions
            # `offload_index` was introduced in https://github.com/huggingface/accelerate/pull/873/
            if "offload_index" in inspect.signature(dispatch_model).parameters:
                dispatch_model_kwargs["offload_index"] = offload_index

            no_split_module_classes = self._no_split_modules

            if device_map != "sequential":
                max_memory = get_balanced_memory(
                    self,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                    low_zero=(device_map == "balanced_low_0"),
                )
            if isinstance(device_map, str):
                device_map = infer_auto_device_map(
                    self, max_memory=max_memory, no_split_module_classes=no_split_module_classes
                )
            dispatch_model(
                self,
                device_map=device_map,
                offload_dir=offload_dir,
                **dispatch_model_kwargs,
            )
            hook = AlignDevicesHook(io_same_device=True)
            if isinstance(self.peft_config[adapter_name], PromptLearningConfig):
                remove_hook_from_submodules(self.prompt_encoder)
            add_hook_to_module(self.get_base_model(), hook)

        # Set model in evaluation mode to deactivate Dropout modules by default
        self.eval()
