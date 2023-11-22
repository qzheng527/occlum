import inspect
import os
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import transformers

import trlx.utils.logging as logging
from solutions.antllm.antllm.utils.modeling_glm_ppo_utils import is_peft_available

logger = logging.get_logger(__name__)

if is_peft_available():
    from peft import (
        PeftConfig,
        PeftModel,
        get_peft_config,
        get_peft_model,
        prepare_model_for_int8_training,
    )


class PreTrainedModelWrapper(nn.Module, transformers.utils.PushToHubMixin):
    """A wrapper around `transformers.PreTrainedModel`

    Reference: @younesbelkada's `PreTrainedModelWrapper`
    https://github.com/lvwerra/trl/blob/4f5c16fafde42d9aca971952bcdcc1f5a0a68cf0/trl/models/modeling_base.py#L2

    Attributes:
        _auto_model_parent_class (transformers.AutoModel): The `transformers.AutoModel`
            type to base the wrapping behavior off of, e.g. `transformers.AutoModelForCausalLM`.
        _supported_modules (List[str]): A list of attribute names for modules of
            the underlying architecture model. This is used, for example, to save
            and load any additional modules by manipulating the state dict.
        _supported_args (List[str]): A list of arguments specific to the underlying
            architecture to separate from arguments that are supported by the
            parent `AutoModel` class. Any arguments that are not supported by the
            underlying model will be passed to the parent `AutoModel` class.
    """

    _auto_model_parent_class: transformers.AutoModel = None
    _supported_modules: List[str] = None
    # TODO (jon-tow): Supported args should come from a `PretrainedConfig` of the
    # specific underlying type similar to how config instances can be used to instantiate
    # `transformers.PreTrainedModel`s.
    _supported_args: List[str] = []

    def __init__(
        self,
        base_model: Optional[transformers.PreTrainedModel] = None,
        peft_config=None,
        **kwargs,
    ):
        super().__init__()
        self.base_model = base_model
        # cache `forward` args for general use (avoids incompatible args across architectures)
        self.forward_kwargs = inspect.getfullargspec(self.base_model.forward).args
        self.is_loaded_in_8bit = getattr(base_model, "is_loaded_in_8bit", False)
        if self.is_loaded_in_8bit:
            # TODO(glerzing): Fully test and support loading in 8-bit
            raise NotImplementedError(
                "`is_loaded_in_8bit` is an experimental feature not yet fully supported. Please do not use it."
            )
        self.peft_config = peft_config
        self.peft_type = peft_config.peft_type if peft_config else None

    @classmethod
    def _split_kwargs(cls, kwargs: Dict[str, Any]):
        """Separates the kwargs from the supported arguments within `supported_args`
        and those that are not
        """
        supported_kwargs = {}
        unsupported_kwargs = {}
        for key, value in kwargs.items():
            if key in cls._supported_args:
                supported_kwargs[key] = value
            else:
                unsupported_kwargs[key] = value
        return supported_kwargs, unsupported_kwargs

    @classmethod
    def from_config(
        cls, config: transformers.PretrainedConfig, peft_config=None, **kwargs
    ):
        """Instantiate the pretrained pytorch model from a configuration.

        Args:
            config (transformers.PretrainedConfig): The configuration to use to
                instantiate the base model.
            peft_config (peft.PeftConfig or dict, *optional*): Configuration for the peft adapter

        NOTE: Loading a model from its configuration file does **not** load the
        model weights. It only affects the model's configuration. Use
        `~transformers.AutoModel.from_pretrained` to load the model weights.
        """
        if kwargs is not None:
            wrapped_model_kwargs, from_config_kwargs = cls._split_kwargs(kwargs)
        else:
            from_config_kwargs = {}
            wrapped_model_kwargs = {}
        base_model = cls._auto_model_parent_class.from_config(
            config, **from_config_kwargs
        )
        if peft_config:
            if isinstance(peft_config, dict):
                peft_config = get_peft_config(peft_config)
            base_model = get_peft_model(base_model, peft_config)
            wrapped_model_kwargs["peft_config"] = peft_config

        model = cls(base_model, **wrapped_model_kwargs)
        return model

    @classmethod
    def from_pretrained(  # noqa: max-complexity
        cls,
        pretrained_model_name_or_path: Union[str, transformers.PreTrainedModel],
        revision=None,
        peft_config=None,
        *model_args,
        **kwargs,
    ):
        """Instantiate a pretrained pytorch model from a pretrained model configuration.
        This method is a wrapper around `transformers.PreTrainedModel.from_pretrained`.
        Please refer to the documentation of `transformers.PreTrainedModel.from_pretrained`
        for more information.

        Args:
            pretrained_model_name_or_path (str or `transformers.PreTrainedModel`):
                The identifier of the pretrained model to load or the pretrained model itself.
            revision (str, *optional*): Optional specific Git branch, tag or commit hash.
            peft_config (peft.PeftConfig or dict, *optional*): The peft configuration to create a peft adapter.
                This is *only useful when creating a new peft adapter, not when loading an already trained adapter*.
                To load an already trained peft adapter, set `pretrained_model_name_or_path` to the directory containing
                the trained adapter, which contains at least 2 files: a config file ("adapter_config.json" by default),
                and a file containing the weights ("adapter_model.bin" by default). If there is a value head,
                it will be loaded from this directory as well.
                For additional argument to give to PeftModel.from_pretrained (such as adapter_name or subdir),
                use the dict argument `peft_from_pretrained_kwargs`. There is also a dict argument
                `peft_int8_kwargs` for specific options with 8-bit models. These arguments will be
                retrieved from kwargs.
            *model_args (sequence of positional arguments, *optional*):
                All remaining positional arguments will be passed to the `_auto_model_parent_class`.
            **kwargs (dict, *optional*):
                Dictionary of keyword arguments to pass to both the underlying `_auto_model_parent_class`
                call (e.g. `transformers.AutoModelForCausalLM.from_pretrained`) and the specific
                instance of the wrapped model.

        NOTE: You must pass in arguments specific to the wrapped model as keyword arguments.
        """
        trans_peft_config = peft_config
        if kwargs is not None:
            peft_from_pretrained_kwargs = kwargs.pop("peft_from_pretrained_kwargs", {})
            peft_int8_kwargs = kwargs.pop("peft_int8_kwargs", {})
            wrapped_model_kwargs, from_pretrained_kwargs = cls._split_kwargs(kwargs)
        else:
            peft_from_pretrained_kwargs = {}
            peft_int8_kwargs = {}
            from_pretrained_kwargs = {}
            wrapped_model_kwargs = {}

        if isinstance(pretrained_model_name_or_path, str):
            is_loaded_in_8bit = (
                from_pretrained_kwargs["load_in_8bit"]
                if "load_in_8bit" in from_pretrained_kwargs
                else False
            )
        else:
            is_loaded_in_8bit = getattr(
                pretrained_model_name_or_path, "is_loaded_in_8bit", False
            )

        if is_loaded_in_8bit:
            # TODO(glerzing): Fully test and support loading in 8-bit
            raise NotImplementedError(
                "`is_loaded_in_8bit` is an experimental feature not yet fully supported. Please do not use it."
            )

        if peft_config is not None:
            if not is_peft_available():
                raise ModuleNotFoundError(
                    "To use the argument peft_config, please install `peft`"
                )
            if not isinstance(peft_config, PeftConfig):
                if isinstance(peft_config, dict):
                    peft_config = get_peft_config(peft_config)
                    trans_peft_config = get_peft_config(trans_peft_config)
                else:
                    raise ValueError(
                        "`peft_config` should be an instance of `peft.PeftConfig` or a dict."
                    )

        if isinstance(pretrained_model_name_or_path, str):
            # Check if there is a local peft adapter
            local_peft_adapter = os.path.exists(
                os.path.join(pretrained_model_name_or_path, "adapter_config.json")
            )

            if local_peft_adapter and not is_peft_available():
                logger.warning(
                    "WARNING: peft adapter detected but peft is not installed. Ignoring the adapter..."
                )

            base_model = None
            if is_peft_available():
                try:
                    trained_adapter_config = PeftConfig.from_pretrained(
                        pretrained_model_name_or_path
                    )
                except ValueError:
                    trained_adapter_config = None

                if trained_adapter_config is not None:
                    if peft_config is not None:
                        logger.warning(
                            f"WARNING: peft config file detected in {pretrained_model_name_or_path}"
                            " also the argument `peft_config` is provided, so peft_config will be ignored. Remove the"
                            " argument `peft_config` to use the trained peft adapter."
                        )
                    peft_config = trained_adapter_config

                    # Use the pretrained (local or remote) peft adapter file "adapter_config.json"
                    base_model = cls._auto_model_parent_class.from_pretrained(
                        pretrained_model_name_or_path,
                        *model_args,
                        **from_pretrained_kwargs,
                    )

                    # Load the peft weights in "adapter_model.bin" and wrap the base model with a PeftModel
                    base_model = PeftModel.from_pretrained(
                        base_model,
                        pretrained_model_name_or_path,
                        is_trainable=True,  # if load from pretrined peft model, set to true by default
                        **peft_from_pretrained_kwargs,
                    )

                    # if lora model, merges the LoRa layers into the base model, so reference should not be sattled
                    if "LORA" in peft_config.peft_type:
                        merged_lora_model = base_model.base_model.merge_and_unload()
                        logger.info("Lora model merged")
                        if trans_peft_config is not None:
                            base_model = get_peft_model(merged_lora_model, trans_peft_config)
                        else:
                            base_model = merged_lora_model
                        peft_config = trans_peft_config

                    logger.info("Trained peft adapter loaded")

                elif peft_config is not None:
                    # Create a new peft adapter with the given config
                    base_model = cls._auto_model_parent_class.from_pretrained(
                        pretrained_model_name_or_path,
                        *model_args,
                        **from_pretrained_kwargs,
                    )

                    if is_loaded_in_8bit:
                        base_model = prepare_model_for_int8_training(
                            base_model,
                            **peft_int8_kwargs,
                        )
                    base_model = get_peft_model(base_model, peft_config)
                    logger.info("peft adapter initialised")

            if base_model is None:
                # No peft
                base_model = cls._auto_model_parent_class.from_pretrained(
                    pretrained_model_name_or_path, *model_args, **from_pretrained_kwargs
                )

        elif isinstance(pretrained_model_name_or_path, transformers.PreTrainedModel):
            base_model = pretrained_model_name_or_path

            if peft_config is not None:
                if is_loaded_in_8bit:
                    base_model = prepare_model_for_int8_training(
                        base_model,
                        **peft_int8_kwargs,
                    )
                base_model = get_peft_model(base_model, peft_config)
                logger.info("peft adapter initialised")
        else:
            raise ValueError(
                f"Invalid type for `base_model_name_or_path`: {type(pretrained_model_name_or_path)}"
                "Expected `str` or `transformers.PreTrainedModel`."
            )

        if peft_config is not None:
            wrapped_model_kwargs["peft_config"] = peft_config

        model = cls(base_model, **wrapped_model_kwargs)

        if isinstance(pretrained_model_name_or_path, str):
            filename = os.path.join(pretrained_model_name_or_path, "head.bin")

            if not os.path.exists(filename):
                state_dict = {}
            else:
                # Merge each shard into a state dict
                # TODO: Optimize this to avoid wasting RAM
                state_dict = {}
                state_dict = torch.load(filename, map_location="cpu")
        else:
            state_dict = pretrained_model_name_or_path.state_dict()

        model.post_init(state_dict=state_dict)
        return model

    def save_pretrained(self, *args, **kwargs):
        """Save the pretrained model to a directory. This method is a wrapper
        around `transformers.PreTrainedModel.save_pretrained`. Please refer to
        the documentation of `transformers.PreTrainedModel.save_pretrained` for
        more information.

        Args:
            *args (`list`, *optional*):
                Positional arguments passed along to the underlying model's
                `save_pretrained` method.
            **kwargs (`dict`, *optional*):
                Keyword arguments passed along to the underlying model's
                `save_pretrained` method.
        """
        state_dict = kwargs.get("state_dict", None)
        if state_dict is None:
            state_dict = self.state_dict()
            kwargs["state_dict"] = state_dict

        if self.peft_type:
            # Save the heads, which are not part of the peft adapter
            os.makedirs(args[0], exist_ok=True)
            save_path = os.path.join(args[0], "head.bin")
            head_state_dict = self.state_dict(heads_only=True)

            torch.save(head_state_dict, save_path)

            # also save the base model itself
            state_dict = self.base_model.base_model.model.state_dict()
            # Filter the peft params ...
            param_keys = list(state_dict.keys())
            peft_param_prefix = "base_model.model."
            for key in param_keys:
                if "lora" in key:
                    state_dict.pop(key)
                elif peft_param_prefix in key:
                    value = state_dict.pop(key)
                    new_key = key.replace(peft_param_prefix, "")
                    state_dict[new_key] = value

            torch.save(state_dict, os.path.join(args[0], "pytorch_model.bin"))

        return self.base_model.save_pretrained(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        """Return the state_dict of the pretrained model."""
        raise NotImplementedError

    def post_init(self, *args, **kwargs):
        """Post initialization method. This method is called after the model is
        instantiated and loaded from a checkpoint. It can be used to perform
        additional operations such as loading the state_dict.
        """
        if self.peft_type:
            # Don't use the interface of the peft model,
            # use the interface of the underlying transformer model instead.
            # (peft adds 2 "base_model" layers)
            self.forward_kwargs = inspect.getfullargspec(
                self.base_model.base_model.base_model.forward
            ).args

    def get_compatible_forward_kwargs(self, **kwargs) -> Dict[str, Any]:
        """Filter out arguments not supported by the specific instance of
        `base_model.transformer.forward`
        """
        # FIXME: This is a hack to get around the fact that the `transformers`
        # architectures we use don't have a consistent API for `forward` parameters.
        return {k: v for k, v in kwargs.items() if k in self.forward_kwargs}
