import math
import re
import warnings
from dataclasses import dataclass, field
from alps.util import logger
from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
from peft.import_utils import is_bnb_available
import peft

from peft.utils import (
    PeftConfig,
    PeftType,
    _get_submodules,
    transpose,
)
from peft import LoraModel

from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING
from ..utils.aggregator import (
    AGGREGATOR_MAPPING,
    StaticAggregatorConfig,
    AggregatorConfig,
    MoEAggregatorConfig,
    MoEV2AggregatorConfig,
)
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING


if is_bnb_available():
    import bitsandbytes as bnb

# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class MultiLoraModel(LoraModel):
    r"""
    改编自LoraModel
    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    def __init__(
        self,
        model,
        config: List[PeftConfig],
        adapter_name: List[str],
        aggregator_config: AggregatorConfig,
    ):
        torch.nn.Module.__init__(self)
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        for adapter_namee in adapter_name:
            logger.info(f"{adapter_namee} weights start initializing...")
            self.add_adapter(adapter_namee, self.peft_config[adapter_namee])
            logger.info(f"{adapter_namee} weights initialized")
        logger.info("aggregator starts initializing")
        self.add_aggregator(aggregator_config)
        logger.info("aggregator initialized")

    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(lora_config.target_modules, str):
                target_module_found = re.fullmatch(lora_config.target_modules, key)
            else:
                target_module_found = any(
                    key.endswith(target_key)
                    for target_key in lora_config.target_modules
                )
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(self.model, key)
                if hasattr(target, "bias"):
                    bias = target.bias is not None

                if isinstance(target, MultiLoraLayer):
                    target.update_layer(
                        adapter_name,
                        lora_config.r,
                        lora_config.lora_alpha,
                        lora_config.lora_dropout,
                        lora_config.init_lora_weights,
                    )
                else:
                    if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                        eightbit_kwargs = kwargs.copy()
                        eightbit_kwargs.update(
                            {
                                "has_fp16_weights": target.state.has_fp16_weights,
                                "memory_efficient_backward": target.state.memory_efficient_backward,
                                "threshold": target.state.threshold,
                                "index": target.index,
                            }
                        )
                        new_module = Linear8bitLt(
                            adapter_name,
                            target.in_features,
                            target.out_features,
                            bias=bias,
                            **eightbit_kwargs,
                        )
                    elif isinstance(target, torch.nn.Embedding):
                        embedding_kwargs = kwargs.copy()
                        embedding_kwargs.pop("fan_in_fan_out", None)
                        in_features, out_features = (
                            target.num_embeddings,
                            target.embedding_dim,
                        )
                        new_module = Embedding(
                            adapter_name, in_features, out_features, **embedding_kwargs
                        )
                    else:
                        if isinstance(target, torch.nn.Linear):
                            in_features, out_features = (
                                target.in_features,
                                target.out_features,
                            )
                            if kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                                    "Setting fan_in_fan_out to False."
                                )
                                kwargs[
                                    "fan_in_fan_out"
                                ] = lora_config.fan_in_fan_out = False
                        elif isinstance(target, Conv1D):
                            in_features, out_features = (
                                target.weight.ds_shape
                                if hasattr(target.weight, "ds_shape")
                                else target.weight.shape
                            )
                            if not kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                                    "Setting fan_in_fan_out to True."
                                )
                                kwargs[
                                    "fan_in_fan_out"
                                ] = lora_config.fan_in_fan_out = True
                        else:
                            raise ValueError(
                                f"Target module {target} is not supported. "
                                f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                            )
                        new_module = MultiLoraLinear(
                            adapter_name,
                            in_features,
                            out_features,
                            bias=bias,
                            **kwargs,
                        )

                    self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, MultiLoraLayer):
                # if module.merged:
                #     warnings.warn(
                #         "Adapter cannot be set when the model is merged. Unmerging the model first."
                #     )
                #     module.unmerge()
                if isinstance(adapter_name, str):
                    adapter_name = [adapter_name]
                module.set_adapters(adapter_name)

    def _set_aggregator_trainable(self, adapter_name, value):
        for module in self.model.modules():
            if isinstance(module, MultiLoraLayer):
                module._set_aggregator_trainable(adapter_name, value)

    def _set_adapter_trainable(self, adapter_name, value):
        for module in self.model.modules():
            if isinstance(module, MultiLoraLayer):
                module._set_adapter_trainable(adapter_name, value)

    def add_aggregator(self, aggregator_config):
        for module in self.model.modules():
            if isinstance(module, MultiLoraLayer):
                module.add_aggregator(aggregator_config)

    def update_aggregator_params(self, **kwargs):
        for module in self.model.modules():
            if isinstance(module, MultiLoraLayer):
                module.update_aggregator_params(**kwargs)


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class MultiLoraLayer(peft.tuners.lora.LoraLayer):
    def __init__(self, in_features: int, out_features: int):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        self.aggregator = None
        # Mark the weight as unmerged
        self.merged = {}
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights
    ):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.merged[adapter_name] = False
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_A.update(
                nn.ModuleDict(
                    {adapter_name: nn.Linear(self.in_features, r, bias=False)}
                )
            )
            self.lora_B.update(
                nn.ModuleDict(
                    {adapter_name: nn.Linear(r, self.out_features, bias=False)}
                )
            )
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def update_layer_embedding(
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights
    ):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_embedding_A.update(
                nn.ParameterDict(
                    {
                        adapter_name: nn.Parameter(
                            self.weight.new_zeros((r, self.in_features))
                        )
                    }
                )
            )
            self.lora_embedding_B.update(
                nn.ParameterDict(
                    {
                        adapter_name: nn.Parameter(
                            self.weight.new_zeros((self.out_features, r))
                        )
                    }
                )
            )
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])

    def add_aggregator(self, aggregator_config: AggregatorConfig):
        kwargs = {}
        if isinstance(aggregator_config, StaticAggregatorConfig):
            pass
        elif isinstance(aggregator_config, MoEAggregatorConfig):
            kwargs["in_features"] = self.in_features
        elif isinstance(aggregator_config, MoEV2AggregatorConfig):
            kwargs["in_features"] = self.in_features
        self.aggregator = AGGREGATOR_MAPPING[aggregator_config.aggregator_type](
            aggregator_config, self.lora_A.keys(), **kwargs
        )

        self.aggregator.to(self.weight.device)

        # # dispatch to correct device
        # for name, module in self.aggregator.named_modules():
        #     if "aggregator_" in name:
        #         module.to(self.weight.device)

    def init_aggregator(self):
        pass

    def set_adapters(self, adapter_names):
        pass

    def set_weight(self, adapter_name, new_weight):
        pass

    def update_aggregator_params(self, **kwargs):
        pass

    def _set_aggregator_trainable(self, adapter_name, value):
        self.aggregator._set_aggregator_trainable(adapter_name, value)

    def _set_adapter_trainable(self, adapter_name, value):
        if adapter_name not in self.lora_A.keys():
            warnings.warn(
                f"{adapter_name} do not exist in {self.lora_A.keys()},\
                          so no change applied. if you use heterogeneous loras, you can ignore this warning."
            )
            return
        for _, param in self.lora_A[adapter_name].named_parameters():
            param.requires_grad = value
        for _, param in self.lora_B[adapter_name].named_parameters():
            param.requires_grad = value


class MultiLoraLinear(nn.Linear, MultiLoraLayer):
    # GatedLora implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        MultiLoraLayer.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
        )
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        self.active_adapter = []
        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)

    def update_layer(
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights
    ):
        super().update_layer(
            adapter_name, r, lora_alpha, lora_dropout, init_lora_weights
        )
        self.active_adapter.append(adapter_name)

    def merge(self):
        for active_adapter in self.active_adapter:
            if active_adapter not in self.lora_A.keys():
                pass
            if self.merged[active_adapter]:
                logger.info(f"{active_adapter} already merged. Nothing to do.")
            if self.r[active_adapter] > 0:
                self.weight.data += (
                    transpose(
                        self.lora_B[active_adapter].weight
                        @ self.lora_A[active_adapter].weight,
                        self.fan_in_fan_out,
                    )
                    * self.scaling[active_adapter]
                    * self.lora_weight[active_adapter]
                )
                self.merged[active_adapter] = True

    def unmerge(self):
        for active_adapter in self.active_adapter:
            if active_adapter not in self.lora_A.keys():
                pass
            if not self.merged:
                logger.info(f"{active_adapter} already unmerged. Nothing to do.")
            if self.r[active_adapter] > 0:
                self.weight.data -= (
                    transpose(
                        self.lora_B[active_adapter].weight
                        @ self.lora_A[active_adapter].weight,
                        self.fan_in_fan_out,
                    )
                    * self.scaling[active_adapter]
                    * self.lora_weight[active_adapter]
                )
                self.merged[active_adapter] = False

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        result = F.linear(
            x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
        )

        if self.disable_adapters:
            for active_adapter in self.active_adapter:
                if self.r[active_adapter] > 0 and self.merged[active_adapter]:
                    self.unmerge()
        else:
            lora_results = []
            if len(self.active_adapter) > 0:
                x = x.to(self.lora_A[self.active_adapter[0]].weight.dtype)
            for active_adapter in self.active_adapter:
                lora_results.append(
                    (
                        self.lora_B[active_adapter](
                            self.lora_A[active_adapter](
                                self.lora_dropout[active_adapter](x)
                            )
                        )
                        * self.scaling[active_adapter]
                    ).unsqueeze(-1)
                )
            lora_results = torch.concat(lora_results, -1)
            result = self.aggregator(x, result, self.active_adapter, lora_results)

        result = result.to(previous_dtype)

        return result

    def set_adapters(self, adapter_names: List[str]):
        self.active_adapter = []
        for adapter_name in adapter_names:
            if adapter_name not in self.r.keys():
                warnings.warn(f"{adapter_name} not in {self.r.keys()}, thus ignored")
            else:
                self.active_adapter.append(adapter_name)

    def set_weight(self, adapter_name, new_weight):
        if adapter_name not in self.r.keys():
            warnings.warn(f"{adapter_name} not in {self.r.keys()}, thus ignored")
        else:
            self.lora_weight[adapter_name] = new_weight

    def update_aggregator_params(self, **kwargs):
        self.aggregator.update_aggregator_params(**kwargs)


class Embedding(nn.Embedding, MultiLoraLayer):
    # LoRA implemented in a Embedding layer
    def __init__(
        self,
        adapter_name: str,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        MultiLoraLayer.__init__(
            self, in_features=num_embeddings, out_features=embedding_dim
        )

        self.weight.requires_grad = False

        nn.Embedding.reset_parameters(self)
        self.update_layer_embedding(
            adapter_name, r, lora_alpha, lora_dropout, init_lora_weights
        )
        self.active_adapter = adapter_name

    def unmerge(self, mode: bool = True):
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= (
                transpose(
                    self.lora_embedding_B[self.active_adapter]
                    @ self.lora_embedding_A[self.active_adapter],
                    True,
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = False

    def merge(self):
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += (
                transpose(
                    self.lora_embedding_B[self.active_adapter]
                    @ self.lora_embedding_A[self.active_adapter],
                    True,
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            if self.r[self.active.adapter] > 0 and self.merged:
                self.weight.data -= (
                    transpose(
                        self.lora_embedding_B[self.active_adapter].weight
                        @ self.lora_embedding_A[self.active_adapter].weight,
                        True,
                    )
                    * self.scaling[self.active_adapter]
                )
                self.merged = False
            return nn.Embedding.forward(self, x)

        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r[self.active_adapter] > 0:
                after_A = F.embedding(
                    x,
                    self.lora_embedding_A[self.active_adapter].T,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )
                result += (
                    after_A @ self.lora_embedding_B[self.active_adapter].T
                ) * self.scaling[self.active_adapter]
            return result
        else:
            return nn.Embedding.forward(self, x)


# 更新下相关mapping
PEFT_TYPE_TO_MODEL_MAPPING["MULTILORA"] = MultiLoraModel
PeftType.MULTILORA = "MULTILORA"


@dataclass
class MultiLoraConfig(PeftConfig):
    """
    因为是在Lor上增加门控，所以其他结构不变
    This is the configuration class to store the configuration of a [`LoraModel`].

    Args:
        r (`int`): Lora attention dimension.
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
        For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out)
        and hence this should be set to `True`.:
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})
    lora_weight: float = field(default=1.0, metadata={"help": "单个lora模块的权重"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={
            "help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"
        },
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"},
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_lora_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the weights of the Lora layers."},
    )

    def __post_init__(self):
        self.peft_type = PeftType.MULTILORA

    @classmethod
    def from_lora_config(cls, lora_config: peft.LoraConfig, adapter_weight):
        return MultiLoraConfig(
            task_type=lora_config.task_type,
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            fan_in_fan_out=lora_config.fan_in_fan_out,
            lora_dropout=lora_config.lora_dropout,
            init_lora_weights=lora_config.init_lora_weights,
            target_modules=lora_config.target_modules,
            modules_to_save=lora_config.modules_to_save,
            bias=lora_config.bias,
            inference_mode=lora_config.inference_mode,
            lora_weight=adapter_weight,
        )


PEFT_TYPE_TO_CONFIG_MAPPING["MULTILORA"] = MultiLoraConfig


if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt, MultiLoraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
        ):
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get(
                    "memory_efficient_backward", False
                ),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            MultiLoraLayer.__init__(
                self, in_features=in_features, out_features=out_features
            )

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            init_lora_weights = kwargs.pop("init_lora_weights", True)
            self.update_layer(
                adapter_name, r, lora_alpha, lora_dropout, init_lora_weights
            )
            self.active_adapter = adapter_name

        def forward(self, x: torch.Tensor):
            result = super().forward(x)

            if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
                return result
            elif self.r[self.active_adapter] > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()
                    output = (
                        self.lora_B[self.active_adapter](
                            self.lora_A[self.active_adapter](
                                self.lora_dropout[self.active_adapter](x)
                            )
                        ).to(expected_dtype)
                        * self.scaling[self.active_adapter]
                    )
                else:
                    output = (
                        self.lora_B[self.active_adapter](
                            self.lora_A[self.active_adapter](
                                self.lora_dropout[self.active_adapter](x)
                            )
                        )
                        * self.scaling[self.active_adapter]
                    )
                result += output
            return result
