import torch
from torch import nn
import enum
from dataclasses import dataclass, field
from typing import Union, Dict, Optional, List
import json
import os
from alps.util import logger
import warnings

CONFIG_NAME = "aggregator.json"
WEIGHTS_NAME = "aggregator_model.bin"
INIT_METHODS_MAP = {
    "identity": nn.init.eye_,
    "kaiming": nn.init.kaiming_normal_,
    "zeros": nn.init.zeros_,
    "xavier": nn.init.xavier_normal_,
}


class LoraAggregator(nn.Module):
    def __init__(self, config, adapter_names, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(self, x, base_res, active_adapters, lora_results):
        pass

    def update_aggregator_params(self, **kwargs):
        pass

    def _set_aggregator_trainable(self, adapter_name, value):
        pass


class StaticWeightAggregator(LoraAggregator):
    def __init__(self, config, adapter_names, **kwargs) -> None:
        super().__init__(config, adapter_names, **kwargs)
        self.aggregator_static_weights = nn.ParameterDict({})
        self._set_aggregator_static_weights(config.static_weights, adapter_names)

    def _set_aggregator_static_weights(self, static_weights, adapter_names):
        if static_weights is None:
            static_weights = {name: 1.0 for name in adapter_names}

        if isinstance(static_weights, float):
            static_weights = {name: static_weights for name in adapter_names}

        for name in static_weights.keys():
            if name in adapter_names:
                self.aggregator_static_weights.update(
                    {name: nn.Parameter(torch.Tensor([static_weights[name]]))}
                )
            else:
                warnings.warn(f"{name} not in {static_weights.keys()}")

    def _print_param(self):
        for name in self.aggregator_static_weights.keys():
            print(f"{name} {self.aggregator_static_weights[name]}")

    def forward(self, x, base_res, active_adapters, lora_results):
        # active_adapters已经由MultiLoraLayer的forward筛选过，不会出现keyError
        weights = torch.concat(
            [
                self.aggregator_static_weights[active_adapter]
                for active_adapter in active_adapters
            ],
        )
        weights_sftmax = torch.nn.functional.softmax(weights, dim=-1)
        for i, active_adapter in enumerate(active_adapters):
            base_res += (weights_sftmax[i] * lora_results[:, :, :, i]).squeeze(-1)

        return base_res

    def update_aggregator_params(self, **kwargs):
        static_weights = kwargs.pop("aggregator_static_weights", None)
        if not isinstance(static_weights, dict):
            logger.warn(
                f"aggregator_static_weights should be a dict, but got {type(static_weights)}, thus nothing happened"
            )
            return
        for k, v in static_weights.items():
            if k in self.aggregator_static_weights.keys():
                device = self.aggregator_static_weights[k].device
                self.aggregator_static_weights[k] = nn.Parameter(
                    torch.Tensor([v]), requires_grad=False
                ).to(device)
            else:
                warnings.warn(
                    f"{k} not in {self.aggregator_static_weights.keys()}\
                        so nothing happened."
                )
        if len(kwargs.keys()):
            warnings.warn(
                f"{__class__}.{__name__} only takes aggregator_static_weights, \
                          following args are ignored \n\t{kwargs}"
            )

    def _set_aggregator_trainable(self, adapter_name, value):
        if adapter_name not in self.aggregator_static_weights.keys():
            warnings.warn(
                f"{adapter_name} not in {self.aggregator_static_weights.keys()}\
                    so nothing happened."
            )
        self.aggregator_static_weights[adapter_name].requires_grad = value


class MoEAggregator(LoraAggregator):
    def __init__(self, config, adapter_names, **kwargs) -> None:
        in_features = kwargs.pop("in_features", None)
        super().__init__(config, adapter_names, **kwargs)
        self.add_noise = config.add_noise
        self.num_adapters = len(adapter_names)
        self.set_top_k(config.top_k)
        self.aggregator_gate = nn.ModuleDict(
            {
                adapter_name: nn.Linear(in_features, 1, bias=config.use_bias)
                for adapter_name in adapter_names
            }
        )
        # TODO 初始化有大讲究，日后再研究
        self._init_weights(config.init_method)

    def _init_weights(self, init_method: Union[str, Dict[str, str]]):
        if init_method is None:
            return
        if isinstance(init_method, str):
            init_method = {"weight": init_method, "bias": init_method}
        if sorted(list(init_method.keys())) != sorted(["weight", "bias"]):
            warnings.warn(
                f"keys of init_method should be {['weight','bias']},but got {list(init_method.keys())}, \
                    no change applied"
            )
            return
        for adapter_name in self.aggregator_gate.keys():
            INIT_METHODS_MAP[init_method["weight"]](
                self.aggregator_gate[adapter_name].weight
            )
            if self.aggregator_gate[adapter_name].bias is not None:
                INIT_METHODS_MAP[init_method["bias"]](
                    self.aggregator_gate[adapter_name].bias
                )

    def forward(self, x: torch.Tensor, base_res, active_adapters, lora_results):
        # pooled_x = torch.mean(x, 1)

        pooled_x = x[:, -1, :]
        gate_weights = []
        activated_adapter_names = []
        # 先算权重，确定激活的lora
        for active_adapter in active_adapters:
            if active_adapter not in self.aggregator_gate.keys():
                continue
            gate_weights.append(self.aggregator_gate[active_adapter](pooled_x))
            activated_adapter_names.append(active_adapter)

        gate_weights = torch.concat(gate_weights, -1)
        if self.add_noise and self.training:
            gate_weights += torch.normal(0, 1, gate_weights.shape).to(
                gate_weights.device
            )
        real_top_k = self.get_real_top_k(len(activated_adapter_names))
        batch_size = lora_results.shape[0]

        if real_top_k is None or (real_top_k >= len(active_adapters)):
            # 不需要稀疏激活
            activated_res = lora_results
            adapter_weights = gate_weights
        else:
            # 为同batch每条样本选择激活的lora
            adapter_weights, adapter_index = torch.topk(gate_weights, k=real_top_k)
            logger.debug(adapter_index)
            activated_res = lora_results.gather(
                3,
                adapter_index.view(batch_size, 1, 1, -1).expand(
                    batch_size, lora_results.size(1), lora_results.size(2), -1
                ),
            )

        # old
        # base_res += torch.sum(
        #     # 把每个权重乘上去
        #     activated_res * adapter_weights.unsqueeze(1).unsqueeze(2),
        #     -1,
        # )
        base_res += torch.sum(
            # 把每个权重乘上去
            activated_res,
            -1,
        )

        return base_res

    def update_aggregator_params(self, **kwargs):
        top_k = kwargs.pop("top_k", None)
        if top_k is not None:
            self.set_top_k(top_k)

    def set_top_k(self, top_k):
        if top_k is None:
            self.top_k = None
        else:
            self.top_k = min(top_k, self.num_adapters)

    def get_real_top_k(self, active_adapter_num):
        if self.top_k is None:
            return None
        else:
            return min(self.top_k, active_adapter_num)

    def _set_aggregator_trainable(self, adapter_name, value):
        if adapter_name not in self.aggregator_gate.keys():
            warnings.warn(
                f"{adapter_name} not in {self.aggregator_gate.keys()}\
                    so nothing happened."
            )
        for _, param in self.aggregator_gate[adapter_name].named_parameters():
            param.requires_grad = value


# class MoE3Aggregator(MoEAggregator):
#     def __init__(self, config, adapter_names, **kwargs) -> None:
#         MoEAggregator.__init__(self, config, adapter_names, **kwargs)

#     def forward(self, x: torch.Tensor, base_res, active_adapters, lora_results):
#         gate_weights = torch.concat(gate_weights, -1)
#         if self.add_noise and self.training:
#             gate_weights += torch.normal(0, 1, gate_weights.shape).to(
#                 gate_weights.device
#             )
#         real_top_k = self.get_real_top_k(len(activated_adapter_names))
#         batch_size = lora_results.shape[0]

#         # if real_top_k is None or (real_top_k >= len(active_adapters)):
#         #     # 不需要稀疏激活
#         #     activated_res = lora_results
#         #     adapter_weights = gate_weights
#         # else:
#         #     # 为同batch每条样本选择激活的lora
#         #     adapter_weights, adapter_index = torch.topk(gate_weights, k=real_top_k)
#         #     logger.debug(adapter_index)
#         #     activated_res = lora_results.gather(
#         #         3,
#         #         adapter_index.view(batch_size, 1, 1, -1).expand(
#         #             batch_size, lora_results.size(1), lora_results.size(2), -1
#         #         ),
#         #     )


#         pooled_x = torch.mean(x, 1)
#         gate_weights = []
#         activated_adapter_names = []
#         # 先算权重，确定激活的lora
#         for active_adapter in active_adapters:
#             if active_adapter not in self.aggregator_gate.keys():
#                 continue
#             gate_weights.append(self.aggregator_gate[active_adapter](pooled_x))
#             activated_adapter_names.append(active_adapter)

#         base_res += torch.sum(
#             # 把每个权重乘上去
#             activated_res,
#             -1,
#         )

#         return base_res

#     def update_aggregator_params(self, **kwargs):
#         top_k = kwargs.pop("top_k", None)
#         if top_k is not None:
#             self.set_top_k(top_k)

#     def set_top_k(self, top_k):
#         if top_k is None:
#             self.top_k = None
#         else:
#             self.top_k = min(top_k, self.num_adapters)

#     def get_real_top_k(self, active_adapter_num):
#         if self.top_k is None:
#             return None
#         else:
#             return min(self.top_k, active_adapter_num)

#     def _set_aggregator_trainable(self, adapter_name, value):
#         if adapter_name not in self.aggregator_gate.keys():
#             warnings.warn(
#                 f"{adapter_name} not in {self.aggregator_gate.keys()}\
#                     so nothing happened."
#             )
#         for _, param in self.aggregator_gate[adapter_name].named_parameters():
#             param.requires_grad = value


class MoEV2Aggregator(MoEAggregator):
    def __init__(self, config, adapter_names, **kwargs) -> None:
        in_features = kwargs.pop("in_features", None)
        LoraAggregator.__init__(self, config, adapter_names, **kwargs)

        self.num_adapters = len(adapter_names)
        self.add_noise = config.add_noise
        self.set_top_k(config.top_k)
        # 1个是激活权重，另一个是缩放权重
        self.aggregator_gate = nn.ModuleDict(
            {
                adapter_name: nn.Linear(in_features, 2, bias=config.use_bias)
                for adapter_name in adapter_names
            }
        )
        # TODO 初始化有大讲究，日后再研究
        self._init_weights(config.init_method)

    def forward(self, x: torch.Tensor, base_res, active_adapters, lora_results):
        pooled_x = x[:, -1, :]
        gate_weights = []
        adapter_weights = []
        activated_adapter_names = []
        # 先算权重，确定激活的lora

        for active_adapter in active_adapters:
            if active_adapter not in self.aggregator_gate.keys():
                continue
            gate_outs = self.aggregator_gate[active_adapter](pooled_x)
            gate_weights.append(gate_outs[:, 0])
            adapter_weights.append(gate_outs[:, 1])
            activated_adapter_names.append(active_adapter)

        gate_weights = torch.stack(gate_weights, -1)
        if self.add_noise and self.training:
            gate_weights += torch.normal(0, 1, gate_weights.shape).to(
                gate_weights.device
            )
        adapter_weights = torch.stack(adapter_weights, -1)
        real_top_k = self.get_real_top_k(len(activated_adapter_names))

        batch_size = lora_results.shape[0]

        if real_top_k is None or (real_top_k >= len(activated_adapter_names)):
            # 不需要稀疏激活
            activated_res = lora_results
        else:
            # 为同batch每条样本选择激活的lora
            _, adapter_index = torch.topk(gate_weights, k=real_top_k)
            logger.debug(adapter_index)
            adapter_weights = adapter_weights.gather(1, adapter_index)
            # adapter_index = adapter_index.view(batch_size, 1, 1, adapter_index.shape[1])
            # numpy真是太深奥了
            activated_res = lora_results.gather(
                3,
                adapter_index.view(batch_size, 1, 1, -1).expand(
                    batch_size, lora_results.size(1), lora_results.size(2), -1
                ),
            )

        base_res += torch.sum(
            # 把每个权重乘上去
            activated_res * adapter_weights.view(batch_size, 1, 1, -1),
            -1,
        )

        return base_res


class DirectAggregator(LoraAggregator):
    def __init__(self, config, adapter_names, **kwargs) -> None:
        super().__init__(config, adapter_names, **kwargs)

    def forward(self, x, base_res, active_adapters, lora_results):
        return base_res + torch.sum(lora_results, -1)

    def _set_aggregator_trainable(self, adapter_name, value):
        warnings.warn("DirectAggregator contains no trainable parameters")


# WIP
class CosineSimilarityAggregator(LoraAggregator):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(self, x):
        pass


class AggregatorType(str, enum.Enum):
    STATIC = "STATIC"
    MOE = "MOE"
    MOE2 = "MOE2"
    COS = "COS"
    DIRECT = "DIRECT"


AGGREGATOR_MAPPING = {
    AggregatorType.STATIC: StaticWeightAggregator,
    AggregatorType.MOE: MoEAggregator,
    AggregatorType.MOE2: MoEV2Aggregator,
    AggregatorType.COS: CosineSimilarityAggregator,
    AggregatorType.DIRECT: DirectAggregator,
}


@dataclass
class AggregatorConfig:
    aggregator_type: Union[str, AggregatorType] = field(default=AggregatorType.STATIC)
    inference_mode: bool = field(default=True)
    trained_wtih_adapters: Optional[List[str]] = field(default=None)

    def save_pretrained(self, save_directory, **kwargs):
        r"""
        This method saves the configuration of your adapter model in a directory.

        Args:
            save_directory (`str`):
                The directory where the configuration will be saved.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the [`~transformers.utils.PushToHubMixin.push_to_hub`]
                method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )

        os.makedirs(save_directory, exist_ok=True)

        output_dict = self.__dict__
        output_path = os.path.join(save_directory, CONFIG_NAME)

        # save it
        with open(output_path, "w") as writer:
            writer.write(json.dumps(output_dict, indent=2, sort_keys=True))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, subfolder=None, **kwargs):
        r"""
        This method loads the configuration of your adapter model from a directory.

        Args:
            pretrained_model_name_or_path (`str`):
                The directory or the Hub repository id where the configuration is saved.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the child class initialization.
        """
        path = (
            os.path.join(pretrained_model_name_or_path, subfolder)
            if subfolder is not None
            else pretrained_model_name_or_path
        )
        if os.path.isfile(os.path.join(path, CONFIG_NAME)):
            config_file = os.path.join(path, CONFIG_NAME)
        else:
            # try:
            #     config_file = hf_hub_download(
            #         pretrained_model_name_or_path, CONFIG_NAME, subfolder=subfolder
            #     )
            # except Exception:
            raise ValueError(
                f"Can't find '{CONFIG_NAME}' at '{pretrained_model_name_or_path}'"
            )

        loaded_attributes = cls.from_json_file(config_file)

        config = cls(**kwargs)

        for key, value in loaded_attributes.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    @classmethod
    def from_json_file(cls, path_json_file, **kwargs):
        r"""
        Loads a configuration file from a json file.

        Args:
            path_json_file (`str`):
                The path to the json file.
        """
        with open(path_json_file, "r") as file:
            json_object = json.load(file)

        return json_object


@dataclass
class DirectAggregatorConfig(AggregatorConfig):
    def __post_init__(self):
        self.aggregator_type = AggregatorType.DIRECT


@dataclass
class StaticAggregatorConfig(AggregatorConfig):
    r"""

    静态参数聚合器配置，静态是指权重不随样本变化

    args:
        `static_weights` 每个adapter设置初始权重，默认全1。
            可传入`float`。
            ```py
                StaticAggregatorConfig(static_weights=1.0)
            ```
            也可传入字典，字典key应为`"weight"`和`"bias"`，例如
            ```py
                StaticAggregatorConfig(static_weights={"lora1":0.9,"lora2":"1.0"})
            ```

    """
    static_weights: Optional[Union[float, Dict[str, float]]] = field(default=1.0)

    def __post_init__(self):
        self.aggregator_type = AggregatorType.STATIC


@dataclass
class MoEAggregatorConfig(AggregatorConfig):
    r"""

    MoE结构配置

    args:
        `top_k` 稀疏激活，选取top_k个lora的值做聚合

        `init_method` `MoEAggregator.aggregator_gate`参数初始化方法，默认全零。
        可传入`str`，可选选项`"identity","kaiming","zeros","xavier"`。

        ```py
            MoEAggregatorConfig(init_method="kaiming")
        ```

        也可传入字典，字典key应为`"weight"`和`"bias"`，例如
        ```py
            MoEAggregatorConfig(init_method={"weight":"kaiming","bias":"zeros"})
        ```

    """
    top_k: Optional[int] = field(default=None)
    init_method: Optional[Union[str, Dict[str, str]]] = field(
        default="zeros", metadata={"help": """初始化参数的方法，默认全零。"""}
    )
    add_noise: Optional[bool] = field(default=True)
    use_bias: Optional[bool] = field(default=True)

    def __post_init__(self):
        self.aggregator_type = AggregatorType.MOE


@dataclass
class MoEV2AggregatorConfig(MoEAggregatorConfig):
    r"""

    MoE结构配置

    args:
        `top_k` 稀疏激活，选取top_k个lora的值做聚合

        `init_method` `MoEAggregator.aggregator_gate`参数初始化方法，默认全零。
        可传入`str`，可选选项`"identity","kaiming","zeros","xavier"`。

        ```py
            MoEAggregatorConfig(init_method="kaiming")
        ```

        也可传入字典，字典key应为`"weight"`和`"bias"`，例如
        ```py
            MoEAggregatorConfig(init_method={"weight":"kaiming","bias":"zeros"})
        ```

    """

    def __post_init__(self):
        self.aggregator_type = AggregatorType.MOE2
