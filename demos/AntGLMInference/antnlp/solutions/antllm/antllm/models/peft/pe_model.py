from peft import PeftModel
from .utils.aggregator import (
    WEIGHTS_NAME as AGG_WEIGHTS_NAME,
    AggregatorConfig,
    DirectAggregatorConfig,
    StaticAggregatorConfig,
)

import os
import inspect
from peft.utils import (
    PeftConfig,
    PromptLearningConfig,
    WEIGHTS_NAME,
    _set_trainable,
)
from peft.tuners.lora import LoraConfig
from peft import PeftType
import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import (
    AlignDevicesHook,
    add_hook_to_module,
    remove_hook_from_submodules,
)
from accelerate.utils import get_balanced_memory
from huggingface_hub import hf_hub_download
from typing import Union, List, Dict, Optional
from .tuner.multi_lora import MultiLoraModel, MultiLoraConfig
from transformers.utils import PushToHubMixin
from alps.util import logger
from dataclasses import dataclass, field


def set_peft_model_state_dict(model, peft_model_state_dict, adapter_name="default"):
    """
    Set the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model.
        peft_model_state_dict (`dict`): The state dict of the Peft model.
    """
    config = model.peft_config[adapter_name]
    state_dict = {}
    if model.modules_to_save is not None:
        for key, value in peft_model_state_dict.items():
            if any(module_name in key for module_name in model.modules_to_save):
                for module_name in model.modules_to_save:
                    if module_name in key:
                        key = key.replace(
                            module_name, f"{module_name}.modules_to_save.{adapter_name}"
                        )
                        break
            state_dict[key] = value
    else:
        state_dict = peft_model_state_dict

    if config.peft_type in (PeftType.LORA, PeftType.ADALORA, "MULTILORA"):
        peft_model_state_dict = {}
        for k, v in state_dict.items():
            if "lora_" in k:
                suffix = k.split("lora_")[1]
                if "." in suffix:
                    suffix_to_replace = ".".join(suffix.split(".")[1:])
                    k = k.replace(
                        suffix_to_replace, f"{adapter_name}.{suffix_to_replace}"
                    )
                else:
                    k = f"{k}.{adapter_name}"
                peft_model_state_dict[k] = v
            else:
                peft_model_state_dict[k] = v
        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                model.resize_modules_by_rank_pattern(rank_pattern, adapter_name)
    elif (
        isinstance(config, PromptLearningConfig)
        or config.peft_type == PeftType.ADAPTION_PROMPT
    ):
        peft_model_state_dict = state_dict
    else:
        raise NotImplementedError

    model.load_state_dict(peft_model_state_dict, strict=False)
    if isinstance(config, PromptLearningConfig):
        model.prompt_encoder[adapter_name].embedding.load_state_dict(
            {"weight": peft_model_state_dict["prompt_embeddings"]}, strict=True
        )


def set_aggregator_state_dict(model, aggregator_state_dict):
    # state_dict = {}
    # if model.modules_to_save is not None:
    #     for key, value in aggregator_state_dict.items():
    #         if any(module_name in key for module_name in model.modules_to_save):
    #             for module_name in model.modules_to_save:
    #                 if module_name in key:
    #                     key = key.replace(
    #                         module_name, f"{module_name}.modules_to_save.{adapter_name}"
    #                     )
    #                     break
    #         state_dict[key] = value
    # else:
    #     state_dict = aggregator_state_dict

    # aggregator_state_dict = {}
    # for k, v in state_dict.items():
    #     if "lora_" in k:
    #         suffix = k.split("lora_")[1]
    #         if "." in suffix:
    #             suffix_to_replace = ".".join(suffix.split(".")[1:])
    #             k = k.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")
    #         else:
    #             k = f"{k}.{adapter_name}"
    #         aggregator_state_dict[k] = v
    #     else:
    #         aggregator_state_dict[k] = v
    # if config.peft_type == PeftType.ADALORA:
    #     rank_pattern = config.rank_pattern
    #     if rank_pattern is not None:
    #         model.resize_modules_by_rank_pattern(rank_pattern, adapter_name)

    model.load_state_dict(aggregator_state_dict, strict=False)


def get_peft_model_state_dict(model, state_dict=None, adapter_name="default"):
    """
    Get the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP,
        the model should be the underlying model/unwrapped model (i.e. model.module).
        state_dict (`dict`, *optional*, defaults to `None`):
            The state dict of the model. If not provided, the state dict of the model
        will be used.
    """
    config = model.peft_config[adapter_name]
    if state_dict is None:
        state_dict = model.state_dict()
    if config.peft_type in (PeftType.LORA, PeftType.ADALORA):
        # to_return = lora_state_dict(model, bias=model.peft_config.bias)
        # adapted from `https://github.com/microsoft/LoRA/blob/main/loralib/utils.py`
        # to be used directly with the state dict which is necessary when using DeepSpeed or FSDP
        bias = config.bias
        if bias == "none":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k}
        elif bias == "all":
            to_return = {
                k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k
            }
        elif bias == "lora_only":
            to_return = {}
            for k in state_dict:
                if "lora_" in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split("lora_")[0] + "bias"
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError
        to_return = {
            k: v
            for k, v in to_return.items()
            if (("lora_" in k and adapter_name in k) or ("bias" in k))
        }
        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                rank_pattern = {
                    k.replace(f".{adapter_name}", ""): v
                    for k, v in rank_pattern.items()
                }
                config.rank_pattern = rank_pattern
                to_return = model.resize_state_dict_by_rank_pattern(
                    rank_pattern, to_return, adapter_name
                )

    elif config.peft_type == PeftType.ADAPTION_PROMPT:
        to_return = {
            k: state_dict[k]
            for k in state_dict
            if k.split(".")[-1].startswith("adaption_")
        }
    elif isinstance(config, PromptLearningConfig):
        to_return = {}
        if config.inference_mode:
            prompt_embeddings = model.prompt_encoder[adapter_name].embedding.weight
        else:
            prompt_embeddings = model.get_prompt_embedding_to_save(adapter_name)
        to_return["prompt_embeddings"] = prompt_embeddings
    else:
        raise NotImplementedError
    if model.modules_to_save is not None:
        for key, value in state_dict.items():
            if any(
                f"{module_name}.modules_to_save.{adapter_name}" in key
                for module_name in model.modules_to_save
            ):
                to_return[key.replace("modules_to_save.", "")] = value

    to_return = {k.replace(f".{adapter_name}", ""): v for k, v in to_return.items()}
    return to_return


def get_aggregator_state_dict(model, state_dict=None):
    if state_dict is None:
        state_dict = model.state_dict()
    # if config.peft_type in (PeftType.LORA, PeftType.ADALORA):
    #     # to_return = lora_state_dict(model, bias=model.peft_config.bias)
    #     # adapted from `https://github.com/microsoft/LoRA/blob/main/loralib/utils.py`
    #     # to be used directly with the state dict which is necessary when using DeepSpeed or FSDP
    #     bias = config.bias
    #     if bias == "none":
    to_return = {k: state_dict[k] for k in state_dict if "aggregator_" in k}
    # elif bias == "all":
    #     to_return = {
    #         k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k
    #     }
    # elif bias == "lora_only":
    #     to_return = {}
    #     for k in state_dict:
    #         if "lora_" in k:
    #             to_return[k] = state_dict[k]
    #             bias_name = k.split("lora_")[0] + "bias"
    #             if bias_name in state_dict:
    #                 to_return[bias_name] = state_dict[bias_name]
    # else:
    #     raise NotImplementedError
    to_return = {k: v for k, v in to_return.items() if (("aggregator_" in k))}
    return to_return


@dataclass
class MultiAdapterConfig:
    """定义了lora们的配置`lora_configs`和所用的聚合器配置`aggregator_config`

    Args:
        lora_configs (`Dict[str, Union[str, LoraConfig]]`):
            lora的地址或者配置，如果是地址就默认是训好的的lora，如果是config就默认是要新训练的lora
        aggregator_config (`Union[str, AggregatorConfig]`):
            可以是训练好的聚合器地址，也可以是配置`AggregatorConfig`，不传就默认使用`DirectAggregatorConfig`
        adapter_name_mapping (`Optional[Dict[str, str]]`): 暂时没用
    """

    lora_configs: Dict[str, Union[str, LoraConfig]] = field(
        metadata={"help": "lora的地址或者配置，如果是地址就默认是训好的的lora，如果是config就默认是要新训练的lora"}
    )
    aggregator_config: Union[str, AggregatorConfig] = field(
        default=DirectAggregatorConfig(), metadata={"help": ""}
    )
    adapter_name_mapping: Optional[Dict[str, str]] = field(
        default=None, metadata={"help": "暂时没用"}
    )


class MultiAdapterModel(PeftModel):
    def __init__(
        self,
        model,
        aggregator_config: AggregatorConfig,
        peft_config: Union[PeftConfig, List[PeftConfig]],
        adapter_name: Union[str, List[str]] = "default",
        adapter_weights: Optional[Union[float, List[float]]] = None,
    ):
        """_summary_

        Args:
            model (_type_): _description_
            aggregator_config (AggregatorConfig): _description_
            peft_config (Union[PeftConfig, List[PeftConfig]]): _description_
            adapter_name (Union[str, List[str]], optional): _description_. Defaults to "default".
            adapter_weights (Optional[Union[float, List[float]]], optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_
        """
        PushToHubMixin.__init__(self)
        torch.nn.Module.__init__(self)

        if adapter_weights is not None:
            logger.warn("adapter_weights以后就删了，请用aggregator_config")

        if isinstance(peft_config, PeftConfig):
            peft_config = [peft_config]

        if isinstance(adapter_name, str):
            adapter_name = [adapter_name]

        assert len(peft_config) == len(
            adapter_name
        ), f"number of model_id {len(peft_config)} != number of adapter_name {len(adapter_name)}"

        self.base_model = model
        self.config = self.base_model.config
        self.modules_to_save = None
        self.peft_config = {}
        self.aggregator_config = aggregator_config
        self.active_adapter = adapter_name
        self.peft_type = peft_config[0].peft_type
        # self.base_model_torch_dtype = getattr(model, "dtype", None)
        for peft_conf, adapter_namee in zip(peft_config, adapter_name):
            if not (
                isinstance(peft_conf, LoraConfig)
                or isinstance(peft_conf, MultiLoraConfig)
            ):
                raise NotImplementedError("当前仅支持Lora")
            self.peft_config[adapter_namee] = peft_conf

        self.base_model = MultiLoraModel(
            model=self.base_model,
            config=self.peft_config,
            adapter_name=adapter_name,
            aggregator_config=self.aggregator_config,
        )
        self.set_additional_trainable_modules(peft_config, adapter_name)

    @classmethod
    def from_pretrained(
        cls,
        model,
        model_id: Union[str, List[str]],
        adapter_name: Union[str, List[str]],
        aggregator_path: Union[str, AggregatorConfig] = None,
        is_trainable=False,
        adapter_weights: Union[float, List[float]] = None,
        **kwargs,
    ):
        """加载已经训练好的lora们和聚合器

        Args:
            model (_type_): 底座模型
            model_id (`Union[str, List[str]]`):
                lora们的地址
            adapter_name (`Union[str, List[str]]`):
                lora们各自的名称，用于权重的匹配
            aggregator_path (`Union[str, AggregatorConfig]`):
                聚合器的地址，也可以传`AggregatorConfig`，这时候就会根据config创建。不传就默认使用
            is_trainable
        Raises:
            NotImplementedError: _description_
        """
        if isinstance(model_id, str):
            model_id = [model_id]

        if isinstance(adapter_name, str):
            adapter_name = [adapter_name]

        assert len(model_id) == len(
            adapter_name
        ), f"number of model_id {len(model_id)} != number of adapter_name {len(adapter_name)}"

        if adapter_weights is not None and aggregator_path is None:
            assert len(model_id) == len(
                adapter_weights
            ), f"number of model_id {len(model_id)} != number of adapter_weights {len(adapter_weights)}"

        from .pe_mapping import (
            MODEL_TYPE_TO_MULTI_ADAPTER_MODEL_MAPPING,
            PEFT_TYPE_TO_CONFIG_MAPPING,
            AGGREGAOTR_TYPE_TO_CONFIG_MAPPING,
        )

        if aggregator_path is None:
            if adapter_weights is not None:
                logger.warning("adapter_weights参数之后就会删除")

            logger.warning("没有传入aggregator_path，默认直接相加融合")
            aggregator_config = StaticAggregatorConfig()
            # aggregator_config = DirectAggregatorConfig()
        else:
            aggregator_config = AGGREGAOTR_TYPE_TO_CONFIG_MAPPING[
                AggregatorConfig.from_pretrained(aggregator_path).aggregator_type
            ].from_pretrained(aggregator_path)
            # 训练aggregator时的lora和读取到的不一致，报错
            if sorted(aggregator_config.trained_wtih_adapters) != sorted(adapter_name):
                raise ValueError(
                    f"""这个聚合器是用以下lora训练的（{aggregator_config.trained_wtih_adapters}），\n
                        但是恢复的时候传入了以下lora（{adapter_name}），请务必保持一致，不然权重恢复会有问题。"""
                )

        # if not isinstance(aggregator_config, DirectAggregatorConfig):
        #     raise ValueError(f"{aggregator_config} currently under development ...")

        if (getattr(model, "hf_device_map", None) is not None) and len(
            set(model.hf_device_map.values()).intersection({"cpu", "disk"})
        ) > 0:
            remove_hook_from_submodules(model)

        # load the config
        configs = []
        for model_idd in model_id:
            config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig.from_pretrained(
                    model_idd, subfolder=kwargs.get("subfolder", None)
                ).peft_type
            ].from_pretrained(model_idd, subfolder=kwargs.get("subfolder", None))

            if isinstance(config, PromptLearningConfig) and is_trainable:
                raise ValueError(
                    "Cannot set a prompt learning adapter to trainable when loading pretrained adapter."
                )
            else:
                config.inference_mode = not is_trainable
            if isinstance(config, LoraConfig):
                pass
            configs.append(config)
        task_type = configs[0].task_type
        if task_type not in MODEL_TYPE_TO_MULTI_ADAPTER_MODEL_MAPPING.keys():
            model = cls(model, aggregator_config, configs, adapter_name)
        else:
            model = MODEL_TYPE_TO_MULTI_ADAPTER_MODEL_MAPPING[task_type](
                model, aggregator_config, configs, adapter_name
            )

        model.load_adapter(model_id, adapter_name, **kwargs)
        load_agg_kwargs = {"from_pretrained": True}
        if aggregator_path is not None:
            model.load_aggregator(aggregator_path, **load_agg_kwargs)
        return model

    def get_base_model(self):
        return self.base_model.model

    def load_adapter(self, model_id, adapter_name, is_trainable=False, **kwargs):
        from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING

        for model_idd, adapter_namee in zip(model_id, adapter_name):
            if adapter_namee not in self.peft_config:
                # load the config
                peft_config = PEFT_TYPE_TO_CONFIG_MAPPING[
                    PeftConfig.from_pretrained(
                        model_idd, subfolder=kwargs.get("subfolder", None)
                    ).peft_type
                ].from_pretrained(model_idd, subfolder=kwargs.get("subfolder", None))
                if isinstance(peft_config, PromptLearningConfig) and is_trainable:
                    raise ValueError(
                        "Cannot set a prompt learning adapter to trainable when loading pretrained adapter."
                    )
                else:
                    peft_config.inference_mode = not is_trainable
                self.add_adapter(adapter_namee, peft_config)

            # load weights if any
            path = (
                os.path.join(model_idd, kwargs["subfolder"])
                if kwargs.get("subfolder", None) is not None
                else model_idd
            )

            if os.path.exists(os.path.join(path, WEIGHTS_NAME)):
                filename = os.path.join(path, WEIGHTS_NAME)
            else:
                try:
                    filename = hf_hub_download(
                        model_idd, WEIGHTS_NAME, subfolder=kwargs.get("subfolder", None)
                    )
                except:  # noqa
                    raise ValueError(
                        f"Can't find weights for {model_idd} in {model_idd} or in the Hugging Face Hub. "
                        f"Please check that the file {WEIGHTS_NAME} is present at {model_idd}."
                    )

            adapters_weights = torch.load(
                filename,
                map_location=torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                ),
            )
            # load the weights into the model
            set_peft_model_state_dict(
                self, adapters_weights, adapter_name=adapter_namee
            )
            logger.info(f"{adapter_namee} weights loaded")
        if (
            (getattr(self, "hf_device_map", None) is not None)
            and (
                len(set(self.hf_device_map.values()).intersection({"cpu", "disk"})) > 0
            )
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
                    self,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                )
            dispatch_model(
                self,
                device_map=device_map,
                offload_dir=offload_dir,
                **dispatch_model_kwargs,
            )
            hook = AlignDevicesHook(io_same_device=True)
            # if isinstance(self.peft_config[adapter_name], PromptLearningConfig):
            #     remove_hook_from_submodules(self.prompt_encoder)
            add_hook_to_module(self.get_base_model(), hook)

        # Set model in evaluation mode to deactivate Dropout modules by default
        self.eval()

    def set_additional_trainable_modules(self, peft_config, adapter_name):
        if not isinstance(peft_config, list):
            peft_config = [peft_config]
        if not isinstance(adapter_name, list):
            adapter_name = [adapter_name]

        for peft_configg, adapter_namee in zip(peft_config, adapter_name):
            if getattr(peft_configg, "modules_to_save", None) is not None:
                if self.modules_to_save is None:
                    self.modules_to_save = set(peft_configg.modules_to_save)
                else:
                    self.modules_to_save.update(peft_configg.modules_to_save)
                _set_trainable(self, adapter_namee)

    def save_pretrained(self, save_directory, **kwargs):
        if os.path.isfile(save_directory):
            raise ValueError(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )
        os.makedirs(save_directory, exist_ok=True)
        self.save_loras(save_directory, **kwargs)
        self.save_aggregator(save_directory, **kwargs)

    def save_loras(self, save_directory, **kwargs):
        for adapter_name, peft_config in self.peft_config.items():
            # save only the trainable weights
            output_state_dict = get_peft_model_state_dict(
                self,
                state_dict=kwargs.get("state_dict", None),
                adapter_name=adapter_name,
            )
            output_dir = (
                os.path.join(save_directory, adapter_name)
                if adapter_name != "default"
                else save_directory
            )
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

    def save_aggregator(self, save_directory, **kwargs):
        output_state_dict = get_aggregator_state_dict(
            self,
            state_dict=kwargs.get("state_dict", None),
        )
        output_dir = save_directory
        os.makedirs(output_dir, exist_ok=True)
        torch.save(output_state_dict, os.path.join(output_dir, AGG_WEIGHTS_NAME))

        self.aggregator_config.trained_wtih_adapters = list(self.peft_config.keys())
        inference_mode = self.aggregator_config.inference_mode
        self.aggregator_config.inference_mode = True
        self.aggregator_config.save_pretrained(output_dir)
        self.aggregator_config.inference_mode = inference_mode

    def set_adapter(self, adapter_name: Union[str, List[str]]):
        """设置激活的lora。只有激活的lora才会参与前馈。不在`adapter_name`中的lora不会激活。

        Args:
            adapter_name (Union[str, List[str]]): 需要激活的lora name(s)

        """
        if isinstance(adapter_name, str):
            adapter_name = [adapter_name]
        else:
            adapter_name = adapter_name
        for adapter_namee in adapter_name:
            if adapter_namee not in self.peft_config:
                raise ValueError(f"Adapter {adapter_namee} not found.")
        self.active_adapter = adapter_name
        self.base_model.set_adapter(self.active_adapter)

    def _set_aggregator_trainable(self, adapter_name, value):
        """_summary_

        Args:
            adapter_name (_type_): _description_
            value (_type_): _description_
        """
        self.base_model._set_aggregator_trainable(adapter_name, value)

    def _set_adapter_trainable(self, adapter_name, value):
        self.base_model._set_adapter_trainable(adapter_name, value)

    def load_aggregator(self, path, **kwargs):
        from_pretrained = kwargs.pop("from_pretrained", False)
        # 不是from_pretrained方法调用的就跳过add_aggregator
        if not from_pretrained:
            from .pe_mapping import (
                AGGREGAOTR_TYPE_TO_CONFIG_MAPPING,
            )

            aggregator_config = AGGREGAOTR_TYPE_TO_CONFIG_MAPPING[
                AggregatorConfig.from_pretrained(
                    path, subfolder=kwargs.get("subfolder", None)
                ).aggregator_type
            ].from_pretrained(path, subfolder=kwargs.get("subfolder", None))
            self.add_aggregator(aggregator_config)

        # 恢复aggregator权重
        path = (
            os.path.join(path, kwargs["subfolder"])
            if kwargs.get("subfolder", None) is not None
            else path
        )

        if os.path.exists(os.path.join(path, AGG_WEIGHTS_NAME)):
            filename = os.path.join(path, AGG_WEIGHTS_NAME)
        else:
            raise ValueError(
                f"Can't find weights for {path} in {path} or in the Hugging Face Hub. "
                f"Please check that the file {AGG_WEIGHTS_NAME} is present at {path}."
            )

        aggregator_weight = torch.load(
            filename,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        # load the weights into the model
        set_aggregator_state_dict(self, aggregator_weight)
        logger.info("aggregator weight loaded")

    def add_aggregator(self, aggregator_config: AggregatorConfig):
        """设置aggregator

        Args:
            aggregator_type (AggregatorType): _description_
        """
        self.base_model.add_aggregator(aggregator_config)

    def update_aggregator_params(self, **kwargs):
        r"""设置聚合器的一些参数

        对于`StaticWeightAggregator`，可以设置`aggregator_static_weights`
        对于`MoEAggregator`，可以设置`top_k`
        若传入不适用的参数则无事发生。

        例子
            ```py
            >>> model.update_aggregator_params(top_k=3) # MoE聚合器
            ```

            ```py
            >>> model.update_aggregator_params(aggregator_static_weights={"lora_1":0.01,"lora_2":0.02}) # 静态权重聚合器
            ```
        """
        self.base_model.update_aggregator_params(**kwargs)

    def aggregator_set_top_k(self, top_k):
        kwargs = {"top_k": top_k}
        self.base_model.update_aggregator_params(**kwargs)


class MultiAdapterModelForCausalLM(MultiAdapterModel):
    def __init__(
        self,
        model,
        aggregator_config,
        peft_config: Union[PeftConfig, List[PeftConfig]],
        adapter_name: Union[str, List[str]] = "default",
        adapter_weights: Union[float, List[float]] = None,
    ):
        super().__init__(
            model, aggregator_config, peft_config, adapter_name, adapter_weights
        )


class MultiAdapterModelForSeq2SeqLM(MultiAdapterModel):
    def __init__(
        self,
        model,
        aggregator_config,
        peft_config: Union[PeftConfig, List[PeftConfig]],
        adapter_name: Union[str, List[str]] = "default",
        adapter_weights: Union[float, List[float]] = None,
    ):
        super().__init__(
            model, aggregator_config, peft_config, adapter_name, adapter_weights
        )


def get_multi_adapter_model(
    model, multi_adapter_config: MultiAdapterConfig, is_trainable: bool = None, **kwargs
) -> MultiAdapterModel:
    """插入多个adapter和aggregator

    Args:
        model (_type_): _description_
        multi_adapter_config (MultiAdapterConfig): _description_
        is_trainable (bool, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        MultiAdapterModel: _description_
    """
    from .pe_mapping import (
        MODEL_TYPE_TO_MULTI_ADAPTER_MODEL_MAPPING,
        PEFT_TYPE_TO_CONFIG_MAPPING,
        AGGREGAOTR_TYPE_TO_CONFIG_MAPPING,
    )

    # load lora config
    adapter_names = []
    lora_configs = []
    trained_paths = []
    trained_names = []
    for adapter_name, lora_config in multi_adapter_config.lora_configs.items():
        if isinstance(lora_config, LoraConfig):
            # config就说明是新增加的lora，要训练的
            adapter_names.append(adapter_name)
            lora_configs.append(lora_config)
        elif isinstance(lora_config, str):
            config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig.from_pretrained(lora_config).peft_type
            ].from_pretrained(lora_config)

            if not isinstance(config, LoraConfig):
                logger.warn(f"config是{type(config)},不是 LoraConfig，因此跳过")
                continue
            if is_trainable is not None:
                lora_config.inference_mode = not is_trainable
            lora_configs.append(config)
            trained_paths.append(lora_config)
            adapter_names.append(adapter_name)
            trained_names.append(adapter_name)
    task_type = lora_configs[0].task_type

    # 加载aggregator config
    aggregator_config = None
    if isinstance(multi_adapter_config.aggregator_config, str):
        aggregator_config = AGGREGAOTR_TYPE_TO_CONFIG_MAPPING[
            AggregatorConfig.from_pretrained(
                multi_adapter_config.aggregator_config
            ).aggregator_type
        ].from_pretrained(multi_adapter_config.aggregator_config)
    elif isinstance(multi_adapter_config.aggregator_config, AggregatorConfig):
        aggregator_config = multi_adapter_config.aggregator_config
    else:
        raise ValueError("")
    model = MODEL_TYPE_TO_MULTI_ADAPTER_MODEL_MAPPING[task_type](
        model, aggregator_config, lora_configs, adapter_names
    )
    if len(trained_paths) > 0:
        model.load_adapter(trained_paths, trained_names, **kwargs)
    if isinstance(multi_adapter_config.aggregator_config, str):
        model.load_aggregator(multi_adapter_config.aggregator_config)

    return model
