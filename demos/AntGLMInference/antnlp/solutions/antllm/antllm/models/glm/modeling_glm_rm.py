from typing import Optional
import os
import json
import torch
import torch.nn as nn
from torch.nn import HuberLoss

from solutions.antllm.antllm.utils.modeling_glm_rm_utils import masked_mean
from solutions.antllm.antllm.utils.modeling_glm_ppo_utils import hf_get_glm_embeddings
from .modeling_glm import GLMModel
from .tokenization_glm import GLMTokenizer
from transformers import AutoTokenizer, AutoModel
from transformers.utils import logging
from peft import get_peft_model, PeftModel
from solutions.antllm.antllm.models.peft.modeling_peft import AntPeftForCausalLM # NOQA


logger = logging.get_logger(__name__)

RM_HYPER_PARAMETERS_SAVE_FILE = "hyper_parameters.json"


def freeze_model(model, num_layers_unfrozen=2, model_type="glm"):
    # freeze layer
    if model_type == "chatglm2":
        hidden_layers = model.encoder.layers
    else:
        hidden_layers = model.transformer.layers
    if num_layers_unfrozen == 0:
        hidden_layers_to_freeze = list(hidden_layers)
    elif num_layers_unfrozen > 0:
        hidden_layers_to_freeze = list(hidden_layers)[:-num_layers_unfrozen]
    else:
        hidden_layers_to_freeze = []

    embeddings_to_freeze = hf_get_glm_embeddings(model)
    for layer in hidden_layers_to_freeze + embeddings_to_freeze:
        layer.requires_grad_(False)


class RewardModel(nn.Module):
    """
    Reward model base class.
    Args:
        model (nn.Module): Reward model.
        value_head (nn.Module): Value head to get reward score.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(
        self,
        model: nn.Module,
        value_head: Optional[nn.Module] = None,
        model_type: str = "glm",
        use_mean_value: bool = False,
        tokenizer: AutoTokenizer = None,
        num_head: int = 1,
        use_position_id: bool = True,
        use_normalized_reward: bool = False,
        eos_token: str = "<|endoftext|>"
    ) -> None:
        super().__init__()
        self.model = model
        self.config = model.config
        self.use_mean_value = use_mean_value
        self.use_position_id = use_position_id
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.num_head = num_head
        # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
        self.n_embd = (
            model.config.hidden_size
            if hasattr(model.config, "hidden_size")
            else model.config.n_embd
        )
        if value_head is not None:
            if value_head.out_features != num_head:
                raise ValueError(
                    "The value head of reward model's output dim should be equal to num_head!"
                )
            self.value_head = value_head
        else:
            self.value_head = nn.Linear(self.n_embd, num_head)
        self.use_normalized_reward = use_normalized_reward
        self.eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bs = input_ids.shape[0]
        # if self.model_type == "glm" and self.use_mean_value:
        #     #  set attention mask as 1 except for <|endoftext|>
        #     attention_mask = ~input_ids.eq(self.tokenizer.pad_token_id)
        #     position_ids = None

        if self.use_position_id:
            outputs = self.model(
                input_ids, position_ids=position_ids, attention_mask=attention_mask
            )
        else:
            outputs = self.model(input_ids, attention_mask=attention_mask)
        # Compatible with the output of the open-source glm, please note that Ant glm should replace modeling_glm.
        if "last_hidden_state" in outputs:
            last_hidden_states = outputs["last_hidden_state"]
            if self.model_type == "chatglm2":
                last_hidden_states = last_hidden_states.transpose(0, 1)
        else:
            last_hidden_states = outputs["last_hidden_states"]

        values = self.value_head(last_hidden_states).squeeze(-1)
        if self.use_mean_value:
            if self.model_type == "glm" or self.model_type == "chatglm2":
                non_pad_attn_single = ~input_ids.eq(self.tokenizer.pad_token_id)
                if self.num_head > 1:
                    non_pad_attn_list = []
                    for i in range(self.num_head):
                        non_pad_attn_list.append(non_pad_attn_single)
                    non_pad_attn = torch.stack(non_pad_attn_list, dim=2)
                    value = masked_mean(values, non_pad_attn.to(values.dtype), dim=1)
                else:
                    value = masked_mean(values, non_pad_attn_single.to(values.dtype), dim=1)
            else:
                # average pooling for all tokens
                if attention_mask is not None:
                    value = masked_mean(values, attention_mask, dim=1)
                else:
                    value = values.mean(dim=1)
        else:
            last_token_values = []
            for i in range(bs):
                eop_inds = (input_ids[i] == self.eos_token_id).nonzero()
                last_index = eop_inds[0].item() if len(eop_inds) > 0 else -1
                last_token_values.append(values[i, last_index])
            value = torch.stack(last_token_values)

        if self.use_normalized_reward:
            value = nn.Sigmoid()(value)
        return value

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        model_type="glm",
        use_mean_value=False,
        lora_config=None,
        use_position_id=True,
        num_head=1,
        use_normalized_reward=False,
        eos_token="<|endoftext|>"
    ):
        if model_type == "chatglm2":
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        else:
            model = GLMModel.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,  # 减少内存占用避免OOM
                torch_dtype="auto"
            )
            tokenizer = GLMTokenizer.from_pretrained(model_path)
        n_embd = (
            model.config.hidden_size
            if hasattr(model.config, "hidden_size")
            else model.config.n_embd
        )
        if lora_config is not None:
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        elif os.path.exists(
            os.path.join(model_path, "adapter_model.bin")
        ) and os.path.exists(os.path.join(model_path, "adapter_config.json")):
            model = PeftModel.from_pretrained(model, model_path, is_trainable=True)

        value_head = nn.Linear(n_embd, num_head)
        if os.path.exists(os.path.join(model_path, "head.bin")):
            value_head_state_dict = torch.load(
                os.path.join(model_path, "head.bin"), map_location="cpu"
            )
            value_head.load_state_dict(value_head_state_dict)

        if os.path.exists(os.path.join(model_path, RM_HYPER_PARAMETERS_SAVE_FILE)):
            with open(os.path.join(model_path, RM_HYPER_PARAMETERS_SAVE_FILE)) as f:
                rm_hyper_parameters = json.load(f)
            num_head = rm_hyper_parameters.get("num_head", num_head)
            use_mean_value = rm_hyper_parameters.get("use_mean_value", use_mean_value)
            use_position_id = rm_hyper_parameters.get("use_position_id", use_position_id)
            use_normalized_reward = rm_hyper_parameters.get("use_normalized_reward", use_normalized_reward)
            eos_token = rm_hyper_parameters.get("eos_token", eos_token)

        rw_model = cls(
            model=model,
            value_head=value_head,
            model_type=model_type,
            use_mean_value=use_mean_value,
            tokenizer=tokenizer,
            num_head=num_head,
            use_position_id=use_position_id,
            use_normalized_reward=use_normalized_reward,
            eos_token=eos_token
        )
        return rw_model


class RewardModelForPairWise(nn.Module):
    def __init__(
        self,
        model_path,
        model_type="glm",
        use_mean_value=False,
        lora_config=None,
        use_position_id=True,
        num_layers_unfrozen=2,
        use_normalized_reward=False
    ):
        super().__init__()
        self.model = RewardModel.from_pretrained(
            model_path=model_path,
            model_type=model_type,
            use_mean_value=use_mean_value,
            lora_config=lora_config,
            use_position_id=use_position_id,
            use_normalized_reward=use_normalized_reward
        )
        self.config = self.model.config
        freeze_model(self.model.model, num_layers_unfrozen=num_layers_unfrozen, model_type=model_type)

    def forward(
        self,
        input_ids_chosen=None,
        input_ids_rejected=None,
        attention_mask_chosen=None,
        attention_mask_rejected=None,
        position_ids_chosen=None,
        position_ids_rejected=None,
    ):
        loss = None
        input_ids_chosen = input_ids_chosen.squeeze(1)
        if attention_mask_chosen is not None:
            attention_mask_chosen = attention_mask_chosen.squeeze(1)
        chosen_reward = self.model(
            input_ids_chosen,
            attention_mask=attention_mask_chosen,
            position_ids=position_ids_chosen,
        )

        input_ids_rejected = input_ids_rejected.squeeze(1)
        if attention_mask_rejected is not None:
            attention_mask_rejected = attention_mask_rejected.squeeze(1)
        rejected_reward = self.model(
            input_ids_rejected,
            attention_mask=attention_mask_rejected,
            position_ids=position_ids_rejected,
        )

        # logits = torch.sigmoid(chosen_reward - rejected_reward)
        # 参考anthropic的loss定义
        logits = 1 + torch.exp(rejected_reward - chosen_reward)
        if self.training:
            # loss = -torch.log(logits).mean()
            loss = torch.log(logits).mean()

            return {"loss": loss, "logits": logits}
        else:
            return {
                "logits": logits,
                "chosen_reward": chosen_reward,
                "rejected_reward": rejected_reward,
            }


class RewardModelForPointWise(nn.Module):
    def __init__(
        self,
        model_path,
        num_head,
        model_type="glm",
        use_mean_value=False,
        lora_config=None,
        use_position_id=True,
        num_layers_unfrozen=2,
        use_normalized_reward=False
    ):
        super().__init__()
        self.model = RewardModel.from_pretrained(
            model_path=model_path,
            model_type=model_type,
            use_mean_value=use_mean_value,
            lora_config=lora_config,
            num_head=num_head,
            use_position_id=use_position_id,
            use_normalized_reward=use_normalized_reward
        )
        self.config = self.model.config
        self.num_head = num_head
        freeze_model(self.model.model, num_layers_unfrozen=num_layers_unfrozen)

    def forward(
        self,
        input_ids_answer: Optional[torch.Tensor] = None,
        attention_mask_answer: Optional[torch.Tensor] = None,
        position_ids_answer=None,
        labels: Optional[torch.Tensor] = None,
    ):
        loss = None
        input_ids_answer = input_ids_answer.squeeze(1)
        if attention_mask_answer is not None:
            attention_mask_answer = attention_mask_answer.squeeze(1)

        reward = self.model(
            input_ids_answer, attention_mask=attention_mask_answer, position_ids=position_ids_answer
        )

        if self.num_head == 1: 
            labels = torch.squeeze(labels)
        labels = labels.to(reward.dtype)

        if self.model.use_normalized_reward:
            loss_fn = nn.BCELoss()
        else:
            loss_fn = HuberLoss()

        if self.num_head == 1: 
            reward_engaged = reward
            labels_engaged = labels
        else:
            reward_engaged = torch.sum(reward, dim=1)
            labels_engaged = torch.sum(labels, dim=1)
        
        loss = torch.tensor(0., device=input_ids_answer.device)
        if self.training:
            loss = loss_fn(reward, labels)

        return {
            "logits": reward_engaged, 
            "labels": labels_engaged, 
            "loss": loss,
        }
