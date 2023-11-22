# coding=utf-8
# Copyright 2022 shunxing1234 The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch GLM model. """

import copy
import inspect
import math
import os
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, LayerNorm, Linear, init
from torch.nn.parameter import Parameter
from solutions.antllm.antllm.utils.activations import ACT2FN
from transformers.generation.beam_constraints import (DisjunctiveConstraint,
                                                      PhrasalConstraint)
from transformers.generation.beam_search import (BeamSearchScorer,
                                                 ConstrainedBeamSearchScorer)
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList, validate_stopping_criteria)
from transformers.generation.utils import (GreedySearchDecoderOnlyOutput,
                                           GreedySearchEncoderDecoderOutput,
                                           GreedySearchOutput,
                                           SampleDecoderOnlyOutput,
                                           SampleEncoderDecoderOutput,
                                           SampleOutput)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions, ModelOutput,
    SequenceClassifierOutput)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (add_code_sample_docstrings,
                                add_start_docstrings,
                                add_start_docstrings_to_model_forward, logging)

from solutions.antllm.antllm.utils.longglm_utils import (LongGLMMemCache,
                                                         mem_apply_update)

from .configuration_glm import GLMConfig

_CHECKPOINT_FOR_DOC = "shunxing1234/GLM"
_CONFIG_FOR_DOC = "GLMConfig"
_TOKENIZER_FOR_DOC = "GLMTokenizer"

GLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "shunxing1234/GLM",
    # See all GLM models at https://huggingface.co/models?filter=glm
]

logger = logging.get_logger(__name__)


def check_fa2():
    try:
        from packaging import version
        import flash_attn
        return version.parse(flash_attn.__version__) >= version.parse("2.0.0")
    except Exception:
        return False


def detect_repetition(input_ids, min_repeat_tokens=3):
    finished_indexes = []
    input_id_list = [input_ids[i].cpu().detach().numpy().tolist()
                     for i in range(len(input_ids))]
    for i in range(len(input_ids)):
        finish_flag = False
        for num_repeat_tokens in range(input_ids[i].shape[0] // min_repeat_tokens, min_repeat_tokens - 1, -1):
            if num_repeat_tokens == 1:
                min_repeat_times = 4
            elif num_repeat_tokens <= 3:
                min_repeat_times = 3
            else:
                min_repeat_times = 2
            substr = ','.join([str(int(i)) for i in input_id_list[i][input_ids[i].shape[0] - num_repeat_tokens:]])
            last_start = input_ids[i].shape[0] - (num_repeat_tokens * 2 + 1)
            last_end = input_ids[i].shape[0] - num_repeat_tokens

            correct_end = -1
            flag = True
            for t in range(min_repeat_times):
                last_substr = ','.join(
                    [str(int(i)) for i in input_id_list[i][last_start:last_end]])
                if last_substr.endswith(substr):
                    correct_end = last_end
                    last_start = last_end - 2 * num_repeat_tokens - 1
                    last_end = last_end - num_repeat_tokens
                elif last_substr.startswith(substr):
                    correct_end = last_start + num_repeat_tokens
                    last_end = last_start
                    last_start = last_start - num_repeat_tokens - 1
                else:
                    flag = False
                    break
            if flag:
                finished_indexes.append(i)
                input_id_list[i] = input_ids[i][:correct_end]
                finish_flag = True
                break
        if not finish_flag:
            input_id_list[i] = input_ids[i]
    return finished_indexes, input_id_list


def check_gpu_sm75_or_greater():
    '''
    Check that the gpu is capable of running flash attention
    ref: https://github.com/pytorch/pytorch/blob/d6dd67a2488c7e17fbf010eee805f1cb2d64ba28/aten/src/ATen/native/transformers/cuda/sdp_utils.h#L356
    '''  # noqa
    if not torch.cuda.is_available():
        return False
    capabilities = torch.cuda.get_device_capability()
    is_sm75 = capabilities[0] == 7 and capabilities[1] == 5
    is_sm8x = capabilities[0] == 8 and capabilities[1] >= 0
    if is_sm75 or is_sm8x:
        return True
    return False


def unscaled_init_method(sigma):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method(mean, std, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = std / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=mean, std=std)

    return init_


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(tensor, num_partitions, partition_sizes=None,
                                contiguous_split_chunks=False):
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    if partition_sizes is None:
        last_dim = tensor.dim() - 1
        last_dim_size = divide(tensor.size()[last_dim], num_partitions)
        # Split.
        tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    else:
        assert isinstance(partition_sizes, (List, Tuple))
        tensor_list = torch.split(tensor, partition_sizes, dim=-1)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class MLP(torch.nn.Module):
    """MLP for GPT2.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform gelu transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.

    Arguments:
        hidden_size: The hidden size of the self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
        output_layer_init_method: output layer initialization. If None,
                                  use `init_method`.
    """

    def __init__(self, hidden_size, output_dropout_prob, init_method,
                 output_layer_init_method=None, intermediate_size=None, use_swiglu=False):
        super(MLP, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        # Project to 4h.
        if use_swiglu:
            # If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
            self.dense_h_to_4h = Linear(hidden_size, 4 * 2 * hidden_size)
        else:
            self.dense_h_to_4h = Linear(hidden_size, 4 * hidden_size)

        # Project back to h.
        self.dense_4h_to_h = Linear(
            intermediate_size if intermediate_size else 4 * hidden_size,
            hidden_size)

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]

        # self.activation_func = swiglu if use_swiglu else gelu
        self.activation_func = ACT2FN["swiglu" if use_swiglu else "gelu"]
        self.dropout = torch.nn.Dropout(output_dropout_prob)

    def forward(self, hidden_states):
        # [b, s, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)

        # [b, s, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        output = self.dropout(output)
        return output


class VocabEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, config):
        super(VocabEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = config.vocab_size
        self.embedding_dim = config.hidden_size
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None

        self.vocab_start_index = 0
        self.vocab_end_index = self.num_embeddings

        # Allocate weights.
        self.weight = Parameter(torch.Tensor(self.num_embeddings,
                                             self.embedding_dim))
        # And initialize.
        init.xavier_normal_(self.weight)

    def forward(self, input_):
        # Get the embeddings.
        output = F.embedding(input_, self.weight,
                             self.padding_idx, self.max_norm,
                             self.norm_type, self.scale_grad_by_freq,
                             self.sparse)
        return output


class PositionalEmbedding(torch.nn.Module):

    def __init__(self, hidden_size):
        super(PositionalEmbedding, self).__init__()

        self.hidden_size = hidden_size

        inv_freq = 1 / (10000 ** (torch.arange(0.0, hidden_size, 2.0) / hidden_size))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        else:
            return pos_emb[None, :, :]


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    # batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, slen, n_rep, num_key_value_heads, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


class SelfAttention(torch.nn.Module):
    """self-attention layer for GLM.

    Self-attention layer takes input with size [b, s, h] where b is
    the batch size, s is the sequence lenght, and h is the hidden size
    and creates output of the same size.
    Arguments:
        hidden_size: total hidden size of the layer (h).
        num_attention_heads: number of attention heads (n). Note that we
                             require n to be divisible by number of GPUs
                             used to parallelize the model. Also, we
                             require hidden size to be divisible by n.
        attention_dropout_prob: dropout probability for the attention scores.
        init_method: weight initialization.
        output_layer_init_method: output layer initialization. If None, use
                                  `init_method`.
    We use the following notation:
        h: hidden_size
        n: num_attention_heads
        p: number of partitions
        np: n/p
        hp: h/p
        hn: h/n
        b: batch size
        s: sequence length
    """

    def __init__(self, hidden_size, num_attention_heads,
                 attention_dropout_prob, output_dropout_prob,
                 init_method, output_layer_init_method=None,
                 attention_scale=1.0,
                 num_key_value_heads=0,
                 rotary_type='none',
                 bf16=False):
        super(SelfAttention, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        # Per attention head and per partition values.
        self.hidden_size = hidden_size
        self.hidden_size_per_attention_head = divide(hidden_size,
                                                     num_attention_heads)

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads > 0 else self.num_attention_heads
        self.no_mha = self.num_key_value_heads != self.num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.attention_scale = attention_scale
        self.rotary_type = rotary_type
        self.bf16 = bf16
        # Strided linear layer.
        if self.no_mha:  # not MHA
            self.query_key_value = Linear(
                hidden_size, hidden_size + 2 * self.hidden_size_per_attention_head * self.num_key_value_heads)
        else:
            self.query_key_value = Linear(hidden_size, 3 * hidden_size)

        self.rotary_type = rotary_type
        if self.rotary_type in ['1d', '2d']:
            self.rotary_emb = RotaryEmbedding(
                self.hidden_size // (self.num_attention_heads * 2),
                base=10000,
                precision=torch.half if not self.bf16 else torch.bfloat16,
                learnable=False
            )
        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        # Output.
        self.dense = Linear(hidden_size,
                            hidden_size)
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size(
        )[:-1] + (self.num_attention_heads, self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states, position_ids, ltor_mask, past_key_value=None, use_cache=False):
        # hidden_states: [b, s, h]
        # ltor_mask: [b,1,s,s]

        # Attention heads. [b, s, hp]
        # query_length = hidden_states.size(1)
        # self attention
        mixed_x_layer = self.query_key_value(hidden_states)
        (mixed_query_layer, mixed_key_layer,
         mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3, [
             self.num_attention_heads * self.hidden_size_per_attention_head,
             self.num_key_value_heads * self.hidden_size_per_attention_head,
             self.num_key_value_heads * self.hidden_size_per_attention_head,
         ])

        query_layer = mixed_query_layer.view(
            *mixed_query_layer.size()[:-1],
            self.num_attention_heads,
            self.hidden_size_per_attention_head
        )
        key_layer = mixed_key_layer.view(
            *mixed_key_layer.size()[:-1],
            self.num_key_value_heads,
            self.hidden_size_per_attention_head
        )
        value_layer = mixed_value_layer.view(
            *mixed_value_layer.size()[:-1],
            self.num_key_value_heads,
            self.hidden_size_per_attention_head
        )

        if self.rotary_type in ['1d', '2d']:
            q1, q2 = query_layer.chunk(2, dim=(query_layer.ndim - 1))
            k1, k2 = key_layer.chunk(2, dim=(key_layer.ndim - 1))
            cos, sin = self.rotary_emb(q1, seq_len=position_ids.max() + 1)
            position_ids, block_position_ids = position_ids[:, 0, :].contiguous(), \
                position_ids[:, 1, :].contiguous()
            q1, k1 = apply_rotary_pos_emb_index(q1, k1, cos, sin, position_ids)
            if self.rotary_type in ['2d']:
                q2, k2 = apply_rotary_pos_emb_index(q2, k2, cos, sin, block_position_ids)
            query_layer = torch.concat([q1, q2], dim=(q1.ndim - 1)).to(value_layer.dtype)  # todo fix output float32
            key_layer = torch.concat([k1, k2], dim=(k1.ndim - 1)).to(value_layer.dtype)

        if past_key_value is not None:
            key_layer = torch.cat([past_key_value[0], key_layer], dim=1)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=1)

        past_key_value = [key_layer, value_layer] if use_cache else None

        if self.no_mha:
            key_layer = repeat_kv(key_layer, self.num_key_value_groups)
            value_layer = repeat_kv(value_layer, self.num_key_value_groups)

        # Reshape and transpose [b, np, s, hn]
        query_layer = query_layer.permute(0, 2, 1, 3)
        key_layer = key_layer.permute(0, 2, 1, 3)
        value_layer = value_layer.permute(0, 2, 1, 3)

        if self.attention_scale > 1.0:
            # Raw attention scores. [b, np, s, s]
            attention_scores = torch.matmul(query_layer / math.sqrt(self.attention_scale),
                                            key_layer.transpose(-1, -2) / math.sqrt(
                                                self.hidden_size_per_attention_head * self.attention_scale))
        else:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2) / math.sqrt(
                self.hidden_size_per_attention_head))

        # Apply the left to right attention mask.
        ltor_mask = ltor_mask.type_as(attention_scores)
        attention_scores = torch.mul(attention_scores, ltor_mask)
        if self.attention_scale > 1.0:
            max_attention_scores = attention_scores.max(
                dim=-1, keepdim=True)[0]
            attention_scores -= max_attention_scores
            attention_scores *= self.attention_scale

        attention_scores = attention_scores + (-65504.0) * (1.0 - ltor_mask)
        # Attention probabilities. [b, np, s, s]
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # with get_cuda_rng_tracker().fork():
        attention_probs = self.attention_dropout(attention_probs)

        # Context layer.
        # [b, np, s, hn]
        context_layer = torch.matmul(attention_probs, value_layer)
        # [b, s, np, hn]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        # [b, s, hp]
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output. [b, s, h]
        output = self.dense(context_layer)
        output = self.output_dropout(output)

        return output, past_key_value


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half, learnable=False):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        # inv_freq 保留float精度，避免bf16损失
        # inv_freq = inv_freq.to(precision)
        self.learnable = learnable
        if learnable:
            self.inv_freq = torch.nn.Parameter(inv_freq)
            self.max_seq_len_cached = None
        else:
            self.register_buffer('inv_freq', inv_freq)
            self.max_seq_len_cached = None
            self.cos_cached = None
            self.sin_cached = None
        self.precision = precision

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        pass

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len
            inv_freq = 1. / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            freqs = torch.einsum('i,j->ij', t, inv_freq.to(x.device))
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()

            # [sx, 1 (b * np), hn]
            cos_cached = emb.cos()[:, None, :]
            sin_cached = emb.sin()[:, None, :]
            if self.precision == torch.bfloat16:
                cos_cached = cos_cached.bfloat16()
                sin_cached = sin_cached.bfloat16()
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]

    def _apply(self, fn):
        if self.cos_cached is not None:
            self.cos_cached = fn(self.cos_cached)
        if self.sin_cached is not None:
            self.sin_cached = fn(self.sin_cached)
        return super()._apply(fn)


class LinearScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with linear scaling."""

    def __init__(
        self,
        dim,
        base=10000,
        precision=torch.half,
        learnable=False,
        max_embedding_length=2048,
        scaling_factor=1.0
    ):
        self.scaling_factor = scaling_factor
        self.max_embedding_length = max_embedding_length
        super().__init__(dim, base, precision, learnable)

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len

            inv_freq = 1. / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            t = t / self.scaling_factor
            freqs = torch.einsum('i,j->ij', t, inv_freq.to(x.device))
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()

            # [sx, 1 (b * np), hn]
            cos_cached = emb.cos()[:, None, :]
            sin_cached = emb.sin()[:, None, :]
            if self.precision == torch.bfloat16:
                cos_cached = cos_cached.bfloat16()
                sin_cached = sin_cached.bfloat16()
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


class NTKScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with Dynamic NTK scaling."""

    def __init__(
        self,
        dim,
        base=10000,
        precision=torch.half,
        learnable=False,
        max_embedding_length=2048,
        scaling_factor=1.0
    ):
        self.scaling_factor = scaling_factor
        self.max_embedding_length = max_embedding_length
        super().__init__(dim, base, precision, learnable)

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len

            base = self.base
            if seq_len > self.max_embedding_length:
                base = self.base * (
                    (self.scaling_factor * seq_len / self.max_embedding_length) - (self.scaling_factor - 1)
                ) ** (self.dim / (self.dim - 2))

            inv_freq = 1. / (base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            freqs = torch.einsum('i,j->ij', t, inv_freq.to(x.device))
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()

            # [sx, 1 (b * np), hn]
            cos_cached = emb.cos()[:, None, :]
            sin_cached = emb.sin()[:, None, :]
            if self.precision == torch.bfloat16:
                cos_cached = cos_cached.bfloat16()
                sin_cached = sin_cached.bfloat16()
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions


@torch.jit.script
def apply_rotary_pos_emb_index(q, k, cos, sin, position_id):
    # position_id: [sq, b], q, k: [sq, b, np, hn], cos: [sq, 1, hn] -> [sq, b, 1, hn]
    cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(2), \
        F.embedding(position_id, sin.squeeze(1)).unsqueeze(2)
    q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    return q, k


class FASelfAttention(torch.nn.Module):
    """SelfAttention layer optimized by flash-attention for HuggingFace GLM.
    source: https://huggingface.co/THUDM/glm-10b/blob/main/modeling_glm.py#L213

    Self-attention layer takes input with size [b, s, h] where b is
    the batch size, s is the sequence lenght, and h is the hidden size
    and creates output of the same size.
    Arguments:
        hidden_size: total hidden size of the layer (h).
        num_attention_heads: number of attention heads (n). Note that we
                             require n to be divisible by number of GPUs
                             used to parallelize the model. Also, we
                             require hidden size to be divisible by n.
        attention_dropout_prob: dropout probability for the attention scores.
        init_method: weight initialization.
        output_layer_init_method: output layer initialization. If None, use
                                  `init_method`.
    We use the following notation:
        h: hidden_size
        n: num_attention_heads
        p: number of partitions
        np: n/p
        hp: h/p
        hn: h/n
        b: batch size
        s: sequence length
    """

    def __init__(
            self,
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            init_method,
            output_layer_init_method=None,
            attention_scale=1.0,
            num_key_value_heads=0,
            flash_attn_function=None,
            rotary_type='none',
            bf16=False
    ):
        super(FASelfAttention, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        # Per attention head and per partition values.
        self.hidden_size = hidden_size
        self.hidden_size_per_attention_head = divide(hidden_size,
                                                     num_attention_heads)
        self.flash_attn_function = flash_attn_function

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads > 0 else self.num_attention_heads
        self.no_mha = self.num_key_value_heads != self.num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.attention_scale = attention_scale
        # Strided linear layer.
        if self.no_mha:  # not MHA
            self.query_key_value = Linear(
                hidden_size, hidden_size + 2 * self.hidden_size_per_attention_head * self.num_key_value_heads)
        else:
            self.query_key_value = Linear(hidden_size, 3 * hidden_size)

        self.rotary_type = rotary_type
        self.bf16 = bf16

        if self.rotary_type in ['1d', '2d']:
            # print('use rotary embedding')
            self.rotary_emb = RotaryEmbedding(
                self.hidden_size // (self.num_attention_heads * 2),
                base=10000,
                precision=torch.half if not self.bf16 else torch.bfloat16,
                learnable=False
            )
        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        # Output.
        self.dense = Linear(hidden_size,
                            hidden_size)
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size(
        )[:-1] + (self.num_attention_heads, self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states, position_ids, ltor_mask, past_key_value=None, use_cache=False):
        # hidden_states: [b, s, h]
        # ltor_mask: [b,1,s,s]

        # Attention heads. [b, s, hp]
        # self attention
        mixed_x_layer = self.query_key_value(hidden_states)
        (mixed_query_layer, mixed_key_layer,
         mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3, [
             self.num_attention_heads * self.hidden_size_per_attention_head,
             self.num_key_value_heads * self.hidden_size_per_attention_head,
             self.num_key_value_heads * self.hidden_size_per_attention_head,
         ])

        # Reshape and transpose [b, s, np, hn]
        query_layer = mixed_query_layer.view(
            *mixed_query_layer.size()[:-1],
            self.num_attention_heads,
            self.hidden_size_per_attention_head
        )
        key_layer = mixed_key_layer.view(
            *mixed_key_layer.size()[:-1],
            self.num_key_value_heads,
            self.hidden_size_per_attention_head
        )
        value_layer = mixed_value_layer.view(
            *mixed_value_layer.size()[:-1],
            self.num_key_value_heads,
            self.hidden_size_per_attention_head
        )

        if self.rotary_type in ['1d', '2d']:
            q1, q2 = query_layer.chunk(2, dim=(query_layer.ndim - 1))
            k1, k2 = key_layer.chunk(2, dim=(key_layer.ndim - 1))
            cos, sin = self.rotary_emb(q1, seq_len=position_ids.max() + 1)
            position_ids, block_position_ids = position_ids[:, 0, :].contiguous(), \
                position_ids[:, 1, :].contiguous()
            # print(f'q1: {q1.dtype}, k1: {k1.dtype}, q2: {q2.dtype}, k2: {k2.dtype}')
            # import pdb; pdb.set_trace()

            q1, k1 = apply_rotary_pos_emb_index(q1, k1, cos, sin, position_ids)
            if self.rotary_type in ['2d']:
                q2, k2 = apply_rotary_pos_emb_index(q2, k2, cos, sin, block_position_ids)
            # print(f'qq1: {q1.dtype}, kk1: {k1.dtype}, qq2: {q2.dtype}, kk2: {k2.dtype}')
            # value_layer = value_layer.half() if not self.bf16 else value_layer.to(torch.bfloat16)
            query_layer = torch.concat([q1, q2], dim=(q1.ndim - 1)).to(value_layer.dtype)  # todo fix output float32
            key_layer = torch.concat([k1, k2], dim=(k1.ndim - 1)).to(value_layer.dtype)
            # print(f'query_layer: {query_layer.dtype}, key_layer: {key_layer.dtype}')

        if past_key_value is not None:
            key_layer = torch.cat([past_key_value[0], key_layer], dim=1)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=1)

        past_key_value = [key_layer, value_layer] if use_cache else None

        if self.no_mha:
            key_layer = repeat_kv(key_layer, self.num_key_value_groups)
            value_layer = repeat_kv(value_layer, self.num_key_value_groups)

        dropout_p = self.attention_dropout.p if self.training else 0.0

        if check_fa2():
            if ltor_mask.dtype != torch.int32:
                ltor_mask = ltor_mask.to(torch.int32)
            context_layer = self.flash_attn_function(
                query_layer, key_layer, value_layer, causal=True, glm_mask=ltor_mask, dropout_p=dropout_p
            )
        else:
            if ltor_mask.dtype != query_layer.dtype:
                ltor_mask = ltor_mask.to(query_layer.dtype)
            context_layer = self.flash_attn_function(
                query_layer, key_layer, value_layer, mask=(-65504.0) * (1.0 - ltor_mask), dropout_p=dropout_p
            )

        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        # [b, s, hp]
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output. [b, s, h]
        # print(f'context_layer: {context_layer.dtype}, {self.dense.weight.dtype}')
        output = self.dense(context_layer)
        output = self.output_dropout(output)

        return output, past_key_value


class LongGLMAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper with FoT modifications"""

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_dropout_prob,
        output_dropout_prob,
        init_method,
        output_layer_init_method=None,
        attention_scale=1.0,
        max_sequence_length=1024,
        flash_attn_function=None,
        positionals: bool = False,
        attention_grouping: Optional[Tuple[int, int]] = (4, 128),
        cache_in_memory: bool = False,
        rotary_type: str = "none",
        bf16: bool = False
    ):
        super().__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        # Per attention head and per partition values.
        self.bf16 = bf16
        self.hidden_size = hidden_size
        self.hidden_size_per_attention_head = divide(hidden_size,
                                                     num_attention_heads)
        self.flash_attn_function = flash_attn_function

        self.num_attention_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.attention_scale = attention_scale
        # Strided linear layer.
        self.query_key_value = Linear(hidden_size, 3 * hidden_size)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        # Output.
        self.dense = Linear(hidden_size,
                            hidden_size)
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)

        # Whether cache kv activations in memory
        self.cache_in_memory = cache_in_memory

        # Rotary Embedding setting
        self.rotary_type = rotary_type

        if self.rotary_type is not None and ("1d" in self.rotary_type or "2d" in self.rotary_type):
            # print('use rotary embedding')
            if "ntk" in self.rotary_type:
                self.rotary_emb = NTKScalingRotaryEmbedding(
                    self.hidden_size // (self.num_attention_heads * 2),
                    base=10000,
                    precision=torch.half if not self.bf16 else torch.bfloat16,
                    learnable=False,
                    max_embedding_length=max_sequence_length,
                    scaling_factor=1
                )
            elif "linear" in self.rotary_type:
                self.rotary_emb = LinearScalingRotaryEmbedding(
                    self.hidden_size // (self.num_attention_heads * 2),
                    base=10000,
                    precision=torch.half if not self.bf16 else torch.bfloat16,
                    learnable=False,
                    max_embedding_length=max_sequence_length,
                    scaling_factor=1
                )
            else:
                self.rotary_emb = RotaryEmbedding(
                    self.hidden_size // (self.num_attention_heads * 2),
                    base=10000,
                    precision=torch.half if not self.bf16 else torch.bfloat16,
                    learnable=False
                )

        # Cache setting.
        self.max_position_embeddings = max_sequence_length
        self.max_cache = self.max_position_embeddings
        self.positionals = positionals
        self.attention_grouping = attention_grouping

        if (self.head_dim * self.num_attention_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_attention_heads})."
            )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        ltor_mask: Optional[torch.Tensor],
        mem: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        external_mem_cache: Optional[LongGLMMemCache] = None,
    ):
        bsz, query_length, _ = hidden_states.size()

        mem_no_local_cache = mem is None and (not use_cache)
        mem_and_local_cache = use_cache
        # positonal embeddings can be disabled for memory layers
        use_positionals = self.positionals
        if mem_no_local_cache:
            # the whole context window will be moved to memory cache after the attention
            if use_positionals:
                # positionally embedd memory content as first token in the sequence
                # rfst_key_states = rotate_as_if_first(key_states, self.rotary_emb)
                raise NotImplementedError(f"The positional FOT need ROPE, which is not implemented yet.")
            else:
                rfst_hidden_states = hidden_states
            # attention_mask [bsz, 1, tgt_seq_len, src_seq_len]
            # we base the mask on the last token in the context window
            mem_update = LongGLMMemCache(
                hiddens=rfst_hidden_states.to(hidden_states.dtype),
                masks=ltor_mask[..., -1, :, None],
            )

        mixed_x_layer = self.query_key_value(hidden_states)
        (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # Reshape and transpose [b, s, np, hn]
        query_layer = query_layer.view(
            *query_layer.size()[:-1],
            self.num_attention_heads,
            self.hidden_size_per_attention_head
        )
        key_layer = key_layer.view(
            *key_layer.size()[:-1],
            self.num_attention_heads,
            self.hidden_size_per_attention_head
        )
        value_layer = value_layer.view(
            *value_layer.size()[:-1],
            self.num_attention_heads,
            self.hidden_size_per_attention_head
        )

        # if self.use_rotary:
        #     if self.position_encoding_2d:
        #         q1, q2 = query_layer.chunk(2, dim=(query_layer.ndim - 1))
        #         k1, k2 = key_layer.chunk(2, dim=(key_layer.ndim - 1))
        #         cos, sin = self.rotary_emb(q1, seq_len=position_ids.max() + 1)
        #         position_ids, block_position_ids = position_ids[:, 0, :].contiguous(), \
        #             position_ids[:, 1, :].contiguous()
        #         # print(f'q1: {q1.dtype}, k1: {k1.dtype}, q2: {q2.dtype}, k2: {k2.dtype}')
        #         # import pdb; pdb.set_trace()
        #         q1, k1 = apply_rotary_pos_emb_index(q1, k1, cos, sin, position_ids)
        #         q2, k2 = apply_rotary_pos_emb_index(q2, k2, cos, sin, block_position_ids)
        #         # print(f'qq1: {q1.dtype}, kk1: {k1.dtype}, qq2: {q2.dtype}, kk2: {k2.dtype}')
        #         # value_layer = value_layer.half() if not self.bf16 else value_layer.to(torch.bfloat16)
        #         query_layer = torch.concat([q1, q2], dim=(q1.ndim - 1)).to(value_layer.dtype)
        #         key_layer = torch.concat([k1, k2], dim=(k1.ndim - 1)).to(value_layer.dtype)
        #         # print(f'query_layer: {query_layer.dtype}, key_layer: {key_layer.dtype}')
        #     else:
        #         position_ids = position_ids.transpose(0, 1)
        #         cos, sin = self.rotary_emb(value_layer, seq_len=position_ids.max() + 1)
        #         # [seq_len, batch, num_attention_heads, hidden_size_per_attention_head]
        #         query_layer, key_layer = apply_rotary_pos_emb_index(query_layer, key_layer, cos, sin, position_ids)

        if "1d" in self.rotary_type or "2d" in self.rotary_type:
            q1, q2 = query_layer.chunk(2, dim=(query_layer.ndim - 1))
            k1, k2 = key_layer.chunk(2, dim=(key_layer.ndim - 1))
            cos, sin = self.rotary_emb(q1, seq_len=position_ids.max() + 1)
            position_ids, block_position_ids = position_ids[:, 0, :].contiguous(), \
                position_ids[:, 1, :].contiguous()

            q1, k1 = apply_rotary_pos_emb_index(q1, k1, cos, sin, position_ids)
            if "2d" in self.rotary_type:
                q2, k2 = apply_rotary_pos_emb_index(q2, k2, cos, sin, block_position_ids)
            query_layer = torch.concat([q1, q2], dim=(q1.ndim - 1)).to(value_layer.dtype)  # todo fix output float32
            key_layer = torch.concat([k1, k2], dim=(k1.ndim - 1)).to(value_layer.dtype)

        if mem is not None:
            assert mem[0].size() == mem[1].size()

            past_local_cache_size = mem[0].shape[1]
            cat_key_layer = torch.cat([mem[0], key_layer], dim=1)
            cat_value_layer = torch.cat([mem[1], value_layer], dim=1)
            # cat_hidden_states = torch.cat((mem, hidden_states), 1)

            if ltor_mask.shape[-2] != hidden_states.shape[1] or ltor_mask.shape[-1] != cat_key_layer.shape[1]:
                raise ValueError("attention_mask should be provided for all key_states in local context")

            # local cache is maintained so that it is <= self.max_cache
            # remaining elements are either dropped or go to memory cache
            if cat_key_layer.shape[1] > self.max_cache:
                num_elems_to_drop = past_local_cache_size
                if mem_and_local_cache:
                    drop_key_layer = cat_key_layer[:, :num_elems_to_drop, ...]
                    drop_value_layer = cat_value_layer[:, :num_elems_to_drop, ...]

                    # as memory mask use the masking of the last key in context
                    # ltor_mask [bsz, 1, tgt_seq_len, src_seq_len]
                    drop_masks = ltor_mask[..., -1, :, None]
                    drop_masks = drop_masks[:, :, :num_elems_to_drop, :]

                    # cache the mems in external mems
                    if self.cache_in_memory:
                        mem_update = LongGLMMemCache(
                            key=drop_key_layer.cpu().to(hidden_states.dtype),
                            value=drop_value_layer.cpu().to(hidden_states.dtype),
                            masks=drop_masks.cpu(),
                        )
                    else:
                        mem_update = LongGLMMemCache(
                            key=drop_key_layer.to(hidden_states.dtype),
                            value=drop_value_layer.to(hidden_states.dtype),
                            masks=drop_masks,
                        )

                    if external_mem_cache is None:
                        external_mem_cache = mem_update
                    else:
                        external_mem_cache = mem_apply_update(
                            prev_external_mem_cache=external_mem_cache, new_mem_content=mem_update
                        )

                cat_key_layer = cat_key_layer[:, num_elems_to_drop:, :]
                cat_value_layer = cat_value_layer[:, num_elems_to_drop:, :]

                ltor_mask = ltor_mask[..., num_elems_to_drop:]

            # hidden_states = cat_hidden_states
            key_layer = cat_key_layer
            value_layer = cat_value_layer

        # FoT additionally stores position_ids to support long inputs
        mem = [key_layer, value_layer] if use_cache else None

        # # Reshape and transpose [b, s, np, hn]
        # query_layer = query_layer.view(
        #     *query_layer.size()[:-1],
        #     self.num_attention_heads,
        #     self.hidden_size_per_attention_head
        # )
        # key_layer = key_layer.view(
        #     *key_layer.size()[:-1],
        #     self.num_attention_heads,
        #     self.hidden_size_per_attention_head
        # )
        # value_layer = value_layer.view(
        #     *value_layer.size()[:-1],
        #     self.num_attention_heads,
        #     self.hidden_size_per_attention_head
        # )

        kv_seq_len = key_layer.shape[1]

        # if self.use_rota

        query_layer = query_layer.transpose(1, 2)
        key_layer = key_layer.transpose(1, 2)
        value_layer = value_layer.transpose(1, 2)

        if self.attention_grouping is not None:
            attn_grouping_h, attn_grouping_q = self.attention_grouping
            if attn_grouping_h <= 0 or attn_grouping_q <= 0:
                raise ValueError("Attention grouping should be positive")
        else:
            attn_grouping_h, attn_grouping_q = self.num_attention_heads, query_length

        if external_mem_cache is not None:
            mixed_external_key_layer = external_mem_cache.key
            mixed_external_value_layer = external_mem_cache.value

            if self.cache_in_memory:
                mixed_external_key_layer = mixed_external_key_layer.to(hidden_states.device)
                mixed_external_value_layer = mixed_external_value_layer.to(hidden_states.device)

            # Reshape and transpose [b, s, np, hn]
            # external_key_layer = mixed_external_key_layer.view(
            #     *mixed_external_key_layer.size()[:-1],
            #     self.num_attention_heads,
            #     self.hidden_size_per_attention_head
            # ).transpose(1, 2)
            # external_value_layer = mixed_external_value_layer.view(
            #     *mixed_external_value_layer.size()[:-1],
            #     self.num_attention_heads,
            #     self.hidden_size_per_attention_head
            # ).transpose(1, 2)
            external_key_layer = mixed_external_key_layer.transpose(1, 2)
            external_value_layer = mixed_external_value_layer.transpose(1, 2)

        if ltor_mask.dtype != query_layer.dtype:
            ltor_mask = ltor_mask.to(query_layer.dtype)

        attn_output_h = []
        for beg_h in range(0, self.num_attention_heads, attn_grouping_h):
            end_h = min(beg_h + attn_grouping_h, self.num_attention_heads)

            attn_output_q = []
            for beg_q in range(0, query_length, attn_grouping_q):
                end_q = min(beg_q + attn_grouping_q, query_length)

                if self.flash_attn_function is not None:
                    if external_mem_cache is not None:
                        attn_keys = torch.concat(
                            [key_layer[:, beg_h: end_h], external_key_layer[:, beg_h: end_h]], dim=-2
                        )
                        attn_values = torch.concat(
                            [value_layer[:, beg_h:end_h], external_value_layer[:, beg_h: end_h]], dim=-2,
                        )

                        mem_mask = external_mem_cache.masks.squeeze(-1).unsqueeze(-2)
                        if self.cache_in_memory:
                            mem_mask = mem_mask.to(hidden_states.device)

                        assert len(mem_mask.shape) == 4
                        assert mem_mask.shape[2] == 1
                        assert mem_mask.shape[3] == external_mem_cache.key.shape[1]
                        mem_mask = torch.broadcast_to(
                            mem_mask, (mem_mask.shape[0], mem_mask.shape[1], end_q - beg_q, mem_mask.shape[3])
                        )
                        attn_mask = torch.concat([ltor_mask[:, :, beg_q:end_q], mem_mask], dim=-1)
                        assert attn_mask.shape[-1] == attn_keys.shape[-2]
                    else:
                        attn_keys = key_layer[:, beg_h: end_h]
                        attn_values = value_layer[:, beg_h: end_h]
                        attn_mask = ltor_mask[:, :, beg_q:end_q]

                    attn_queries = query_layer[:, beg_h: end_h, beg_q: end_q]

                    dropout_p = self.attention_dropout.p if self.training else 0.0
                    attn_output = self.flash_attn_function(
                        attn_queries.transpose(1, 2), attn_keys.transpose(1, 2),
                        attn_values.transpose(1, 2), mask=(-65504.0) * (1.0 - attn_mask),
                        dropout_p=dropout_p).transpose(1, 2)
                    attn_output_q.append(attn_output)
                else:
                    if self.attention_scale > 1.0:
                        attn_weights = torch.matmul(
                            query_layer[:, beg_h:end_h, beg_q:end_q] / math.sqrt(self.attention_scale),
                            key_layer[:, beg_h:end_h].transpose(2, 3) / math.sqrt(
                                self.hidden_size_per_attention_head * self.attention_scale))
                    else:
                        attn_weights = torch.matmul(
                            query_layer[:, beg_h:end_h, beg_q:end_q],
                            key_layer[:, beg_h:end_h].transpose(2, 3)
                        ) / math.sqrt(self.hidden_size_per_attention_head)

                    if attn_weights.size() != (bsz, end_h - beg_h, end_q - beg_q, kv_seq_len):
                        raise ValueError(
                            f"Attention weights should be of size {(bsz, end_h - beg_h, end_q - beg_q, kv_seq_len)},"
                            f" but is {attn_weights.size()}."
                        )

                    if ltor_mask.size() != (bsz, 1, query_length, kv_seq_len):
                        raise ValueError(
                            f"Attention mask should be of size {(bsz, 1, query_length, kv_seq_len)},"
                            f" but is {ltor_mask.size()}."
                        )
                    attn_mask = ltor_mask[:, :, beg_q:end_q]
                    attn_weights = attn_weights + (-65504.0) * (1.0 - attn_mask)
                    if external_mem_cache is not None:
                        mem_mask = external_mem_cache.masks.to(query_layer.dtype).squeeze(-1).unsqueeze(-2)
                        if self.cache_in_memory:
                            mem_mask = mem_mask.to(hidden_states.device)

                        if self.attention_scale > 1.0:
                            mem_attn_weights = torch.matmul(
                                query_layer[:, beg_h:end_h, beg_q:end_q] / math.sqrt(self.attention_scale),
                                external_key_layer[:, beg_h:end_h].transpose(2, 3)) / math.sqrt(
                                    self.hidden_size_per_attention_head * self.attention_scale)
                        else:
                            mem_attn_weights = torch.matmul(
                                query_layer[:, beg_h:end_h, beg_q:end_q],
                                external_key_layer[:, beg_h:end_h].transpose(2, 3)
                            ) / math.sqrt(self.hidden_size_per_attention_head)

                        assert mem_mask.shape[2] == 1
                        mem_attn_weights = mem_attn_weights + (-65504.0) * (1.0 - mem_mask)

                        attn_weights = torch.concat([attn_weights, mem_attn_weights], dim=-1)
                        combined_value_states = torch.concat(
                            [value_layer[:, beg_h:end_h], external_value_layer[:, beg_h:end_h]],
                            dim=-2,
                        )
                    else:
                        combined_value_states = value_layer[:, beg_h:end_h]
                    # upcast attention to fp32
                    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
                    attn_weights = self.attention_dropout(attn_weights)
                    attn_output = torch.matmul(attn_weights, combined_value_states)
                    assert attn_output.shape[-2] == end_q - beg_q
                    attn_output_q.append(attn_output)
            attn_output_h.append(torch.concat(attn_output_q, dim=-2))

        attn_output = torch.concat(attn_output_h, dim=-3)

        if attn_output.size() != (bsz, self.num_attention_heads, query_length, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_attention_heads, query_length, self.head_dim)},"
                f" but is {attn_output.size()}"
            )

        attn_output = attn_output.permute(0, 2, 1, 3)
        attn_output = attn_output.reshape(bsz, query_length, self.hidden_size)

        attn_output = self.dense(attn_output)
        attn_output = self.output_dropout(attn_output)

        if mem_no_local_cache:
            if external_mem_cache is not None:
                external_mem_cache = mem_apply_update(
                    prev_external_mem_cache=external_mem_cache, new_mem_content=mem_update
                )
            else:
                external_mem_cache = mem_update

        return attn_output, mem, external_mem_cache


class RMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        return (self.weight * hidden_states).to(input_dtype)


class GLMBlock(torch.nn.Module):
    """A single layer transformer for GLM.

    We use the following notation:
        h: hidden size
        n: number of attention heads
        b: batch size
        s: sequence length
    Transformore layer takes input with size [b, s, h] and returns an
    output of the same size.

    Arguments:
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
        output_layer_init_method: output layers (attention output and
                                  mlp output) initialization. If None,
                                  use `init_method`.
    """

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_dropout_prob,
                 output_dropout_prob,
                 layernorm_epsilon,
                 init_method,
                 output_layer_init_method=None,
                 attention_scale=1.0,
                 num_key_value_heads=0,
                 intermediate_size=None,
                 rotary_type='none',
                 use_rmsnorm=False,
                 use_swiglu=False,
                 bf16=False,
                 focused_attention: bool = False,
                 attention_grouping: Optional[Tuple[int, int]] = (4, 128),
                 cache_in_memory: bool = False
                 ):
        super(GLMBlock, self).__init__()
        # Set output layer initialization if not provided.
        self.focused_attention = focused_attention
        self.cache_in_memory = cache_in_memory
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        # Layernorm on the input data.
        LayerNormFunc = RMSNorm if use_rmsnorm else LayerNorm
        self.input_layernorm = LayerNormFunc(hidden_size, eps=layernorm_epsilon)

        # Self attention.
        try:
            if not check_gpu_sm75_or_greater():
                if torch.cuda.is_available():
                    device_model = torch.cuda.get_device_properties(
                        torch.cuda.current_device()).name
                else:
                    device_model = "cpu"
                raise Exception(
                    f'Current device {device_model} is not allowed to use flash attention. '
                    'Please use Ampere arch or Hopper arch gpus such as A100, A800, H800, etc.'
                )
            if self.focused_attention:
                from atorch.modules.transformer.layers import \
                    flash_attn_with_mask_bias
                self.attention = LongGLMAttention(
                    hidden_size,
                    num_attention_heads,
                    attention_dropout_prob,
                    output_dropout_prob,
                    init_method,
                    output_layer_init_method=output_layer_init_method,
                    attention_scale=attention_scale,
                    flash_attn_function=flash_attn_with_mask_bias,
                    attention_grouping=attention_grouping,
                    rotary_type=rotary_type,
                    cache_in_memory=cache_in_memory,
                    bf16=bf16
                )
            else:
                if check_fa2():
                    from flash_attn import flash_attn_func
                    flash_attn_function = flash_attn_func
                else:
                    from atorch.modules.transformer.layers import \
                        flash_attn_with_mask_bias
                    flash_attn_function = flash_attn_with_mask_bias
                self.attention = FASelfAttention(
                    hidden_size,
                    num_attention_heads,
                    attention_dropout_prob,
                    output_dropout_prob,
                    init_method,
                    output_layer_init_method=output_layer_init_method,
                    attention_scale=attention_scale,
                    num_key_value_heads=num_key_value_heads,
                    flash_attn_function=flash_attn_function,
                    rotary_type=rotary_type,
                    bf16=bf16)
        except Exception:
            if self.focused_attention:
                self.attention = LongGLMAttention(
                    hidden_size,
                    num_attention_heads,
                    attention_dropout_prob,
                    output_dropout_prob,
                    init_method,
                    output_layer_init_method=output_layer_init_method,
                    attention_scale=attention_scale,
                    flash_attn_function=None,
                    attention_grouping=attention_grouping,
                    rotary_type=rotary_type,
                    cache_in_memory=cache_in_memory,
                    bf16=bf16
                )
            else:
                self.attention = SelfAttention(
                    hidden_size,
                    num_attention_heads,
                    attention_dropout_prob,
                    output_dropout_prob,
                    init_method,
                    output_layer_init_method=output_layer_init_method,
                    attention_scale=attention_scale,
                    rotary_type=rotary_type,
                    num_key_value_heads=num_key_value_heads)

        # Layernorm on the input data.
        self.post_attention_layernorm = LayerNormFunc(hidden_size,
                                                      eps=layernorm_epsilon)

        # MLP
        self.mlp = MLP(
            hidden_size,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method,
            intermediate_size=intermediate_size,
            use_swiglu=use_swiglu)

    def forward(
        self, hidden_states, position_ids, ltor_mask,
        past_key_value=None, use_cache=False, external_mem_cache=None
    ):
        # hidden_states: [b, s, h]
        # ltor_mask: [b,1, s,s]

        # Layer norm at the begining of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        if self.focused_attention:
            attention_output, present_key_value, present_external_mem_cache = self.attention(
                layernorm_output, position_ids, ltor_mask, mem=past_key_value,
                use_cache=use_cache, external_mem_cache=external_mem_cache
            )
        else:
            attention_output, present_key_value = self.attention(layernorm_output, position_ids, ltor_mask,
                                                                 past_key_value=past_key_value, use_cache=use_cache)
            present_external_mem_cache = None
        # Residual connection.
        layernorm_input = hidden_states + attention_output
        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        # MLP.
        mlp_output = self.mlp(layernorm_output)
        # Second residual connection.
        output = layernorm_input + mlp_output

        return (output, present_key_value, present_external_mem_cache)


class GLMStack(torch.nn.Module):
    """GLM transformer.

    This module takes input from embedding layer and it's output can
    be used directly by a logit layer. It consists of L (num-layers)
    blocks of:
        layer norm
        self attention
        residual connection
        layer norm
        mlp
        residual connection
    followed by a final layer norm.

    Arguments:
        num_layers: Number of transformer layers.
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        checkpoint_activations: if True, checkpoint activations.
        checkpoint_num_layers: number of layers to checkpoint. This
                               is basically the chunk size in checkpoitning.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method_std: standard deviation of the init method which has
                         the form N(0, std).
        use_scaled_init_for_output_weights: If Ture use 1/sqrt(2*num_layers)
                                            scaling for the output weights (
                                            output of self attention and mlp).
    """

    def __init__(self,
                 num_layers,
                 hidden_size,
                 num_attention_heads,
                 max_sequence_length,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 layernorm_epsilon=1.0e-5,
                 init_method_std=0.02,
                 use_scaled_init_for_output_weights=True,
                 block_position_encoding=False,
                 attention_scale=1.0,
                 num_key_value_heads=0,
                 max_memory_length=0,
                 bf16=False,
                 intermediate_size=None,
                 rotary_type='none',
                 use_rmsnorm=False,
                 use_swiglu=False,
                 focused_attention: bool = False,
                 cache_in_memory: bool = False,
                 attention_grouping: Optional[Tuple[int, int]] = (4, 128)
                 ):
        super(GLMStack, self).__init__()
        self.hidden_size = hidden_size
        # Store activation checkpoiting flag.
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.max_memory_length = max_memory_length
        self.focused_attention = focused_attention
        self.cache_in_memory = cache_in_memory
        self.attention_grouping = attention_grouping
        self.bf16 = bf16

        output_layer_init_method = None
        if use_scaled_init_for_output_weights:
            output_layer_init_method = scaled_init_method(0.0, init_method_std,
                                                          num_layers)
        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)
        self.block_position_encoding = block_position_encoding
        self.rotary_type = rotary_type.lower() if rotary_type else rotary_type

        if '1d' not in self.rotary_type and '2d' not in self.rotary_type:
            # Position embedding (serial).
            if block_position_encoding:
                self.position_embeddings = torch.nn.Embedding(max_sequence_length + 1, hidden_size)
                self.block_position_embeddings = torch.nn.Embedding(max_sequence_length + 1, hidden_size)
                torch.nn.init.normal_(self.block_position_embeddings.weight, mean=0.0, std=init_method_std)
            else:
                self.position_embeddings = torch.nn.Embedding(max_sequence_length, hidden_size)
            # Initialize the position embeddings.
            torch.nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=init_method_std)

        def get_layer():

            return GLMBlock(
                hidden_size,
                num_attention_heads,
                attention_dropout_prob,
                output_dropout_prob,
                layernorm_epsilon,
                unscaled_init_method(init_method_std),
                output_layer_init_method=output_layer_init_method,
                attention_scale=attention_scale,
                num_key_value_heads=num_key_value_heads,
                intermediate_size=intermediate_size,
                rotary_type=self.rotary_type,
                use_rmsnorm=use_rmsnorm,
                use_swiglu=use_swiglu,
                bf16=self.bf16,
                focused_attention=self.focused_attention,
                cache_in_memory=self.cache_in_memory,
                attention_grouping=self.attention_grouping)

        # Transformer layers.
        self.layers = torch.nn.ModuleList(
            [get_layer() for _ in range(num_layers)])

        # Final layer norm before output.
        LayerNormFunc = RMSNorm if use_rmsnorm else LayerNorm
        self.final_layernorm = LayerNormFunc(hidden_size, eps=layernorm_epsilon)

    def forward(
        self,
        hidden_states,
        position_ids,
        attention_mask,
        past_key_values=None,
        use_cache=True,
        external_memory_states=None,
        output_hidden_states=False,
        **kwargs
    ):
        batch_size, query_length = hidden_states.size()[:2]
        past_key_value_length = 0
        if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
            past_key_value_length = past_key_values[0][0].shape[1]

        # attention mask is the beginning postion of B region, \in [0, query_len)
        is_scalar = torch.numel(attention_mask) == 1
        is_sep = is_scalar or torch.numel(attention_mask) == batch_size
        if is_sep:
            sep = attention_mask.item() if is_scalar else attention_mask

            # conventional transformer
            def build_mask_matrix(seq_length, sep, past_key_value_length=0):
                # https://github.com/pytorch/pytorch/issues/101932, fix triu/tril bf16 support
                m = hidden_states.new_ones((1, seq_length, seq_length))
                mask = torch.arange(
                    1, m.shape[-1] + 1).reshape(1, -1, 1).to(m.device)
                ids = torch.arange(
                    1, m.shape[-1] + 1).reshape(1, 1, -1).expand(1, m.shape[-1], -1).to(m.device)
                m = (ids <= mask).type_as(m)

                if is_scalar:
                    m[0, :, :int(sep)] = 1
                else:
                    m = m.expand(batch_size, -1, -1)
                    ids = torch.arange(seq_length, device=sep.device, dtype=sep.dtype).view(1, -1)
                    mask = ids < sep.view(-1, 1)
                    m = m.masked_fill(mask.unsqueeze(1).expand_as(m), 1)
                if past_key_value_length > 0:
                    m = m.expand(batch_size, -1, -1)
                    m = torch.cat((hidden_states.new_ones((batch_size, seq_length, past_key_value_length)), m), dim=2)
                m = m.unsqueeze(1)
                return m
            if not check_fa2():
                attention_mask = build_mask_matrix(query_length, sep, past_key_value_length=past_key_value_length)
        else:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask[:, :, :, -query_length - past_key_value_length:]

        if '1d' not in self.rotary_type and '2d' not in self.rotary_type:
            if self.block_position_encoding:
                position_ids, block_position_ids = position_ids[:, 0], position_ids[:, 1]
            position_embeddings = self.position_embeddings(position_ids)

            hidden_states = hidden_states + position_embeddings
            if self.block_position_encoding:
                block_position_embeddings = self.block_position_embeddings(block_position_ids)
                hidden_states = hidden_states + block_position_embeddings

        hidden_states = self.embedding_dropout(hidden_states)

        def check_detach(_hidden_states):
            return _hidden_states.detach() if _hidden_states is not None else None

        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = None
        external_mem_layers = None
        if use_cache:
            next_decoder_cache = []
            if self.focused_attention:
                external_mem_layers = []

        for i, layer in enumerate(self.layers):

            # 适配PPO逻辑
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            args = [hidden_states, position_ids, attention_mask]

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, None)

                return custom_forward

            past_key_value = past_key_values[i] if past_key_values is not None and len(past_key_values) > 0 else None
            external_mem_i_cache = external_memory_states[i] \
                if external_memory_states and len(past_key_values) > 0 else None

            if self.checkpoint_activations and self.training:
                if self.focused_attention:
                    raise NotImplementedError("The focused attention is not supported for gradient checkpointing.")
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    *args, past_key_value,
                )
            else:
                layer_outputs = layer(
                    hidden_states, position_ids, attention_mask,
                    past_key_value=past_key_value, use_cache=use_cache, external_mem_cache=external_mem_i_cache
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                if not self.focused_attention:
                    next_decoder_cache.append(layer_outputs[1])
                else:
                    external_mem_layers.append(layer_outputs[2])
                    if layer_outputs[1] is not None:
                        next_decoder_cache.append([
                            check_detach(layer_outputs[1][0]),
                            check_detach(layer_outputs[1][1]),
                        ])
                    else:
                        next_decoder_cache = None

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Final layer norm.
        output = self.final_layernorm(hidden_states)
        next_cache = next_decoder_cache if use_cache else None
        return output, next_cache, external_mem_layers, all_hidden_states

    def update_mems(self, hiddens, mems, return_memory=False):
        # todo tp fix
        memory_length = mems[0].size(1) if mems else 0
        query_length = hiddens[0].size(1)
        new_memory_length = memory_length + query_length
        if not return_memory:
            new_memory_length = min(self.max_memory_length, new_memory_length)
        new_mems = []
        # with torch.no_grad():
        for i in range(len(hiddens)):
            if new_memory_length <= query_length:
                new_mems.append(hiddens[i][:, -new_memory_length:])
            else:
                new_mems.append(torch.cat((mems[i][:, -new_memory_length + query_length:],
                                           hiddens[i].to(mems[i].device)), dim=1))
        return new_mems


class GLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = GLMConfig
    base_model_prefix = "glm"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self._no_split_modules = ["GLMBlock"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, torch.nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GLMModel):
            module.gradient_checkpointing = value
            module.transformer.checkpoint_activations = value

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        # Delete and cache the original base model prefix.
        GLMPreTrainedModel.cache_base_model_prefix = GLMPreTrainedModel.base_model_prefix
        # delattr(GLMPreTrainedModel, "base_model_prefix")

        return super(GLMPreTrainedModel, cls).from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    @classmethod
    def _load_pretrained_model(
            cls,
            *model_args,
            **kwargs
    ):
        # Restore the base model prefix.
        if hasattr(GLMPreTrainedModel, "cache_base_model_prefix"):
            GLMPreTrainedModel.base_model_prefix = GLMPreTrainedModel.cache_base_model_prefix
            delattr(GLMPreTrainedModel, "cache_base_model_prefix")
        else:
            GLMPreTrainedModel.base_model_prefix = "glm"

        return super()._load_pretrained_model(
            *model_args, **kwargs
        )


GLM_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config ([`~GLMConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

GLM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`GLMTokenizer`].
            See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range `[0, config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert *input_ids* indices into associated vectors
            than the model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare GLM Model transformer outputting raw hidden-states without any specific head on top.",
    GLM_START_DOCSTRING,
)
class GLMModel(GLMPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    `is_decoder` argument of the configuration set to `True`.
    To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder`
    argument and `add_cross_attention` set to `True`; an
    `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.output_predict = config.output_predict
        # Word embeddings (parallel).
        self.word_embeddings = VocabEmbedding(config)
        self.output_hidden_states = config.output_hidden_states
        # Transformer
        self.transformer = GLMStack(config.num_layers,
                                    config.hidden_size,
                                    config.num_attention_heads,
                                    config.max_sequence_length,
                                    config.embedding_dropout_prob,
                                    config.attention_dropout_prob,
                                    config.output_dropout_prob,
                                    config.checkpoint_activations,
                                    config.checkpoint_num_layers,
                                    attention_scale=config.attention_scale,
                                    num_key_value_heads=config.num_key_value_heads,
                                    block_position_encoding=config.block_position_encoding,
                                    max_memory_length=config.max_memory_length,
                                    intermediate_size=config.intermediate_size,
                                    rotary_type=config.rotary_type,
                                    use_rmsnorm=config.use_rmsnorm,
                                    use_swiglu=config.use_swiglu,
                                    bf16=config.bf16,
                                    focused_attention=config.focused_attention,
                                    cache_in_memory=config.cache_in_memory,
                                    attention_grouping=config.attention_grouping)

        # Initialize the gradient checkpointing config
        self.gradient_checkpointing = False

        # Initialize the gradient checkpointing config
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(GLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        mems=None,
        external_mems=None,
        use_cache=False,
        output_hidden_states=None,
        **kwargs
    ):
        batch_size = input_ids.size(0)
        if inputs_embeds is not None:
            words_embeddings = inputs_embeds
        else:
            words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings

        device = input_ids.device
        input_shape = input_ids.size()

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if position_ids is None:
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
            block_position_ids = torch.zeros(input_shape[-1], dtype=torch.long, device=device)
            position_ids = torch.stack((position_ids, block_position_ids), dim=0).unsqueeze(0)
        if attention_mask is None:
            attention_mask = torch.zeros(batch_size)
        # Transformer.
        transformer_output = self.transformer(
            embeddings, position_ids, attention_mask,
            past_key_values=mems,
            external_memory_states=external_mems,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            **kwargs
        )
        last_hidden_states, mems, external_mems, output_hidden_states = transformer_output
        logits = None
        if self.output_predict:
            last_hidden_states = last_hidden_states.to(self.word_embeddings.weight.device)  # for test
            logits = F.linear(last_hidden_states, self.word_embeddings.weight)

        return ModelOutput(
            last_hidden_states=last_hidden_states,
            logits=logits,
            mems=mems,
            external_mems=external_mems,
            hidden_states=output_hidden_states
        )


@add_start_docstrings(
    """GLM Model transformer for multiple choice classification""",
    GLM_START_DOCSTRING
)
class GLMForMultipleChoice(GLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.glm = GLMModel(config)
        self.post_init()

    def forward(
            self,
            input_ids=None,
            position_ids=None,
            attention_mask=None,
            choice_ids=None,
            choice_indices=None,
            labels=None,
            mems=None,
            **kwargs
    ):
        model_output = self.glm(input_ids, position_ids,
                                attention_mask, mems=mems, **kwargs)
        lm_logits = model_output.logits
        log_probs = []
        for output, choices, choice_index in zip(F.log_softmax(lm_logits, dim=-1), choice_ids, choice_indices):
            log_probs_single = []
            for choice, choice_target_id in zip(choices, choice_index):
                tmp = output[choice_target_id, choice]
                log_probs_single.append(tmp.sum())
            log_probs.append(torch.stack(log_probs_single))
        log_probs = torch.stack(log_probs)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(log_probs, labels)
        return ModelOutput(
            loss=loss,
            logits=log_probs,
            lm_logits=lm_logits,
            mems=model_output.mems,
            external_mems=model_output.external_mems
        )


@add_start_docstrings(
    """GLM Model transformer with a `language modeling` head on top""",
    GLM_START_DOCSTRING,
)
class GLMForConditionalGeneration(GLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        config.output_predict = False
        self.config = config
        self.lm_head = torch.nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.quantized = False
        self.focused_attention = config.focused_attention
        self.glm = GLMModel(config)
        self.use_cache = config.use_cache
        self.max_sequence_length = config.max_sequence_length
        self.tie_weights()
        self.post_init()

    def get_input_embeddings(self):
        return self.glm.word_embeddings

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.glm.word_embeddings = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            return past
        reordered_decoder_past = ()
        for layer_past in past:
            reordered_decoder_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_decoder_past

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.Tensor,
            past: torch.Tensor = None,
            past_key_values: torch.Tensor = None,
            position_ids: torch.Tensor = None,
            generation_attention_mask: torch.Tensor = None,
            external_mems: torch.Tensor = None,
            **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        attention_mask = generation_attention_mask
        seq_length = input_ids.shape[1]
        if past is not None or past_key_values is not None:
            if position_ids is not None:
                position_ids = position_ids[:, :, seq_length - 1].unsqueeze(-1)
            if attention_mask is not None:
                attention_mask = attention_mask[:, :, seq_length - 1, :seq_length].unsqueeze(-2)
            input_ids = input_ids[:, -1].unsqueeze(-1)
        else:
            if position_ids is not None:
                position_ids = position_ids[:, :, :seq_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :, :seq_length, :seq_length]

        if position_ids is not None and input_ids.size(0) > position_ids.size(0):
            batch_size = position_ids.size(0)
            num_beams = input_ids.size(0) // batch_size
            position_ids = position_ids.unsqueeze(1).expand(-1, num_beams, -1, -1)
            position_ids = position_ids.reshape(batch_size * num_beams, *position_ids.shape[-2:])

        if attention_mask is not None and input_ids.size(0) > attention_mask.size(0):
            batch_size = attention_mask.size(0)
            num_beams = input_ids.size(0) // batch_size
            attention_mask = attention_mask.unsqueeze(1).expand(-1, num_beams, -1, -1, -1)
            attention_mask = attention_mask.reshape(batch_size * num_beams, *attention_mask.shape[-3:])

        if past is None:
            past = past_key_values

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "mems": past,
            "external_mems": external_mems,
            "use_cache": kwargs.get("use_cache")
        }

    # 临时方法，适配inference时保留mems操作让推理结果正确
    def set_max_memory_length(self, max_memory_length):
        if max_memory_length > 0:
            self.glm.transformer.max_memory_length = max_memory_length

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        labels=None,
        inputs_embeds=None,
        mems=None,
        external_mems=None,
        inner_loop: bool = False,
        **kwargs
    ):
        seq_length = input_ids.size(1)
        external_last_hidden_states = None

        if self.focused_attention is True:
            kwargs["use_cache"] = self.use_cache

        if inner_loop is False and self.focused_attention is True and seq_length > self.max_sequence_length:
            (input_ids, position_ids, attention_mask, inputs_embeds, mems, external_mems,
                external_last_hidden_states) = self._handle_long_input(
                input_ids, position_ids, attention_mask, inputs_embeds=inputs_embeds,
                mems=mems, external_mems=external_mems, context_window_length=self.max_sequence_length,
                last_context_length=self.max_sequence_length
            )
        model_output = self.glm(
            input_ids, position_ids, attention_mask,
            inputs_embeds=inputs_embeds, mems=mems,
            external_mems=external_mems, **kwargs
        )
        last_hidden_states = model_output.last_hidden_states
        if external_last_hidden_states is not None:
            last_hidden_states = torch.cat([
                external_last_hidden_states, last_hidden_states], dim=1)

        lm_logits = self.lm_head(last_hidden_states)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        return ModelOutput(
            loss=loss,
            logits=lm_logits,
            mems=model_output.mems,
            last_hidden_states=last_hidden_states,
            hidden_states=model_output.hidden_states,
            external_mems=model_output.external_mems
        )

    def _handle_long_input(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        mems=None,
        external_mems=None,
        context_window_length=1024,
        last_context_length=1024
    ):
        batch_size, input_length = input_ids.shape

        # First we load prefix to memory cache
        mem_input_length = max(input_length - last_context_length, 0)
        last_hidden_states = []
        outputs_list = []
        if mem_input_length > 0:
            for i in range(0, mem_input_length, context_window_length):
                beg, end = i, min(mem_input_length, i + context_window_length)

                if attention_mask is not None:
                    if mems is not None:
                        local_cache_size = mems[0][0].shape[1]
                    else:
                        local_cache_size = 0
                    attn_length = attention_mask.shape[-1]
                    attn_beg = beg - local_cache_size
                    attn_end = end
                    assert attn_end <= attn_length
                    assert attn_beg >= 0 and attn_end > attn_beg

                # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn, external_mems)
                outputs = self(
                    input_ids=input_ids[..., beg: end] if input_ids is not None else None,
                    attention_mask=attention_mask[..., beg: end, attn_beg: attn_end] if attention_mask is not None else None,  # noqa
                    position_ids=position_ids[..., beg: end],
                    mems=mems,
                    inputs_embeds=inputs_embeds[..., beg: end, :] if inputs_embeds is not None else None,
                    external_mems=external_mems,
                    inner_loop=True
                )
                if i > 0:
                    if external_mems is not None and mems is None:
                        for mc_layer in external_mems:
                            if mc_layer is not None:
                                del mc_layer.hiddens
                                del mc_layer.masks

                external_mems = outputs.external_mems
                last_hidden_states.append(outputs.last_hidden_states)
                outputs.external_mems = None
                mems = outputs.mems
                outputs.mems = None
                outputs_list.append(outputs)

        last_hidden_states = torch.cat(last_hidden_states, dim=1)
        remaining_input_length = input_length - mem_input_length
        beg = mem_input_length
        attn_length = remaining_input_length
        if mems is not None:
            attn_length += mems[0][0].shape[1]

        input_ids = input_ids[..., beg:] if input_ids is not None else None
        position_ids = position_ids[..., beg:]
        attention_mask = attention_mask[..., beg:, -attn_length:] if attention_mask is not None else None
        inputs_embeds = inputs_embeds[..., beg:, :] if inputs_embeds is not None else None

        return input_ids, position_ids, attention_mask, inputs_embeds, mems, external_mems, last_hidden_states

    @torch.no_grad()
    def generate_stream(
            self,
            inputs=None,
            generation_config=None,
            logits_processor=None,
            stopping_criteria=None,
            prefix_allowed_tokens_fn=None,
            synced_gpus=False,
            stream=True,
            **kwargs,
    ):
        r"""

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            kwargs:
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GreedySearchDecoderOnlyOutput`],
                    - [`~generation.SampleDecoderOnlyOutput`],
                    - [`~generation.BeamSearchDecoderOnlyOutput`],
                    - [`~generation.BeamSampleDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GreedySearchEncoderDecoderOutput`],
                    - [`~generation.SampleEncoderDecoderOutput`],
                    - [`~generation.BeamSearchEncoderDecoderOutput`],
                    - [`~generation.BeamSampleEncoderDecoderOutput`]
        """
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation -- update the generation config
            # model attribute accordingly, if it was created from the model config
            if self.generation_config._from_model_config:
                new_generation_config = GenerationConfig.from_model_config(
                    self.config)
                if new_generation_config != self.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use a generation configuration file (see"
                        " https://huggingface.co/docs/transformers/main_classes/text_generation)"
                    )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        # All unused kwargs must be model kwargs
        model_kwargs = generation_config.update(**kwargs)
        self._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(
                f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        # 3. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        # 4. Define other model kwargs
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        model_kwargs["use_cache"] = generation_config.use_cache

        accepts_attention_mask = "attention_mask" in set(
            inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        # decoder-only models should use left-padding for generation
        if not self.config.is_encoder_decoder:
            if (
                    generation_config.pad_token_id is not None
                    and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids = self._prepare_decoder_input_ids_for_generation(
                batch_size,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                bos_token_id=generation_config.bos_token_id,
                model_kwargs=model_kwargs,
                device=inputs_tensor.device,
            )
        else:
            # if decoder-only then inputs_tensor has to be `input_ids`
            input_ids = inputs_tensor

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_seq_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get(
            "max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                "Neither `max_length` nor `max_new_tokens` has been set, `max_length` will default to"
                f" {generation_config.max_length} (`generation_config.max_length`). Controlling `max_length` via the"
                " config is deprecated and `max_length` will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif has_default_max_length and generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
        elif not has_default_max_length and generation_config.max_new_tokens is not None:
            raise ValueError(
                "Both `max_new_tokens` and `max_length` have been set but they serve the same purpose -- setting a"
                " limit to the generated output length. Remove one of those arguments. Please refer to the"
                " documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
            )

        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            raise ValueError(
                f"Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than"
                f" the maximum length ({generation_config.max_length})"
            )
        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        # 7. determine generation mode，streaming generate only support greedy gen mode
        is_constraint_gen_mode = (generation_config.constraints is not None
                                  or generation_config.force_words_ids is not None)

        is_contrastive_search_gen_mode = (generation_config.top_k is not None and generation_config.top_k > 1
                                          and generation_config.do_sample is False
                                          and generation_config.penalty_alpha is not None
                                          and generation_config.penalty_alpha > 0
                                          )

        is_greedy_gen_mode = ((generation_config.num_beams == 1) and (generation_config.num_beam_groups == 1)
                              and generation_config.do_sample is False
                              and not is_constraint_gen_mode
                              and not is_contrastive_search_gen_mode
                              )
        is_sample_gen_mode = ((generation_config.num_beams == 1) and (generation_config.num_beam_groups == 1)
                              and generation_config.do_sample is True
                              and not is_constraint_gen_mode
                              and not is_contrastive_search_gen_mode
                              )
        is_beam_gen_mode = ((generation_config.num_beams > 1) and (generation_config.num_beam_groups == 1)
                            and generation_config.do_sample is False
                            and not is_constraint_gen_mode
                            and not is_contrastive_search_gen_mode
                            )
        is_beam_sample_gen_mode = ((generation_config.num_beams > 1) and (generation_config.num_beam_groups == 1)
                                   and generation_config.do_sample is True
                                   and not is_constraint_gen_mode
                                   and not is_contrastive_search_gen_mode
                                   )
        is_group_beam_gen_mode = ((generation_config.num_beams > 1) and (generation_config.num_beam_groups > 1)
                                  and not is_constraint_gen_mode
                                  and not is_contrastive_search_gen_mode
                                  )

        if generation_config.num_beam_groups > generation_config.num_beams:
            raise ValueError(
                "`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and generation_config.do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        # 9. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        # 10. go into different generation modes
        if is_greedy_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {generation_config.num_return_sequences} when doing"
                    " greedy search."
                )

            # 11. run greedy search
            if stream:
                for token in self.greedy_search_stream(
                        input_ids,
                        logits_processor=logits_processor,
                        stopping_criteria=stopping_criteria,
                        pad_token_id=generation_config.pad_token_id,
                        eos_token_id=generation_config.eos_token_id,
                        output_scores=generation_config.output_scores,
                        return_dict_in_generate=generation_config.return_dict_in_generate,
                        synced_gpus=synced_gpus,
                        stream=stream,
                        **model_kwargs,
                ):
                    yield token

            else:
                return self.greedy_search(
                    input_ids,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=generation_config.pad_token_id,
                    eos_token_id=generation_config.eos_token_id,
                    output_scores=generation_config.output_scores,
                    return_dict_in_generate=generation_config.return_dict_in_generate,
                    synced_gpus=synced_gpus,
                    **model_kwargs,
                )

        elif is_contrastive_search_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {generation_config.num_return_sequences} when doing"
                    " contrastive search."
                )

            return self.contrastive_search(
                input_ids,
                top_k=generation_config.top_k,
                penalty_alpha=generation_config.penalty_alpha,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_sample_gen_mode:
            # 11. prepare logits warper
            logits_warper = self._get_logits_warper(generation_config)

            # 12. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            if stream:
                for token in self.sample_stream(
                        input_ids,
                        logits_processor=logits_processor,
                        logits_warper=logits_warper,
                        stopping_criteria=stopping_criteria,
                        pad_token_id=generation_config.pad_token_id,
                        eos_token_id=generation_config.eos_token_id,
                        output_scores=generation_config.output_scores,
                        return_dict_in_generate=generation_config.return_dict_in_generate,
                        synced_gpus=synced_gpus,
                        stream=stream,
                        **model_kwargs,
                ):
                    yield token
            else:
                return self.sample(
                    input_ids,
                    logits_processor=logits_processor,
                    logits_warper=logits_warper,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=generation_config.pad_token_id,
                    eos_token_id=generation_config.eos_token_id,
                    output_scores=generation_config.output_scores,
                    return_dict_in_generate=generation_config.return_dict_in_generate,
                    synced_gpus=synced_gpus,
                    **model_kwargs,
                )

        elif is_beam_gen_mode:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError(
                    "`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError(
                    "`max_length` needs to be a stopping_criteria for now.")

            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_beam_sample_gen_mode:
            # 11. prepare logits warper
            logits_warper = self._get_logits_warper(generation_config)

            if stopping_criteria.max_length is None:
                raise ValueError(
                    "`max_length` needs to be a stopping_criteria for now.")
            # 12. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size * generation_config.num_return_sequences,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
            )

            # 13. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams * generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 14. run beam sample
            return self.beam_sample(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_group_beam_gen_mode:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError(
                    "`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if generation_config.num_beams % generation_config.num_beam_groups != 0:
                raise ValueError(
                    "`num_beams` should be divisible by `num_beam_groups` for group beam search.")

            if stopping_criteria.max_length is None:
                raise ValueError(
                    "`max_length` needs to be a stopping_criteria for now.")

            has_default_typical_p = kwargs.get(
                "typical_p") is None and generation_config.typical_p == 1.0
            if not has_default_typical_p:
                raise ValueError(
                    "Decoder argument `typical_p` is not supported with beam groups.")

            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                max_length=stopping_criteria.max_length,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                num_beam_groups=generation_config.num_beam_groups,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            return self.group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_constraint_gen_mode:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError(
                    "`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError(
                    "`max_length` needs to be a stopping_criteria for now.")

            if generation_config.num_beams <= 1:
                raise ValueError(
                    "`num_beams` needs to be greater than 1 for constrained generation.")

            if generation_config.do_sample:
                raise ValueError(
                    "`do_sample` needs to be false for constrained generation.")

            if generation_config.num_beam_groups is not None and generation_config.num_beam_groups > 1:
                raise ValueError(
                    "`num_beam_groups` not supported yet for constrained generation.")

            final_constraints = []
            if generation_config.constraints is not None:
                final_constraints = generation_config.constraints

            if generation_config.force_words_ids is not None:

                def typeerror():
                    raise ValueError(
                        "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]`"
                        f"of positive integers, but is {generation_config.force_words_ids}."
                    )

                if (
                        not isinstance(generation_config.force_words_ids, list)
                        or len(generation_config.force_words_ids) == 0
                ):
                    typeerror()

                for word_ids in generation_config.force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any(not isinstance(token_ids, list) for token_ids in word_ids):
                            typeerror()
                        if any(
                                any((not isinstance(token_id, int) or token_id < 0)
                                    for token_id in token_ids)
                                for token_ids in word_ids
                        ):
                            typeerror()

                        constraint = DisjunctiveConstraint(word_ids)
                    else:
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                            typeerror()

                        constraint = PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)

            # 11. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=final_constraints,
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            return self.constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

    def greedy_search(
            self,
            input_ids: torch.LongTensor,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[Union[int, List[int]]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            synced_gpus: Optional[bool] = False,
            finish_if_repetition=False,
            **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
        used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.greedy_search`] directly. Use generate()
        instead. For an overview of generation strategies and code examples, check the [following
        guide](./generation_strategies).

        </Tip>


        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.

            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GreedySearchDecoderOnlyOutput`], [`~generation.GreedySearchEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GreedySearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GreedySearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
        >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

        >>> input_prompt = "It might be possible to"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> outputs = model.greedy_search(
        ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(
                stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get(
                "attentions") if output_attentions else None
            encoder_hidden_states = (model_kwargs["encoder_outputs"].get("hidden_states")
                                     if output_hidden_states else None
                                     )

        this_peer_finished = False  # used by synced_gpus only
        finished_input_ids = [0] * len(input_ids)
        index_map = [i for i in range(len(input_ids))]

        def recursive_gen_unfinished_params(params, length, unfinished_indexes):
            if isinstance(params, torch.Tensor):
                if len(params) == length:
                    new_params = torch.index_select(
                        params, 0, unfinished_indexes.long().to(params.device))
                    return new_params
                else:
                    return params
            elif isinstance(params, list):
                results = []
                for param in params:
                    results.append(recursive_gen_unfinished_params(
                        param, length, unfinished_indexes))
                return results
            elif isinstance(params, dict):
                result = {}
                for key, value in params.items():
                    result[key] = recursive_gen_unfinished_params(
                        value, length, unfinished_indexes)
                return result
            else:
                return params

        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(
                    0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (
                            outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                finished_index = []
                for i in range(len(next_tokens)):
                    for eos_id in eos_token_id:
                        if next_tokens[i] == eos_id:
                            finished_index.append(i)
                            break

            if finish_if_repetition:
                input_id_list = [input_ids[i].detach()
                                 for i in range(input_ids.shape[0])]
                repeat_finished_index, input_id_list = detect_repetition(
                    input_id_list, min_repeat_tokens=1)

                finished_index = list(
                    set(repeat_finished_index) | set(finished_index))

            if finished_index:
                batch_size = len(input_ids)
                unfinished_indexes = [i for i in range(
                    batch_size) if i not in finished_index]
                unfinished_indexes = torch.Tensor(unfinished_indexes).long()

                new_input_ids = []
                new_index_map = []
                # prepare new input_ids
                for index in range(batch_size):
                    if index in finished_index:
                        if finish_if_repetition:
                            finished_input_ids[index_map[index]] = input_id_list[index]
                        else:
                            finished_input_ids[index_map[index]] = input_ids[index]
                    else:
                        new_input_ids.append(input_ids[index].reshape(1, -1))
                        new_index_map.append(index_map[index])

                outputs = recursive_gen_unfinished_params(
                    outputs, batch_size, unfinished_indexes)
                outputs = ModelOutput(
                    loss=outputs['loss'], logits=outputs['logits'],
                    mems=outputs['mems'], external_mems=outputs['external_mems']
                )
                model_kwargs = recursive_gen_unfinished_params(
                    model_kwargs, batch_size, unfinished_indexes)

                index_map = new_index_map
                if new_input_ids:
                    input_ids = torch.cat(new_input_ids, dim=0)
                else:
                    input_ids = None

            # update generated ids, model inputs, and length for next step
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if self.focused_attention:
                model_kwargs["external_mems"] = outputs.external_mems

            if not index_map or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        if input_ids is not None:
            for index in range(len(input_ids)):
                finished_input_ids[index_map[index]] = input_ids[index]
        max_len = max(len(seq) for seq in finished_input_ids)
        for index in range(len(finished_input_ids)):
            finished_input_ids[index] = torch.cat([finished_input_ids[index], torch.ones(
                max_len - len(finished_input_ids[index])).long().to(finished_input_ids[index].device) * pad_token_id])
        finished_input_ids = [ids.reshape(1, -1) for ids in finished_input_ids]
        input_ids = torch.cat(finished_input_ids, dim=0)

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids

    def sample_stream(
            self,
            input_ids: torch.LongTensor,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            logits_warper: Optional[LogitsProcessorList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[Union[int, List[int]]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            synced_gpus: bool = False,
            streamer: Optional["BaseStreamer"] = None,
            stream: bool = False,
            **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.sample`] directly. Use generate() instead.
        For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            is_stream:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.SampleDecoderOnlyOutput`], [`~generation.SampleEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.SampleDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.SampleEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     TopKLogitsWarper,
        ...     TemperatureLogitsWarper,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
        >>> model.config.pad_token_id = model.config.eos_token_id
        >>> model.generation_config.pad_token_id = model.config.eos_token_id

        >>> input_prompt = "Today is a beautiful day, and"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> # instantiate logits processors
        >>> logits_warper = LogitsProcessorList(
        ...     [
        ...         TopKLogitsWarper(50),
        ...         TemperatureLogitsWarper(0.7),
        ...     ]
        ... )

        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> torch.manual_seed(0)  # doctest: +IGNORE_RESULT
        >>> outputs = model.sample(
        ...     input_ids,
        ...     logits_processor=logits_processor,
        ...     logits_warper=logits_warper,
        ...     stopping_criteria=stopping_criteria,
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Today is a beautiful day, and we must do everything possible to make it a day of celebration.']
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(
                stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(
            input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get(
                "attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get(
                    "hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(
            input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        # auto-regressive generation
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(
                    0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (
                            outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            from torch import nn
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(
                        eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True
            if stream:
                yield next_tokens[:, None]

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return SampleEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return SampleDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids

    def greedy_search_stream(
            self,
            input_ids: torch.LongTensor,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[Union[int, List[int]]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            synced_gpus: Optional[bool] = False,
            stream=True,
            **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
        used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.greedy_search`] directly. Use generate()
        instead. For an overview of generation strategies and code examples, check the [following
        guide](./generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.

            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GreedySearchDecoderOnlyOutput`], [`~generation.GreedySearchEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GreedySearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GreedySearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
        >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

        >>> input_prompt = "It might be possible to"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> outputs = model.greedy_search(
        ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(
                stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get(
                "attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get(
                    "hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(
                    0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (
                            outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    (sum(next_tokens != i for i in eos_token_id)).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True
            if stream:
                yield next_tokens[:, None]

        if not stream:
            if return_dict_in_generate:
                if self.config.is_encoder_decoder:
                    return GreedySearchEncoderDecoderOutput(
                        sequences=input_ids,
                        scores=scores,
                        encoder_attentions=encoder_attentions,
                        encoder_hidden_states=encoder_hidden_states,
                        decoder_attentions=decoder_attentions,
                        cross_attentions=cross_attentions,
                        decoder_hidden_states=decoder_hidden_states,
                    )
                else:
                    return GreedySearchDecoderOnlyOutput(
                        sequences=input_ids,
                        scores=scores,
                        attentions=decoder_attentions,
                        hidden_states=decoder_hidden_states,
                    )
            else:
                return input_ids


@add_start_docstrings(
    """GLM Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    GLM_START_DOCSTRING,
)
class GLMForSequenceClassification(GLMPreTrainedModel):
    def __init__(self, config: GLMConfig, hidden_dropout=None, num_class=1):
        super().__init__(config)
        self.pool_token = config.pool_token
        self.glm = GLMModel(config)
        self.glm.output_predict = False
        self.num_class = num_class
        # Multi-choice head.
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.output_dropout_prob
        )
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.out_proj = torch.nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(GLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(self,
                input_ids=None,
                position_ids=None,
                attention_mask=None,
                labels=None):

        num_choices = None

        if len(input_ids.shape) == 3:
            batch_size, num_choices = input_ids.shape[:2]
            input_ids = input_ids.reshape(-1, input_ids.size(-1))
            attention_mask = attention_mask.reshape(
                -1, *attention_mask.size()[2:])
            position_ids = position_ids.reshape(-1, *position_ids.size()[2:])
        model_out = self.glm(input_ids, position_ids, attention_mask)
        # outputs, mems = model_out.last_hidden_states, model_out.mems
        outputs, _ = model_out.last_hidden_states, model_out.mems

        output = outputs[:, 0, :]
        output = self.dropout(output)
        output = torch.tanh(self.dense(output))
        output = self.dropout(output)
        logits = self.out_proj(output)
        if num_choices is not None:
            logits = logits.view(-1, num_choices)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        # loss = F.cross_entropy(logits.contiguous().float(), labels.long())
        return SequenceClassifierOutput(loss=loss,
                                        logits=logits,
                                        hidden_states=outputs)
