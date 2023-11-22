import torch
from typing import Tuple, Union, List

from transformers.modeling_outputs import ModelOutput
from solutions.antllm.antllm.models.embeddings.distributed_embedding import (
    embedding_gather,
    get_rank
)
from solutions.antllm.antllm.models.glm.modeling_glm import (
    GLMPreTrainedModel,
    GLMModel
)


class BaseEmbeddingModel(torch.nn.Module):
    def __init__(self) -> None:
        pass

    def _inbatch_loss(self):
        pass

    def constractive_training(
        self,
        hiddens: torch.Tensor,
        labels: torch.LongTensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def _compute_embedding(
        self,
        hiddens: torch.Tensor,
        mask: torch.LongTensor = None,
        reduction: str = "last"
    ) -> torch.Tensor:
        raise NotImplementedError


class GLMForEmbedding(GLMPreTrainedModel, BaseEmbeddingModel):
    def __init__(self, config):
        super(BaseEmbeddingModel, self).__init__()
        super(GLMPreTrainedModel, self).__init__(config)

        self.glm = GLMModel(config)
        try:
            self.temperature = config.temperature
        except Exception:
            self.temperature = 0.01
        self.label_smoothing = 0.0
        self.post_init()
    
    def _compute_embedding(
        self,
        hiddens: torch.Tensor,
        mask: torch.LongTensor = None,
        seq_length: Union[torch.LongTensor, List[int]] = None,
        reduction: str = "mean"
    ) -> torch.Tensor:
        if mask is None:
            mask = hiddens.new_ones(*hiddens.size())
        else:
            mask = mask.unsqueeze(-1)

        hiddens = hiddens.masked_fill(~mask.bool(), 0.0)

        if reduction == "mean":
            embedding = torch.sum(hiddens, dim=1)
            denominator = torch.sum(mask, dim=1)
            embedding = embedding / denominator

        elif reduction == "sum":
            embedding = torch.sum(hiddens, dim=1)

        elif reduction == "last":
            batch_size, length, _ = hiddens.size()
            if seq_length is None:
                seq_length = batch_size * [length - 1]

            embedding = hiddens[torch.arange(batch_size), seq_length]

        else:
            raise NotImplementedError(f"The reduction method {reduction} is not inplemented.")

        return embedding

    def _inbatch_loss(
        self, 
        q_embeddings: torch.Tensor,
        p_embeddings: torch.Tensor,
        simlarity_method: str = "cos"
    ) -> torch.Tensor:
        batch_size = q_embeddings.size(0)
        labels = torch.arange(0, batch_size, dtype=torch.long, device=q_embeddings.device)

        # Support for distributed training
        gather_p_embeddings = embedding_gather(p_embeddings)
        labels = labels + get_rank() * batch_size

        if simlarity_method == "cos":
            scores = torch.cosine_similarity(
                gather_p_embeddings[None, ...], q_embeddings[:, None], dim=-1) / self.temperature
        else:
            scores = torch.einsum("id, jd->ij", q_embeddings / self.temperature, gather_p_embeddings)
        loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing)

        return loss

    def constractive_training(self, hiddens, labels):
        pass

    def forward(
        self,
        query_ids: torch.Tensor,
        query_position_ids: torch.Tensor = None,
        query_attention_mask: torch.Tensor = None,
        query_mask: torch.Tensor = None,
        query_mems: Tuple[torch.Tensor] = None,
        passage_ids: torch.Tensor = None,
        passage_position_ids: torch.Tensor = None,
        passage_attention_mask: torch.Tensor = None,
        passage_mask: torch.Tensor = None,
        passage_mems: Tuple[torch.Tensor] = None,
        reduction: str = "mean",
        **kwargs
    ):
        query_output = self.glm(query_ids, query_position_ids,
                                query_attention_mask, mems=query_mems, **kwargs)
        query_hiddens = query_output.last_hidden_states
        query_embeddings = self._compute_embedding(query_hiddens, query_mask, query_attention_mask, reduction)

        loss = None
        passage_embeddings = None
        if passage_ids is not None:
            passage_output = self.glm(
                passage_ids, passage_position_ids, passage_attention_mask, mems=passage_mems, **kwargs)
            passage_hiddens = passage_output.last_hidden_states
            passage_embeddings = self._compute_embedding(
                passage_hiddens, passage_mask, passage_attention_mask, reduction)

            loss = self._inbatch_loss(query_embeddings, passage_embeddings)

        return ModelOutput(
            loss=loss,
            query_embeddings=query_embeddings,
            passage_embeddings=passage_embeddings
        )
