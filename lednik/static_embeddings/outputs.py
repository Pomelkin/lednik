from dataclasses import dataclass

import torch


@dataclass(slots=True, frozen=True)
class StaticEmbeddingsPostTrainingOutput:
    embeddings: torch.Tensor


@dataclass(slots=True, frozen=True)
class StaticEmbeddingsOutput:
    embeddings: torch.Tensor
    pos_weights: torch.Tensor
    sentence_embeddings: torch.Tensor


@dataclass(slots=True, frozen=True)
class StaticEmbeddingsSequenceClassifierOutput:
    logits: torch.Tensor
    pos_weights: torch.Tensor
    embeddings: torch.Tensor
    sentence_embeddings: torch.Tensor
