from dataclasses import dataclass

import torch


@dataclass(slots=True)
class StaticEmbeddingsOutput:
    """Output class for static embeddings models."""

    embeddings: torch.Tensor
    sentence_embeddings: torch.Tensor
    pos_weights: torch.Tensor | None = None


@dataclass(slots=True)
class StaticEmbeddingsSequenceClassifierOutput:
    """Output class for static embeddings sequence classifier models."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None
