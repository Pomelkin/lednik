from dataclasses import dataclass

import torch


@dataclass(slots=True)
class StaticEmbeddingsOutput:
    """Output class for static embeddings models."""

    embeddings: torch.Tensor
    sentence_embeddings: torch.Tensor
    pos_weights: torch.Tensor | None = None

    def __getitem__(self, item) -> torch.Tensor:
        """Enable indexing into the output object. 0: embeddings, 1: sentence_embeddings."""
        return (self.embeddings, self.sentence_embeddings)[item]


@dataclass(slots=True)
class StaticEmbeddingsSequenceClassifierOutput:
    """Output class for static embeddings sequence classifier models."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None
