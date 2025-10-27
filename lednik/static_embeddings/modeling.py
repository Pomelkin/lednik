from typing import override

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import logging

from lednik.static_embeddings.config import StaticEmbeddingsConfig
from dataclasses import dataclass

logger = logging.get_logger(__name__)


class StaticEmbeddingsPreTrainedModel(PreTrainedModel):
    config: StaticEmbeddingsConfig

    @override
    def _init_weights(self, module: nn.Module) -> None:
        match module:
            case nn.Embedding() if module.weight.dtype != torch.int8:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            case nn.LayerNorm():
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            case nn.RMSNorm():
                nn.init.ones_(module.weight)
            case nn.Linear():
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            case _:
                pass
        return


class StaticEmbeddingsModelForPostTraining(StaticEmbeddingsPreTrainedModel):
    """Static Embeddings for post-training only."""

    def __init__(self, config: StaticEmbeddingsConfig) -> None:
        """Initialize model."""
        super().__init__(config)
        if config.qat:
            self.embeddings = nn.Embedding(
                config.vocab_size,
                config.embedding_dim,
                padding_idx=config.pad_token_id,
            )
            self.dequant_scales = nn.Embedding(
                config.vocab_size,
                1,
                padding_idx=config.pad_token_id,
            )
            if not config.trainable_scales:
                self.dequant_scales.weight.requires_grad = False
        else:
            self.embeddings = nn.Embedding(
                config.vocab_size, config.embedding_dim, padding_idx=config.pad_token_id
            )

        self.register_buffer("token_weights", torch.ones(config.vocab_size))

        match config.norm_type:
            case "layernorm":
                self.norm = nn.LayerNorm(config.embedding_dim)
            case "rmsnorm":
                self.norm = nn.RMSNorm(config.embedding_dim)
            case None:
                self.norm = nn.Identity()
            case _:
                raise ValueError(f"Unsupported norm type: {config.norm_type}")

        if config.dropout_p > 0.0:
            self.dropout = nn.Dropout(config.dropout_p)
        else:
            self.dropout = nn.Identity()
        self.config = config
        return

    def update_embeddings(self, new_embeddings: nn.Embedding) -> None:
        """Replace the current model embeddings with given one."""
        self.embeddings = new_embeddings
        return

    def update_tokens_weights(self, new_token_weights: torch.Tensor) -> None:
        """Replace the current model token weights with given one."""
        self.token_weights = new_token_weights
        return

    @torch.no_grad()
    def quantize_embeddings(
        self, update_scales: bool, convert_to_int8: bool
    ) -> torch.Tensor:
        """
        Apply quantization to the embeddings, but keep them in float format. If scales are not trainable, set them as well.

        Args:
            update_scales: Whether to update the dequantization scales.
            convert_to_int8: Whether to convert the embeddings to int8. If False, embeddings remain in float format.

        Returns:
            The computed scales for the embeddings.

        """
        if not self.config.quat:
            raise ValueError("Embeddings are not set to use int8 quantization.")

        embeddings_weight = self.embeddings.weight.data

        scales = embeddings_weight.abs().amax(dim=1, keepdim=True) / 127.0
        embeddings_weight = (embeddings_weight / scales).round().clamp(-128, 127)
        if convert_to_int8:
            embeddings_weight = embeddings_weight.to(torch.int8)
        self.embeddings.weight.data = embeddings_weight

        if update_scales:
            self.dequant_scales.weight.data = scales.to(
                self.dequant_scales.weight.data.dtype
            ).log()
        return scales

    @torch.compile(dynamic=True)
    def compiled_postprocessing(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply normalization and dropout in a compiled manner."""
        return self.dropout(self.norm(embeddings))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        embeddings = self.embeddings(input_ids)

        if self.config.qat:
            scales = self.dequant_scales(input_ids).exp()
            embeddings = embeddings * scales

        embeddings = (
            self.compiled_postprocessing(embeddings)
            if self.config.model_compile
            else self.dropout(self.norm(embeddings))
        )
        return embeddings


@dataclass
class StaticEmbeddingsOutput:
    embeddings: torch.Tensor
    weights: torch.Tensor
    sentence_embeddings: torch.Tensor


class StaticEmbeddingsModel(StaticEmbeddingsPreTrainedModel):
    """Static Embeddings Model for inference."""

    def __init__(self, config: StaticEmbeddingsConfig) -> None:
        """Initialize model."""
        super().__init__(config)
        self.embeddings = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
            padding_idx=config.pad_token_id,
            dtype=torch.int8 if config.embeddings_int8 else torch.get_default_dtype(),
        )

        if config.embeddings_int8:
            self.dequant_scales = nn.Embedding(
                config.vocab_size,
                1,
                padding_idx=config.pad_token_id,
            )

        self.register_buffer("token_weights", torch.ones(config.vocab_size))

        match config.norm_type:
            case "layernorm":
                self.norm = nn.LayerNorm(config.embedding_dim)
            case "rmsnorm":
                self.norm = nn.RMSNorm(config.embedding_dim)
            case None:
                self.norm = nn.Identity()
            case _:
                raise ValueError(f"Unsupported norm type: {config.norm_type}")

        self.config = config
        return

    def forward(
        self, input_ids: torch.Tensor, mask: torch.Tensor
    ) -> StaticEmbeddingsOutput:
        """Forward pass."""
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        embeddings = self.embeddings(input_ids)

        if self.config.embeddings_int8:
            scales = self.dequant_scales(input_ids).exp()
            embeddings = embeddings * scales

        embeddings = self.norm(embeddings)
        token_weights = self.token_weights[input_ids].unsqueeze(-1)  # type: ignore
        masked_embeddings = embeddings * mask.unsqueeze(-1)
        sentence_embeddings = (embeddings * token_weights).sum(dim=1) / mask.sum(
            dim=1, keepdim=True
        )
        return StaticEmbeddingsOutput(
            embeddings=masked_embeddings,
            weights=token_weights,
            sentence_embeddings=sentence_embeddings,
        )
