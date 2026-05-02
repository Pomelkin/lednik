from typing import TypeVar
from typing import override

import torch
import torch.nn.functional as F
from kostyl.ml.integrations.lightning import LightningCheckpointLoaderMixin
from kostyl.utils import setup_logger
from torch import nn
from transformers import PreTrainedModel
from transformers import SentencePieceBackend
from transformers import TokenizersBackend

from .configuration_static import StaticEmbeddingsConfig
from .outputs import StaticEmbeddingsOutput
from .outputs import StaticEmbeddingsSequenceClassifierOutput


logger = setup_logger(fmt="only_message")

TPreTrainedModel = TypeVar("TPreTrainedModel", bound="PreTrainedModel")


class StaticEmbeddingsPreTrainedModel(LightningCheckpointLoaderMixin, PreTrainedModel):
    """
    An abstract base class for static embedding models, inheriting from `PreTrainedModel`.

    This class handles the initialization of weights, tokenizer integration, and provides
    overridden methods for loading and saving pretrained models that include tokenizer handling.
    """

    config: StaticEmbeddingsConfig  # type: ignore
    base_model_prefix = "model"

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


class StaticEmbeddingsModel(StaticEmbeddingsPreTrainedModel):
    """Static Embeddings Model."""

    def __init__(self, config: StaticEmbeddingsConfig) -> None:
        """Initialize model."""
        super().__init__(config)
        self.embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.token_pos_weights = nn.Embedding(
            config.vocab_size,
            1,
        )
        self.output_proj = (
            nn.Linear(config.hidden_size, config.output_hidden_size)
            if config.output_hidden_size is not None
            else nn.Identity()
        )
        self.norm = nn.RMSNorm(config.output_hidden_size or config.hidden_size)
        self.dropout = (
            nn.Dropout(config.embedding_dropout)
            if config.embedding_dropout > 0.0
            else nn.Identity()
        )

        self.config = config
        self.tokenizer: TokenizersBackend | SentencePieceBackend | None = None

        # Initialize weights and apply final processing
        self.post_init()
        return

    def replace_embeddings(self, new_embeddings: torch.Tensor | nn.Parameter) -> None:
        """Replace the current model embeddings weights with given one."""
        if new_embeddings.numel() != self.embeddings.weight.numel():
            raise ValueError(
                f"new_embeddings should have numel {self.embeddings.weight.numel()}, "
                f"but got {new_embeddings.numel()}."
            )
        if new_embeddings.size() != self.embeddings.weight.size():
            raise ValueError(
                f"new_embeddings should have size {self.embeddings.weight.size()}, "
                f"but got {new_embeddings.size()}."
            )

        new_embeddings = new_embeddings.clone()  # use clone() to avoid weight tying
        if isinstance(new_embeddings, torch.Tensor):
            new_embeddings = nn.Parameter(new_embeddings)  # avoiding weight ties
        self.embeddings.weight = new_embeddings
        return

    def replace_pos_weights(self, new_pos_weights: torch.Tensor | nn.Parameter) -> None:
        """Replace the current model position weights with given one."""
        if new_pos_weights.numel() != self.token_pos_weights.weight.numel():
            raise ValueError(
                f"new_pos_weights should have numel {self.token_pos_weights.weight.numel()}, "
                f"but got {new_pos_weights.numel()}."
            )
        if new_pos_weights.size() != self.token_pos_weights.weight.size():
            raise ValueError(
                f"new_pos_weights should have size {self.token_pos_weights.weight.size()}, "
                f"but got {new_pos_weights.size()}."
            )

        new_pos_weights = new_pos_weights.clone()  # use clone() to avoid weight tying
        if isinstance(new_pos_weights, torch.Tensor):
            new_pos_weights = nn.Parameter(new_pos_weights)  # avoiding weight ties
        self.token_pos_weights.weight = new_pos_weights
        return

    @classmethod
    def initialize(
        cls,
        config: StaticEmbeddingsConfig,
        embeddings: torch.Tensor | nn.Parameter,
    ) -> "StaticEmbeddingsModel":
        """Initialize model with given embeddings."""
        model = cls(config)
        model.replace_embeddings(embeddings)
        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> StaticEmbeddingsOutput:
        """Forward pass."""
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        embeddings = self.output_proj(self.dropout(self.embeddings(input_ids)))
        token_weights = self.token_pos_weights(input_ids)
        embeddings = self.norm(embeddings)

        masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
        sentence_embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(
            dim=-1, keepdim=True
        ).clamp(min=1e-9)
        output = StaticEmbeddingsOutput(
            token_embeddings=embeddings,
            pos_weights=token_weights,
            sentence_embeddings=sentence_embeddings,
        )
        return output


class StaticEmbeddingsClassificationHead(nn.Module):
    """Classification head for static embeddings."""

    def __init__(self, config: StaticEmbeddingsConfig) -> None:
        """Initialize classifier."""
        super().__init__()
        self.dropout = (
            nn.Dropout(config.classifier_dropout)
            if config.classifier_dropout > 0.0
            else nn.Identity()
        )
        self.norm = nn.RMSNorm(config.hidden_size)
        self.hidden_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.num_labels)

        self.config = config
        return

    def forward(self, sentence_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.dropout(F.relu(self.hidden_proj(self.norm(sentence_embeddings))))
        logits = self.head(x)
        return logits


class StaticEmbeddingsForSequenceClassification(StaticEmbeddingsPreTrainedModel):
    """Static Embeddings Model for sequence classification."""

    def __init__(self, config: StaticEmbeddingsConfig) -> None:
        """Initialize model."""
        super().__init__(config)
        if not self.config.is_tokenizer_customized:
            raise ValueError(
                "The tokenizer is not customized. Please use "
                "`lednik.distill.initialization.tokenizer_utils.customize_tokenizer` "
                "to customize the tokenizer before initializing the model."
            )

        self.model = StaticEmbeddingsModel(config)
        self.classifier = StaticEmbeddingsClassificationHead(config)
        self.config = config

        # Initialize weights and apply final processing
        self.post_init()
        return

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> StaticEmbeddingsSequenceClassifierOutput:
        """Forward pass."""
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        embeddings_output = self.model(input_ids, attention_mask)
        logits = self.classifier(embeddings_output.sentence_embeddings)
        loss = self.loss_function(logits, labels) if labels is not None else None  #  type: ignore
        output = StaticEmbeddingsSequenceClassifierOutput(
            logits=logits,
            loss=loss,  # type: ignore
        )
        return output
