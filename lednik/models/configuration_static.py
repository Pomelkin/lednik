from typing import Any

from huggingface_hub.dataclasses import strict
from transformers import PreTrainedConfig


@strict
class StaticEmbeddingsConfig(PreTrainedConfig):
    """Configuration class for Static Embeddings Model."""

    model_type = "StaticEmbeddingsPreTrainedModel"

    # Vocabulary size for lookup table.
    vocab_size: int = 30522
    # Padding token id.
    pad_token_id: int = 30125
    # Embedding vector dimensionality.
    hidden_size: int = 300
    # Output sentence embedding dimensionality.
    output_hidden_size: int | None = None
    # Dropout on token embeddings.
    embedding_dropout: float = 0.0
    # Dropout before classifier.
    classifier_dropout: float = 0.0
    # Whether tokenizer was customized.
    is_tokenizer_customized: bool = True

    def __post_init__(self, **kwargs: Any) -> None:  # noqa: D105
        self.num_attention_heads = 0
        self.num_hidden_layers = 0
        super().__post_init__(**kwargs)
        return
