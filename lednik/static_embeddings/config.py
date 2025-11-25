from typing import Any
from typing import Literal

from transformers import PretrainedConfig


class StaticEmbeddingsConfig(PretrainedConfig):
    """Configuration class for Static Embeddings Model."""

    model_type = "StaticEmbeddingsPreTrainedModel"

    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        embedding_dim: int = 300,
        embedding_dropout: float = 0.0,
        classifier_dropout: float = 0.0,
        dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16",
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Initialize the configuration for static embeddings.

        Args:
            vocab_size: The size of the vocabulary.
            embedding_dim: The dimensionality of the embeddings. Defaults to 300.
            norm_type: The type of normalization to apply. Defaults to None.
            embedding_dropout: Embedding dropout probability. Defaults to 0.0.
            pad_token_id: The ID of the padding token. Defaults to None.
            classifier_dropout: Classifier dropout probability. Defaults to 0.0.
            dtype: The data type for the embeddings. Defaults to "float32".
            kwargs: Additional keyword arguments passed to the parent class.

        """
        super().__init__(**kwargs)  # type: ignore
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_dropout = embedding_dropout
        self.pad_token_id = pad_token_id
        self.classifier_dropout = classifier_dropout
        self.dtype = dtype
        return
