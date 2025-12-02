from typing import Any

from transformers import PretrainedConfig


class StaticEmbeddingsConfig(PretrainedConfig):
    """Configuration class for Static Embeddings Model."""

    model_type = "StaticEmbeddingsPreTrainedModel"

    def __init__(
        self,
        vocab_size: int = 30522,
        pad_token_id: int = 30125,
        embedding_dim: int = 300,
        embedding_dropout: float = 0.0,
        classifier_dropout: float = 0.0,
        num_labels: int = 1,
        id2label: dict[int, str] | None = None,
        label2id: dict[str, int] | None = None,
        tokenizer_modified: bool = True,
        **kwargs: Any,
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
            num_labels: Number of labels for classification tasks. Defaults to 1.
            id2label: A mapping from label IDs to label names. Defaults to None.
            label2id: A mapping from label names to label IDs. Defaults to None.
            tokenizer_modified: Whether the tokenizer has been modified. Defaults to True.
            kwargs: Additional keyword arguments passed to the parent class.

        """
        super().__init__(
            pad_token_id=pad_token_id,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_dropout = embedding_dropout
        self.classifier_dropout = classifier_dropout
        self.tokenizer_modified = tokenizer_modified
        return
