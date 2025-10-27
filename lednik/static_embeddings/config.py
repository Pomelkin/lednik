from typing import Any
from typing import Literal

from transformers import PretrainedConfig


class StaticEmbeddingsConfig(PretrainedConfig):
    model_type = "StaticEmbeddingsPreTrainedModel"

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        norm_type: Literal["layernorm", "rmsnorm"] | None = None,
        dropout_p: float = 0.0,
        pad_token_id: int | None = None,
        model_compile: bool = False,
        dtype: Literal["float32", "float16", "bfloat16"] = "float32",
        embeddings_int8: bool = False,
        trainable_scales: bool = False,
        qat: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Initialize the configuration for static embeddings.

        Args:
            vocab_size: The size of the vocabulary.
            embedding_dim: The dimensionality of the embeddings. Defaults to 300.
            norm_type: The type of normalization to apply. Defaults to None.
            dropout_p: Dropout probability. Defaults to 0.0.
            pad_token_id: The ID of the padding token. Defaults to None.
            model_compile: Whether to compile the model. Defaults to False.
            dtype: The data type for the embeddings. Defaults to "float32".
            embeddings_int8: Whether to use int8 quantization for embeddings. Defaults to False.
            trainable_scales: Whether the scales are trainable. Defaults to False.
            qat: Whether to use quantization-aware training. Defaults to False.
            kwargs: Additional keyword arguments passed to the parent class.

        """
        super().__init__(**kwargs)  # type: ignore
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.norm_type = norm_type
        self.dropout_p = dropout_p
        self.pad_token_id = pad_token_id
        self.dtype = dtype
        self.model_compile = model_compile
        self.embeddings_int8 = embeddings_int8
        self.trainable_scales = trainable_scales
        self.qat = qat
        return
