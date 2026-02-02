from typing import Any
from typing import Literal

from kostyl.utils.logging import setup_logger
from transformers import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters


logger = setup_logger(fmt="only_message")


class LednikConfig(PreTrainedConfig):
    """Configuration class for Lednik Model."""

    model_type = "LednikPreTrainedModel"

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 384,
        embeddings_dropout: float = 0.0,
        num_attention_heads: int = 6,
        num_hidden_layers: int = 1,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        out_attn_dropout: float = 0.0,
        rope_parameters: RopeParameters | None = None,
        max_position_embeddings: int = 1024,
        hidden_act: str = "silu",
        intermediate_size: int = 576,
        mlp_bias: bool = False,
        mlp_dropout: float = 0.0,
        pad_token_id: int = 30125,
        classifier_pooling: Literal["cls", "mean"] = "cls",
        classifier_dropout: float | None = 0.0,
        classifier_bias: bool | None = False,
        classifier_activation: str | None = "gelu",
        **kwargs: Any,
    ) -> None:
        """
        Initializes the Lednik configuration.

        Args:
            vocab_size (`int`, *optional*, defaults to 30522):
                Vocabulary size of the Lednik model. Defines the number of different tokens that can be represented by the
                `inputs_ids` passed when calling [`LednikModel`].
            hidden_size (`int`, *optional*, defaults to 384):
                Dimensionality of the encoder layers and the pooler layer.
            embeddings_dropout (`float`, *optional*, defaults to 0.0):
                The dropout ratio for the embeddings.
            num_attention_heads (`int`, *optional*, defaults to 6):
                Number of attention heads for each attention layer in the Transformer encoder.
            num_hidden_layers (`int`, *optional*, defaults to 1):
                Number of hidden layers in the Transformer encoder.
            attention_dropout (`float`, *optional*, defaults to 0.0):
                The dropout ratio for the attention probabilities.
            attention_bias (`bool`, *optional*, defaults to `False`):
                Whether to use bias in the query, key, and value projections in the attention layer.
            out_attn_dropout (`float`, *optional*, defaults to 0.0):
                The dropout ratio for the output of the attention layer.
            rope_parameters (`RopeParameters` | `None`, *optional*, defaults to `None`):
                The parameters for the Rotary Position Embeddings (RoPE).
            max_position_embeddings (`int`, *optional*, defaults to 1024):
                The maximum sequence length that this model might ever be used with.
            hidden_act (`str`, *optional*, defaults to `"silu"`):
                The non-linear activation function (function or string) in the encoder and pooler.
            intermediate_size (`int`, *optional*, defaults to 576):
                Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
            mlp_bias (`bool`, *optional*, defaults to `False`):
                Whether to use bias in the MLP layers.
            mlp_dropout (`float`, *optional*, defaults to 0.0):
                The dropout ratio for the MLP layers.
            pad_token_id (`int`, *optional*, defaults to 30125):
                The id of the padding token.
            classifier_pooling (`Literal["cls", "mean"]`, *optional*, defaults to `"cls"`):
                Pooling strategy for the classifier.
            classifier_dropout (`float` | `None`, *optional*, defaults to 0.0):
                The dropout ratio for the classifier.
            classifier_bias (`bool` | `None`, *optional*, defaults to `False`):
                Whether to use bias in the classifier.
            classifier_activation (`str` | `None`, *optional*, defaults to `"gelu"`):
                Activation function for the classifier.
            **kwargs (`Any`):
                Additional keyword arguments passed to `PreTrainedConfig`.

        """
        if classifier_pooling not in {"cls", "mean"}:
            raise ValueError(
                f"Invalid classifier_pooling: {classifier_pooling}. "
                "Must be one of {'cls', 'mean'}."
            )
        if hidden_act not in {"gelu", "silu"}:
            raise ValueError(
                f"Invalid hidden_activation: {hidden_act}. "
                "Must be one of {'gelu', 'silu'}."
            )
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )
        if hidden_size % 8 != 0:
            new_hidden_size = (-hidden_size % 8) + hidden_size
            logger.info(
                f"Adjusting hidden_size from {hidden_size} to {new_hidden_size} to be divisible by 8"
            )
            hidden_size = new_hidden_size
        if intermediate_size % 8 != 0:
            new_intermediate_size = (-intermediate_size % 8) + intermediate_size
            logger.info(
                f"Adjusting intermediate_size from {intermediate_size} to {new_intermediate_size} to be divisible by 8"
            )
            intermediate_size = new_intermediate_size
        head_dim = hidden_size // num_attention_heads
        if head_dim not in {32, 64, 128, 256}:
            raise ValueError(
                f"head_dim ({head_dim}) must be one of {32, 64, 128, 256}. "
                "Please adjust hidden_size or num_attention_heads accordingly."
            )
        self.embeddings_dropout = embeddings_dropout
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.pad_token_id = pad_token_id
        self.intermediate_size = intermediate_size
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.out_attn_dropout = out_attn_dropout
        self.rope_parameters = rope_parameters
        self.hidden_act = hidden_act
        self.mlp_bias = mlp_bias
        self.mlp_dropout = mlp_dropout
        self.classifier_pooling = classifier_pooling
        self.classifier_dropout = classifier_dropout
        self.classifier_bias = classifier_bias
        self.classifier_activation = classifier_activation

        super().__init__(**kwargs)
        return
