from typing import Literal

from huggingface_hub.dataclasses import strict
from kostyl.utils.logging import setup_logger
from transformers import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters


logger = setup_logger(fmt="only_message")


@strict
class LednikConfig(PreTrainedConfig):
    """Configuration class for Lednik Model."""

    model_type = "LednikPreTrainedModel"
    # Vocabulary size for token embeddings.
    vocab_size: int = 30522
    # Hidden size of encoder states.
    hidden_size: int = 384
    # Output embedding size produced by model.
    output_hidden_size: int | None = None
    # Dropout applied on input embeddings.
    embeddings_dropout: float = 0.0
    # Number of attention heads.
    num_attention_heads: int = 6
    # Number of transformer layers.
    num_hidden_layers: int = 1
    # Dropout on attention probabilities.
    attention_dropout: float = 0.0
    # Enable bias in attention projections.
    attention_bias: bool = False
    # Dropout after attention output projection.
    out_attn_dropout: float = 0.0
    # RoPE parameters dictionary.
    rope_parameters: RopeParameters
    # Maximum supported sequence length.
    max_position_embeddings: int = 1024
    # Activation function in feed-forward.
    hidden_act: str = "silu"
    # Intermediate FFN size.
    intermediate_size: int = 576
    # Enable bias in MLP projections.
    mlp_bias: bool = False
    # Dropout inside MLP block.
    mlp_dropout: float = 0.0
    # Padding token id.
    pad_token_id: int = 30125
    # Pooling strategy for classifier head.
    classifier_pooling: Literal["cls", "mean"] = "cls"
    # Dropout before classifier projection.
    classifier_dropout: float | None = 0.0
    # Enable bias in classifier.
    classifier_bias: bool | None = False
    # Classifier activation function.
    classifier_activation: str | None = "gelu"

    def __post_init__(self, **kwargs) -> None:  # noqa: D105
        if self.classifier_pooling not in {"cls", "mean"}:
            raise ValueError(
                f"Invalid classifier_pooling: {self.classifier_pooling}. "
                "Must be one of {'cls', 'mean'}."
            )
        if self.hidden_act not in {"gelu", "silu"}:
            raise ValueError(
                f"Invalid hidden_activation: {self.hidden_act}. "
                "Must be one of {'gelu', 'silu'}."
            )
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.hidden_size % 8 != 0:
            new_hidden_size = (-self.hidden_size % 8) + self.hidden_size
            logger.info(
                f"Adjusting hidden_size from {self.hidden_size} to {new_hidden_size} to be divisible by 8"
            )
            self.hidden_size = new_hidden_size
        if self.intermediate_size % 8 != 0:
            new_intermediate_size = (
                -self.intermediate_size % 8
            ) + self.intermediate_size
            logger.info(
                f"Adjusting intermediate_size from {self.intermediate_size} to {new_intermediate_size} to be divisible by 8"
            )
            self.intermediate_size = new_intermediate_size
        head_dim = self.hidden_size // self.num_attention_heads
        if head_dim not in {32, 64, 128, 256}:
            raise ValueError(
                f"head_dim ({head_dim}) must be one of {32, 64, 128, 256}. "
                "Please adjust hidden_size or num_attention_heads accordingly."
            )
        super().__post_init__(**kwargs)
        return
