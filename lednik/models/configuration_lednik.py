from dataclasses import field
from typing import Literal

from kostyl.utils.logging import setup_logger
from transformers import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters


logger = setup_logger(fmt="only_message")


class LednikConfig(PreTrainedConfig):
    """Configuration class for Lednik Model."""

    model_type = "LednikPreTrainedModel"
    # Initializer range for weight initialization.
    initializer_range = 0.02
    # Vocabulary size for token embeddings.
    vocab_size: int = 30522
    # Hidden size of encoder states.
    hidden_size: int = 384
    # Output embedding size produced by model.
    output_hidden_size: int | None = None
    # Dropout applied on input embeddings.
    embedding_dropout: float = 0.0
    # Number of attention heads.
    num_attention_heads: int = 6
    # Size of each attention head.
    head_dim: int = 64
    # Dropout on attention probabilities.
    attention_dropout: float = 0.0
    # Enable bias in attention projections.
    attention_bias: bool = False
    # Dropout after attention output projection.
    out_attn_dropout: float = 0.0
    # RoPE parameters dictionary.
    rope_parameters: RopeParameters | None = None
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

    # Attention gating (gating for full attention)
    use_gated_attention: bool = True

    # List of layers to include in the model architecture.
    layers: list[Literal["full-attention", "gated-delta-net"]] = field(
        default_factory=list
    )

    # Gated Delta Net (GDN) specific parameters.
    gdn_bidir: bool = True
    gdn_expand_v: float = 2.0
    gdn_conv_size: int = 4
    gdn_conv_bias: bool = False
    gdn_head_dim: int = 64
    gdn_num_heads: int = 6
    gdn_use_short_conv: bool = True
    gdn_allow_neg_eigval: bool = False
    use_mlp_after_gdn: bool = True

    # Pooling strategy for classifier head.
    classifier_pooling: Literal["cls", "mean"] = "cls"
    # Dropout before classifier projection.
    classifier_dropout: float | None = 0.0
    # Enable bias in classifier.
    classifier_bias: bool | None = False
    # Classifier activation function.
    classifier_activation: str | None = "gelu"

    @property
    def num_hidden_layers(self) -> int:
        """Number of transformer layers."""
        return len(self.layers)

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

        if self.hidden_size % 8 != 0:
            new_hidden_size = (-self.hidden_size % 8) + self.hidden_size
            logger.info(
                f"Adjusting hidden_size from {self.hidden_size} to {new_hidden_size} to be divisible by 8"
            )
            self.hidden_size = new_hidden_size
        if self.head_dim % 8 != 0:
            new_head_dim = (-self.head_dim % 8) + self.head_dim
            logger.info(
                f"Adjusting head_dim from {self.head_dim} to {new_head_dim} to be divisible by 8"
            )
            self.head_dim = new_head_dim
        if self.intermediate_size % 8 != 0:
            new_intermediate_size = (
                -self.intermediate_size % 8
            ) + self.intermediate_size
            logger.info(
                f"Adjusting intermediate_size from {self.intermediate_size} to {new_intermediate_size} to be divisible by 8"
            )
            self.intermediate_size = new_intermediate_size
        super().__post_init__(**kwargs)
        return
