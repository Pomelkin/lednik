import pytest

from lednik.models import LednikConfig
from lednik.models import StaticEmbeddingsConfig


def test_num_hidden_layers_derived_from_layers() -> None:
    config = LednikConfig(
        layers=["full-attention", "gated-delta-net", "full-attention"]
    )
    assert config.num_hidden_layers == 3


def test_dims_are_rounded_up_to_multiple_of_8() -> None:
    config = LednikConfig(hidden_size=100, head_dim=60, intermediate_size=570)
    assert config.hidden_size == 104
    assert config.head_dim == 64
    assert config.intermediate_size == 576


def test_dims_multiple_of_8_are_kept() -> None:
    config = LednikConfig(hidden_size=384, head_dim=64, intermediate_size=576)
    assert config.hidden_size == 384
    assert config.head_dim == 64
    assert config.intermediate_size == 576


def test_invalid_hidden_act_raises() -> None:
    with pytest.raises(ValueError, match="hidden_activation"):
        LednikConfig(hidden_act="relu")


def test_invalid_classifier_pooling_raises() -> None:
    with pytest.raises(ValueError, match="classifier_pooling"):
        LednikConfig(classifier_pooling="max")


def test_static_config_has_no_layers() -> None:
    config = StaticEmbeddingsConfig(vocab_size=50, pad_token_id=0)
    assert config.num_hidden_layers == 0
    assert config.num_attention_heads == 0
