import json
from pathlib import Path

import pytest
import torch

from lednik.models import AutoLednikModel
from lednik.models import LednikModel
from lednik.models import StaticEmbeddingsConfig
from lednik.models import StaticEmbeddingsModel
from lednik.models import get_model_class
from lednik.models import is_lednik_checkpoint


def test_model_registry_resolves_registered_classes() -> None:
    assert get_model_class("LednikModel") is LednikModel
    assert get_model_class("StaticEmbeddingsModel") is StaticEmbeddingsModel


def test_model_registry_raises_for_unknown_architecture() -> None:
    with pytest.raises(KeyError, match="not registered"):
        get_model_class("BertModel")


def test_ckpt_file_is_always_lednik(tmp_path: Path) -> None:
    ckpt = tmp_path / "epoch=0-step=1.ckpt"
    ckpt.touch()
    assert is_lednik_checkpoint(ckpt)


def test_directory_is_matched_by_architectures(tmp_path: Path) -> None:
    lednik_dir = tmp_path / "lednik"
    lednik_dir.mkdir()
    (lednik_dir / "config.json").write_text(
        json.dumps({"architectures": ["StaticEmbeddingsModel"]})
    )
    assert is_lednik_checkpoint(lednik_dir)

    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()
    (plain_dir / "config.json").write_text(json.dumps({"architectures": ["BertModel"]}))
    assert not is_lednik_checkpoint(plain_dir)


def test_directory_without_config_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="config.json"):
        is_lednik_checkpoint(tmp_path)


def test_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        is_lednik_checkpoint(tmp_path / "missing")


def test_save_load_roundtrip_resolves_concrete_class(tmp_path: Path) -> None:
    config = StaticEmbeddingsConfig(vocab_size=50, hidden_size=16, pad_token_id=0)
    model = StaticEmbeddingsModel(config)
    model.save_pretrained(tmp_path)

    assert is_lednik_checkpoint(tmp_path)

    loaded = AutoLednikModel.from_pretrained(tmp_path)
    assert isinstance(loaded, StaticEmbeddingsModel)
    assert torch.equal(loaded.embeddings.weight, model.embeddings.weight)
