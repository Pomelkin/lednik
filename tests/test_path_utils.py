import json
from pathlib import Path

import pytest

import lednik.path_utils as path_utils
from lednik.path_utils import determine_path


@pytest.fixture(autouse=True)
def _local_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep resolution local: no ClearML lookups and no HF Hub requests."""
    monkeypatch.setattr(path_utils, "CLEAR_ML_AVAILABLE", False)
    monkeypatch.setattr(path_utils, "repo_exists", lambda *args, **kwargs: False)


def test_resolves_local_model_directory(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({}))
    assert determine_path(str(tmp_path), is_tokenizer=False) == tmp_path.resolve()


def test_resolves_local_ckpt_file(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.ckpt"
    ckpt.touch()
    assert determine_path(str(ckpt), is_tokenizer=False) == ckpt.resolve()


def test_resolves_local_tokenizer_directory(tmp_path: Path) -> None:
    (tmp_path / "tokenizer.json").write_text(json.dumps({}))
    assert determine_path(str(tmp_path), is_tokenizer=True) == tmp_path.resolve()


def test_model_directory_without_config_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a valid"):
        determine_path(str(tmp_path), is_tokenizer=False)


def test_model_directory_is_not_a_tokenizer(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({}))
    with pytest.raises(ValueError, match="tokenizer"):
        determine_path(str(tmp_path), is_tokenizer=True)


def test_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        determine_path(str(tmp_path / "missing"), is_tokenizer=False)
