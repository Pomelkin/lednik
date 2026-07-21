from pathlib import Path

from huggingface_hub import repo_exists
from huggingface_hub import snapshot_download
from kostyl.utils import setup_logger


try:
    from clearml import InputModel

    CLEAR_ML_AVAILABLE = True
except ImportError:
    CLEAR_ML_AVAILABLE = False

logger = setup_logger(fmt="detailed")


def determine_path(path_name_or_id: str, is_tokenizer: bool) -> Path:
    """
    Resolve a model or tokenizer reference to a validated local path.

    Resolution order:
        1. ClearML model ID (when the `clearml` package is available) — the
           artifact is fetched into the local ClearML cache.
        2. Hugging Face Hub repo id — resolved via `snapshot_download`.
        3. Local filesystem path, used as-is.

    Args:
        path_name_or_id: Local path, ClearML model ID or HF Hub repo id.
        is_tokenizer: Validate the result as a tokenizer directory (must contain
            `tokenizer.json`) instead of a model checkpoint (a directory with
            `config.json` or a Lightning `.ckpt` file).

    Returns:
        Absolute path to the resolved artifact.

    Raises:
        FileNotFoundError: If nothing was resolved and the string is not an existing path.
        ValueError: If the resolved path fails content validation.
    """
    path: Path | None = None
    # Then check if it's a ClearML model ID
    if CLEAR_ML_AVAILABLE:
        model_id = path_name_or_id
        try:
            clearml_model = InputModel(model_id=model_id)
            local_path = clearml_model.get_local_copy()
            path = Path(local_path)
            logger.info(
                f"Resolved ClearML model ID '{model_id}' to local path '{path}'."
            )
        except Exception:  # noqa: S110
            pass

    # If neither, try to check if it's a Hugging Face model ID
    if repo_exists(path_name_or_id):
        repo_id = path_name_or_id
        try:
            local_path = snapshot_download(repo_id=repo_id)
            path = Path(local_path)
            logger.info(
                f"Resolved Hugging Face model ID '{repo_id}' to local path '{path}'."
            )
        except Exception:  # noqa: S110
            pass

    # Check if the provided string is a local path
    if path is None:
        path = Path(path_name_or_id)
        logger.info(f"Using provided local path '{path}'.")

    if not path.exists():
        raise FileNotFoundError(
            f"Path '{path}' does not exist. Please provide a valid local path, ClearML model ID, or Hugging Face model ID."
        )

    # Path content validation
    # tokenizer validation
    if is_tokenizer:
        if not (path.is_dir() and (path / "tokenizer.json").exists()):
            raise ValueError(f"Path '{path}' is not a valid tokenizer directory.")
    # model validation
    elif not (
        (path.is_dir() and (path / "config.json").exists())
        or (path.is_file() and path.suffix == ".ckpt")
    ):
        raise ValueError(
            f"Path '{path}' is not a valid Transformers checkpoint directory or Lightning .ckpt file."
        )
    return path.resolve()
