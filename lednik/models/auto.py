from pathlib import Path
from threading import Lock

import orjson
import torch
from transformers.configuration_utils import PreTrainedConfig

from .base import LednikPreTrainedModel


_registry_guard = Lock()
LEDNIK_MODEL_REGISTRY: dict[str, type[LednikPreTrainedModel]] = {}
LEDNIK_CONFIG_REGISTRY: dict[str, type[PreTrainedConfig]] = {}


def register_model[T: LednikPreTrainedModel](cls: type[T]) -> type[T]:
    """Class decorator that registers a model class in `LEDNIK_MODEL_REGISTRY` under its class name."""
    with _registry_guard:
        LEDNIK_MODEL_REGISTRY[cls.__name__] = cls
    return cls


def get_model_class(model_arch: str) -> type[LednikPreTrainedModel]:
    """Return the registered model class for the given architecture name."""
    try:
        return LEDNIK_MODEL_REGISTRY[model_arch]
    except KeyError as e:
        raise KeyError(
            f"Model architecture '{model_arch}' is not registered. "
            "Please ensure the model class is registered using the @register_model decorator."
        ) from e


def register_config[T: PreTrainedConfig](cls: type[T]) -> type[T]:
    """Class decorator that registers a config class in `LEDNIK_CONFIG_REGISTRY` under its `model_type`."""
    with _registry_guard:
        LEDNIK_CONFIG_REGISTRY[cls.model_type] = cls
    return cls


def get_config_class(model_type: str) -> type[PreTrainedConfig]:
    """Return the registered config class for the given `model_type`."""
    try:
        return LEDNIK_CONFIG_REGISTRY[model_type]
    except KeyError as e:
        raise KeyError(
            f"Model type '{model_type}' is not registered. "
            "Please ensure the config class is registered using the @register_config decorator."
        ) from e


def is_lednik_checkpoint(path: str | Path) -> bool:
    """
    Check whether a checkpoint belongs to a Lednik model (as opposed to a plain Transformers one).

    A Lightning `.ckpt` file is always considered a Lednik checkpoint; for a checkpoint
    directory the config's `architectures`/`model_type` are matched against the registries.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Path '{path}' does not exist.")

    if path.is_file() and path.suffix == ".ckpt":
        return True

    config_path = path / "config.json"
    if not config_path.exists():
        raise ValueError(f"Path '{path}' does not contain a config.json file.")

    config_dict = orjson.loads(config_path.read_bytes())
    architectures: list[str] = config_dict.get("architectures") or []
    return any(arch in LEDNIK_MODEL_REGISTRY for arch in architectures)


class AutoLednikModel:
    """Factory that resolves a registered Lednik model class from a checkpoint and instantiates it."""

    @staticmethod
    def from_pretrained(path: str | Path, *args, **kwargs) -> LednikPreTrainedModel:
        """
        Load a model from a Transformers checkpoint directory or a Lightning `.ckpt` file.

        The concrete class is resolved from the checkpoint config's `architectures` entry
        via the model registry; extra arguments are forwarded to the loader.
        """

        def get_model_class_from_config_dict(
            config_dict: dict,
        ) -> type[LednikPreTrainedModel]:
            architectures: list[str] | None = config_dict.get("architectures")
            if architectures is None:
                raise KeyError(
                    "The provided configuration dictionary does not contain an 'architectures' key. "
                    "Please ensure that the configuration dictionary is valid and contains the necessary information."
                )
            return get_model_class(architectures[0])

        if isinstance(path, str):
            path = Path(path)

        if not path.exists():
            raise ValueError(f"Path '{path}' does not exist.")

        # Transformers checkpoint format
        if path.is_dir():
            config_dict = orjson.loads((path / "config.json").read_bytes())
            model_type = get_model_class_from_config_dict(config_dict)

            # Del from_lightning_checkpoint kwargs to avoid passing them to the Transformers loader
            kwargs.pop("weights_prefix", None)
            kwargs.pop("strict_prefix", None)

            model_instance = model_type.from_pretrained(path, *args, **kwargs)

        # Lightning checkpoint format
        elif path.is_file() and path.suffix == ".ckpt":
            ckpt = torch.load(path, map_location="meta", weights_only=False)

            config_dict = ckpt.get("config", None)
            if config_dict is None:
                raise KeyError(
                    f"The provided checkpoint file '{path}' does not contain a 'config' key. "
                    "Please ensure that the checkpoint file is valid and contains the necessary information."
                )

            model_type = get_model_class_from_config_dict(config_dict)
            model_instance = model_type.from_lightning_checkpoint(path, *args, **kwargs)

        else:
            raise ValueError(
                f"Path '{path}' is neither a directory nor a .ckpt file. "
                "Please provide a valid path to a pretrained model or checkpoint."
            )
        return model_instance
