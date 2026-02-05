from dataclasses import dataclass
from dataclasses import fields
from pathlib import Path
from typing import Literal

from clearml import Task
from kostyl.ml.configs.training_settings import SUPPORTED_STRATEGIES
from kostyl.ml.configs.training_settings import CheckpointConfig
from kostyl.ml.configs.training_settings import DDPStrategyConfig
from kostyl.ml.configs.training_settings import EarlyStoppingConfig
from kostyl.ml.configs.training_settings import FSDP1StrategyConfig
from kostyl.ml.configs.training_settings import FSDP2StrategyConfig
from kostyl.ml.configs.training_settings import SingleDeviceStrategyConfig
from kostyl.ml.integrations.clearml import ClearMLCheckpointUploader
from kostyl.ml.integrations.lightning.callbacks import setup_checkpoint_callback
from kostyl.ml.integrations.lightning.callbacks import setup_early_stopping_callback
from kostyl.ml.integrations.lightning.loggers import setup_tb_logger
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.strategies import ModelParallelStrategy
from pydantic import BaseModel


@dataclass
class Callbacks:
    """Dataclass to hold PyTorch Lightning callbacks."""

    checkpoint: ModelCheckpoint
    lr_monitor: LearningRateMonitor
    early_stopping: EarlyStopping | None = None

    def to_list(self) -> list[Callback]:
        """Convert dataclass fields to a list of Callbacks. None values are omitted."""
        callbacks: list[Callback] = [
            getattr(self, field.name)
            for field in fields(self)
            if getattr(self, field.name) is not None
        ]
        return callbacks


class CheckpointUploaderConfig(BaseModel):
    """
    Configuration for uploading model checkpoints.

    Attributes:
        model_name: The name for the newly created model.
        label_enumeration: The label enumeration dictionary of string (label) to integer (value) pairs.
        config_dict: Optional configuration dictionary to associate with the model.
        tags: A list of strings which are tags for the model.
        comment: A comment / description for the model.
        framework: The framework of the model (e.g., "PyTorch", "TensorFlow").
        base_model_id: Optional ClearML model ID to use as a base for the new model
        upload_as_new_model: Whether to create a new ClearML model
            for every upload or update weights of the same model. When True,
            each checkpoint is uploaded as a separate model with timestamp added to the name.
            When False, weights of the same model are updated.
        verbose: Whether to log messages during upload.
        upload_strategy: Checkpoint upload mode:
            - "only-best": only the best checkpoint is uploaded
            - "every-checkpoint": every saved checkpoint is uploaded

    """

    model_name: str
    config_dict: dict[str, str] | None = None
    label_enumeration: dict[str, int] | None = None
    tags: list[str] | None = None
    comment: str | None = None
    framework: str | None = None
    base_model_id: str | None = None
    upload_as_new_model: bool = True
    verbose: bool = True
    upload_strategy: Literal["only-best", "every-checkpoint"] = "only-best"


def setup_callbacks(
    task: Task,
    root_path: Path,
    checkpoint_cfg: CheckpointConfig,
    early_stopping_cfg: EarlyStoppingConfig | None,
    checkpoint_uploader_config: CheckpointUploaderConfig,
) -> Callbacks:
    """
    Creates and configures a set of callbacks including checkpoint saving,
    learning rate monitoring, model registry uploading, and optional early stopping.
    """  # noqa: D205
    lr_monitor = LearningRateMonitor(
        logging_interval="step", log_weight_decay=True, log_momentum=False
    )

    uploader_kwargs = checkpoint_uploader_config.model_dump()
    upload_strategy = uploader_kwargs.pop("upload_strategy")
    model_uploader = ClearMLCheckpointUploader(**uploader_kwargs)

    checkpoint_callback = setup_checkpoint_callback(
        root_path / "checkpoints" / task.name / task.id,
        checkpoint_cfg,
        checkpoint_uploader=model_uploader,
        upload_strategy=upload_strategy,
    )

    if early_stopping_cfg is not None:
        early_stopping_callback = setup_early_stopping_callback(early_stopping_cfg)
    else:
        early_stopping_callback = None

    callbacks = Callbacks(
        checkpoint=checkpoint_callback,
        lr_monitor=lr_monitor,
        early_stopping=early_stopping_callback,
    )
    return callbacks


def setup_loggers(task: Task, root_path: Path) -> list[TensorBoardLogger]:
    """Set up PyTorch Lightning loggers for training."""
    loggers = [
        setup_tb_logger(root_path / "runs" / task.name / task.id),
    ]
    return loggers


def setup_strategy(
    strategy_settings: SUPPORTED_STRATEGIES,
    devices: list[int] | int,
) -> Literal["auto"] | ModelParallelStrategy | DDPStrategy | FSDPStrategy:
    """Configure and return a PyTorch Lightning training strategy."""
    if isinstance(devices, list):
        num_devices = len(devices)
    else:
        num_devices = devices

    match strategy_settings:
        case FSDP1StrategyConfig():
            if num_devices < 2:
                raise ValueError("FSDP strategy requires multiple devices.")

            strategy = FSDPStrategy()
        case DDPStrategyConfig():
            if num_devices < 2:
                raise ValueError("DDP strategy requires at least two devices.")
            strategy = DDPStrategy(
                find_unused_parameters=strategy_settings.find_unused_parameters
            )
        case FSDP2StrategyConfig():
            if num_devices < 2:
                raise ValueError("FSDP strategy requires multiple devices.")
            strategy = ModelParallelStrategy(
                data_parallel_size=num_devices,
                tensor_parallel_size=1,
                save_distributed_checkpoint=False,
            )
        case SingleDeviceStrategyConfig():
            if num_devices != 1:
                raise ValueError("SingleDevice strategy requires exactly one device.")
            strategy = "auto"
        case _:
            raise ValueError(
                f"Unsupported strategy type: {type(strategy_settings.trainer.strategy)}"
            )
    return strategy
