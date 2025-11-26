from kostyl.ml_core.clearml.config_mixin import ClearMLConfigMixin
from kostyl.ml_core.configs.training_params import CheckpointConfig
from kostyl.ml_core.configs.training_params import EarlyStoppingConfig
from kostyl.ml_core.configs.training_params import LightningTrainerParameters
from pydantic import BaseModel
from pydantic import Field

from lednik.distill.training.configs import TrainConfig


class ClearMLTrainConfig(TrainConfig, ClearMLConfigMixin):
    """Configuration schema for the training process with ClearML functionality."""

    pass


class DataConfig(BaseModel, ClearMLConfigMixin):
    """Data configuration."""

    train_datasets: dict[str, str]  # except eval
    val_datasets: dict[str, str]
    batch_size: int
    num_workers: int = Field(ge=1)
    train_tokens_column: str
    val_tokens_column: str
    val_label_column: str


class TrainingParams(BaseModel, ClearMLConfigMixin):
    """Training parameters configuration."""

    teacher_model_id: str
    student_model_id: str
    tokenizer_id: str
    trainer: LightningTrainerParameters
    early_stopping: EarlyStoppingConfig | None = None
    checkpoint: CheckpointConfig
    data: DataConfig
