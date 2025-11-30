from kostyl.ml.configs import CheckpointConfig
from kostyl.ml.configs import EarlyStoppingConfig
from kostyl.ml.configs import KostylBaseModel
from kostyl.ml.configs.training_settings import LightningTrainerParameters
from pydantic import BaseModel
from pydantic import Field

from lednik.distill.training.configs import TrainConfig as LednikTrainConfig


class TrainConfig(LednikTrainConfig, KostylBaseModel):
    """Configuration schema for the training process with ClearML functionality."""

    embedding_dropout: float = 0.0


class DataConfig(BaseModel):
    """Data configuration."""

    train_datasets: dict[str, str]  # except eval
    val_datasets: dict[str, str]
    batch_size: int
    num_workers: int = Field(ge=1)
    train_tokens_column: str
    val_tokens_column: str
    val_label_column: str


class TrainingSettings(KostylBaseModel):
    """Training parameters configuration."""

    teacher_model_id: str
    student_model_id: str
    tokenizer_id: str
    trainer: LightningTrainerParameters
    early_stopping: EarlyStoppingConfig | None = None
    checkpoint: CheckpointConfig = CheckpointConfig()
    data: DataConfig
