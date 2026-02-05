from typing import Literal

from kostyl.ml.configs import CheckpointConfig
from kostyl.ml.configs import EarlyStoppingConfig
from kostyl.ml.configs import LightningTrainerParameters
from kostyl.ml.configs.mixins import ConfigLoadingMixin
from kostyl.ml.integrations.clearml import ConfigSyncingClearmlMixin
from pydantic import BaseModel
from pydantic import Field

from lednik.distill.training.configs import DistillationConfig as BaseDistillationConfig


class DistillationConfig(
    BaseDistillationConfig, ConfigSyncingClearmlMixin, ConfigLoadingMixin
):
    """Configuration schema for the training process with ClearML functionality."""


class LednikModelTrainConfig(BaseModel):
    """Configuration for training a Lednik model."""

    model_type: Literal["lednik"]
    embeddings_dropout: float = 0.0
    attention_dropout: float = 0.0
    out_attn_dropout: float = 0.0
    mlp_dropout: float = 0.0


class StaticEmbeddingsTrainConfig(BaseModel):
    """Configuration for training a static embeddings model."""

    model_type: Literal["static_embeddings"]
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
    val_num_labels: int | None = Field(
        default=None,
        gt=0,
        validate_default=False,
    )
    max_length: int = Field(gt=0)


class TrainingSettings(BaseModel, ConfigSyncingClearmlMixin, ConfigLoadingMixin):
    """Training parameters configuration."""

    teacher_model_id: str
    student_model_id: str
    model_cfg: LednikModelTrainConfig | StaticEmbeddingsTrainConfig
    tokenizer_id: str
    trainer: LightningTrainerParameters
    early_stopping: EarlyStoppingConfig | None = None
    checkpoint: CheckpointConfig = CheckpointConfig()
    data: DataConfig
