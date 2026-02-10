from typing import Literal

from kostyl.ml.configs import CheckpointConfig
from kostyl.ml.configs import EarlyStoppingConfig
from kostyl.ml.configs import LightningTrainerParameters
from kostyl.ml.configs.mixins import ConfigLoadingMixin
from kostyl.ml.integrations.clearml import ConfigSyncingClearmlMixin
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

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
    ignore_mismatched_sizes: bool = True
    output_hidden_size: int | None = None


class StaticEmbeddingsTrainConfig(BaseModel):
    """Configuration for training a static embeddings model."""

    model_type: Literal["static_embeddings"]
    embedding_dropout: float = 0.0
    ignore_mismatched_sizes: bool = True


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
    max_length: int | None = Field(default=None, gt=0, validate_default=False)


class TrainingSettings(BaseModel, ConfigSyncingClearmlMixin, ConfigLoadingMixin):
    """Training parameters configuration."""

    teacher_model_id: str
    student_model_id: str
    tokenizer_id: str
    is_student_lightning_checkpoint: bool = False
    checkpoint_weight_prefix: str | None = None
    model_cfg: LednikModelTrainConfig | StaticEmbeddingsTrainConfig
    trainer: LightningTrainerParameters
    early_stopping: EarlyStoppingConfig | None = None
    checkpoint: CheckpointConfig = CheckpointConfig()
    data: DataConfig

    @model_validator(mode="after")
    def _validate_checkpoint_settigs(self) -> "TrainingSettings":
        if (
            self.is_student_lightning_checkpoint
            and self.checkpoint_weight_prefix is None
        ):
            raise ValueError(
                "checkpoint_weight_prefix must be provided when is_student_lightning_checkpoint is True."
            )
        return self
