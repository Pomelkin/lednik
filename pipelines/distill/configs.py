from typing import Literal

from kostyl.ml.configs import CheckpointConfig
from kostyl.ml.configs import EarlyStoppingConfig
from kostyl.ml.configs import LightningTrainerParameters
from kostyl.ml.configs.mixins import ConfigLoadingMixin
from kostyl.ml.integrations.clearml import ConfigSyncingClearmlMixin
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from lednik.distill.configs import DistillationConfig as BaseDistillationConfig
from lednik.distill.validation.structs import RedisConfig


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
    output_hidden_size: int | None = None


class StaticEmbeddingsTrainConfig(BaseModel):
    """Configuration for training a static embeddings model."""

    model_type: Literal["static_embeddings"]
    embedding_dropout: float = 0.0


class DataConfig(BaseModel):
    """Data configuration."""

    datasets: dict[str, str]
    batch_size: int
    num_workers: int = Field(ge=1)

    query_colname: str
    pos_colname: str
    neg_colname: str

    val_label_colname: str | None = None
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
    model_cfg: LednikModelTrainConfig | StaticEmbeddingsTrainConfig

    is_student_lightning_checkpoint: bool = False
    checkpoint_weight_prefix: str | None = None

    trainer: LightningTrainerParameters
    early_stopping: EarlyStoppingConfig | None = None
    checkpoint: CheckpointConfig = CheckpointConfig()
    data: DataConfig

    redis: RedisConfig | None = None

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
