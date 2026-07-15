from typing import Literal, Any

from kostyl.ml.configs import CheckpointConfig
from kostyl.ml.configs import EarlyStoppingConfig
from kostyl.ml.configs import LightningTrainerParameters
from kostyl.ml.configs.mixins import ConfigLoadingMixin
from kostyl.ml.integrations.clearml import ConfigSyncingClearmlMixin
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from lednik.distill.configs import DistillationConfig as BaseDistillationConfig
from lednik.distill.validation import EvaluationRunnerConfig
from lednik.distill.validation.structs import RedisConfig


class DistillationConfig(
    BaseDistillationConfig, ConfigSyncingClearmlMixin, ConfigLoadingMixin
):
    """Configuration schema for the training process with ClearML functionality."""


class ModelTrainingConfig(BaseModel):
    """Configuration for training a static embeddings model."""

    model_type: Literal["static_embeddings", "lednik"]
    override_params: dict[str, Any] = Field(default_factory=dict)


class DataConfig(BaseModel):
    """Data configuration."""

    datasets: dict[str, str]
    batch_size: int
    num_workers: int = Field(ge=1)

    query_tok_colname: str
    query_text_colname: str
    pos_tok_colname: str
    pos_text_colname: str
    neg_tok_colname: str | None = None
    neg_text_colname: str | None = None

    aug_prob: float = Field(default=0.0, ge=0.0, le=1.0)

    query_teacher_embedding_colname: str
    pos_teacher_embedding_colname: str
    neg_teacher_embedding_colname: str | None = None

    val_label_colname: str | None = None
    val_num_labels: int | None = Field(
        default=None,
        gt=0,
        validate_default=False,
    )
    max_length: int | None = Field(default=None, gt=0, validate_default=False)

    @model_validator(mode="after")
    def _validate_negative_columns(self) -> "DataConfig":
        if (self.neg_tok_colname is None) != (
            self.neg_teacher_embedding_colname is None
        ):
            raise ValueError(
                "neg_tok_colname and neg_teacher_embedding_colname must be provided together or omitted."
            )
        return self


class TrainingSettings(BaseModel, ConfigSyncingClearmlMixin, ConfigLoadingMixin):
    """Training parameters configuration."""

    teacher_model_id: str
    student_model_id: str
    tokenizer_id: str
    model_cfg: ModelTrainingConfig

    is_student_lightning_checkpoint: bool = False
    checkpoint_weight_prefix: str | None = None

    trainer: LightningTrainerParameters
    early_stopping: EarlyStoppingConfig | None = None
    checkpoint: CheckpointConfig = CheckpointConfig()
    data: DataConfig

    redis: RedisConfig | None = None
    runner_config: EvaluationRunnerConfig | None = None

    distill_config: DistillationConfig

    @model_validator(mode="after")
    def _validate_settings(self) -> "TrainingSettings":
        if (
            self.is_student_lightning_checkpoint
            and self.checkpoint_weight_prefix is None
        ):
            raise ValueError(
                "checkpoint_weight_prefix must be provided when is_student_lightning_checkpoint is True."
            )
        if self.redis is None and self.runner_config is None:
            raise ValueError(
                "At least one of redis or runner_config must be provided for:"
                "\n- If redis is not provided, runner_config must be provided for local evaluation."
                "\n- If runner_config is not provided, redis must be provided for remote evaluation."
                "\n- If both are provided, the dispatcher will attempt to use Redis for dispatching and fall back to local evaluation if Redis is unavailable."
            )
        return self
