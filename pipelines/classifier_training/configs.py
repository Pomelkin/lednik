from kostyl.ml.configs import CheckpointConfig
from kostyl.ml.configs import EarlyStoppingConfig
from kostyl.ml.configs import LightningTrainerParameters
from kostyl.ml.configs.mixins import ConfigLoadingMixin
from kostyl.ml.integrations.clearml import ConfigSyncingClearmlMixin
from pydantic import BaseModel
from pydantic import Field

from lednik.distill.training.configs import (
    ClassifierHyperparamsConfig as BaseClassifierHyperparamsConfig,
)


class ClassifierHyperparamsConfig(
    BaseClassifierHyperparamsConfig, ConfigLoadingMixin, ConfigSyncingClearmlMixin
):
    """Configuration for classifier training."""

    embedding_dropout: float = 0.0
    classifier_dropout: float = 0.0


class DataConfig(BaseModel):
    """Data configuration."""

    datasets: dict[str, str]  # except eval
    batch_size: int
    num_workers: int = Field(ge=1)
    tokens_column: str
    label_column: str
    max_length: int | None = None


class TrainingSettings(BaseModel, ConfigSyncingClearmlMixin, ConfigLoadingMixin):
    """Training parameters configuration."""

    model_id: str
    tokenizer_id: str | None = None
    weights_prefix: str | None = None
    trainer: LightningTrainerParameters
    early_stopping: EarlyStoppingConfig | None = None
    checkpoint: CheckpointConfig = CheckpointConfig()
    data: DataConfig
