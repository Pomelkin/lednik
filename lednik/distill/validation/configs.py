from typing import Literal

from kostyl.ml.configs.mixins import ConfigLoadingMixin
from pydantic import BaseModel


class RedisConfig(BaseModel, ConfigLoadingMixin["RedisConfig"]):
    """Configuration for Redis connection."""

    host: str
    port: int
    stream_name: str


class MRRConfig(BaseModel):
    """Configuration for retrieval evaluation."""

    qdrant_url: str
    qdrant_port: int
    mmr_top_k: int


class LogRegConfig(BaseModel):
    """Configuration for logistic regression evaluation."""

    lr: float = 3e-4
    weight_decay: float = 0.01
    solver: Literal["LBFGS", "Muon", "Adam"] = "LBFGS"
    tol: float = 1e-4
    batch_size: int = 128
    total_steps: int = 1000


class KNNConfig(BaseModel):
    """Configuration for k-NN evaluation."""

    k: int


class EvaluationRunnerConfig(BaseModel, ConfigLoadingMixin["EvaluationRunnerConfig"]):
    """Configuration for retrieval evaluation."""

    mrr_config: MRRConfig | None = None
    knn_config: KNNConfig | None = None
    logreg_config: LogRegConfig | None = None
    scatter_num_points: int


class EvaluationWorkerConfig(BaseModel, ConfigLoadingMixin["EvaluationWorkerConfig"]):
    """Configuration for the evaluation worker."""

    redis: RedisConfig
    runner_config: EvaluationRunnerConfig
