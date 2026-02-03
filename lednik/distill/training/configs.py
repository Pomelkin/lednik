from typing import Literal

from kostyl.ml.configs import HyperparamsConfig
from kostyl.ml.configs.hyperparams import OPTIMIZER_CONFIG
from kostyl.ml.configs.hyperparams import Lr
from kostyl.ml.configs.hyperparams import WeightDecay
from kostyl.utils.logging import setup_logger
from pydantic import BaseModel
from pydantic import Field


logger = setup_logger(fmt="only_message")


class DirectDistillationConfig(BaseModel):
    """Configuration for direct distillation."""

    type: Literal["direct-distillation"] = "direct-distillation"
    loss_type: Literal["cosine", "mse"] = "cosine"
    student_to_teacher_intermediate_dim: int | None = Field(
        default=None,
        ge=1,
        validate_default=False,
    )
    proj_dropout: float = Field(default=0.0, ge=0.0, le=1.0)


class DinoDistillationConfig(BaseModel):
    """
    Configuration for DINO-style distillation.

    This configuration covering student/teacher embedding dimensions, temperature scheduling, momentum ramp-up,
    Sinkhorn-Knopp iterations, and prototype head layout parameters.
    """

    type: Literal["dino"] = "dino"

    student_to_teacher_intermediate_dim: int | None = Field(
        default=None,
        ge=1,
        validate_default=False,
    )
    proj_dropout: float = Field(default=0.0, ge=0.0, le=1.0)

    start_teacher_temp: float
    peak_teacher_temp: float
    warmup_teacher_temp_steps_ratio: float
    student_temp: float
    start_teacher_momentum: float
    final_teacher_momentum: float
    sinkhorn_knopp_n_iters: int

    head_nlayers: int
    head_bottleneck_dim: int
    head_hidden_dim: int
    head_n_prototypes: int


AVAILABLE_DISTILLATION_METHODS = DinoDistillationConfig | DirectDistillationConfig


class DistillationConfig(HyperparamsConfig):
    """
    Configuration schema for the training process.

    This class defines the hyperparameters and settings required for training the distillation model,
    including learning rate schedules, dimension reduction strategies, and model dimensions.
    """

    distillation_method: AVAILABLE_DISTILLATION_METHODS | None = None
    teacher_pooling_method: Literal["cls", "mean", "last"]
    student_pooling_method: Literal["cls", "mean", "last"] | None = None
    grad_clip_val: float | None = Field(default=None, gt=0, validate_default=False)
    optimizer: OPTIMIZER_CONFIG
    lr: Lr
    weight_decay: WeightDecay


class ClassifierTrainConfig(HyperparamsConfig):
    """Configuration schema for classifier training hyperparameters."""

    label2id: dict[str, int]
    class_weights: list[float] | None = None

    @property
    def num_labels(self) -> int:
        """Get the number of labels."""
        return len(self.label2id)

    @property
    def id2label(self) -> dict[int, str]:
        """Get the mapping from label IDs to label names."""
        return {v: k for k, v in self.label2id.items()}
