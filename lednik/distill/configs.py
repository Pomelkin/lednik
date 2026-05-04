from typing import Literal

from kostyl.ml.configs import HyperparamsConfig
from kostyl.ml.configs.hyperparams import OPTIMIZER_CONFIG
from kostyl.ml.configs.hyperparams import Lr
from kostyl.ml.configs.hyperparams import WeightDecay
from kostyl.utils.logging import setup_logger
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator


logger = setup_logger(fmt="only_message")


class DirectDistillationConfig(BaseModel):
    """Configuration for direct distillation."""

    type: Literal["direct-distillation"] = "direct-distillation"
    per_token_loss_type: Literal["cosine", "mse"] = "cosine"  # noqa: S105
    contrastive_loss_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    temperature: float | None = Field(default=None, gt=0.0, validate_default=False)

    @model_validator(mode="after")
    def _validate_temp(self) -> "DirectDistillationConfig":
        if self.contrastive_loss_weight > 0.0 and self.temperature is None:
            raise ValueError(
                "Temperature must be provided when contrastive_loss_weight is greater than 0."
            )
        return self


AVAILABLE_DISTILLATION_METHODS = DirectDistillationConfig


class DistillationConfig(HyperparamsConfig):
    """
    Configuration schema for the training process.

    This class defines the hyperparameters and settings required for training the distillation model,
    including learning rate schedules, dimension reduction strategies, and model dimensions.
    """

    distillation_method: AVAILABLE_DISTILLATION_METHODS
    teacher_pooling_method: Literal["cls", "mean", "last"]

    freeze_student_emb_steps_ratio: float | None = Field(
        default=None, ge=0.0, le=1.0, validate_default=False
    )

    embeddings_lr_multiplier: float | None = Field(
        default=None, gt=0.0, validate_default=False
    )
    attn_proj_wd_multiplier: float | None = Field(
        default=None, gt=0.0, validate_default=False
    )
    grad_clip_val: float | None = Field(default=None, gt=0, validate_default=False)
    optimizer: OPTIMIZER_CONFIG
    lr: Lr
    weight_decay: WeightDecay
