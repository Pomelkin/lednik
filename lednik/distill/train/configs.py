from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from lednik.utils.logging import setup_logger

logger = setup_logger(fmt="only_message")


class TrainConfig(BaseModel):
    warmup_iters: int | None = Field(default=None, gt=0, validate_default=False)
    warmup_lr: float | None = Field(default=None, gt=0, validate_default=False)
    base_lr: float
    weight_decay: float = 0.0
    teacher_dim_reduction_type: Literal["pca", "autoencoder"] | None = None
    student_dim: int
    teacher_dim: int

    @model_validator(mode="after")
    def validate_warmup(self) -> "TrainConfig":
        """Validate that warmup parameters are set correctly."""
        if (self.warmup_iters is None) != (self.warmup_lr is None):
            raise ValueError(
                "Both warmup_iters and warmup_value must be set together or both be None."
            )
        return self

    @model_validator(mode="after")
    def validate_dim_reduction(self) -> "TrainConfig":
        """Validate that dimension reduction parameters are set correctly."""
        if (self.student_dim != self.teacher_dim) and (
            self.teacher_dim_reduction_type is None
        ):
            raise ValueError(
                "teacher_dim_reduction_type must be set when student_dim and teacher_dim differ."
            )
        if self.teacher_dim < self.student_dim:
            raise ValueError(
                "teacher_dim must be greater than or equal to student_dim."
            )
        if (self.teacher_dim_reduction_type != "autoencoder") and (
            self.weight_decay > 0.0
        ):
            self.weight_decay = 0.0
            logger.warning(
                "Weight decay is only applicable with 'autoencoder' dimension reduction. Setting weight_decay to 0.0."
            )
        return self
