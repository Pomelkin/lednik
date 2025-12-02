from typing import Literal

from kostyl.ml.configs import HyperparamsConfig
from kostyl.utils.logging import setup_logger
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator


logger = setup_logger(fmt="only_message")


class FinetuningConfig(BaseModel):
    """
    Configuration schema for the training process.

    This class defines the hyperparameters and settings required for training the distillation model,
    including learning rate schedules, dimension reduction strategies, and model dimensions.
    """

    grad_clip_val: float | None = Field(default=None, gt=0, validate_default=False)
    warmup_iters: int | None = Field(default=None, gt=0, validate_default=False)
    warmup_lr: float | None = Field(default=None, gt=0, validate_default=False)
    base_lr: float
    weight_decay: float = 0.0

    teacher_dim_reduction_type: Literal["pca", "autoencoder"] | None = None

    student_dim: int
    student_freeze_iters: int = Field(default=0, ge=0)
    reduction_dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    reconstruction_loss_boost_while_frozen: float | None = Field(
        default=None, ge=0, validate_default=False
    )
    reconstruction_loss_weight: float | None = Field(
        default=None, ge=0, validate_default=False
    )

    semantic_loss_weight: float

    teacher_dim: int
    teacher_pooling_method: Literal["cls", "mean", "last"]

    @model_validator(mode="after")
    def validate_reconstruction_params(self) -> "FinetuningConfig":
        """Validate that student freeze parameters are set correctly."""
        if (self.teacher_dim_reduction_type != "autoencoder") and (
            self.reduction_dropout > 0.0
        ):
            logger.warning(
                "reduction_dropout is only applicable with 'autoencoder' dimension reduction. Setting reduction_dropout to 0.0."
            )
            self.reduction_dropout = 0.0
        if (self.reconstruction_loss_weight is not None) and (
            self.teacher_dim_reduction_type != "autoencoder"
        ):
            logger.warning(
                "reconstruction_loss_weight is only applicable with 'autoencoder' dimension reduction. Setting reconstruction_loss_weight to None."
            )
            self.reconstruction_loss_weight = None

        if (self.reconstruction_loss_weight is None) and (
            self.teacher_dim_reduction_type == "autoencoder"
        ):
            raise ValueError(
                "reconstruction_loss_weight must be set when using 'autoencoder' dimension reduction."
            )

        if (self.teacher_dim_reduction_type != "autoencoder") and (
            self.student_freeze_iters > 0
        ):
            logger.warning(
                "student_freeze_iters is only applicable with 'autoencoder' dimension reduction. Setting student_freeze_iters to 0."
            )
            self.student_freeze_iters = 0

        if (self.reconstruction_loss_boost_while_frozen is not None) and (
            self.reconstruction_loss_weight is None
        ):
            logger.warning(
                "reconstruction_loss_boost_while_frozen is set, but reconstruction_loss_weight is None. The boost will have no effect."
            )
            self.reconstruction_loss_boost_while_frozen = None

        if (self.student_freeze_iters == 0) and (
            self.reconstruction_loss_boost_while_frozen is not None
        ):
            logger.warning(
                "reconstruction_loss_boost_while_frozen is set, but student_freeze_iters is 0. The boost will have no effect."
            )
            self.reconstruction_loss_boost_while_frozen = None
        return self

    @model_validator(mode="after")
    def validate_warmup(self) -> "FinetuningConfig":
        """Validate that warmup parameters are set correctly."""
        if (self.warmup_iters is None) != (self.warmup_lr is None):
            raise ValueError(
                "Both warmup_iters and warmup_value must be set together or both be None."
            )
        return self

    @model_validator(mode="after")
    def validate_dim_reduction(self) -> "FinetuningConfig":
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
