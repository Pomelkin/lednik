from typing import Literal

from kostyl.ml.configs import HyperparamsConfig
from kostyl.utils.logging import setup_logger
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator


logger = setup_logger(fmt="only_message")


class DirectDistillationConfig(BaseModel):
    """
    Configuration for direct embedding-to-embedding distillation.

    This configuration defines parameters for knowledge distillation using direct alignment
    of student and teacher embeddings via cosine similarity loss. The method supports optional
    dimensionality reduction of teacher embeddings when student dimension differs from teacher
    dimension, using either PCA or an autoencoder.
    """

    teacher_dim_reduction_type: Literal["pca", "autoencoder"] | None = None
    student_freeze_iters: int = Field(default=0, ge=0)
    reduction_dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    reconstruction_loss_boost_while_frozen: float | None = Field(
        default=None,
        ge=0,
        validate_default=False,
    )
    reconstruction_loss_weight: float | None = Field(
        default=None,
        ge=0,
        validate_default=False,
    )
    semantic_loss_weight: float
    student_dim: int
    teacher_dim: int

    @model_validator(mode="after")
    def validate_reconstruction_params(self) -> "DirectDistillationConfig":  # noqa: C901
        """Validate that student freeze parameters are set correctly."""
        if self.student_dim > self.teacher_dim:
            raise ValueError("student_dim must be less than or equal to teacher_dim.")
        if (
            self.student_dim != self.teacher_dim
            and self.teacher_dim_reduction_type is None
        ):
            raise ValueError(
                "teacher_dim_reduction_type must be set when student_dim and teacher_dim differ."
            )
        if self.teacher_dim_reduction_type != "autoencoder":
            if self.reduction_dropout > 0.0:
                logger.warning(
                    "reduction_dropout is only applicable with 'autoencoder' dimension reduction. Setting reduction_dropout to 0.0."
                )
                self.reduction_dropout = 0.0

            if self.reconstruction_loss_weight is not None:
                logger.warning(
                    "reconstruction_loss_weight is only applicable with 'autoencoder' dimension reduction. Setting reconstruction_loss_weight to None."
                )
                self.reconstruction_loss_weight = None
            if self.student_freeze_iters > 0:
                logger.warning(
                    "student_freeze_iters is only applicable with 'autoencoder' dimension reduction. Setting student_freeze_iters to 0."
                )
                self.student_freeze_iters = 0
            if self.reconstruction_loss_boost_while_frozen is not None:  # fmt: skip
                logger.warning(
                    "reconstruction_loss_boost_while_frozen is set, but dimension reduction is not 'autoencoder'. Setting reconstruction_loss_boost_while_frozen to None."
                )
                self.reconstruction_loss_boost_while_frozen = None

        if self.teacher_dim_reduction_type == "autoencoder":
            if self.reconstruction_loss_weight is None:
                raise ValueError(
                    "reconstruction_loss_weight must be set when using 'autoencoder' dimension reduction."
                )

        if (self.student_freeze_iters == 0) and (
            self.reconstruction_loss_boost_while_frozen is not None
        ):
            logger.warning(
                "reconstruction_loss_boost_while_frozen is set, but student_freeze_iters is 0. The boost will have no effect."
            )
            self.reconstruction_loss_boost_while_frozen = None
        return self


class DinoDistillationConfig(BaseModel):
    """
    Configuration for DINO-style distillation.

    This configuration covering student/teacher embedding dimensions, temperature scheduling, momentum ramp-up,
    Sinkhorn-Knopp iterations, and prototype head layout parameters.
    """

    student_dim: int
    teacher_dim: int

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


class FinetuningConfig(BaseModel):
    """
    Configuration schema for the training process.

    This class defines the hyperparameters and settings required for training the distillation model,
    including learning rate schedules, dimension reduction strategies, and model dimensions.
    """

    distillation_method: DinoDistillationConfig | DirectDistillationConfig
    grad_clip_val: float | None = Field(default=None, gt=0, validate_default=False)
    warmup_iters_ratio: float | None = Field(default=None, gt=0, validate_default=False)
    warmup_lr: float | None = Field(default=None, gt=0, validate_default=False)
    weight_decay: float = 0.0
    peak_lr: float
    final_lr: float | None = None
    student_dim: int
    teacher_dim: int
    teacher_pooling_method: Literal["cls", "mean", "last"]

    @model_validator(mode="after")
    def validate_lr(self) -> "FinetuningConfig":
        """Validate learning rate parameters."""
        if (self.warmup_iters_ratio is None) != (self.warmup_lr is None):
            raise ValueError(
                "Both warmup_iters and warmup_lr must be set together or both be None."
            )
        if self.final_lr is not None and self.final_lr > self.peak_lr:
            raise ValueError("final_lr must be less than or equal to peak_lr.")
        if self.final_lr is None:
            self.final_lr = self.peak_lr
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
