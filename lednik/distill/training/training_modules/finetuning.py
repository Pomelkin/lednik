from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any
from typing import Literal
from typing import cast
from typing import override

import plotly.graph_objects as go
import torch
import torch.distributed as dist
import torch.nn.functional as F
from clearml import Task
from kostyl.ml.dist_utils import scale_lrs_by_world_size
from kostyl.ml.lightning.extenstions import KostylLightningModule
from kostyl.ml.lightning.steps_estimation import estimate_total_steps
from kostyl.ml.params_groups import create_params_groups
from kostyl.ml.schedulers.cosine import CosineParamScheduler
from kostyl.ml.schedulers.cosine import CosineScheduler
from kostyl.ml.schedulers.linear import LinearParamScheduler
from kostyl.utils.logging import setup_logger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torchmetrics import Accuracy
from torchmetrics import CosineSimilarity
from torchmetrics import F1Score
from torchmetrics import MeanSquaredError
from torchmetrics import MetricCollection
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import BaseModelOutput

from lednik.distill.dim_reduction import PCA
from lednik.distill.dim_reduction import Autoencoder
from lednik.distill.extraction_utils import get_sentence_embedding
from lednik.distill.training.configs import DinoDistillationConfig
from lednik.distill.training.configs import DirectDistillationConfig
from lednik.distill.training.configs import FinetuningConfig
from lednik.distill.training.knn import knn_predict
from lednik.distill.training.layers import DINOHead
from lednik.distill.training.losses import dino_ce_loss
from lednik.distill.training.losses import sinkhorn_knopp_teacher
from lednik.static_embeddings.modeling import StaticEmbeddingsModel
from lednik.static_embeddings.outputs import StaticEmbeddingsOutput


logger = setup_logger(add_rank=True)


@dataclass
class _BaseStepOutput:
    loss: torch.Tensor
    semantic_loss: torch.Tensor
    teacher_embeddings: torch.Tensor
    student_embeddings: torch.Tensor
    student_sentence_embeddings: torch.Tensor
    teacher_sentence_embeddings: torch.Tensor
    reconstruction_loss: torch.Tensor | None = None


@dataclass
class _DirectDistillationOutput:
    loss: torch.Tensor
    semantic_loss: torch.Tensor
    reconstruction_loss: torch.Tensor | None = None


@dataclass
class _DinoDistillationOutput:
    loss: torch.Tensor
    semantic_loss: torch.Tensor


@dataclass
class _EvalResult:
    teacher_embeddings: torch.Tensor
    student_embeddings: torch.Tensor
    labels: torch.Tensor

    def __setattr__(self, name: str, value: Any) -> None:
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Attribute {name} must be a torch.Tensor")
        value = value.detach().cpu()
        super().__setattr__(name, value)
        return


def metric_factory() -> MetricCollection:
    """Create a collection of metrics for evaluation."""
    collection = MetricCollection(
        {
            "RMSE": MeanSquaredError(squared=False),
            "CosineSimilarity": CosineSimilarity(reduction="mean"),
        }
    )
    return collection


def _classification_metric_factory(
    num_labels: int, prefix_str: str = ""
) -> MetricCollection:
    prefix = f"{prefix_str}_" if prefix_str else ""
    if num_labels == 1:
        metric_collection = MetricCollection(
            {
                f"{prefix}F1": F1Score("binary"),
                f"{prefix}Accuracy": Accuracy("binary"),
            }
        )
    else:
        metric_collection = MetricCollection(
            {
                f"{prefix}F1-Macro": F1Score(
                    task="multiclass", num_classes=num_labels, average="macro"
                ),
                f"{prefix}Accuracy": Accuracy(
                    task="multiclass", num_classes=num_labels
                ),
            }
        )
    return metric_collection


class FineTuningModule(KostylLightningModule):
    """A PyTorch Lightning module for fine-tuning a static embeddings model via knowledge distillation."""

    def __init__(
        self,
        teacher: PreTrainedModel,
        static_model: StaticEmbeddingsModel,
        train_cfg: FinetuningConfig,
        tokenizer: PreTrainedTokenizerBase,
        task: Task | None = None,
        num_labels: int | None = None,
    ) -> None:
        """
        Initialize Fine-Tuning Lightning Module.

        Args:
            teacher : The pre-trained teacher hf model.
            static_model : The static embeddings model to be trained.
            train_cfg : Training configuration.
            tokenizer : The tokenizer corresponding to the teacher model.
            task : ClearML Task for logging (optional).
            num_labels : Number of classification labels for KNN evaluation metrics.
                If provided, enables KNN-based evaluation during validation.
                If None, KNN metrics are disabled.

        """
        super().__init__()
        self.teacher = teacher
        self.static_model = static_model.train()

        self.student_dino_head: DINOHead | None = None
        self.teacher_dino_head: DINOHead | None = None
        self.student_to_teacher_proj: nn.Linear | nn.Identity | None = None
        self.dim_reducer: PCA | Autoencoder | None = None
        self.teacher_temp_scheduler: LinearParamScheduler | None = None
        self.teacher_momentum_scheduler: CosineParamScheduler | None = None
        self.teacher_head_params_list: list[nn.Parameter] = []
        self.student_head_params_list: list[nn.Parameter] = []

        distillation_cfg = train_cfg.distillation_method
        if isinstance(distillation_cfg, DinoDistillationConfig):
            if distillation_cfg.student_dim != distillation_cfg.teacher_dim:
                self.student_to_teacher_proj = nn.Linear(
                    distillation_cfg.student_dim,
                    distillation_cfg.teacher_dim,
                    bias=True,
                )
                nn.init.trunc_normal_(self.student_to_teacher_proj.weight, std=0.02)
                nn.init.zeros_(self.student_to_teacher_proj.bias)
            else:
                self.student_to_teacher_proj = nn.Identity()
            dino_head_in_dim = distillation_cfg.teacher_dim
            head = partial(
                DINOHead,
                in_dim=dino_head_in_dim,
                out_dim=distillation_cfg.head_n_prototypes,
                nlayers=distillation_cfg.head_nlayers,
                bottleneck_dim=distillation_cfg.head_bottleneck_dim,
                hidden_dim=distillation_cfg.head_hidden_dim,
            )
            self.student_dino_head = head()
            self.teacher_dino_head = head().eval()
            for param in self.teacher_dino_head.parameters():
                param.requires_grad = False
        elif isinstance(distillation_cfg, DirectDistillationConfig):
            match distillation_cfg.teacher_dim_reduction_type:
                case "pca":
                    self.dim_reducer = PCA(n_components=distillation_cfg.student_dim)
                case "autoencoder":
                    self.dim_reducer = Autoencoder(
                        input_dim=distillation_cfg.teacher_dim,
                        latent_dim=distillation_cfg.student_dim,
                        dropout=distillation_cfg.reduction_dropout,
                    )
                case None:
                    self.dim_reducer = None
                case _:
                    raise ValueError(
                        f"Unsupported dimension reduction type: {train_cfg.teacher_dim_reduction_type}"
                    )
        else:
            raise ValueError(
                f"Unsupported distillation method config type: {type(distillation_cfg)}"
            )

        self.train_cfg = train_cfg
        self.distillation_cfg = distillation_cfg

        self.tokenizer = tokenizer
        self.train_metrics = metric_factory()
        self.val_metrics = metric_factory()
        if num_labels is not None:
            self.use_knn_in_val = True
            self.use_logprob = True
            self.teacher_knn_metrics = _classification_metric_factory(
                prefix_str="teacher", num_labels=num_labels
            )
            self.student_knn_metrics = _classification_metric_factory(
                prefix_str="student", num_labels=num_labels
            )
            self.num_labels = num_labels
        else:
            self.use_knn_in_val = False
            self.use_logprob = True
            self.teacher_knn_metrics = None
            self.student_knn_metrics = None
            self.num_labels = None

        self.task = task
        self.freeze_model("teacher")

        self.eval_outputs: list[_EvalResult] = []
        return

    @property
    @override
    def grad_clip_val(self) -> float | None:
        return self.train_cfg.grad_clip_val

    @property
    @override
    def model_instance(self) -> PreTrainedModel | nn.Module:
        """Returns the underlying model."""
        return self.static_model

    def freeze_model(self, target_model: Literal["teacher", "student"]) -> None:
        """Freezes model so its parameters become non-trainable."""
        match target_model:
            case "teacher":
                for param in self.teacher.parameters():
                    param.requires_grad = False
            case "student":
                for param in self.static_model.parameters():
                    param.requires_grad = False
            case _:
                raise ValueError(f"Unknown model to freeze: {target_model}")
        return

    def unfreeze_model(self, target_model: Literal["teacher", "student"]) -> None:
        """Unfreezes model so its parameters become trainable again."""
        match target_model:
            case "teacher":
                for param in self.teacher.parameters():
                    param.requires_grad = True
            case "student":
                for param in self.static_model.parameters():
                    param.requires_grad = True
            case _:
                raise ValueError(f"Unknown model to unfreeze: {target_model}")
        return

    def is_frozen(self, target_model: Literal["teacher", "student"]) -> bool:
        """Return True if all parameters of the specified model are frozen."""
        match target_model:
            case "teacher":
                return all(
                    not param.requires_grad for param in self.teacher.parameters()
                )
            case "student":
                return all(
                    not param.requires_grad for param in self.static_model.parameters()
                )
            case _:
                raise ValueError(
                    f"Unknown model to check freeze status: {target_model}"
                )

    @override
    def configure_optimizers(self) -> OptimizerLRScheduler:
        if dist.is_initialized():
            lrs = {
                "warmup_lr": self.train_cfg.warmup_lr,
                "base_lr": self.train_cfg.base_lr,
            }
            if self.train_cfg.final_lr is not None:
                lrs["final_lr"] = self.train_cfg.final_lr

            scaled_lrs = scale_lrs_by_world_size(
                lrs=lrs, group=self.get_process_group(), verbose="only-zero-rank"
            )
            for key, value in scaled_lrs.items():
                setattr(self.train_cfg, key, value)

        total_steps = estimate_total_steps(
            trainer=self.trainer, process_group=self.get_process_group()
        )

        params = [
            {
                "params": list(self.static_model.parameters()),
                "lr": self.train_cfg.base_lr,
                "weight_decay": 0.0,
            }
        ]
        if self.dim_reducer is not None:
            params += create_params_groups(
                model=self.dim_reducer,
                lr=self.train_cfg.base_lr,
                weight_decay=self.train_cfg.weight_decay,
            )
        if self.student_to_teacher_proj is not None:
            params += create_params_groups(
                model=self.student_to_teacher_proj,
                lr=self.train_cfg.base_lr,
                weight_decay=self.train_cfg.weight_decay,
            )
        if self.student_dino_head is not None:
            params += create_params_groups(
                model=self.student_dino_head,
                lr=self.train_cfg.base_lr,
                weight_decay=self.train_cfg.weight_decay,
            )

        optim = AdamW(params)

        if self.train_cfg.final_lr is None:
            raise ValueError("final_lr must be set for the scheduler configuration")

        scheduler = CosineScheduler(
            optimizer=optim,
            num_iters=total_steps,
            param_group_field="lr",
            warmup_value=self.train_cfg.warmup_lr,
            warmup_ratio=self.train_cfg.warmup_iters_ratio,
            base_value=self.train_cfg.base_lr,
            final_value=self.train_cfg.final_lr,
        )

        if isinstance(self.distillation_cfg, DinoDistillationConfig):
            self.teacher_temp_scheduler = LinearParamScheduler(
                param_name="teacher_temp",
                start_value=self.distillation_cfg.start_teacher_temp,
                final_value=self.distillation_cfg.peak_teacher_temp,
                num_iters=int(
                    self.distillation_cfg.warmup_teacher_temp_steps_ratio * total_steps
                ),
            )
            self.teacher_momentum_scheduler = CosineParamScheduler(
                param_name="teacher_momentum",
                base_value=self.distillation_cfg.start_teacher_momentum,
                final_value=self.distillation_cfg.final_teacher_momentum,
                num_iters=total_steps,
            )

        if (
            isinstance(self.distillation_cfg, DirectDistillationConfig)
            and self.distillation_cfg.student_freeze_iters > 0
        ):
            self.freeze_model("student")

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }  # pyright: ignore[reportReturnType]

    @override
    def lr_scheduler_step(
        self,
        scheduler: CosineScheduler | LinearParamScheduler,
        metric: Any | None,
    ) -> None:
        scheduler.step(self.global_step)
        return

    @override
    def optimizer_zero_grad(
        self, epoch: int, batch_idx: int, optimizer: Optimizer
    ) -> None:
        super().optimizer_zero_grad(epoch, batch_idx, optimizer)
        if isinstance(self.distillation_cfg, DinoDistillationConfig):
            self._update_teacher_head()
        return

    @torch.no_grad()
    def _update_teacher_head(self) -> None:
        if self.teacher_momentum_scheduler is None:
            raise ValueError("Teacher momentum scheduler is not initialized")
        if self.student_dino_head is None or self.teacher_dino_head is None:
            raise ValueError(
                "DINO update_teacher_head called but DINO heads are not initialized"
            )
        mom = self.teacher_momentum_scheduler.step(self.global_step)
        if (
            len(self.teacher_head_params_list) == 0
            or len(self.student_head_params_list) == 0
        ):
            for teacher_param, student_param in zip(
                self.teacher_dino_head.parameters(),
                self.student_dino_head.parameters(),
                strict=True,
            ):
                self.teacher_head_params_list.append(teacher_param)
                self.student_head_params_list.append(student_param)
        torch._foreach_mul_(self.teacher_head_params_list, mom)  # pyright: ignore[reportArgumentType, reportCallIssue]
        torch._foreach_add_(
            self.teacher_head_params_list,  # pyright: ignore[reportArgumentType]
            self.student_head_params_list,  # pyright: ignore[reportArgumentType]
            alpha=1 - mom,
        )  # type: ignore
        return

    @torch.inference_mode()
    def _get_teacher_outputs(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        teacher_outputs: BaseModelOutput = self.teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        teacher_embeddings: torch.Tensor = teacher_outputs[0]
        teacher_sentence_embeddings = get_sentence_embedding(
            teacher_embeddings,
            attention_mask,
            pooling_method=self.train_cfg.teacher_pooling_method,
        )
        return {
            "teacher_embeddings": teacher_embeddings,
            "teacher_sentence_embeddings": teacher_sentence_embeddings,
        }

    def _get_student_outputs(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        student_output: StaticEmbeddingsOutput = self.static_model(
            input_ids,
            attention_mask,
            apply_token_weights=False,
        )
        student_embeddings = student_output.embeddings
        student_sentence_embeddings = student_output.sentence_embeddings
        return {
            "student_embeddings": student_embeddings,
            "student_sentence_embeddings": student_sentence_embeddings,
        }

    def _direct_distillation_step(
        self,
        flattened_student_embeddings: torch.Tensor,
        flattened_teacher_embeddings: torch.Tensor,
    ) -> _DirectDistillationOutput:
        if not isinstance(self.distillation_cfg, DirectDistillationConfig):
            raise ValueError(
                "Direct distillation step called but distillation config is not DirectDistillationConfig"
            )

        if self.dim_reducer is not None:
            dim_reduction_output = self.dim_reducer.transform(
                flattened_teacher_embeddings
            )
            flattened_teacher_embeddings = dim_reduction_output.reduced_data
        else:
            dim_reduction_output = None

        flattened_teacher_embeddings = flattened_teacher_embeddings.contiguous()
        flattened_student_embeddings = flattened_student_embeddings.contiguous()
        targets = flattened_teacher_embeddings.new_ones(
            flattened_teacher_embeddings.size(0)
        )
        semantic_loss = F.cosine_embedding_loss(
            flattened_student_embeddings, flattened_teacher_embeddings, targets
        )
        loss = semantic_loss * self.distillation_cfg.semantic_loss_weight
        reconstruction_loss = None
        if dim_reduction_output is not None:
            if dim_reduction_output.reconstruction_loss is not None:
                reconstruction_loss = dim_reduction_output.reconstruction_loss
                if self.is_frozen("student"):
                    if (
                        self.distillation_cfg.reconstruction_loss_boost_while_frozen
                        is not None
                    ):
                        weight = (
                            self.distillation_cfg.reconstruction_loss_boost_while_frozen
                        )
                    else:
                        weight = self.distillation_cfg.reconstruction_loss_weight
                        weight = cast(float, weight)
                    loss = reconstruction_loss * weight
                else:
                    weight = self.distillation_cfg.reconstruction_loss_weight
                    weight = cast(float, weight)
                    loss = loss + reconstruction_loss * weight
        return _DirectDistillationOutput(
            loss=loss,
            semantic_loss=semantic_loss,
            reconstruction_loss=reconstruction_loss,
        )

    def _dino_distillation_step(
        self,
        flattened_student_embeddings: torch.Tensor,
        flattened_teacher_embeddings: torch.Tensor,
    ) -> _DinoDistillationOutput:
        if not isinstance(self.distillation_cfg, DinoDistillationConfig):
            raise ValueError(
                "DINO distillation step called but distillation config is not DinoDistillationConfig"
            )
        if self.student_dino_head is None or self.teacher_dino_head is None:
            raise ValueError(
                "DINO distillation step called but DINO heads are not initialized"
            )
        if self.student_to_teacher_proj is None:
            raise ValueError(
                "DINO distillation step called but student to teacher projection is not initialized"
            )
        if self.teacher_temp_scheduler is None:
            raise ValueError(
                "DINO distillation step called but teacher temperature scheduler is not initialized"
            )
        student_embeddings = self.student_to_teacher_proj(flattened_student_embeddings)
        with torch.no_grad():
            teacher_temp = self.teacher_temp_scheduler.step(self.global_step)
            teacher_logits = self.teacher_dino_head(flattened_teacher_embeddings)
            teacher_probs = sinkhorn_knopp_teacher(
                teacher_logits,
                teacher_temp,
                n_iterations=self.distillation_cfg.sinkhorn_knopp_n_iters,
                process_group=self.get_process_group(),
            )
        student_logits = self.student_dino_head(student_embeddings)
        loss = dino_ce_loss(
            teacher_probs=teacher_probs,
            student_logits=student_logits,
            student_temp=self.distillation_cfg.student_temp,
        )
        return _DinoDistillationOutput(loss=loss, semantic_loss=loss)

    def _base_step(self, batch: dict[str, torch.Tensor]) -> _BaseStepOutput:
        """Performs a single training step for knowledge distillation."""
        student_outputs = self._get_student_outputs(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        teacher_outputs = self._get_teacher_outputs(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        student_embeddings = student_outputs["student_embeddings"]
        teacher_embeddings = teacher_outputs["teacher_embeddings"]

        flattened_input_ids = batch["input_ids"].flatten()

        special_tokens_mask_list = self.tokenizer.get_special_tokens_mask(
            flattened_input_ids.tolist(),
            already_has_special_tokens=True,
        )
        special_tokens_mask = torch.tensor(special_tokens_mask_list, dtype=torch.bool)

        B, T, _ = student_embeddings.size()
        student_embeddings = student_embeddings.view(B * T, -1)[~special_tokens_mask]
        teacher_embeddings = teacher_embeddings.view(B * T, -1)[~special_tokens_mask]

        if isinstance(self.distillation_cfg, DirectDistillationConfig):
            direct_distillation_output = self._direct_distillation_step(
                flattened_student_embeddings=student_embeddings,
                flattened_teacher_embeddings=teacher_embeddings,
            )
            loss = direct_distillation_output.loss
            semantic_loss = direct_distillation_output.semantic_loss
            reconstruction_loss = direct_distillation_output.reconstruction_loss
        elif isinstance(self.distillation_cfg, DinoDistillationConfig):
            dino_distillation_output = self._dino_distillation_step(
                flattened_student_embeddings=student_embeddings,
                flattened_teacher_embeddings=teacher_embeddings,
            )
            loss = dino_distillation_output.loss
            semantic_loss = dino_distillation_output.semantic_loss
            reconstruction_loss = None
        else:
            raise ValueError(
                f"Unsupported distillation method config type: {type(self.distillation_cfg)}"
            )

        output = _BaseStepOutput(
            loss=loss,
            semantic_loss=semantic_loss,
            reconstruction_loss=reconstruction_loss,
            teacher_embeddings=teacher_embeddings,
            student_embeddings=student_embeddings,
            student_sentence_embeddings=student_outputs["student_sentence_embeddings"],
            teacher_sentence_embeddings=teacher_outputs["teacher_sentence_embeddings"],
        )
        return output

    @override
    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        if isinstance(self.distillation_cfg, DirectDistillationConfig):
            if self.distillation_cfg.student_freeze_iters > 0:
                student_frozen = self.is_frozen("student")
                self.log(
                    name="student_frozen",
                    value=int(student_frozen),
                    prog_bar=True,
                    on_step=True,
                    logger=True,
                    on_epoch=True,
                )
                if (self.global_step >= self.distillation_cfg.student_freeze_iters) and student_frozen:  # fmt: off
                    self.unfreeze_model("student")
                    logger.info(
                        f"Unfreezing student model at global step {self.global_step}"
                    )
        output = self._base_step(batch)
        metrics = self.train_metrics(
            output.student_embeddings,
            output.teacher_embeddings,
        )
        metrics["loss"] = output.loss.detach()
        if output.reconstruction_loss is not None:
            metrics["reconstruction_loss"] = output.reconstruction_loss.detach()
            metrics["semantic_loss"] = output.semantic_loss.detach()

        self.log_dict(
            metrics,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=False,
            stage="train",
        )
        self.log(
            "train_loss",
            metrics["loss"],
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=False,
            sync_dist=True,
        )

        if self.teacher_momentum_scheduler is not None:
            val = self.teacher_momentum_scheduler.current_value()
            self.log_dict(
                val,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                logger=True,
                sync_dist=False,
            )
        if self.teacher_temp_scheduler is not None:
            val = self.teacher_temp_scheduler.current_value()
            self.log_dict(
                val,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                logger=True,
                sync_dist=False,
            )
        return output.loss

    @override
    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        output = self._base_step(batch)
        metrics = self.val_metrics(
            output.student_embeddings,
            output.teacher_embeddings,
        )
        metrics["loss"] = output.loss.detach()
        if output.reconstruction_loss is not None:
            metrics["reconstruction_loss"] = output.reconstruction_loss.detach()
            metrics["semantic_loss"] = output.semantic_loss.detach()

        if self.trainer.is_global_zero:
            self.eval_outputs.append(
                _EvalResult(
                    teacher_embeddings=output.teacher_sentence_embeddings,
                    student_embeddings=output.student_sentence_embeddings,
                    labels=batch["labels"],
                )
            )

        self.log_dict(
            metrics,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=True,
            stage="val",
        )
        self.log(
            "val_loss",
            metrics["loss"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=False,
            sync_dist=True,
        )
        return output.loss

    @override
    @torch.inference_mode()
    def on_validation_epoch_end(self) -> None:
        if self.trainer.is_global_zero and self.task is not None:
            NUM_POINTS2LOG = 200
            logger = self.task.get_logger()

            teacher_embeddings_list = []
            student_embeddings_list = []
            labels_list = []
            for output in self.eval_outputs:
                teacher_embeddings_list.append(output.teacher_embeddings)
                student_embeddings_list.append(output.student_embeddings)
                labels_list.append(output.labels)

            teacher_embeddings = torch.cat(teacher_embeddings_list, dim=0)
            student_embeddings = torch.cat(student_embeddings_list, dim=0)
            labels = torch.cat(labels_list, dim=0)

            if self.use_knn_in_val and (self.num_labels is not None):
                metrics: dict[str, torch.Tensor] = {}
                teacher_knn_preds = knn_predict(
                    embeddings=teacher_embeddings,
                    labels=labels,
                    num_labels=self.num_labels,
                    k_neighbors=5,
                )

                student_knn_preds = knn_predict(
                    embeddings=student_embeddings,
                    labels=labels,
                    num_labels=self.num_labels,
                    k_neighbors=5,
                )

                teacher_metrics = self.teacher_knn_metrics(  # type: ignore
                    teacher_knn_preds,
                    labels,
                )
                metrics.update(teacher_metrics)
                student_metrics = self.student_knn_metrics(  # type: ignore
                    student_knn_preds,
                    labels,
                )
                metrics.update(student_metrics)
                for key, value in metrics.items():
                    logger.report_scalar(
                        title="KNN Evaluation Metrics",
                        series=key,
                        value=value.item(),
                        iteration=self.global_step,
                    )
            if self.use_logprob and (self.num_labels is not None):
                student_embeddings_np = student_embeddings.cpu().numpy()
                labels_np = labels.cpu().numpy()

                X_train, X_test, y_train, y_test = train_test_split(
                    student_embeddings_np,
                    labels_np,
                    test_size=0.2,
                    random_state=42,
                    stratify=labels_np,
                )
                logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
                logreg.fit(X_train, y_train)
                y_pred = logreg.predict(X_test)

                average = "binary" if self.num_labels == 2 else "macro"

                logreg_accuracy = logreg.score(X_test, y_test)
                logreg_f1 = f1_score(y_test, y_pred, average=average)
                logreg_precision = precision_score(y_test, y_pred, average=average)
                logreg_recall = recall_score(y_test, y_pred, average=average)

                for name, value in [
                    ("Accuracy", logreg_accuracy),
                    ("F1", logreg_f1),
                    ("Precision", logreg_precision),
                    ("Recall", logreg_recall),
                ]:
                    logger.report_scalar(
                        title="Logistic Regression Metrics",
                        series=name,
                        value=value,
                        iteration=self.global_step,
                    )
                del (
                    student_embeddings_np,
                    labels_np,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    logreg,
                    y_pred,
                )

            teacher_embeddings = teacher_embeddings[:NUM_POINTS2LOG, :]
            student_embeddings = student_embeddings[:NUM_POINTS2LOG, :]
            labels = labels[:NUM_POINTS2LOG]

            teacher_embeddings = teacher_embeddings.to(self.device)
            student_embeddings = student_embeddings.to(self.device)

            pca = PCA(n_components=2)

            teacher_embeddings_2d = pca.transform(teacher_embeddings).reduced_data.cpu()
            student_embeddings_2d = pca.transform(student_embeddings).reduced_data.cpu()

            labels_list = labels.tolist()
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Teacher embeddings", "Student embeddings"),
                horizontal_spacing=0.08,
            )
            fig.add_trace(
                go.Scatter(
                    x=teacher_embeddings_2d[:, 0].float().tolist(),
                    y=teacher_embeddings_2d[:, 1].float().tolist(),
                    mode="markers",
                    marker={
                        "color": labels_list,
                        "colorscale": "Plotly3",
                        "showscale": True,
                        "opacity": 0.7,
                    },
                    name="Teacher",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=student_embeddings_2d[:, 0].float().tolist(),
                    y=student_embeddings_2d[:, 1].float().tolist(),
                    mode="markers",
                    marker={
                        "color": labels_list,
                        "colorscale": "Plotly3",
                        "showscale": False,
                        "opacity": 0.7,
                    },
                    name="Student",
                ),
                row=1,
                col=2,
            )

            fig.update_xaxes(title_text="dim0", row=1, col=1)
            fig.update_yaxes(title_text="dim1", row=1, col=1)
            fig.update_xaxes(title_text="dim0", row=1, col=2)
            fig.update_yaxes(title_text="dim1", row=1, col=2)

            self.task.get_logger().report_plotly(
                title="Embeddings Plots",
                series="Embeddings Teacher vs Student Scatter",
                figure=fig,
                iteration=self.global_step,
            )
            self.eval_outputs = []

        if dist.is_initialized():
            dist.barrier()
        return
