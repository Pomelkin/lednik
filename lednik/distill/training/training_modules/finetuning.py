from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Literal
from typing import cast
from typing import override

import plotly.graph_objects as go
import torch
import torch.distributed as dist
from clearml import Task
from kostyl.ml.dist_utils import scale_lrs_by_world_size
from kostyl.ml.lightning.extenstions import KostylLightningModule
from kostyl.ml.params_groups import create_params_groups
from kostyl.utils.logging import setup_logger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from plotly.subplots import make_subplots
from torch import nn
from torch.distributed import Work
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
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
from lednik.distill.training.configs import FinetuningConfig
from lednik.distill.training.knn import knn_predict
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


def _knn_metric_factory(
    prefix_str: Literal["teacher", "student"], num_labels: int
) -> MetricCollection:
    if num_labels == 1:
        metric_collection = MetricCollection(
            {
                f"{prefix_str}_F1": F1Score("binary"),
                f"{prefix_str}_Accuracy": Accuracy("binary"),
            }
        )
    else:
        metric_collection = MetricCollection(
            {
                f"{prefix_str}_F1-Macro": F1Score(
                    task="multiclass", num_classes=num_labels, average="macro"
                ),
                f"{prefix_str}_Accuracy": Accuracy(
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
        self.static_model = static_model

        match train_cfg.teacher_dim_reduction_type:
            case "pca":
                self.dim_reduction = PCA(n_components=train_cfg.student_dim)
            case "autoencoder":
                self.dim_reduction = Autoencoder(
                    input_dim=train_cfg.teacher_dim,
                    latent_dim=train_cfg.student_dim,
                    dropout=train_cfg.reduction_dropout,
                )
            case None:
                self.dim_reduction = None
            case _:
                raise ValueError(
                    f"Unsupported dimension reduction type: {train_cfg.teacher_dim_reduction_type}"
                )

        self.loss = torch.nn.CosineEmbeddingLoss()
        self.train_cfg = train_cfg

        self.tokenizer = tokenizer
        self.train_metrics = metric_factory()
        self.val_metrics = metric_factory()
        if num_labels is not None:
            self.use_knn_in_val = True
            self.teacher_knn_metrics = _knn_metric_factory(
                prefix_str="teacher", num_labels=num_labels
            )
            self.student_knn_metrics = _knn_metric_factory(
                prefix_str="student", num_labels=num_labels
            )
            self.num_labels = num_labels
        else:
            self.use_knn_in_val = False
            self.teacher_knn_metrics = None
            self.student_knn_metrics = None
            self.num_labels = None
        self.task = task

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
        self.freeze_model("teacher")
        if dist.is_initialized():
            lrs = {
                "warmup_lr": self.train_cfg.warmup_lr,
                "base_lr": self.train_cfg.base_lr,
            }
            scaled_lrs = scale_lrs_by_world_size(
                lrs=lrs, group=self.get_process_group(), verbose="only-zero-rank"
            )
            for key, value in scaled_lrs.items():
                setattr(self.train_cfg, key, value)

        params = [
            {
                "params": list(self.static_model.parameters()),
                "lr": self.train_cfg.base_lr,
                "weight_decay": 0.0,
            }
        ]
        if self.dim_reduction is not None:
            params += create_params_groups(
                model=self.dim_reduction,
                lr=self.train_cfg.base_lr,
                weight_decay=self.train_cfg.weight_decay,
            )
        optim = AdamW(params)

        if self.train_cfg.warmup_iters is None or self.train_cfg.warmup_lr is None:
            return optim
        scheduler = LinearLR(
            optimizer=optim,
            start_factor=self.train_cfg.warmup_lr / self.train_cfg.base_lr,
            total_iters=self.train_cfg.warmup_iters,
        )

        if self.train_cfg.student_freeze_iters > 0:
            self.freeze_model("student")

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _base_step(self, batch: dict[str, torch.Tensor]) -> _BaseStepOutput:
        """Performs a single training step for knowledge distillation."""
        student_output: StaticEmbeddingsOutput = self.static_model(
            batch["input_ids"],
            batch["attention_mask"],
            apply_token_weights=False,
        )
        student_embeddings = student_output.embeddings
        student_sentence_embeddings = student_output.sentence_embeddings

        with torch.no_grad():
            teacher_outputs: BaseModelOutput = self.teacher(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            teacher_embeddings: torch.Tensor = teacher_outputs[0]
            teacher_sentence_embeddings = get_sentence_embedding(
                teacher_embeddings,
                batch["attention_mask"],
                pooling_method=self.train_cfg.teacher_pooling_method,
            )

        flattened_input_ids = batch["input_ids"].flatten()

        special_tokens_mask_list = self.tokenizer.get_special_tokens_mask(
            flattened_input_ids.tolist(),
            already_has_special_tokens=True,
        )
        special_tokens_mask = torch.tensor(special_tokens_mask_list, dtype=torch.bool)

        B, T, _ = student_embeddings.size()
        student_embeddings = student_embeddings.view(B * T, -1)[~special_tokens_mask]
        teacher_embeddings = teacher_embeddings.view(B * T, -1)[~special_tokens_mask]

        if self.dim_reduction is not None:
            dim_reduce_output = self.dim_reduction.transform(teacher_embeddings)
            teacher_embeddings = dim_reduce_output.reduced_data
        else:
            dim_reduce_output = None

        target = torch.ones(
            student_embeddings.size(0),
            device=student_embeddings.device,
        )
        student_embeddings = student_embeddings.contiguous()
        teacher_embeddings = teacher_embeddings.contiguous()
        semantic_loss = self.loss(
            student_embeddings,
            teacher_embeddings,
            target,
        )

        loss = semantic_loss * self.train_cfg.semantic_loss_weight
        reconstruction_loss = None
        if dim_reduce_output is not None:
            if dim_reduce_output.reconstruction_loss is not None:
                student_frozen = self.is_frozen("student")
                if (
                    self.train_cfg.reconstruction_loss_boost_while_frozen is not None
                ) and student_frozen:
                    weight = self.train_cfg.reconstruction_loss_boost_while_frozen
                else:
                    weight = self.train_cfg.reconstruction_loss_weight

                weight = cast(float, weight)
                reconstruction_loss = dim_reduce_output.reconstruction_loss

                if student_frozen:
                    loss = reconstruction_loss * weight
                else:
                    loss = loss + reconstruction_loss * weight

        output = _BaseStepOutput(
            loss=loss,
            semantic_loss=semantic_loss,
            reconstruction_loss=reconstruction_loss,
            teacher_embeddings=teacher_embeddings,
            student_embeddings=student_embeddings,
            student_sentence_embeddings=student_sentence_embeddings,
            teacher_sentence_embeddings=teacher_sentence_embeddings,
        )
        return output

    @override
    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        if self.train_cfg.student_freeze_iters > 0:
            student_frozen = self.is_frozen("student")
            self.log(
                name="student_frozen",
                value=int(student_frozen),
                prog_bar=True,
                on_step=True,
                logger=True,
                on_epoch=True,
            )
            if (self.global_step >= self.train_cfg.student_freeze_iters) and student_frozen:  # fmt: off
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

        if (self.num_labels is not None) and self.use_knn_in_val:
            if dist.is_initialized():
                teacher_sentence_embeddings_global = [
                    torch.zeros_like(output.teacher_sentence_embeddings)
                    for _ in range(dist.get_world_size(self.get_process_group()))
                ]
                teacher_emb_handler: Work = dist.all_gather(  # type: ignore
                    teacher_sentence_embeddings_global,
                    output.teacher_sentence_embeddings.contiguous(),
                    group=self.get_process_group(),
                    async_op=True,
                )
                student_sentence_embeddings_global = [
                    torch.zeros_like(output.student_sentence_embeddings)
                    for _ in range(dist.get_world_size(self.get_process_group()))
                ]
                student_emb_handler: Work = dist.all_gather(  # type: ignore
                    student_sentence_embeddings_global,
                    output.student_sentence_embeddings.contiguous(),
                    group=self.get_process_group(),
                    async_op=True,
                )
                labels_global = [
                    torch.zeros_like(batch["labels"])
                    for _ in range(dist.get_world_size(self.get_process_group()))
                ]
                labels_handler: Work = dist.all_gather(  # type: ignore
                    labels_global,
                    batch["labels"].contiguous(),
                    group=self.get_process_group(),
                    async_op=True,
                )

                teacher_emb_handler.wait()
                teacher_sentence_embeddings = torch.cat(
                    teacher_sentence_embeddings_global, dim=0
                )
                student_emb_handler.wait()
                student_sentence_embeddings = torch.cat(
                    student_sentence_embeddings_global, dim=0
                )
                labels_handler.wait()
                labels = torch.cat(labels_global, dim=0)
            else:
                teacher_sentence_embeddings = output.teacher_sentence_embeddings
                student_sentence_embeddings = output.student_sentence_embeddings
                labels = batch["labels"]

            teacher_knn_preds = knn_predict(
                embeddings=teacher_sentence_embeddings,
                labels=labels,
                num_labels=self.num_labels,
                k_neighbors=5,
            )

            student_knn_preds = knn_predict(
                embeddings=student_sentence_embeddings,
                labels=labels,
                num_labels=self.num_labels,
                k_neighbors=5,
            )

            teacher_metrics = self.teacher_knn_metrics(  # type: ignore
                teacher_knn_preds,
                labels=labels,
            )
            metrics.update(teacher_metrics)
            student_metrics = self.student_knn_metrics(  # type: ignore
                student_knn_preds,
                labels=labels,
            )
            metrics.update(student_metrics)

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
    def on_validation_epoch_end(self) -> None:
        if self.trainer.is_global_zero and self.task is not None:
            num_points2log = 200

            all_teacher_embeddings_list = []
            all_student_embeddings_list = []
            all_labels_list = []
            for output in self.eval_outputs[:num_points2log]:
                all_teacher_embeddings_list.append(output.teacher_embeddings)
                all_student_embeddings_list.append(output.student_embeddings)
                all_labels_list.append(output.labels)

            all_teacher_embeddings = torch.cat(all_teacher_embeddings_list, dim=0)
            all_student_embeddings = torch.cat(all_student_embeddings_list, dim=0)
            all_labels = torch.cat(all_labels_list, dim=0)

            pca = PCA(n_components=2)

            teacher_embeddings_2d = pca.transform(
                all_teacher_embeddings.to(self.device)
            ).reduced_data.cpu()
            student_embeddings_2d = pca.transform(
                all_student_embeddings.to(self.device)
            ).reduced_data.cpu()

            labels_list = all_labels.tolist()
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
