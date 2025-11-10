from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from typing import override

import lightning as L
import torch
import torch.distributed as dist
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.distributed import ProcessGroup
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torchmetrics import CosineSimilarity
from torchmetrics import MeanAbsolutePercentageError
from torchmetrics import MeanSquaredError
from torchmetrics import Metric
from torchmetrics import MetricCollection
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import BaseModelOutput

from lednik.distill.dim_reduction import Autoencoder
from lednik.distill.dim_reduction import PCA
from lednik.distill.train.configs import TrainConfig
from lednik.distill.train.dist_utils import scale_lrs_by_world_size
from lednik.distill.train.metrics_formatting import apply_suffix
from lednik.static_embeddings.modeling import StaticEmbeddingsModelForPostTraining
from lednik.static_embeddings.outputs import StaticEmbeddingsOutput
from lednik.utils.logging import setup_logger

logger = setup_logger(add_rank=True)


@dataclass
class BaseStepOutput:
    loss: torch.Tensor
    semantic_loss: torch.Tensor
    teacher_embeddings: torch.Tensor
    student_embeddings: torch.Tensor
    reconstruction_loss: torch.Tensor | None = None


def metric_factory() -> MetricCollection:
    """Create a collection of metrics for evaluation."""
    collection = MetricCollection(
        {
            "RMSE": MeanSquaredError(squared=False),
            "MAPE": MeanAbsolutePercentageError(),
            "CosineSimilarity": CosineSimilarity(),
        }
    )
    return collection


class PostTrainModule(L.LightningModule):
    def __init__(
        self,
        teacher: PreTrainedModel,
        static_model: StaticEmbeddingsModelForPostTraining,
        train_cfg: TrainConfig,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        """
        Initialize Post-Training Lightning Module.

        Args:
            teacher : The pre-trained teacher hf model.
            static_model : The static embeddings model to be trained.
            train_cfg : Training configuration.
            tokenizer : The tokenizer corresponding to the teacher model.

        """
        super().__init__()
        self.teacher = teacher
        self.static_model = static_model
        self.train_cfg = train_cfg

        match train_cfg.teacher_dim_reduction_type:
            case "pca":
                self.dim_reduction = PCA(n_components=train_cfg.student_dim)
            case "autoencoder":
                self.dim_reduction = Autoencoder(
                    input_dim=train_cfg.teacher_dim,
                    latent_dim=train_cfg.student_dim,
                )
            case None:
                self.dim_reduction = None
            case _:
                raise ValueError(
                    f"Unsupported dimension reduction type: {train_cfg.teacher_dim_reduction_type}"
                )

        self.loss = torch.nn.CosineEmbeddingLoss()

        self.tokenizer = tokenizer
        self.train_metrics = metric_factory()
        self.val_metrics = metric_factory()
        return

    def freeze_teacher(self) -> None:
        """Freeze the teacher model parameters."""
        for param in self.teacher.parameters():
            param.requires_grad = False
        return

    @property
    def process_group(self) -> ProcessGroup | None:
        """Get the process group in distributed training."""
        if not dist.is_initialized():
            return None

        if self.device_mesh is not None:
            dp_mesh = self.device_mesh["data_parallel"]
            if dp_mesh.size() == 1:
                logger.warning("Data parallel mesh size is 1, returning None")
                return None
            dp_pg = dp_mesh.get_group()
        else:
            dp_pg = dist.group.WORLD
        return dp_pg

    @override
    def log_dict(
        self,
        dictionary: Mapping[str, Metric | torch.Tensor | int | float]
        | MetricCollection,
        prog_bar: bool = False,
        logger: bool | None = None,
        on_step: bool | None = None,
        on_epoch: bool | None = None,
        reduce_fx: str | Callable[..., Any] = "mean",
        enable_graph: bool = False,
        sync_dist: bool = False,
        sync_dist_group: Any | None = None,
        add_dataloader_idx: bool = True,
        batch_size: int | None = None,
        rank_zero_only: bool = False,
        stage: str | None = None,
    ) -> None:
        if stage is not None:
            dictionary = apply_suffix(
                metrics=dictionary,
                suffix=stage,
                add_dist_rank=False,
            )

        super().log_dict(
            dictionary,
            prog_bar,
            logger,
            on_step,
            on_epoch,
            reduce_fx,
            enable_graph,
            sync_dist,
            sync_dist_group,
            add_dataloader_idx,
            batch_size,
            rank_zero_only,
        )
        return

    @override
    def configure_optimizers(self) -> OptimizerLRScheduler:
        self.freeze_teacher()
        if dist.is_initialized():
            scale_lrs_by_world_size(
                lr_config=self.train_cfg,
                group=self.process_group,
            )

        params = [
            {
                "params": list(self.static_model.parameters()),
                "lr": self.train_cfg.base_lr,
                "weight_decay": 0.0,
            }
        ]
        if self.dim_reduction is not None:
            for name, param in self.dim_reduction.named_parameters():
                if "bias" in name or "norm" in name:
                    params[0]["params"].append(param)
                else:
                    if len(params) == 1:
                        params.append(
                            {
                                "params": [param],
                                "lr": self.train_cfg.base_lr,
                                "weight_decay": self.train_cfg.weight_decay,
                            }
                        )
                    else:
                        params[1]["params"].append(param)
        optim = AdamW(params)

        if self.train_cfg.warmup_iters is None or self.train_cfg.warmup_lr is None:
            return optim
        scheduler = LinearLR(
            optimizer=optim,
            start_factor=self.train_cfg.warmup_lr / self.train_cfg.base_lr,
            total_iters=self.train_cfg.warmup_iters,
        )

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def base_step(self, batch: dict[str, torch.Tensor]) -> BaseStepOutput:
        """
        Performs a single training step for knowledge distillation.

        Computes embeddings from the student (static) model and teacher model using the input batch.
        Applies masking to exclude padding tokens, optionally reduces dimensions of teacher embeddings,
        and calculates the distillation loss. If dimension reduction includes reconstruction loss,
        it is added to the total loss.

        Args:
            batch (dict[str, torch.Tensor]): A dictionary containing 'input_ids' and 'attention_mask'
                tensors for the input sequence.

        Returns:
            BaseStepOutput: An object containing the computed loss, teacher embeddings, and student embeddings.

        """
        student_output: StaticEmbeddingsOutput = self.static_model(batch["input_ids"])
        student_embeddings = student_output.embeddings

        with torch.no_grad():
            teacher_outputs: BaseModelOutput = self.teacher(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            teacher_embeddings: torch.Tensor = teacher_outputs[0]

        flattened_input_ids = batch["input_ids"].flatten()
        special_tokens_mask_list = self.tokenizer.get_special_tokens_mask(
            flattened_input_ids.tolist(), already_has_special_tokens=True
        )
        special_tokens_mask = torch.tensor(special_tokens_mask_list, dtype=torch.bool)

        student_embeddings = student_embeddings.flatten(start_dim=0, end_dim=1)[
            ~special_tokens_mask
        ]
        teacher_embeddings = teacher_embeddings.flatten(start_dim=0, end_dim=1)[
            ~special_tokens_mask
        ]

        if self.dim_reduction is not None:
            dim_reduce_output = self.dim_reduction.transform(teacher_embeddings)
            teacher_embeddings = dim_reduce_output.reduced_data
        else:
            dim_reduce_output = None

        target = torch.ones(
            student_embeddings.size(0), device=student_embeddings.device
        )
        semantic_loss = self.loss(
            student_embeddings,
            teacher_embeddings,
            target,
        )

        loss = semantic_loss
        reconstruction_loss = None
        if dim_reduce_output is not None:
            if dim_reduce_output.reconstruction_loss is not None:
                reconstruction_loss = dim_reduce_output.reconstruction_loss * 0.3
                loss = (
                    loss + dim_reduce_output.reconstruction_loss * reconstruction_loss
                )

        output = BaseStepOutput(
            loss=loss,
            semantic_loss=semantic_loss,
            reconstruction_loss=reconstruction_loss,
            teacher_embeddings=teacher_embeddings,
            student_embeddings=student_embeddings,
        )
        return output

    @override
    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        output = self.base_step(batch)
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
        output = self.base_step(batch)
        metrics = self.val_metrics(
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
