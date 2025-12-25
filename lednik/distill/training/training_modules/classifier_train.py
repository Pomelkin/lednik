from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from typing import cast
from typing import override

import polars as pl
import torch
import torch.distributed as dist
from clearml import Task
from kostyl.ml.dist_utils import scale_lrs_by_world_size
from kostyl.ml.lightning import KostylLightningModule
from kostyl.ml.lightning.training_utils import estimate_total_steps
from kostyl.ml.params_groups import create_params_groups
from kostyl.ml.schedulers import CompositeScheduler
from kostyl.ml.schedulers import CosineScheduler
from kostyl.utils.logging import setup_logger
from torch import nn
from torchmetrics import Accuracy
from torchmetrics import F1Score
from torchmetrics import MetricCollection
from torchmetrics import Precision
from torchmetrics import Recall
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerBase

from lednik.distill.training.configs import ClassifierTrainConfig
from lednik.static_embeddings import StaticEmbeddingsForSequenceClassification
from lednik.static_embeddings import (
    StaticEmbeddingsSequenceClassifierOutput as SEOutput,
)


logger = setup_logger()


def _metric_factory(
    prefix: str, num_classes: int, process_group: dist.ProcessGroup | None = None
) -> MetricCollection:
    if prefix[-1] != "/":
        prefix += "/"
    if num_classes < 2:
        collection = MetricCollection(
            [
                Accuracy(
                    task="binary",
                    process_group=process_group,
                ),
                Precision(
                    task="binary",
                    process_group=process_group,
                ),
                Recall(
                    task="binary",
                    process_group=process_group,
                ),
                F1Score(
                    task="binary",
                    process_group=process_group,
                ),
            ],
            prefix=prefix,
        )
    else:
        collection = MetricCollection(
            [
                Accuracy(
                    task="multiclass",
                    num_classes=num_classes,
                    process_group=process_group,
                ),
                Precision(
                    task="multiclass",
                    num_classes=num_classes,
                    process_group=process_group,
                ),
                Recall(
                    task="multiclass",
                    num_classes=num_classes,
                    process_group=process_group,
                ),
                F1Score(
                    task="multiclass",
                    num_classes=num_classes,
                    process_group=process_group,
                ),
            ],
            prefix=prefix,
        )
    return collection


@dataclass(slots=True)
class _BaseStepOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor


@dataclass(slots=True)
class _ValStepOutput:
    gt_label: int
    pred_label: int
    confidence: float
    input_ids: torch.Tensor

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
        super().__setattr__(name, value)
        return


class ClassifierTrainingModule(KostylLightningModule):
    """Lightning module that trains and validates a static-embedding classifier."""

    def __init__(
        self,
        config: ClassifierTrainConfig,
        model: StaticEmbeddingsForSequenceClassification,
        task: Task | None = None,
    ) -> None:
        """
        Initialize the classifier training module with metrics, loss, configuration, and model.

        Args:
            config (ClassifierTrainConfig): Training configuration containing loss parameters,
                optional class weights, and other settings.
            model (StaticEmbeddingsForSequenceClassification): Sequence classification model whose
                label configuration drives metric initialization and loss choice.
            task (Task | None, optional): Optional task metadata used for logging or organization.

        """
        super().__init__()
        if config.class_weights is not None:
            if len(config.class_weights) != model.config.num_labels:
                raise ValueError("Length of class_weights must match number of labels.")
            class_weights_tensor = torch.tensor(config.class_weights)
        else:
            class_weights_tensor = None

        match model.config.num_labels:
            case 1:
                self.loss = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor)
            case _:
                self.loss = nn.CrossEntropyLoss(weight=class_weights_tensor)

        self.train_cfg = config
        self.model = model
        self.task = task

        self.val_outputs: list[_ValStepOutput] = []
        self.train_metrics: MetricCollection | None = None
        self.val_metrics: MetricCollection | None = None
        return None

    @override
    def on_train_start(self) -> None:
        self.train_metrics = _metric_factory(
            prefix="train/",
            num_classes=self.model.config.num_labels,
            process_group=self.get_process_group(),
        ).to(self.device)
        self.val_metrics = _metric_factory(
            prefix="val/",
            num_classes=self.model.config.num_labels,
            process_group=self.get_process_group(),
        ).to(self.device)
        return None

    @override
    def configure_optimizers(self) -> dict[str, Any]:
        if dist.is_initialized():
            lrs = {
                "warmup_lr": self.train_cfg.lr.warmup_value,
                "base_lr": self.train_cfg.lr.base_value,
                "final_lr": self.train_cfg.lr.final_value,
            }
            scaled_lrs = scale_lrs_by_world_size(
                lrs, verbose="world", group=self.get_process_group()
            )
            for key, value in scaled_lrs.items():
                attr_name = key.replace("_lr", "_value")
                setattr(self.train_cfg.lr, attr_name, value)

        total_steps = estimate_total_steps(
            trainer=self.trainer, process_group=self.get_process_group()
        )

        pgs = create_params_groups(
            model=self.model,
            weight_decay=self.train_cfg.weight_decay.base_value,
            lr=self.train_cfg.lr.base_value,
        )

        betas = (
            self.train_cfg.optimizer.adamw_beta1,
            self.train_cfg.optimizer.adamw_beta2,
        )
        optimizer = torch.optim.AdamW(
            pgs,
            betas=betas,
        )

        if not self.train_cfg.lr.use_scheduler:
            raise NotImplementedError(
                "Training without LR scheduler is not implemented yet."
            )

        schedulers: dict[str, CosineScheduler] = {}

        lr_scheduler = CosineScheduler(
            optimizer=optimizer,
            param_group_field="lr",
            num_iters=total_steps,
            base_value=self.train_cfg.lr.base_value,
            final_value=self.train_cfg.lr.final_value,  # type: ignore
            warmup_ratio=self.train_cfg.lr.warmup_iters_ratio,
            warmup_value=self.train_cfg.lr.warmup_value,
        )
        schedulers["lr_scheduler"] = lr_scheduler

        if self.train_cfg.weight_decay.use_scheduler:
            weight_decay_scheduler = CosineScheduler(
                optimizer=optimizer,
                param_group_field="weight_decay",
                num_iters=total_steps,
                base_value=self.train_cfg.weight_decay.base_value,
                final_value=self.train_cfg.weight_decay.final_value,  # type: ignore
            )
            schedulers["wd_scheduler"] = weight_decay_scheduler

        if len(schedulers) > 1:
            scheduler = CompositeScheduler(optimizer=optimizer, **schedulers)
        else:
            scheduler = lr_scheduler

        optimization_config = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
        return optimization_config

    @override
    def lr_scheduler_step(
        self, scheduler: CompositeScheduler | CosineScheduler, metric: Any | None
    ) -> None:
        scheduler.step(self.global_step)
        return

    @property
    @override
    def grad_clip_val(self) -> float | None:
        return self.train_cfg.grad_clip_val

    @property
    @override
    def model_instance(self) -> PreTrainedModel | nn.Module:
        """Returns the underlying model."""
        return self.model

    def _base_step(self, batch: dict[str, torch.Tensor]) -> _BaseStepOutput:
        labels = batch["labels"]
        output: SEOutput = self.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )

        loss = self.loss(output.logits, labels)

        return _BaseStepOutput(loss=loss, logits=output.logits, labels=labels)

    @override
    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        output = self._base_step(batch)
        if self.train_metrics is not None:
            self.train_metrics.update(output.logits, output.labels)
            self.log_dict(
                self.train_metrics,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                logger=True,
                sync_dist=False,
            )
        detached_loss = output.loss.detach()
        self.log(
            "train/loss",
            detached_loss,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=False,
        )
        self.log(
            "train_loss",
            detached_loss,
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
        if self.val_metrics is not None:
            metrics = self.val_metrics(output.logits, output.labels)
            self.log_dict(
                metrics,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=False,
            )
        detached_loss = output.loss.detach()
        self.log(
            "val/loss",
            detached_loss,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_loss",
            detached_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=False,
            sync_dist=True,
        )
        if self.trainer.is_global_zero:
            pred_distr = output.logits.softmax(dim=-1)
            pred_label = pred_distr.argmax(dim=-1).tolist()
            confidence = pred_distr.amax(dim=-1).tolist()
            labels = output.labels.tolist()
            input_ids_b = batch["input_ids"]

            for label, plabel, conf, input_ids in zip(
                labels, pred_label, confidence, input_ids_b, strict=True
            ):
                self.val_outputs.append(
                    _ValStepOutput(
                        gt_label=label,
                        pred_label=plabel,
                        confidence=conf,
                        input_ids=input_ids,
                    )
                )
        return output.loss

    @override
    def on_validation_epoch_end(self) -> None:
        if self.trainer.is_global_zero and self.task is not None:
            NUM_ROWS2LOG = 20

            if not hasattr(self.trainer, "datamodule"):
                logger.warning_once(
                    "Trainer has no datamodule; cannot log validation predictions."
                )
                if dist.is_initialized():
                    dist.barrier()
                return
            if not hasattr(self.trainer.datamodule, "tokenizer"):  # type: ignore
                logger.warning_once(
                    "Datamodule has no tokenizer; cannot log validation predictions."
                )
                if dist.is_initialized():
                    dist.barrier()
                return
            tokenizer = self.trainer.datamodule.tokenizer  # type: ignore
            tokenizer = cast(PreTrainedTokenizerBase, tokenizer)

            df_data = defaultdict(list)
            id2label = self.model.config.id2label
            for output in self.val_outputs[:NUM_ROWS2LOG]:
                if id2label is not None:
                    gt_label = id2label[output.gt_label]
                    pred_label = id2label[output.pred_label]
                else:
                    gt_label = str(output.gt_label)
                    pred_label = str(output.pred_label)
                text = tokenizer.decode(
                    output.input_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                confidence = output.confidence
                df_data["text"].append(text)
                df_data["ground_truth_label"].append(gt_label)
                df_data["predicted_label"].append(pred_label)
                df_data["confidence"].append(confidence)
            df = pl.DataFrame(df_data)
            self.task.get_logger().report_table(
                title="Validation Predictions",
                series="Dataframe",
                table_plot=df,
                iteration=self.global_step,
            )
            self.val_outputs = []

        if dist.is_initialized():
            dist.barrier()
        return None
