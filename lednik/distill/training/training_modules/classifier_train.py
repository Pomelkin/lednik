from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from typing import Literal
from typing import cast
from typing import override

import polars as pl
import torch
import torch.distributed as dist
from clearml import Task
from kostyl.ml.dist_utils import scale_lrs_by_world_size
from kostyl.ml.integrations.lightning import KostylLightningModule
from kostyl.ml.integrations.lightning.utils import estimate_total_steps
from kostyl.ml.optim.factory import create_optimizer
from kostyl.ml.optim.factory import create_scheduler
from kostyl.ml.optim.schedulers import BaseScheduler
from kostyl.ml.optim.schedulers import CompositeScheduler
from kostyl.ml.params_groups import create_params_groups
from kostyl.utils.logging import setup_logger
from lightning.pytorch.strategies import ParallelStrategy
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torchmetrics import Accuracy
from torchmetrics import F1Score
from torchmetrics import MetricCollection
from torchmetrics import Precision
from torchmetrics import Recall
from torchmetrics.functional.classification import accuracy
from torchmetrics.functional.classification import f1_score
from torchmetrics.functional.classification import precision
from torchmetrics.functional.classification import recall
from transformers import PreTrainedConfig
from transformers import PreTrainedModel
from transformers import SentencePieceBackend
from transformers import TokenizersBackend

from lednik.distill.training.configs import ClassifierTrainConfig
from lednik.models import StaticEmbeddingsForSequenceClassification
from lednik.models import StaticEmbeddingsSequenceClassifierOutput as SEOutput


logger = setup_logger()


def _metric_factory(
    num_classes: int,
    process_group: dist.ProcessGroup | None = None,
    compute_groups: bool | list[list[str]] = True,
    prefix: str | None = None,
) -> MetricCollection:
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
            compute_groups=compute_groups,
        )
    else:
        collection = MetricCollection(
            [
                Accuracy(
                    task="multiclass",
                    num_classes=num_classes,
                    process_group=process_group,
                    average="macro",
                ),
                Precision(
                    task="multiclass",
                    num_classes=num_classes,
                    process_group=process_group,
                    average="macro",
                ),
                Recall(
                    task="multiclass",
                    num_classes=num_classes,
                    process_group=process_group,
                    average="macro",
                ),
                F1Score(
                    task="multiclass",
                    num_classes=num_classes,
                    process_group=process_group,
                    average="macro",
                ),
            ],
            prefix=prefix,
            compute_groups=compute_groups,
        )
    return collection


@dataclass
class _BaseStepOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor


@dataclass
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

        self.classification_type: Literal["binary", "multiclass"] = (
            "binary" if model.config.num_labels < 3 else "multiclass"
        )

        self.loss = (
            nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor)
            if self.classification_type == "binary"
            else nn.CrossEntropyLoss(weight=class_weights_tensor)
        )

        self.train_cfg = config
        self.model = model
        self.task = task
        self.val_outputs: list[_ValStepOutput] = []
        self.val_metrics: MetricCollection | None = None
        return None

    @property
    @override
    def model_config(self) -> PreTrainedConfig | None:
        return self.model.config

    @property
    @override
    def grad_clip_val(self) -> float | None:
        return self.train_cfg.grad_clip_val

    @property
    @override
    def model_instance(self) -> PreTrainedModel | nn.Module:
        """Returns the underlying model."""
        return self.model

    @property
    def _data_parallel_group(self) -> dist.ProcessGroup | None:
        if not dist.is_initialized():
            return None
        strategy = cast(ParallelStrategy, self.trainer.strategy)
        device_mesh: DeviceMesh | None = getattr(strategy, "device_mesh", None)
        if device_mesh is not None:
            return device_mesh.get_group("data")
        return dist.group.WORLD

    @override
    def on_train_start(self) -> None:
        self.val_metrics = _metric_factory(
            num_classes=self.model.config.num_labels,
            process_group=self._data_parallel_group,
            prefix="val/",
        ).to(self.device)
        return None

    @override
    def configure_optimizers(self) -> OptimizerLRScheduler:
        if dist.is_initialized():
            lrs = {
                "base_value": self.train_cfg.lr.base_value,
            }
            if self.train_cfg.lr.final_value is not None:
                lrs["final_value"] = self.train_cfg.lr.final_value
            if self.train_cfg.lr.warmup_value is not None:
                lrs["warmup_value"] = self.train_cfg.lr.warmup_value
            scaled_lrs = scale_lrs_by_world_size(
                lrs=lrs,
                verbose_level="only-zero-rank",
                group=self._data_parallel_group,
            )
            for key, value in scaled_lrs.items():
                setattr(self.train_cfg.lr, key, value)

        total_steps = estimate_total_steps(
            trainer=self.trainer,
            dp_process_group=self._data_parallel_group,
        )

        pg = create_params_groups(
            model=self.model,
            weight_decay=self.train_cfg.weight_decay.base_value,
            lr=self.train_cfg.lr.base_value,
        )

        optimizer = create_optimizer(
            parameters_groups=pg,
            optimizer_config=self.train_cfg.optimizer,
            lr=self.train_cfg.lr.base_value,
            weight_decay=self.train_cfg.weight_decay.base_value,
        )

        schedulers: dict[str, BaseScheduler] = {}
        if self.train_cfg.lr.scheduler_type is not None:
            scheduler = create_scheduler(
                config=self.train_cfg.lr,
                optim=optimizer,
                num_iters=total_steps,
                param_group_field="lr",
            )
            schedulers[scheduler.param_name] = scheduler
        if self.train_cfg.weight_decay.scheduler_type is not None:
            scheduler = create_scheduler(
                config=self.train_cfg.weight_decay,
                optim=optimizer,
                num_iters=total_steps,
                param_group_field="weight_decay",
            )
            schedulers[scheduler.param_name] = scheduler

        if len(schedulers) == 0:
            return optimizer
        elif len(schedulers) == 1:
            scheduler = next(iter(schedulers.values()))
        else:
            scheduler = CompositeScheduler(optimizer=optimizer, **schedulers)

        return {  # pyrefly: ignore[bad-return]
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    @override
    def lr_scheduler_step(  # type: ignore
        self, scheduler: BaseScheduler
    ) -> None:
        scheduler.step(self.global_step)
        return

    def _base_step(self, batch: dict[str, torch.Tensor]) -> _BaseStepOutput:
        labels = batch["labels"]
        output: SEOutput = self.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        loss = self.loss(output.logits, labels)
        return _BaseStepOutput(loss=loss, logits=output.logits, labels=labels)

    @override
    def training_step(  # type: ignore
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        output = self._base_step(batch)
        num_classes = (
            None
            if self.classification_type == "binary"
            else self.model.config.num_labels
        )
        metrics = {
            "Accuracy": accuracy(
                output.logits,
                output.labels,
                task=self.classification_type,
                num_classes=num_classes,
                average="macro",
            ),
            "F1": f1_score(
                output.logits,
                output.labels,
                task=self.classification_type,
                num_classes=num_classes,
                average="macro",
            ),
            "Precision": precision(
                output.logits,
                output.labels,
                task=self.classification_type,
                num_classes=num_classes,
                average="macro",
            ),
            "Recall": recall(
                output.logits,
                output.labels,
                task=self.classification_type,
                num_classes=num_classes,
                average="macro",
            ),
            "loss": output.loss.detach(),
        }
        self.log_dict(
            metrics,
            enable_graph=False,
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
            sync_dist_group=self._data_parallel_group,
        )
        return output.loss

    @override
    def validation_step(  # type: ignore
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        output = self._base_step(batch)
        if self.val_metrics is not None:
            self.val_metrics.update(output.logits, output.labels)
            self.log_dict(
                self.val_metrics,
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
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            sync_dist_group=self._data_parallel_group,
        )
        self.log(
            "val_loss",
            detached_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=False,
            sync_dist=True,
            sync_dist_group=self._data_parallel_group,
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
            NUM_ROWS2LOG = 40

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
            tokenizer = getattr(self.trainer.datamodule, "tokenizer", None)
            if tokenizer is not None:
                tokenizer = cast(SentencePieceBackend | TokenizersBackend, tokenizer)

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
                    table_plot=df.to_pandas(use_pyarrow_extension_array=True),
                    iteration=self.global_step,
                )
                self.val_outputs = []
        if dist.is_initialized():
            dist.barrier()
        return None
