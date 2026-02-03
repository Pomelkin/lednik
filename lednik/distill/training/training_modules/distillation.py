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
from clearml import Task
from kostyl.ml.dist_utils import scale_lrs_by_world_size
from kostyl.ml.integrations.lightning import KostylLightningModule
from kostyl.ml.integrations.lightning.utils import estimate_total_steps
from kostyl.ml.optim.factory import create_optimizer
from kostyl.ml.optim.factory import create_scheduler
from kostyl.ml.optim.schedulers import BaseScheduler
from kostyl.ml.optim.schedulers import CompositeScheduler
from kostyl.ml.optim.schedulers import CosineParamScheduler
from kostyl.ml.optim.schedulers import LinearParamScheduler
from kostyl.ml.params_groups import create_params_groups
from kostyl.utils.logging import setup_logger
from lightning.pytorch.strategies import ParallelStrategy
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torchmetrics.functional import accuracy as torchmetrics_accuracy
from torchmetrics.functional import cosine_similarity
from torchmetrics.functional import f1_score as torchmetrics_f1_score
from torchmetrics.functional import mean_squared_error
from transformers import PreTrainedConfig
from transformers import PreTrainedModel
from transformers import SentencePieceBackend
from transformers import TokenizersBackend
from transformers.modeling_outputs import BaseModelOutput

from lednik.distill.dim_reduction import PCA
from lednik.distill.emb_utils import get_sentence_embedding
from lednik.distill.training.configs import DinoDistillationConfig
from lednik.distill.training.configs import DirectDistillationConfig
from lednik.distill.training.configs import DistillationConfig
from lednik.distill.training.knn import knn_predict
from lednik.distill.training.layers import DINOHead
from lednik.distill.training.losses import dino_ce_loss
from lednik.distill.training.losses import sinkhorn_knopp_teacher
from lednik.models import LednikModel
from lednik.models import StaticEmbeddingsModel
from lednik.models.outputs import StaticEmbeddingsOutput


logger = setup_logger()


@dataclass
class _BaseStepOutput:
    loss: torch.Tensor
    teacher_embeddings: torch.Tensor
    student_embeddings: torch.Tensor
    student_sentence_embeddings: torch.Tensor
    teacher_sentence_embeddings: torch.Tensor


@dataclass
class _DirectDistillationOutput:
    loss: torch.Tensor
    teacher_embeddings: torch.Tensor
    student_embeddings: torch.Tensor


@dataclass
class _DinoDistillationOutput:
    loss: torch.Tensor
    teacher_embeddings: torch.Tensor
    student_embeddings: torch.Tensor


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


def _build_dim_proj(
    input_dim: int,
    output_dim: int,
    dropout: float,
    intermediate_dim: int | None = None,
) -> nn.Sequential:
    intermediate_dim = output_dim if intermediate_dim is None else intermediate_dim
    layers: list[nn.Module] = [
        nn.Linear(input_dim, intermediate_dim),
        nn.GELU(approximate="tanh"),
        nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        nn.Linear(intermediate_dim, output_dim),
    ]
    for layer in layers:
        if isinstance(layer, nn.Linear):
            nn.init.trunc_normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)
    return nn.Sequential(*layers)


class DistillationModule(KostylLightningModule):
    """A PyTorch Lightning module for fine-tuning a static embeddings model via knowledge distillation."""

    def __init__(
        self,
        teacher: PreTrainedModel,
        student: StaticEmbeddingsModel | LednikModel,
        train_cfg: DistillationConfig,
        tokenizer: SentencePieceBackend | TokenizersBackend,
        task: Task | None = None,
        num_labels: int | None = None,
    ) -> None:
        """
        Initialize Fine-Tuning Lightning Module.

        Args:
            teacher : The pre-trained teacher hf model.
            student : The static embeddings model to be trained.
            train_cfg : Training configuration.
            tokenizer : The tokenizer corresponding to the teacher model.
            task : ClearML Task for logging (optional).
            num_labels : Number of classification labels for KNN evaluation metrics.
                If provided, enables KNN-based evaluation during validation.
                If None, KNN metrics are disabled.

        """
        super().__init__()
        self.teacher = teacher.eval()
        self.student = student.train()

        self.train_cfg = train_cfg
        self.distill_method_cfg = train_cfg.distillation_method

        self.tokenizer = tokenizer
        self.num_labels = num_labels if num_labels is not None else 0
        self.task = task
        self.spec_tok_to_ids = {
            tok: tokenizer.convert_tokens_to_ids(v)
            for tok, v in tokenizer.special_tokens_map.items()
        }
        self.register_buffer(
            "spec_tok_buff",
            torch.tensor(list(self.spec_tok_to_ids.values())),
            persistent=False,
        )

        ### DINO specific attributes ###
        self.student_dino_head: DINOHead | None = None
        self.teacher_dino_head: DINOHead | None = None
        self.teacher_temp_scheduler: LinearParamScheduler | None = None
        self.teacher_momentum_scheduler: CosineParamScheduler | None = None
        self.teacher_head_params_list: list[nn.Parameter] = []
        self.student_head_params_list: list[nn.Parameter] = []

        ### Direct distillation specific attributes ###
        self.student_to_teacher_proj: nn.Sequential | nn.Identity | None = None
        self.direct_loss: _Loss | None = None

        if isinstance(self.distill_method_cfg, DinoDistillationConfig):
            (
                self.student_to_teacher_proj,
                self.student_dino_head,
                self.teacher_dino_head,
            ) = self._init_dino(  # type: ignore
                self.distill_method_cfg
            )
            self.pipeline_type = "dino"
        elif isinstance(self.distill_method_cfg, DirectDistillationConfig):
            (
                self.student_to_teacher_proj,
                self.direct_loss,
            ) = self._init_direct(  # type: ignore
                self.distill_method_cfg
            )
            self.pipeline_type = "direct"
        else:
            raise ValueError(
                f"Unsupported distillation method config type: {self.distill_method_cfg.__class__.__name__}"
                f"Supported types: DinoDistillationConfig, DirectDistillationConfig"
            )

        self._set_model_freeze_state("teacher", freeze=True)
        self.eval_outputs: list[_EvalResult] = []
        return

    def _init_dino(
        self,
        config: DinoDistillationConfig,
        teacher_hidden_size: int,
        student_hidden_size: int,
    ) -> tuple[nn.Sequential | nn.Identity, DINOHead, DINOHead]:
        if teacher_hidden_size != student_hidden_size:
            student_to_teacher_proj = _build_dim_proj(
                input_dim=student_hidden_size,
                intermediate_dim=config.student_to_teacher_intermediate_dim,
                output_dim=teacher_hidden_size,
                dropout=config.proj_dropout,
            )
        else:
            student_to_teacher_proj = nn.Identity()

        head = partial(
            DINOHead,
            in_dim=teacher_hidden_size,
            out_dim=config.head_n_prototypes,
            nlayers=config.head_nlayers,
            bottleneck_dim=config.head_bottleneck_dim,
            hidden_dim=config.head_hidden_dim,
        )
        student_dino_head = head()
        teacher_dino_head = head().eval()
        for param in teacher_dino_head.parameters():
            param.requires_grad = False

        output = (
            student_to_teacher_proj,
            student_dino_head,
            teacher_dino_head,
        )
        return output

    def _init_direct(
        self,
        config: DirectDistillationConfig,
        teacher_hidden_size: int,
        student_hidden_size: int,
    ) -> tuple[nn.Sequential | nn.Identity, _Loss]:
        if student_hidden_size != teacher_hidden_size:
            student_to_teacher_proj = _build_dim_proj(
                input_dim=student_hidden_size,
                intermediate_dim=config.student_to_teacher_intermediate_dim,
                output_dim=teacher_hidden_size,
                dropout=config.proj_dropout,
            )
        else:
            student_to_teacher_proj = nn.Identity()
        match config.loss_type:
            case "cosine":
                loss = nn.CosineEmbeddingLoss()
            case "mse":
                loss = nn.MSELoss()
            case _:
                raise ValueError(
                    f"Unsupported loss type: {config.loss_type} "
                    f"Supported types: 'cosine', 'mse'"
                )
        return student_to_teacher_proj, loss

    @property
    @override
    def model_config(self) -> PreTrainedConfig | None:
        return self.student.config

    @property
    @override
    def grad_clip_val(self) -> float | None:
        return self.train_cfg.grad_clip_val

    @property
    @override
    def model_instance(self) -> PreTrainedModel | nn.Module:
        """Returns the underlying model."""
        return self.student

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
    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        cfg = self.model_config
        if cfg is not None:
            checkpoint["config"] = cfg.to_dict()
        state_dict = checkpoint["state_dict"]
        for key in list(state_dict.keys()):
            if key.startswith("teacher."):
                state_dict.pop(key)
        return

    def _set_model_freeze_state(
        self, target_model: Literal["teacher", "student"], freeze: bool
    ) -> None:
        match target_model:
            case "teacher":
                for param in self.teacher.parameters():
                    param.requires_grad = not freeze
            case "student":
                for param in self.student.parameters():
                    param.requires_grad = not freeze
            case _:
                raise ValueError(
                    f"Unknown model to set freeze state: {target_model}. "
                    "Supported models: 'teacher', 'student'"
                )
        return

    def is_frozen(self, target_model: Literal["teacher", "student"]) -> bool:
        """Return True if all parameters of the specified model are frozen."""
        match target_model:
            case "teacher":
                frozen_flag = all(
                    not param.requires_grad for param in self.teacher.parameters()
                )
            case "student":
                frozen_flag = all(
                    not param.requires_grad for param in self.student.parameters()
                )
            case _:
                raise ValueError(
                    f"Unknown model to check freeze status: {target_model}. "
                    "Supported models: 'teacher', 'student'"
                )
        return frozen_flag

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

        params = create_params_groups(
            model=self,
            lr=self.train_cfg.lr.base_value,
            weight_decay=self.train_cfg.weight_decay.base_value,
            no_decay_keywords={"emb"},
        )

        optim = create_optimizer(
            parameters_groups=params,
            optimizer_config=self.train_cfg.optimizer,
            lr=self.train_cfg.lr.base_value,
            weight_decay=self.train_cfg.weight_decay.base_value,
        )
        schedulers: dict[str, BaseScheduler] = {}
        if self.train_cfg.lr.scheduler_type is not None:
            scheduler = create_scheduler(
                config=self.train_cfg.lr,
                optim=optim,
                num_iters=total_steps,
                param_group_field="lr",
            )
            schedulers[scheduler.param_name] = scheduler
        if self.train_cfg.weight_decay.scheduler_type is not None:
            scheduler = create_scheduler(
                config=self.train_cfg.weight_decay,
                optim=optim,
                num_iters=total_steps,
                param_group_field="weight_decay",
            )
            schedulers[scheduler.param_name] = scheduler

        if isinstance(self.distill_method_cfg, DinoDistillationConfig):
            self.teacher_temp_scheduler = LinearParamScheduler(
                param_name="teacher_temp",
                initial_value=self.distill_method_cfg.start_teacher_temp,
                final_value=self.distill_method_cfg.peak_teacher_temp,
                num_iters=int(
                    self.distill_method_cfg.warmup_teacher_temp_steps_ratio
                    * total_steps
                ),
            )
            self.teacher_momentum_scheduler = CosineParamScheduler(
                param_name="teacher_momentum",
                base_value=self.distill_method_cfg.start_teacher_momentum,
                final_value=self.distill_method_cfg.final_teacher_momentum,
                num_iters=total_steps,
            )

        if len(schedulers) == 0:
            return optim
        elif len(schedulers) == 1:
            scheduler = next(iter(schedulers.values()))
        else:
            scheduler = CompositeScheduler(optimizer=optim, **schedulers)

        return {  # pyrefly: ignore[bad-return]
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    @override
    def lr_scheduler_step(  # pyrefly: ignore[bad-override]
        self,
        scheduler: BaseScheduler,
        metric: Any | None,
    ) -> None:
        scheduler.step(self.global_step)
        return

    @override
    def optimizer_zero_grad(
        self, epoch: int, batch_idx: int, optimizer: Optimizer
    ) -> None:
        super().optimizer_zero_grad(epoch, batch_idx, optimizer)
        if isinstance(self.distill_method_cfg, DinoDistillationConfig):
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
        torch._foreach_mul_(self.teacher_head_params_list, mom)  # type: ignore
        torch._foreach_add_(  # type: ignore
            self.teacher_head_params_list,
            self.student_head_params_list,
            alpha=1 - mom,
        )
        return

    @torch.no_grad()
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
        if isinstance(self.student, StaticEmbeddingsModel):
            student_output: StaticEmbeddingsOutput = self.student(
                input_ids,
                attention_mask,
                apply_token_weights=False,
            )
            student_embeddings = student_output.embeddings
            student_sentence_embeddings = student_output.sentence_embeddings
        else:
            outputs = self.student(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            student_embeddings = outputs[0]
            if self.train_cfg.student_pooling_method is not None:
                pooling_method = self.train_cfg.student_pooling_method
            else:
                pooling_method = self.train_cfg.teacher_pooling_method
            student_sentence_embeddings = get_sentence_embedding(
                student_embeddings,
                attention_mask,
                pooling_method=pooling_method,
            )
        return {
            "student_embeddings": student_embeddings,
            "student_sentence_embeddings": student_sentence_embeddings,
        }

    def _direct_distillation_step(
        self,
        flat_student_embeddings: torch.Tensor,  # [b*seq_len, dim]
        flat_teacher_embeddings: torch.Tensor,  # [b*seq_len, dim]
    ) -> dict[str, torch.Tensor]:
        if self.direct_loss is None or self.student_to_teacher_proj is None:
            raise ValueError(
                "Direct distillation step called but direct loss or "
                "student to teacher projection is not initialized"
            )

        flat_student_embeddings = self.student_to_teacher_proj(flat_student_embeddings)
        if isinstance(self.direct_loss, nn.CosineEmbeddingLoss):
            loss = self.direct_loss(
                flat_student_embeddings,
                flat_teacher_embeddings,
                torch.ones(flat_student_embeddings.size(0), device=self.device),
            )
        elif isinstance(self.direct_loss, nn.MSELoss):
            loss = self.direct_loss(
                flat_student_embeddings,
                flat_teacher_embeddings,
            )
        else:
            raise ValueError(
                f"Unsupported loss: {type(self.direct_loss.__class__.__name__)}"
            )
        return {
            "loss": loss,
            "teacher_embeddings": flat_teacher_embeddings,
            "student_embeddings": flat_student_embeddings,
        }

    def _dino_distillation_step(
        self,
        flattened_student_embeddings: torch.Tensor,  # [b*seq_len, dim]
        flattened_teacher_embeddings: torch.Tensor,  # [b*seq_len, dim]
    ) -> dict[str, torch.Tensor]:
        if not isinstance(self.distill_method_cfg, DinoDistillationConfig):
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
        flattened_student_embeddings = self.student_to_teacher_proj(
            flattened_student_embeddings
        )
        with torch.no_grad():
            teacher_temp = self.teacher_temp_scheduler.step(self.global_step)
            teacher_logits = self.teacher_dino_head(flattened_teacher_embeddings)
            teacher_probs = sinkhorn_knopp_teacher(
                teacher_logits,
                teacher_temp,
                n_iterations=self.distill_method_cfg.sinkhorn_knopp_n_iters,
                process_group=self._data_parallel_group,
            )
        student_logits = self.student_dino_head(flattened_student_embeddings)
        loss = dino_ce_loss(
            teacher_probs=teacher_probs,
            student_logits=student_logits,
            student_temp=self.distill_method_cfg.student_temp,
        )
        return {
            "loss": loss,
            "teacher_embeddings": flattened_teacher_embeddings,
            "student_embeddings": flattened_student_embeddings,
        }

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

        mask = batch["attention_mask"].flatten() == 0

        if isinstance(self.student, StaticEmbeddingsModel):
            flattened_input_ids = batch["input_ids"].flatten()

            special_tokens_mask = torch.isin(
                flattened_input_ids, cast(torch.Tensor, self.spec_tok_buff)
            )
            mask = mask | special_tokens_mask

        bs, seqlen, *_ = student_embeddings.size()
        student_embeddings = student_embeddings.view(bs * seqlen, -1)[~mask]
        teacher_embeddings = teacher_embeddings.view(bs * seqlen, -1)[~mask]

        if self.pipeline_type == "direct":
            output = self._direct_distillation_step(
                flat_student_embeddings=student_embeddings,
                flat_teacher_embeddings=teacher_embeddings,
            )
        elif self.pipeline_type == "dino":
            output = self._dino_distillation_step(
                flattened_student_embeddings=student_embeddings,
                flattened_teacher_embeddings=teacher_embeddings,
            )
        else:
            raise ValueError(f"Unsupported distillation method: {self.pipeline_type}")

        output = _BaseStepOutput(
            loss=output["loss"],
            teacher_embeddings=output["teacher_embeddings"],
            student_embeddings=output["student_embeddings"],
            student_sentence_embeddings=student_outputs["student_sentence_embeddings"],
            teacher_sentence_embeddings=teacher_outputs["teacher_sentence_embeddings"],
        )
        return output

    @override
    def training_step(  # type: ignore
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        output = self._base_step(batch)

        cosine_similarity_value = cosine_similarity(
            output.student_embeddings.detach(),
            output.teacher_embeddings.detach(),
            reduction="mean",
        )
        rmse_value = mean_squared_error(
            output.student_embeddings.detach(),
            output.teacher_embeddings.detach(),
            squared=False,
        )
        metrics = {
            "CosineSimilarity": cosine_similarity_value,
            "RMSE": rmse_value,
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
            enable_graph=False,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=False,
            sync_dist=True,
            sync_dist_group=self._data_parallel_group,
        )

        if self.teacher_momentum_scheduler is not None:
            val = self.teacher_momentum_scheduler.current_value()
            self.log_dict(
                val,
                enable_graph=False,
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
                enable_graph=False,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                logger=True,
                sync_dist=False,
            )
        return output.loss

    @override
    def validation_step(  # type: ignore
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        output = self._base_step(batch)

        cosine_similarity_value = cosine_similarity(
            output.student_embeddings.detach(),
            output.teacher_embeddings.detach(),
            reduction="mean",
        )
        rmse_value = mean_squared_error(
            output.student_embeddings.detach(),
            output.teacher_embeddings.detach(),
            squared=False,
        )
        metrics = {
            "CosineSimilarity": cosine_similarity_value,
            "RMSE": rmse_value,
            "loss": output.loss.detach(),
        }
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
            enable_graph=False,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False,
            stage="val",
        )
        self.log(
            "val_loss",
            metrics["loss"],
            enable_graph=False,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=False,
            sync_dist=True,
            sync_dist_group=self._data_parallel_group,
        )
        return output.loss

    @override
    @torch.inference_mode()
    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return

        if self.trainer.is_global_zero and self.task is not None:
            NUM_POINTS2LOG = 200
            clearml_logger = self.task.get_logger()

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

            if self.num_labels > 0:
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

                teacher_f1 = torchmetrics_f1_score(
                    teacher_knn_preds,
                    labels,
                    task="multiclass" if self.num_labels > 2 else "binary",
                    num_classes=self.num_labels,
                    average="macro",
                )
                teacher_accuracy = torchmetrics_accuracy(
                    teacher_knn_preds,
                    labels,
                    task="multiclass" if self.num_labels > 2 else "binary",
                    num_classes=self.num_labels,
                )
                student_f1 = torchmetrics_f1_score(
                    student_knn_preds,
                    labels,
                    task="multiclass" if self.num_labels > 2 else "binary",
                    num_classes=self.num_labels,
                    average="macro",
                )
                student_accuracy = torchmetrics_accuracy(
                    student_knn_preds,
                    labels,
                    task="multiclass" if self.num_labels > 2 else "binary",
                    num_classes=self.num_labels,
                )
                metrics = {
                    "Teacher_F1": teacher_f1,
                    "Teacher_Accuracy": teacher_accuracy,
                    "Student_F1": student_f1,
                    "Student_Accuracy": student_accuracy,
                }
                for key, value in metrics.items():
                    clearml_logger.report_scalar(
                        title="KNN Evaluation Metrics",
                        series=key,
                        value=value.item(),
                        iteration=self.global_step,
                    )
            if self.num_labels > 0:
                try:
                    student_embeddings_np = student_embeddings.float().cpu().numpy()
                    labels_np = labels.float().cpu().numpy()

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
                    logreg_f1 = f1_score(
                        y_test,
                        y_pred,
                        average=average,
                    )
                    logreg_precision = precision_score(y_test, y_pred, average=average)
                    logreg_recall = recall_score(y_test, y_pred, average=average)

                    for name, value in [
                        ("Accuracy", logreg_accuracy),
                        ("F1", logreg_f1),
                        ("Precision", logreg_precision),
                        ("Recall", logreg_recall),
                    ]:
                        clearml_logger.report_scalar(
                            title="Logistic Regression Metrics",
                            series=name,
                            value=value,
                            iteration=self.global_step,
                        )
                except Exception as e:
                    logger.warning(
                        f"Logistic regression evaluation failed at step {self.global_step} with error: {e}"
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

            clearml_logger.report_plotly(
                title="Embeddings Plots",
                series="Embeddings Teacher vs Student Scatter",
                figure=fig,
                iteration=self.global_step,
            )
            self.eval_outputs = []

        if dist.is_initialized():
            dist.barrier()
        return
