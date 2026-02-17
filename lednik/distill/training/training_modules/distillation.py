from __future__ import annotations

import multiprocessing as mp
import threading
from dataclasses import dataclass
from typing import Any
from typing import Literal
from typing import cast
from typing import override

import numpy as np
import plotly.graph_objects as go
import torch
import torch.distributed as dist
import torch.nn.functional as F
from clearml import Task
from kostyl.ml.configs.training_settings import SUPPORTED_STRATEGIES
from kostyl.ml.configs.training_settings import FSDP1StrategyConfig
from kostyl.ml.configs.training_settings import FSDP2StrategyConfig
from kostyl.ml.dist_utils import scale_lrs_by_world_size
from kostyl.ml.integrations.lightning import KostylLightningModule
from kostyl.ml.integrations.lightning.utils import estimate_total_steps
from kostyl.ml.optim.factory import create_optimizer
from kostyl.ml.optim.factory import create_scheduler
from kostyl.ml.optim.schedulers import BaseScheduler
from kostyl.ml.optim.schedulers import CompositeScheduler
from kostyl.ml.params_groups import create_params_groups
from kostyl.utils.logging import setup_logger
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.strategies import ModelParallelStrategy
from lightning.pytorch.strategies import ParallelStrategy
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from plotly.subplots import make_subplots
from torch import nn
from torch.distributed._composable.replicate_with_fsdp import replicate
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import fully_shard
from torch.nn.modules.loss import _Loss
from torchmetrics.functional import accuracy as torchmetrics_accuracy
from torchmetrics.functional import cosine_similarity
from torchmetrics.functional import f1_score as torchmetrics_f1_score
from torchmetrics.functional import mean_squared_error
from transformers import PreTrainedConfig
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerBase
from transformers import SentencePieceBackend
from transformers import TokenizersBackend
from transformers.modeling_outputs import BaseModelOutput

from lednik.distill.dim_reduction import PCA
from lednik.distill.emb_utils import get_sentence_embedding
from lednik.distill.training.configs import DirectDistillationConfig
from lednik.distill.training.configs import DistillationConfig
from lednik.distill.training.dist_utils import DistributedMMFunction
from lednik.distill.training.dist_utils import get_fsdp1_policies
from lednik.distill.training.dist_utils import get_fsdp2_policies
from lednik.distill.training.dist_utils import get_transformer_wrap_classes
from lednik.distill.training.dist_utils import select_wrap_policy
from lednik.distill.training.knn import knn_predict
from lednik.models import LednikModel
from lednik.models import StaticEmbeddingsModel
from lednik.models.outputs import StaticEmbeddingsOutput


logger = setup_logger()


@dataclass
class _BaseStepOutput:
    loss: torch.Tensor
    contrastive_loss: torch.Tensor
    per_token_loss: torch.Tensor
    student_sentence_embeddings: torch.Tensor
    teacher_sentence_embeddings: torch.Tensor
    teacher_embeddings: torch.Tensor | None = None
    student_embeddings: torch.Tensor | None = None


@dataclass
class _EvalResult:
    teacher_sentence_embeddings: torch.Tensor
    student_sentence_embeddings: torch.Tensor
    labels: torch.Tensor

    def __setattr__(self, name: str, value: Any) -> None:
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Attribute {name} must be a torch.Tensor")
        value = value.detach().cpu()
        super().__setattr__(name, value)
        return


class DistillationModule(KostylLightningModule):
    """A PyTorch Lightning module for fine-tuning a static embeddings model via knowledge distillation."""

    def __init__(
        self,
        teacher: PreTrainedModel,
        student: StaticEmbeddingsModel | LednikModel,
        train_cfg: DistillationConfig,
        strategy_config: SUPPORTED_STRATEGIES,
        tokenizer: SentencePieceBackend | TokenizersBackend | PreTrainedTokenizerBase,
        task: Task | None = None,
        num_labels: int | None = None,
    ) -> None:
        """
        Initialize Fine-Tuning Lightning Module.

        Args:
            teacher : The pre-trained teacher hf model.
            student : The static embeddings model to be trained.
            train_cfg : Training configuration.
            strategy_config : Configuration for the training strategy.
            tokenizer : The tokenizer corresponding to the teacher model.
            task : ClearML Task for logging (optional).
            num_labels : Number of classification labels for KNN evaluation metrics.
                If provided, enables KNN-based evaluation during validation.
                If None, KNN metrics are disabled.

        """
        super().__init__()
        self.teacher = teacher
        self.student = student

        self.train_cfg = train_cfg
        self.strategy_config = strategy_config
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

        (
            self.student_to_teacher_proj,
            self.direct_loss,
        ) = self._init_direct(
            self.distill_method_cfg,
            teacher_hidden_size=teacher.config.hidden_size,
            student_hidden_size=self._get_student_hidden_size(),
            device=student.device,
            dtype=student.dtype,
        )

        self.eval_outputs: list[_EvalResult] = []
        self.logprobing_thread: threading.Thread | None = None
        self.configured_flag = False
        return

    def _get_student_hidden_size(self) -> int:
        output_hidden_size = getattr(self.student.config, "output_hidden_size", None)
        if output_hidden_size is not None:
            return output_hidden_size
        return self.student.config.hidden_size

    @override
    def configure_model(self) -> None:  # noqa: C901
        if self.configured_flag:
            return

        modules_shard = {"teacher": self.teacher, "student": self.student}
        module_no_shard = {
            "student_to_teacher_proj": self.student_to_teacher_proj,
        }
        strategy = self.trainer.strategy
        strategy_config = self.strategy_config
        if isinstance(strategy, FSDPStrategy):
            if not isinstance(strategy_config, FSDP1StrategyConfig):
                raise ValueError(
                    f"Expected FSDPStrategy but got {strategy_config.__class__.__name__} "
                    f"for {strategy.__class__.__name__} strategy"
                )
            policies = get_fsdp1_policies(strategy_config)
            for name, module in modules_shard.items():
                wrap_policy = select_wrap_policy(module)
                fsdp_model = FSDP(
                    module=module,
                    use_orig_params=True,
                    device_id=strategy.root_device.index,
                    sharding_strategy=strategy.sharding_strategy,
                    auto_wrap_policy=wrap_policy,
                    **policies,
                )
                setattr(self, name, fsdp_model)

            for name, module in module_no_shard.items():
                if (module is None) or isinstance(module, nn.Identity):
                    continue
                fsdp_module = FSDP(
                    module=module,
                    use_orig_params=True,
                    device_id=strategy.root_device.index,
                    sharding_strategy=ShardingStrategy.NO_SHARD,
                    **policies,
                )
                setattr(self, name, fsdp_module)

            self_fsdp_wrapped = FSDP(
                module=self,
                use_orig_params=True,
                device_id=strategy.root_device.index,
                sharding_strategy=strategy.sharding_strategy,
                **policies,
            )
            strategy.model = self_fsdp_wrapped
        elif isinstance(strategy, ModelParallelStrategy):
            if not isinstance(strategy_config, FSDP2StrategyConfig):
                raise ValueError(
                    f"Expected ModelParallelStrategy but got {strategy_config.__class__.__name__} "
                    f"for {strategy.__class__.__name__} strategy"
                )

            mesh = self.device_mesh
            if mesh is None:
                raise ValueError(
                    "Device mesh is not initialized for ModelParallelStrategy"
                )
            dp_mesh = mesh["data_parallel"]
            tp_mesh = mesh["tensor_parallel"]
            if tp_mesh.size() > 1:
                raise ValueError(
                    "Tensor parallelism is not supported in this distillation module"
                )

            policies = get_fsdp2_policies(strategy_config)

            for name, module in modules_shard.items():
                modules_to_shard = get_transformer_wrap_classes(model=module)
                if modules_to_shard is None:  # TODO: add size-based wrap fallback
                    logger.warning(
                        f"Failed to get modules to shard for {name} module in ModelParallelStrategy. "
                        "Module will be wrapped into one FSDP instance."
                    )
                else:
                    modules_to_shard = tuple(modules_to_shard)
                    for child_module in module.modules():
                        if isinstance(child_module, modules_to_shard):
                            fully_shard(child_module, mesh=dp_mesh, **policies)
                fully_shard(module=module, mesh=dp_mesh, **policies)

            for _, module in module_no_shard.items():
                if (module is None) or isinstance(module, nn.Identity):
                    continue
                replicate(
                    module,
                    mesh=dp_mesh,
                    mp_policy=policies["mp_policy"],
                )
        self.configured_flag = True
        return

    def _init_direct(
        self,
        config: DirectDistillationConfig,
        teacher_hidden_size: int,
        student_hidden_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[nn.Linear | nn.Identity, _Loss]:
        if student_hidden_size != teacher_hidden_size:
            student_to_teacher_proj = nn.Linear(
                student_hidden_size, teacher_hidden_size, device=device, dtype=dtype
            )
            if device != torch.device("meta"):
                nn.init.trunc_normal_(student_to_teacher_proj.weight, std=0.02)
                nn.init.zeros_(student_to_teacher_proj.bias)
                student_to_teacher_proj.weight._is_hf_initialized = True  # type: ignore
                student_to_teacher_proj.bias._is_hf_initialized = True  # type: ignore
        else:
            student_to_teacher_proj = nn.Identity()
        match config.per_token_loss_type:
            case "cosine":
                loss = nn.CosineEmbeddingLoss()
            case "mse":
                loss = nn.MSELoss()
            case _:
                raise ValueError(
                    f"Unsupported loss type: {config.per_token_loss_type} "
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
            return device_mesh.get_group("data_parallel")
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
                    param.requires_grad_(not freeze)
            case "student":
                for param in self.student.parameters():
                    param.requires_grad_(not freeze)
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
        self._set_model_freeze_state("teacher", freeze=True)
        self.teacher.eval()  # set teacher to eval mode to disable dropout if any, just in case
        self.student.train()  # ensure student is in train mode for optimization

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

    def _calculate_per_token_similarity_loss(
        self,
        flat_student_embeddings: torch.Tensor,  # [b*seq_len, dim]
        flat_teacher_embeddings: torch.Tensor,  # [b*seq_len, dim]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return loss, flat_teacher_embeddings, flat_student_embeddings

    def _calculate_contrastive_loss(
        self,
        temp: float,
        student_sentence_embeddings: torch.Tensor,
        teacher_sentence_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        student_sentence_embeddings = self.student_to_teacher_proj(
            student_sentence_embeddings
        )
        student_sentence_embeddings = (
            student_sentence_embeddings
            / student_sentence_embeddings.norm(p=2, dim=-1, keepdim=True)
            / temp
        )
        teacher_sentence_embeddings = (
            teacher_sentence_embeddings
            / teacher_sentence_embeddings.norm(p=2, dim=-1, keepdim=True)
            / temp
        )

        if dist.is_initialized():
            sim_matrix = DistributedMMFunction.apply(
                student_sentence_embeddings,
                teacher_sentence_embeddings,
                self._data_parallel_group,
                multi_by_world_size=True,
            )
        else:
            sim_matrix = student_sentence_embeddings @ teacher_sentence_embeddings.T

        targets = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

        teacher2student_loss = F.cross_entropy(sim_matrix, targets, reduction="mean")
        student2teacher_loss = F.cross_entropy(sim_matrix.T, targets, reduction="mean")
        loss = (teacher2student_loss + student2teacher_loss) / 2
        return loss

    def _base_step(self, batch: dict[str, torch.Tensor]) -> _BaseStepOutput:
        """Performs a single training step for knowledge distillation."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        if isinstance(self.student, StaticEmbeddingsModel):
            special_tokens_mask = torch.isin(  # Exclude spec tokens (POS, BOS, EOS, CLS and etc) from processing because they are static (cannot be sentence invariant)
                input_ids, cast(torch.Tensor, self.spec_tok_buff), invert=True
            ).to(device=attention_mask.device, dtype=attention_mask.dtype)
            teacher_attention_mask = attention_mask
            student_attention_mask = attention_mask * special_tokens_mask
        else:
            teacher_attention_mask = attention_mask
            student_attention_mask = attention_mask

        student_outputs = self._get_student_outputs(
            input_ids=input_ids,
            attention_mask=student_attention_mask,
        )
        teacher_outputs = self._get_teacher_outputs(
            input_ids=input_ids,
            attention_mask=teacher_attention_mask,
        )
        student_embeddings = student_outputs["student_embeddings"]
        teacher_embeddings = teacher_outputs["teacher_embeddings"]

        mask = student_attention_mask.flatten() != 0

        bs, seqlen, *_ = student_embeddings.size()
        student_embeddings = student_embeddings.view(bs * seqlen, -1)[mask]
        teacher_embeddings = teacher_embeddings.view(bs * seqlen, -1)[mask]

        contrastive_weight = self.distill_method_cfg.contrastive_loss_weight
        if contrastive_weight < 1.0:
            per_token_loss, flat_teacher_embeddings, flat_student_embeddings = (
                self._calculate_per_token_similarity_loss(
                    flat_student_embeddings=student_embeddings,
                    flat_teacher_embeddings=teacher_embeddings,
                )
            )
        else:
            flat_teacher_embeddings = None
            flat_student_embeddings = None
            per_token_loss = student_embeddings.new_tensor(0.0)

        if contrastive_weight > 0.0:
            temp = self.distill_method_cfg.temperature
            if temp is None:
                raise ValueError("Temperature must be specified for contrastive loss.")

            contrastive_loss = self._calculate_contrastive_loss(
                student_sentence_embeddings=student_outputs[
                    "student_sentence_embeddings"
                ],
                teacher_sentence_embeddings=teacher_outputs[
                    "teacher_sentence_embeddings"
                ],
                temp=temp,
            )
        else:
            contrastive_loss = student_embeddings.new_tensor(0.0)

        loss = (
            contrastive_weight * contrastive_loss
            + (1 - contrastive_weight) * per_token_loss
        )

        output = _BaseStepOutput(
            loss=loss,
            contrastive_loss=contrastive_loss,
            per_token_loss=per_token_loss,
            teacher_embeddings=flat_teacher_embeddings,
            student_embeddings=flat_student_embeddings,
            student_sentence_embeddings=student_outputs["student_sentence_embeddings"],
            teacher_sentence_embeddings=teacher_outputs["teacher_sentence_embeddings"],
        )
        return output

    @override
    def training_step(  # type: ignore
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        output = self._base_step(batch)
        metrics = {
            "loss": output.loss.detach(),
        }
        if (
            output.student_embeddings is not None
            and output.teacher_embeddings is not None
        ):
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
            metrics["CosineSimilarity"] = cosine_similarity_value
            metrics["RMSE"] = rmse_value
        if output.contrastive_loss.item() > 0.0:
            metrics["ContrastiveLoss"] = output.contrastive_loss.detach()
        if output.per_token_loss.item() > 0.0:
            metrics["PerTokenLoss"] = output.per_token_loss.detach()

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
        return output.loss

    @override
    def validation_step(  # type: ignore
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        output = self._base_step(batch)
        metrics = {
            "loss": output.loss.detach(),
        }
        if (
            output.student_embeddings is not None
            and output.teacher_embeddings is not None
        ):
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
            metrics["CosineSimilarity"] = cosine_similarity_value
            metrics["RMSE"] = rmse_value
        if output.contrastive_loss.item() > 0.0:
            metrics["ContrastiveLoss"] = output.contrastive_loss.detach()
        if output.per_token_loss.item() > 0.0:
            metrics["PerTokenLoss"] = output.per_token_loss.detach()

        if self.trainer.is_global_zero:
            self.eval_outputs.append(
                _EvalResult(
                    teacher_sentence_embeddings=output.teacher_sentence_embeddings,
                    student_sentence_embeddings=output.student_sentence_embeddings,
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

            teacher_embeddings_list = []
            student_embeddings_list = []
            labels_list = []
            for output in self.eval_outputs:
                teacher_embeddings_list.append(output.teacher_sentence_embeddings)
                student_embeddings_list.append(output.student_sentence_embeddings)
                labels_list.append(output.labels)

            teacher_embeddings = torch.cat(teacher_embeddings_list, dim=0).to(
                self.device
            )
            student_embeddings = torch.cat(student_embeddings_list, dim=0).to(
                self.device
            )
            labels = torch.cat(labels_list, dim=0).to(self.device)

            if self.num_labels > 0:
                self._log_knn_metrics(
                    teacher_embeddings=teacher_embeddings,
                    student_embeddings=student_embeddings,
                    labels=labels,
                    k=5,
                )

                if self.logprobing_thread is not None:
                    self.logprobing_thread.join()

                student_embeddings_np = student_embeddings.float().cpu().numpy()
                labels_np = labels.float().cpu().numpy()
                self.logprobing_thread = self._launch_async_probing(
                    student_embeddings_np=student_embeddings_np,
                    labels_np=labels_np,
                )

            teacher_embeddings = teacher_embeddings[:NUM_POINTS2LOG, :]
            student_embeddings = student_embeddings[:NUM_POINTS2LOG, :]
            labels = labels[:NUM_POINTS2LOG]

            self._log_embeddings_scatter(
                teacher_embeddings=teacher_embeddings,
                student_embeddings=student_embeddings,
                labels=labels,
            )
            self.eval_outputs = []

        if dist.is_initialized():
            dist.barrier()
        return

    def _log_knn_metrics(
        self,
        teacher_embeddings: torch.Tensor,
        student_embeddings: torch.Tensor,
        labels: torch.Tensor,
        k: int = 5,
    ) -> None:
        task = cast(Task, self.task)

        metrics: dict[str, torch.Tensor] = {}
        teacher_knn_preds = knn_predict(
            embeddings=teacher_embeddings,
            labels=labels,
            num_labels=self.num_labels,
            k_neighbors=k,
        )

        student_knn_preds = knn_predict(
            embeddings=student_embeddings,
            labels=labels,
            num_labels=self.num_labels,
            k_neighbors=k,
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
            task.get_logger().report_scalar(
                title="KNN Evaluation Metrics",
                series=key,
                value=value.item(),
                iteration=self.global_step,
            )
        return

    def _launch_async_probing(
        self,
        student_embeddings_np: np.ndarray,
        labels_np: np.ndarray,
    ) -> threading.Thread:
        if self.num_labels == 0:
            raise ValueError(
                "Cannot launch async probing without labels. num_labels is set to 0."
            )
        task = cast(Task, self.task)
        iteration = self.global_step
        ctx = mp.get_context("spawn")
        out_q = ctx.Queue(maxsize=1)

        payload = {
            "student_embeddings": student_embeddings_np,
            "labels": labels_np,
            "num_labels": self.num_labels,
        }
        p = ctx.Process(target=_probing_worker, args=(payload, out_q))
        p.start()

        def _wait_and_log() -> None:
            try:
                metrics = out_q.get(timeout=600)
                for metric_name, metric_value in metrics.items():
                    task.get_logger().report_scalar(
                        title="Logistic Regression Metrics",
                        series=metric_name,
                        value=metric_value,
                        iteration=iteration,
                    )
            except Exception as e:
                logger.error(f"Probing worker failed with exception: {e}")
            finally:
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
            return

        t = threading.Thread(target=_wait_and_log, daemon=True)
        t.start()
        return t

    def _log_embeddings_scatter(
        self,
        teacher_embeddings: torch.Tensor,
        student_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        task = cast(Task, self.task)

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

        task.get_logger().report_plotly(
            title="Embeddings Plots",
            series="Embeddings Teacher vs Student Scatter",
            figure=fig,
            iteration=self.global_step,
        )
        return


def _probing_worker(payload: dict, out_q: mp.Queue) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.model_selection import train_test_split

    X = payload["student_embeddings"]
    y = payload["labels"]
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
    logreg = LogisticRegression(max_iter=10000, class_weight="balanced")
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)

    avg = "binary" if payload["num_labels"] <= 2 else "macro"
    res = {
        "logreg_accuracy": float(logreg.score(X_test, y_test)),
        "logreg_f1": float(f1_score(y_test, y_pred, average=avg)),
        "logreg_precision": float(precision_score(y_test, y_pred, average=avg)),
        "logreg_recall": float(recall_score(y_test, y_pred, average=avg)),
    }
    out_q.put(res)
    return
