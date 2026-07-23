from dataclasses import dataclass
from typing import Any
from typing import cast
from typing import override

import torch
import torch.distributed as dist
import torch.nn.functional as F
from clearml import Task
from kostyl.ml.configs.structs.training_settings import SUPPORTED_STRATEGIES
from kostyl.ml.configs.structs.training_settings import FSDP1StrategyConfig
from kostyl.ml.configs.structs.training_settings import FSDP2StrategyConfig
from kostyl.ml.dist_utils import scale_lrs_by_world_size
from kostyl.ml.integrations.lightning import KostylLightningModule
from kostyl.ml.integrations.lightning.utils import estimate_total_steps
from kostyl.ml.optim.factory import create_optimizer
from kostyl.ml.optim.factory import create_scheduler
from kostyl.ml.optim.schedulers import BaseScheduler
from kostyl.ml.optim.schedulers import CompositeScheduler
from kostyl.utils.logging import setup_logger
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.strategies import ModelParallelStrategy
from lightning.pytorch.strategies import ParallelStrategy
from torch import nn
from torch.distributed._composable.replicate_with_fsdp import replicate
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import fully_shard
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torchmetrics.functional import cosine_similarity
from torchmetrics.functional import mean_squared_error
from transformers import PreTrainedTokenizerBase
from transformers import SentencePieceBackend
from transformers import TokenizersBackend
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_utils import PreTrainedModel

from lednik.dist_utils import GatherSentenceEmbeddings
from lednik.dist_utils import get_fsdp1_policies
from lednik.dist_utils import get_fsdp2_policies
from lednik.dist_utils import get_fsdp_wrap_classes
from lednik.dist_utils import select_wrap_policy
from lednik.distill.configs import DistillationConfig
from lednik.distill.validation import EvaluationDispatcher
from lednik.distill.validation import EvaluationRunner
from lednik.distill.validation import EvaluationRunnerConfig
from lednik.distill.validation import RedisConfig
from lednik.distill.validation import ValidationContract
from lednik.models import LednikPreTrainedModel
from lednik.models import StaticEmbeddingsModel
from lednik.models.outputs import LednikModelOutput
from lednik.models.outputs import StaticEmbeddingsOutput

from .collator import CollatorOutput
from .param_groups import create_param_groups


logger = setup_logger()


@dataclass
class _BaseStepOutput:
    loss: torch.Tensor
    contrastive_loss: torch.Tensor
    distill_loss: torch.Tensor
    student_sentence_embeddings: torch.Tensor
    teacher_sentence_embeddings: torch.Tensor
    student_sentence_embeddings_proj: torch.Tensor | None = None


@dataclass
class _EvalResult:
    teacher_sentence_embeddings: torch.Tensor
    student_sentence_embeddings: torch.Tensor
    queries_mask: torch.Tensor
    positives_mask: torch.Tensor
    labels: torch.Tensor

    def __setattr__(self, name: str, value: Any) -> None:
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Attribute {name} must be a torch.Tensor")
        value = value.detach().cpu()
        super().__setattr__(name, value)
        return


def is_fp8_supported() -> bool:
    """Check if the current environment supports FP8 training."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) >= (8, 9)


def is_torchao_available() -> bool:
    """Check if torchao library is available for advanced optimizers and schedulers."""
    try:
        import torchao  # noqa: F401

        return True
    except ImportError:
        return False


def _get_special_tokens_ids(
    tokenizer: SentencePieceBackend | TokenizersBackend,
) -> list[int]:
    special_tokens = tokenizer.special_tokens_map.values()
    return [tokenizer.convert_tokens_to_ids(token) for token in special_tokens]  # ty:ignore[invalid-return-type]


def _unwrap_model(model: nn.Module) -> nn.Module:
    while True:
        if isinstance(model, (FSDP, DDP)):
            model = model.module
            continue

        # для некоторых wrappers / accelerate-like объектов
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod  # ty:ignore[invalid-assignment]
            continue

        return model


class DistillationModule(KostylLightningModule):
    """A PyTorch Lightning module for distillation Lednik models from teacher models."""

    def __init__(
        self,
        student: LednikPreTrainedModel,
        tokenizer: SentencePieceBackend | TokenizersBackend | PreTrainedTokenizerBase,
        teacher_hidden_size: int,
        train_cfg: DistillationConfig,
        strategy_config: SUPPORTED_STRATEGIES,
        redis_config: RedisConfig | None = None,
        runner_config: EvaluationRunnerConfig | None = None,
        task: Task | None = None,
        num_labels: int | None = None,
    ) -> None:
        """
        Initialize Fine-Tuning Lightning Module.

        Args:
            student : The static embeddings model to be trained.
            train_cfg : Training configuration.
            strategy_config : Configuration for the training strategy.
            teacher_hidden_size : The hidden size of the teacher model.
            tokenizer : The tokenizer corresponding to the teacher model.
            task : ClearML Task for logging (optional).
            num_labels : Number of classification labels for KNN evaluation metrics.
                If provided, enables KNN-based evaluation during validation.
                If None, KNN metrics are disabled.
            redis_config : Configuration for Redis-based evaluation dispatching (optional).
            runner_config : Configuration for the evaluation runner (optional).

        """
        if runner_config is None and redis_config is None:
            raise ValueError(
                "At least one of runner_config or redis_config must be provided for:"
                "\n- If runner_config is not provided, redis_config must be provided for remote evaluation."
                "\n- If redis_config is not provided, runner_config must be provided for local evaluation."
                "\n- If both are provided, the dispatcher will attempt to use Redis for dispatching and fall back to local evaluation if Redis is unavailable."
            )

        super().__init__()
        self.student = student
        self.register_buffer(
            "spec_tok_buff",
            torch.tensor(_get_special_tokens_ids(tokenizer), dtype=torch.long),  # ty:ignore[invalid-argument-type]
            persistent=False,
        )

        if self.student.config.hidden_size != teacher_hidden_size:
            self.student_to_teacher_proj = nn.Linear(
                self.student.config.hidden_size,
                teacher_hidden_size,
                device=self.student.device,
                dtype=self.student.dtype,
                bias=True,
            )
            if self.student.device != torch.device("meta"):
                nn.init.trunc_normal_(self.student_to_teacher_proj.weight, std=0.02)
                nn.init.zeros_(self.student_to_teacher_proj.bias)
                self.student_to_teacher_proj.weight._is_hf_initialized = True  # ty:ignore[unresolved-attribute]
                self.student_to_teacher_proj.bias._is_hf_initialized = True  # ty:ignore[unresolved-attribute]
        else:
            self.student_to_teacher_proj = nn.Identity()

        match train_cfg.distillation_method.distill_loss_type:
            case "cosine":
                direct_loss = nn.CosineEmbeddingLoss()
            case "mse":
                direct_loss = nn.MSELoss()
            case _:
                raise ValueError(
                    f"Unsupported loss type: {train_cfg.distillation_method.distill_loss_type} "
                    f"Supported types: 'cosine', 'mse'"
                )
        self.direct_loss = direct_loss

        self.train_cfg = train_cfg
        self.strategy_config = strategy_config
        self.distill_method_cfg = train_cfg.distillation_method
        self.tokenizer = tokenizer
        self.num_labels = num_labels if num_labels is not None else 0
        self.task = task

        if runner_config is not None:
            eval_runner = EvaluationRunner(config=runner_config)
        else:
            eval_runner = None
        self.evaluation_dispatcher = EvaluationDispatcher(
            evaluation_runner=eval_runner, redis_config=redis_config
        )

        self.eval_outputs: list[_EvalResult] = []
        self.cpu_group: dist.ProcessGroup | None = None
        self._model_configured = False
        return

    @override
    def configure_model(self) -> None:  # noqa: C901
        if self._model_configured:
            return

        shard_modules = {"student": self.student}
        no_shard_modules = {
            "student_to_teacher_proj": self.student_to_teacher_proj,
        }
        strategy = self.trainer.strategy
        strategy_config = self.strategy_config

        ### FSDP Wrapping ###
        if isinstance(strategy, FSDPStrategy):
            if not isinstance(strategy_config, FSDP1StrategyConfig):
                raise ValueError(
                    f"Expected  FSDP1StrategyConfig but got {strategy_config.__class__.__name__} "
                    f"for {strategy.__class__.__name__} strategy"
                )

            policies = get_fsdp1_policies(strategy_config)

            for name, module in shard_modules.items():
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

            for name, module in no_shard_modules.items():
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

        ### FSDP2 (Model Parallel) Wrapping ###
        elif isinstance(strategy, ModelParallelStrategy):
            if not isinstance(strategy_config, FSDP2StrategyConfig):
                raise ValueError(
                    f"Expected FSDP2StrategyConfig but got {strategy_config.__class__.__name__} "
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
                raise ValueError("Tensor parallelism is not yet supported.")

            policies = get_fsdp2_policies(strategy_config)

            for name, module in shard_modules.items():
                modules_to_shard = get_fsdp_wrap_classes(model=module)
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
                fully_shard(module=module, mesh=dp_mesh, **policies)  # ty:ignore[missing-argument]

            for module in no_shard_modules.values():
                if (module is None) or isinstance(module, nn.Identity):
                    continue
                replicate(
                    module,
                    mesh=dp_mesh,
                    mp_policy=policies["mp_policy"],
                )
        logger.info(
            f"Using attn implementation: {self.student.config._attn_implementation}"
        )
        self._model_configured = True
        return

    @property
    @override
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    @override
    def dtype(self) -> torch.dtype:  # pyrefly: ignore
        return next(self.parameters()).dtype

    @property
    @override
    def model_instance(self) -> PreTrainedModel:
        """Returns the underlying model."""
        return self.student

    @property
    @override
    def model_config(self) -> PreTrainedConfig:
        if self.model_instance is None:
            raise ValueError("Model instance is not initialized.")
        return self.model_instance.config

    @property
    @override
    def grad_clip_val(self) -> float | None:
        return self.train_cfg.grad_clip_val

    @property
    def dp_group(self) -> dist.ProcessGroup | None:
        """Returns the data parallel process group for distributed training."""
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

    def _set_freeze_state(self, freeze: bool) -> None:
        for param in self.student.parameters():
            param.requires_grad_(not freeze)
        return

    def is_frozen(self) -> bool:
        """Return True if all parameters of the specified model are frozen."""
        frozen_flag = all(
            not param.requires_grad for param in self.student.parameters()
        )
        return frozen_flag

    @override
    def configure_optimizers(self) -> dict[str, Any] | Optimizer:  # ty:ignore[invalid-method-override]
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
                verbosity_level="rank-zero-only",
                group=self.dp_group,
            )
            for key, value in scaled_lrs.items():
                setattr(self.train_cfg.lr, key, value)

        total_steps = estimate_total_steps(
            trainer=self.trainer,
            dp_process_group=self.dp_group,
        )
        freeze_student_embeddings = (
            self.train_cfg.freeze_student_emb_steps_ratio is not None
        ) and self.train_cfg.freeze_student_emb_steps_ratio > 0.0

        params = create_param_groups(
            model=self,
            lr=self.train_cfg.lr.base_value,
            weight_decay=self.train_cfg.weight_decay.base_value,
            no_decay_keywords={"emb", "token_pos_weights", "dt_bias", "A_log"},
            freeze_student_embeddings=freeze_student_embeddings,
            embeddings_lr_multiplier=self.train_cfg.embeddings_lr_multiplier,
        )

        optim = create_optimizer(
            parameters_groups=params,
            optimizer_config=self.train_cfg.optimizer,
            lr=self.train_cfg.lr.base_value,
            weight_decay=self.train_cfg.weight_decay.base_value,
        )

        schedulers: dict[str, BaseScheduler] = {}
        if self.train_cfg.lr.scheduler_type is not None:
            if freeze_student_embeddings:
                emb_scheduler = create_scheduler(
                    config=self.train_cfg.lr,
                    optim=optim,
                    num_iters=total_steps,
                    param_group_field="lr",
                    apply_if_field="is_embedding",
                    multiplier_field="lr_multiplier",
                    freeze_ratio=self.train_cfg.freeze_student_emb_steps_ratio,
                    plateau_ratio=self.train_cfg.lr.plateau_ratio
                    - self.train_cfg.freeze_student_emb_steps_ratio,  # ty:ignore[unsupported-operator]
                )
                schedulers["embedding_lr"] = emb_scheduler

                model_scheduler = create_scheduler(
                    config=self.train_cfg.lr,
                    optim=optim,
                    num_iters=total_steps,
                    param_group_field="lr",
                    ignore_if_field="is_embedding",
                    multiplier_field=None,
                )
                schedulers["model_lr"] = model_scheduler
            else:
                scheduler = create_scheduler(
                    config=self.train_cfg.lr,
                    optim=optim,
                    num_iters=total_steps,
                    param_group_field="lr",
                    multiplier_field="lr_multiplier",
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

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    @override
    def lr_scheduler_step(
        self,
        scheduler: BaseScheduler,
        metric: Any | None,
    ) -> None:  # ty:ignore[invalid-method-override]
        scheduler.step(self.global_step)
        return

    def _get_student_outputs(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if isinstance(_unwrap_model(self.student), StaticEmbeddingsModel):
            student_output: StaticEmbeddingsOutput = self.student(
                input_ids,
                attention_mask,
            )
            student_embeddings = student_output.token_embeddings
            student_sentence_embeddings = student_output.sentence_embeddings
        else:
            outputs: LednikModelOutput = self.student(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            student_embeddings = outputs.last_hidden_state
            student_sentence_embeddings = outputs.sentence_embeddings
        return {
            "student_embeddings": student_embeddings,
            "student_sentence_embeddings": student_sentence_embeddings,
        }

    def _calculate_distill_loss(
        self,
        student_embeddings: torch.Tensor,  # [b*seq_len, dim]
        teacher_embeddings: torch.Tensor,  # [b*seq_len, dim]
    ) -> torch.Tensor:
        if isinstance(self.direct_loss, nn.CosineEmbeddingLoss):
            loss = self.direct_loss(
                student_embeddings,
                teacher_embeddings,
                torch.ones(student_embeddings.size(0), device=self.device),
            )
        elif isinstance(self.direct_loss, nn.MSELoss):
            loss = self.direct_loss(
                student_embeddings,
                teacher_embeddings,
            )
        else:
            raise ValueError(
                f"Unsupported loss: {type(self.direct_loss.__class__.__name__)}"
            )
        return loss

    def _calculate_contrastive_loss(
        self,
        sentence_embeddings: torch.Tensor,
        temp: float,
        queries_mask: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        sentence_embeddings = sentence_embeddings / sentence_embeddings.norm(
            p=2, dim=-1, keepdim=True
        )

        queries_emb = sentence_embeddings[queries_mask]
        pos_emb = sentence_embeddings[pos_mask]
        if neg_mask is None:
            neg_emb = None
        else:
            neg_emb = sentence_embeddings[neg_mask]
            if neg_emb.numel() == 0:
                neg_emb = None
        if queries_emb.size() != pos_emb.size():
            raise ValueError(
                f"Anchors and positives have different sizes. Found {queries_emb.size()} and {pos_emb.size()} respectively."
            )

        if dist.is_initialized():
            rank = dist.get_rank(group=self.dp_group)
            global_anchors, global_positives, global_negatives = (
                GatherSentenceEmbeddings.apply(
                    queries_emb,
                    pos_emb,
                    neg_emb,
                    self.dp_group,
                )
            )
        else:
            rank = 0
            global_anchors = queries_emb
            global_positives = pos_emb
            global_negatives = neg_emb

        bsz_queries = queries_emb.size(0)
        bsz_positives = pos_emb.size(0)
        local_anchors = global_anchors[rank * bsz_queries : (rank + 1) * bsz_queries]
        local_positives = global_positives[
            rank * bsz_positives : (rank + 1) * bsz_positives
        ]

        targets = torch.arange(
            rank * bsz_positives,
            (rank + 1) * bsz_positives,
            device=local_anchors.device,
        )

        if global_negatives is None:
            candidate_embeddings = global_positives
        else:
            candidate_embeddings = torch.cat(
                (global_positives, global_negatives), dim=0
            )

        anchor2pos_sim = local_anchors @ candidate_embeddings.T / temp
        pos2achors_sim = local_positives @ global_anchors.T / temp

        anchor2pos_loss = F.cross_entropy(anchor2pos_sim, targets, reduction="mean")
        pos2achors_loss = F.cross_entropy(pos2achors_sim, targets, reduction="mean")
        loss = (anchor2pos_loss + pos2achors_loss) / 2
        return loss

    def _base_step(self, batch: CollatorOutput) -> _BaseStepOutput:
        """Performs a single training step for knowledge distillation."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        if isinstance(_unwrap_model(self.student), StaticEmbeddingsModel):
            non_special_tokens_mask = torch.isin(
                input_ids, cast(torch.Tensor, self.spec_tok_buff), invert=True
            ).to(device=attention_mask.device, dtype=torch.long)
            attention_mask = attention_mask * non_special_tokens_mask

        student_outputs = self._get_student_outputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        teacher_sentence_embeddings = batch["teacher_sentence_embeddings"]
        student_sentence_embeddings = student_outputs["student_sentence_embeddings"]

        contrastive_weight = self.distill_method_cfg.contrastive_loss_weight
        if contrastive_weight < 1.0:
            student_sentence_embeddings_proj = self.student_to_teacher_proj(
                student_sentence_embeddings
            )

            distill_loss = self._calculate_distill_loss(
                student_embeddings=student_sentence_embeddings_proj,
                teacher_embeddings=teacher_sentence_embeddings,
            )
        else:
            distill_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            student_sentence_embeddings_proj = student_sentence_embeddings.clone()

        if contrastive_weight > 0.0:
            temp = self.distill_method_cfg.temperature
            if temp is None:
                raise ValueError("Temperature must be specified for contrastive loss.")

            contrastive_loss = self._calculate_contrastive_loss(
                sentence_embeddings=student_sentence_embeddings,
                queries_mask=batch["queries_mask"],
                pos_mask=batch["positives_mask"],
                neg_mask=batch["negatives_mask"],
                temp=temp,
            )
        else:
            contrastive_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        loss = (
            contrastive_weight * contrastive_loss
            + (1 - contrastive_weight) * distill_loss
        )

        output = _BaseStepOutput(
            loss=loss,
            contrastive_loss=contrastive_loss,
            distill_loss=distill_loss,
            student_sentence_embeddings=student_sentence_embeddings,
            teacher_sentence_embeddings=teacher_sentence_embeddings,
            student_sentence_embeddings_proj=student_sentence_embeddings_proj,
        )
        return output

    @override
    def training_step(self, batch: CollatorOutput, batch_idx: int) -> torch.Tensor:
        output = self._base_step(batch)
        metrics = {
            "loss": output.loss.detach(),
        }
        teacher_embeddings = output.teacher_sentence_embeddings
        student_embeddings_proj = output.student_sentence_embeddings_proj

        if student_embeddings_proj is not None:
            cosine_similarity_value = cosine_similarity(
                student_embeddings_proj.detach(),
                teacher_embeddings.detach(),
                reduction="mean",
            )
            rmse_value = mean_squared_error(
                student_embeddings_proj.detach(),
                teacher_embeddings.detach(),
                squared=False,
            )
            metrics["CosineSimilarity"] = cosine_similarity_value
            metrics["RMSE"] = rmse_value

        if output.contrastive_loss.item() > 0.0:
            metrics["ContrastiveLoss"] = output.contrastive_loss.detach()
        if output.distill_loss.item() > 0.0:
            metrics["DistillLoss"] = output.distill_loss.detach()

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
            sync_dist_group=self.dp_group,
        )
        return output.loss

    @override
    def validation_step(self, batch: CollatorOutput, batch_idx: int) -> torch.Tensor:
        output = self._base_step(batch)
        metrics = {
            "loss": output.loss.detach(),
        }
        student_embeddings = output.student_sentence_embeddings
        teacher_embeddings = output.teacher_sentence_embeddings
        student_embeddings_proj = output.student_sentence_embeddings_proj

        if student_embeddings_proj is not None:
            cosine_similarity_value = cosine_similarity(
                student_embeddings_proj.detach(),
                teacher_embeddings.detach(),
                reduction="mean",
            )
            rmse_value = mean_squared_error(
                student_embeddings_proj.detach(),
                teacher_embeddings.detach(),
                squared=False,
            )
            metrics["CosineSimilarity"] = cosine_similarity_value
            metrics["RMSE"] = rmse_value

        if output.contrastive_loss.item() > 0.0:
            metrics["ContrastiveLoss"] = output.contrastive_loss.detach()
        if output.distill_loss.item() > 0.0:
            metrics["DistillLoss"] = output.distill_loss.detach()

        if self.task is not None and self.evaluation_dispatcher is not None:
            self.eval_outputs.append(
                _EvalResult(
                    teacher_sentence_embeddings=teacher_embeddings,
                    student_sentence_embeddings=student_embeddings,
                    labels=batch["labels"],
                    queries_mask=batch["queries_mask"],
                    positives_mask=batch["positives_mask"],
                )
            )

        self.log_dict(
            metrics,
            enable_graph=False,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            stage="val",
            sync_dist_group=self.dp_group,
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
            sync_dist_group=self.dp_group,
        )
        return output.loss

    @override
    def on_fit_end(self) -> None:
        if self.cpu_group is not None:
            dist.destroy_process_group(self.cpu_group)
        return

    @override
    @torch.inference_mode()
    def on_validation_epoch_end(self) -> None:
        if (
            self.trainer.sanity_checking
            or self.task is None
            or self.evaluation_dispatcher is None
        ):
            return

        teacher_embeddings_list = []
        student_embeddings_list = []
        labels_list = []
        queries_mask_list = []
        pos_mask_list = []
        for output in self.eval_outputs:
            teacher_embeddings_list.append(output.teacher_sentence_embeddings)
            student_embeddings_list.append(output.student_sentence_embeddings)
            labels_list.append(output.labels)
            queries_mask_list.append(output.queries_mask)
            pos_mask_list.append(output.positives_mask)

        ### TORCH TENSORS ###
        teacher_embeddings = torch.cat(teacher_embeddings_list, dim=0)
        student_embeddings = torch.cat(student_embeddings_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        queries_mask = torch.cat(queries_mask_list, dim=0)
        pos_mask = torch.cat(pos_mask_list, dim=0)

        if self.trainer.is_global_zero:
            contract = ValidationContract(
                task_id=self.task.id,
                current_step=self.global_step,
                teacher_embeddings=teacher_embeddings,
                student_embeddings=student_embeddings,
                labels=labels,
                queries_mask=queries_mask,
                pos_mask=pos_mask,
                num_classes=self.num_labels,
            )
            self.evaluation_dispatcher.dispatch(contract=contract)

        self.eval_outputs = []
        if dist.is_initialized():
            dist.barrier()
        return
