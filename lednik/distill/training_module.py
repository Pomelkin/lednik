from dataclasses import dataclass
from typing import Any
from typing import Literal
from typing import cast
from typing import override

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
from torch import nn
from torch.distributed._composable.replicate_with_fsdp import replicate
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp import fully_shard
from torch.nn.modules.loss import _Loss
from torchmetrics.functional import cosine_similarity
from torchmetrics.functional import mean_squared_error
from transformers import PreTrainedConfig
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerBase
from transformers import SentencePieceBackend
from transformers import TokenizersBackend
from transformers.modeling_outputs import BaseModelOutput

from lednik.dist_utils import GatherSentenceEmbeddings
from lednik.dist_utils import get_fsdp1_policies
from lednik.dist_utils import get_fsdp2_policies
from lednik.dist_utils import get_fsdp_wrap_classes
from lednik.dist_utils import select_wrap_policy
from lednik.distill.configs import DirectDistillationConfig
from lednik.distill.configs import DistillationConfig
from lednik.distill.validation import EvaluationDispatcher
from lednik.distill.validation import RedisConfig
from lednik.distill.validation import ValidationContract
from lednik.emb_utils import get_sentence_embedding
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
    return [tokenizer.convert_tokens_to_ids(token) for token in special_tokens]  # type: ignore


class DistillationModule(KostylLightningModule):
    """A PyTorch Lightning module for fine-tuning a static embeddings model via knowledge distillation."""

    def __init__(
        self,
        teacher: PreTrainedModel,
        student: StaticEmbeddingsModel | LednikModel,
        tokenizer: SentencePieceBackend | TokenizersBackend | PreTrainedTokenizerBase,
        train_cfg: DistillationConfig,
        strategy_config: SUPPORTED_STRATEGIES,
        redis_config: RedisConfig | None = None,
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
            redis_config : Configuration for Redis-based evaluation dispatching (optional).

        """
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.register_buffer(
            "spec_tok_buff",
            torch.tensor(_get_special_tokens_ids(tokenizer), dtype=torch.long),  # type: ignore
            persistent=False,
        )

        (
            self.student_to_teacher_proj,
            self.direct_loss,
        ) = self._init_direct(
            train_cfg.distillation_method,
            teacher_hidden_size=teacher.config.hidden_size,
            student_hidden_size=student.config.hidden_size,
            device=student.device,
            dtype=student.dtype,
        )

        self.train_cfg = train_cfg
        self.strategy_config = strategy_config
        self.distill_method_cfg = train_cfg.distillation_method
        self.tokenizer = tokenizer
        self.num_labels = num_labels if num_labels is not None else 0
        self.task = task

        self.evaluation_dispatcher = (
            EvaluationDispatcher(redis_config=redis_config)
            if redis_config is not None
            else None
        )

        self.eval_outputs: list[_EvalResult] = []
        self._model_configured = False
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

    @override
    def configure_model(self) -> None:  # noqa: C901
        if self._model_configured:
            return

        shard_modules = {"teacher": self.teacher, "student": self.student}
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
                fully_shard(module=module, mesh=dp_mesh, **policies)

            for module in no_shard_modules.values():
                if (module is None) or isinstance(module, nn.Identity):
                    continue
                replicate(
                    module,
                    mesh=dp_mesh,
                    mp_policy=policies["mp_policy"],
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
    def model_instance(self) -> PreTrainedModel | nn.Module:
        """Returns the underlying model."""
        return self.student

    @property
    @override
    def model_config(self) -> PreTrainedConfig | None:
        return self.student.config

    @property
    @override
    def grad_clip_val(self) -> float | None:
        return self.train_cfg.grad_clip_val

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

    def _set_freeze_state(
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
        self._set_freeze_state("teacher", freeze=True)
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
                verbosity_level="only-zero-rank",
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
            no_decay_keywords={"emb", "token_pos_weights"},
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

        return {  # pyrefly: ignore
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    @override
    def lr_scheduler_step(  # pyrefly: ignore
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
            )
            student_embeddings = student_output.token_embeddings
            student_sentence_embeddings = student_output.sentence_embeddings
        else:
            outputs = self.student(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            student_embeddings = outputs[0]

            pooling_method = (
                self.train_cfg.student_pooling_method
                if self.train_cfg.student_pooling_method is not None
                else self.train_cfg.teacher_pooling_method
            )

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
    ) -> torch.Tensor:
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
        return loss

    def _calculate_contrastive_loss(
        self,
        sentence_embeddings: torch.Tensor,
        temp: float,
        queries_mask: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_mask: torch.Tensor,
    ) -> torch.Tensor:
        sentence_embeddings = sentence_embeddings / sentence_embeddings.norm(
            p=2, dim=-1, keepdim=True
        )

        queries_emb = sentence_embeddings[queries_mask]
        pos_emb = sentence_embeddings[pos_mask]
        neg_emb = sentence_embeddings[neg_mask]
        if queries_emb.size() != pos_emb.size():
            raise ValueError(
                f"Anchors and positives have different sizes. Found {queries_emb.size()} and {pos_emb.size()} respectively."
            )

        if dist.is_initialized():
            rank = dist.get_rank(group=self._data_parallel_group)
            global_anchors, global_positives, global_negatives = (  # type: ignore
                GatherSentenceEmbeddings.apply(
                    queries_emb,
                    pos_emb,
                    neg_emb,
                    self._data_parallel_group,
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

        anchor2pos_sim = (
            local_anchors
            @ torch.cat((global_positives, global_negatives), dim=0).T
            / temp
        )
        pos2achors_sim = local_positives @ global_anchors.T / temp

        anchor2pos_loss = F.cross_entropy(anchor2pos_sim, targets, reduction="mean")
        pos2achors_loss = F.cross_entropy(pos2achors_sim, targets, reduction="mean")
        loss = (anchor2pos_loss + pos2achors_loss) / 2
        return loss

    def _base_step(self, batch: dict[str, torch.Tensor]) -> _BaseStepOutput:
        """Performs a single training step for knowledge distillation."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        non_special_tokens_mask = torch.isin(
            input_ids, cast(torch.Tensor, self.spec_tok_buff), invert=True
        ).to(device=attention_mask.device)

        # Keep special tokens in the student's forward + pooling, but exclude them from
        # token-level distillation to avoid biasing towards service tokens.
        per_token_loss_mask = attention_mask * non_special_tokens_mask.long()

        # if student is static embeddings model,
        # we need to exclude special tokens from sentence embedding (we use mask for pooling in forward)
        # because they are static (cannot be sentence-dependent) and would degrade the quality of sentence embeddings.
        student_attention_mask = (
            attention_mask
            if not isinstance(self.student, StaticEmbeddingsModel)
            else per_token_loss_mask
        )
        teacher_attention_mask = attention_mask

        student_outputs = self._get_student_outputs(
            input_ids=input_ids,
            attention_mask=student_attention_mask,
        )
        teacher_outputs = self._get_teacher_outputs(
            input_ids=input_ids,
            attention_mask=teacher_attention_mask,
        )

        contrastive_weight = self.distill_method_cfg.contrastive_loss_weight
        if contrastive_weight < 1.0:
            student_embeddings = student_outputs["student_embeddings"]
            teacher_embeddings = teacher_outputs["teacher_embeddings"]

            student_embeddings = self.student_to_teacher_proj(
                student_embeddings[per_token_loss_mask != 0]
            )
            teacher_embeddings = teacher_embeddings[per_token_loss_mask != 0]

            per_token_loss = self._calculate_per_token_similarity_loss(
                flat_student_embeddings=student_embeddings,
                flat_teacher_embeddings=teacher_embeddings,
            )
        else:
            student_embeddings = None
            teacher_embeddings = None
            per_token_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        if contrastive_weight > 0.0:
            temp = self.distill_method_cfg.temperature
            if temp is None:
                raise ValueError("Temperature must be specified for contrastive loss.")

            contrastive_loss = self._calculate_contrastive_loss(
                sentence_embeddings=student_outputs["student_sentence_embeddings"],
                queries_mask=batch["queries_mask"],
                pos_mask=batch["positives_mask"],
                neg_mask=batch["negatives_mask"],
                temp=temp,
            )
        else:
            contrastive_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        loss = (
            contrastive_weight * contrastive_loss
            + (1 - contrastive_weight) * per_token_loss
        )

        output = _BaseStepOutput(
            loss=loss,
            contrastive_loss=contrastive_loss,
            per_token_loss=per_token_loss,
            teacher_embeddings=teacher_embeddings,
            student_embeddings=student_embeddings,
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

        if self.task is not None and self.evaluation_dispatcher is not None:
            self.eval_outputs.append(
                _EvalResult(
                    teacher_sentence_embeddings=output.teacher_sentence_embeddings,
                    student_sentence_embeddings=output.student_sentence_embeddings,
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
            sync_dist_group=self._data_parallel_group,
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
    def on_validation_epoch_end(self) -> None:  # type: ignore
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

        if dist.is_initialized():
            teacher_embeddings = teacher_embeddings.to(device=self.device)
            student_embeddings = student_embeddings.to(device=self.device)
            labels = labels.to(device=self.device)
            queries_mask = queries_mask.to(device=self.device)
            pos_mask = pos_mask.to(device=self.device)

            group = self._data_parallel_group
            world_size = dist.get_world_size(group)
            rank = dist.get_rank(group)
            works = []

            def _gather_to_rank0(tensor: torch.Tensor) -> list[torch.Tensor] | None:
                if rank == 0:
                    gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
                else:
                    gather_list = None
                work = dist.gather(
                    tensor,
                    gather_list=gather_list,
                    dst=0,
                    group=group,
                    async_op=True,
                )
                works.append(cast(dist.Work, work))
                return gather_list  # unused on non-zero

            teacher_embeddings_list = _gather_to_rank0(teacher_embeddings)
            student_embeddings_list = _gather_to_rank0(student_embeddings)
            labels_list = _gather_to_rank0(labels)
            queries_mask_list = _gather_to_rank0(queries_mask)
            pos_mask_list = _gather_to_rank0(pos_mask)

            for work in works:
                work.wait()

            if self.trainer.is_global_zero:
                teacher_embeddings = torch.cat(
                    cast(list[torch.Tensor], teacher_embeddings_list), dim=0
                ).cpu()
                student_embeddings = torch.cat(
                    cast(list[torch.Tensor], student_embeddings_list), dim=0
                ).cpu()
                labels = torch.cat(cast(list[torch.Tensor], labels_list), dim=0).cpu()
                queries_mask = torch.cat(
                    cast(list[torch.Tensor], queries_mask_list), dim=0
                ).cpu()
                pos_mask = torch.cat(
                    cast(list[torch.Tensor], pos_mask_list), dim=0
                ).cpu()

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
