from functools import partial
from typing import Any
from typing import TypedDict

import torch
import torch.distributed as dist
from kostyl.ml.configs.training_settings import FSDP1StrategyConfig
from kostyl.ml.configs.training_settings import FSDP2StrategyConfig
from kostyl.utils import setup_logger
from torch import nn
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import CPUOffloadPolicy
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.fsdp import OffloadPolicy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import PreTrainedModel


logger = setup_logger()


def _find_modules_to_exclude_from_wrapping(
    module_list: set[type[nn.Module]], target_substrings: list[str] | None = None
) -> set[type[nn.Module]]:
    """
    Find modules to exclude from individual FSDP wrapping based on their class name substrings.

    These modules are typically embeddings or output heads that share weights.
    Excluding them from leaf-wrapping ensures they end up in the root FSDP container together.
    """
    default_substrings = ["embedding", "lmhead"]
    target_substrings = (
        target_substrings + default_substrings
        if target_substrings is not None
        else default_substrings
    )
    target_substrings = [x.lower() for x in target_substrings]

    modules_to_exclude = set()
    for module in module_list:
        for substring in target_substrings:
            if substring in module.__name__.lower():
                logger.debug(
                    f"Excluding {module.__name__} from sharding due to name contains '{substring}'"
                )
                modules_to_exclude.add(module)
                break
    return modules_to_exclude


def get_transformer_wrap_classes(
    model: PreTrainedModel, exclude_sharding_substrings: list[str] | None = None
) -> set[type[nn.Module]] | None:
    """
    Identifies transformer modules that should to be sharded during distributed training.

    This function inspects the provided `model` for a `_no_split_modules` attribute,
    which typically contains a list of class names (strings) that should be kept intact
    on a single device (e.g., Transformer blocks). It then resolves these names to
    actual module classes present in the model instance.

    It also filters out specific sub-modules that might have been captured but should
    be excluded from the no-shard set based on internal logic.

    Args:
        model (PreTrainedModel): The Hugging Face PreTrainedModel instance to inspect.
        exclude_sharding_substrings (list[str] | None, optional): A list of substrings to match
            against module names. If a module's name contains any of these substrings, it will
            be excluded from the no-shard set. If None, a default set of substrings will be used.
            Default set: ["embedding", "lmhead"]. Comparison will be case-insensitive.

    Returns:
        set[type[nn.Module]] | None: A set of unique module classes that should
        not be sharded, or None if the model does not define `_no_split_modules`.

    """
    shard_modules_names: list[str] | None = getattr(model, "_no_split_modules", None)
    if shard_modules_names is None or len(shard_modules_names) == 0:
        return None

    shard_modules: set[type[nn.Module]] = set()
    for module in model.modules():
        if module.__class__.__name__ in shard_modules_names:
            shard_modules.add(type(module))

    modules_to_exclude = _find_modules_to_exclude_from_wrapping(
        shard_modules,
        exclude_sharding_substrings,
    )
    shard_modules = shard_modules - modules_to_exclude

    if len(shard_modules) > 0:
        logger.debug(
            f"Identified {len(shard_modules)} shard modules: "
            f"{[module.__name__ for module in shard_modules]}"
        )
        result = shard_modules
    else:
        logger.debug("No shard modules identified after filtering.")
        result = None
    return result


def select_wrap_policy(
    model: PreTrainedModel, exclude_sharding_substrings: list[str] | None = None
) -> partial | ModuleWrapPolicy:
    """
    Selects an appropriate wrapping policy for FSDP1 (Fully Sharded Data Parallel) based on the model architecture.

    This function determines which modules within the model should be wrapped for sharding.
    It prioritizes explicit shard modules if defined for the specific model type. If not found,
    it falls back to a size-based auto-wrap policy, while excluding specific module types that are known
    to cause issues with sharding (e.g., small layers or specific embeddings) determined dynamically.

    Args:
        model (PreTrainedModel): The model to be wrapped.
        exclude_sharding_substrings (list[str] | None, optional): A list of substrings to match
            against module names. If a module's name contains any of these substrings, it will
            be excluded from the shard set. If None, a default set of substrings will be used.
            Default set: ["embedding", "lmhead"]. Comparison will be case-insensitive.

    Returns:
        partial | ModuleWrapPolicy: A callable policy or a `ModuleWrapPolicy` instance that dictates
        how the model's layers should be wrapped by FSDP.

    """
    shard_modules = get_transformer_wrap_classes(model, exclude_sharding_substrings)
    if shard_modules is not None:
        return ModuleWrapPolicy(module_classes=shard_modules)

    logger.debug(
        "No explicit shard modules found. "
        "Falling back to size-based auto-wrap policy with exclusions."
    )

    modules = {type(module) for module in model.modules()}
    modules_to_exclude = _find_modules_to_exclude_from_wrapping(
        modules, exclude_sharding_substrings
    )
    logger.debug(
        f"Excluding {len(modules_to_exclude)} modules from size-based wrapping: "
        f"{[module.__name__ for module in modules_to_exclude]}"
    )

    EXCLUDE_WRAP_MODULES: set[type[nn.Module]] = (
        size_based_auto_wrap_policy.EXCLUDE_WRAP_MODULES  # type: ignore
    ).union(modules_to_exclude)
    FORCE_LEAF_MODULES: set[type[nn.Module]] = (
        size_based_auto_wrap_policy.FORCE_LEAF_MODULES  # type: ignore
    )
    return partial(
        size_based_auto_wrap_policy,
        force_leaf_modules=FORCE_LEAF_MODULES,
        exclude_wrap_modules=EXCLUDE_WRAP_MODULES,
    )


def _get_optional_attribute(obj: object, attr_name: str | None) -> Any | None:
    if attr_name is None:
        return None
    return getattr(obj, attr_name, None)


class FSDP2PolicyDict(TypedDict):  # noqa: D101
    mp_policy: MixedPrecisionPolicy
    offload_policy: CPUOffloadPolicy | OffloadPolicy


def get_fsdp2_policies(
    strategy_config: FSDP2StrategyConfig,
) -> FSDP2PolicyDict:
    """Create a MixedPrecisionPolicy for FSDP2 based on the provided strategy configuration."""
    kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=_get_optional_attribute(torch, strategy_config.param_dtype),
            reduce_dtype=_get_optional_attribute(torch, strategy_config.reduce_dtype),
            output_dtype=_get_optional_attribute(torch, strategy_config.output_dtype),
        ),
        "offload_policy": CPUOffloadPolicy(pin_memory=True)
        if strategy_config.use_cpu_offload
        else OffloadPolicy(),
    }
    return kwargs  # type: ignore


class FSDP1PolicyDict(TypedDict):  # noqa: D101
    mixed_precision: MixedPrecision
    cpu_offload: CPUOffload | None


def get_fsdp1_policies(
    strategy_config: FSDP1StrategyConfig,
) -> FSDP1PolicyDict:
    """Create a MixedPrecisionPolicy for FSDP1 based on the provided strategy configuration."""
    kwargs = {
        "mixed_precision": MixedPrecision(
            param_dtype=_get_optional_attribute(torch, strategy_config.param_dtype),
            reduce_dtype=_get_optional_attribute(torch, strategy_config.reduce_dtype),
            buffer_dtype=_get_optional_attribute(torch, strategy_config.buffer_dtype),
            keep_low_precision_grads=strategy_config.keep_low_precision_grads,
        ),
        "cpu_offload": CPUOffload(offload_params=True)
        if strategy_config.use_cpu_offload
        else None,
    }
    return kwargs  # type: ignore


class DistributedMMFunction(torch.autograd.Function):
    """
    Custom autograd function for distributed matrix multiplication.

    This function performs matrix multiplication across multiple distributed processes,
    gathering tensors from all ranks before computing the result. It implements custom
    forward and backward passes to handle gradient computation in a distributed setting.
    """

    @staticmethod
    def forward(  # type: ignore[override]  # noqa: D102
        ctx: torch.autograd.Function,
        x: torch.Tensor,
        y: torch.Tensor,
        group: dist.ProcessGroup | None = None,
        multi_by_world_size: bool = True,
    ) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("Input x must be a 2D tensor.")
        if not dist.is_initialized():
            raise RuntimeError("Distributed process group is not initialized.")

        handlers: list[dist.Work] = []
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)

        y_buff = y.new_empty((y.size(0) * world_size, y.size(1)))
        y_handler = dist.all_gather_into_tensor(y_buff, y, group=group, async_op=True)
        handlers.append(y_handler)  # type: ignore

        x_buff = x.new_empty((x.size(0) * world_size, x.size(1)))
        x_handler = dist.all_gather_into_tensor(x_buff, x, group=group, async_op=True)
        handlers.append(x_handler)  # type: ignore

        for handler in handlers:
            handler.wait()

        res = x_buff @ y_buff.T

        ctx.save_for_backward(x_buff, y_buff)
        ctx.x_offset = y.size(0) * rank  # type: ignore
        ctx.x_batch_size = x.size(0)  # type: ignore
        ctx.y_offset = y.size(0) * rank  # type: ignore
        ctx.y_batch_size = y.size(0)  # type: ignore
        ctx.world_size = world_size  # type: ignore
        ctx.multi_by_world_size = multi_by_world_size  # type: ignore
        return res

    @staticmethod
    def backward(  # type: ignore[override] # noqa: D102
        ctx: Any, dres: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        x, y = ctx.saved_tensors

        dx = dres @ y
        dy = dres.T @ x

        dx = dx[ctx.x_offset : ctx.x_offset + ctx.x_batch_size]
        dy = dy[ctx.y_offset : ctx.y_offset + ctx.y_batch_size]
        if ctx.multi_by_world_size:
            dx *= ctx.world_size
            dy *= ctx.world_size
        return dx, dy, None, None
