from functools import partial
from typing import TypedDict

import torch
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


class FSDP2PolicyDict(TypedDict):  # noqa: D101
    mp_policy: MixedPrecisionPolicy
    offload_policy: CPUOffloadPolicy | OffloadPolicy


def get_fsdp2_policies(
    strategy_config: FSDP2StrategyConfig,
) -> FSDP2PolicyDict:
    """Create a MixedPrecisionPolicy for FSDP2 based on the provided strategy configuration."""
    kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=getattr(torch, strategy_config.param_dtype, None),  # type: ignore
            reduce_dtype=getattr(torch, strategy_config.reduce_dtype, None),  # type: ignore
            output_dtype=getattr(torch, strategy_config.output_dtype, None),  # type: ignore
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
            param_dtype=getattr(torch, strategy_config.param_dtype, None),  # type: ignore
            reduce_dtype=getattr(torch, strategy_config.reduce_dtype, None),  # type: ignore
            buffer_dtype=getattr(torch, strategy_config.buffer_dtype, None),  # type: ignore
        ),
        "cpu_offload": CPUOffload(offload_params=True)
        if strategy_config.use_cpu_offload
        else None,
    }
    return kwargs  # type: ignore
