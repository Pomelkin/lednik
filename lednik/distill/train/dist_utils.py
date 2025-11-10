import math
import os

import torch.distributed as dist

from lednik.distill.train.configs import TrainConfig
from lednik.utils.logging import setup_logger

logger = setup_logger(add_rank=True)


def scale_lrs_by_world_size(
    lr_config: TrainConfig,
    group: dist.ProcessGroup | None = None,
    config_name: str = "",
    inv_scale: bool = False,
) -> None:
    """
    Scale learning-rate configuration values to match the active distributed world size.

    Args:
        lr_config (TrainConfig): Learning-rate configuration whose values will be scaled.
        group (dist.ProcessGroup | None): Optional process group used to determine
            the target world size. Defaults to the global process group.
        config_name (str): Human-readable identifier included in log messages. Helpful when several lr config
            objects are being used in the same training run. Defaults to an empty string.
        inv_scale (bool): If True, use the inverse square-root scale factor.

    Returns:
        None

    """
    world_size = dist.get_world_size(group=group)

    if inv_scale:
        scale = 1 / math.sqrt(world_size)
    else:
        scale = math.sqrt(world_size)

    logger.info(f"Scaling learning rates for world size: {world_size}")
    logger.info(f"Scale factor: {scale:.4f}")
    old_base = lr_config.base_lr
    lr_config.base_lr *= scale
    logger.info(f"New {config_name} lr BASE: {lr_config.base_lr}; OLD: {old_base}")

    if lr_config.warmup_lr is not None:
        old_warmup_lr = lr_config.warmup_lr
        lr_config.warmup_lr *= scale
        logger.info(
            f"New {config_name} lr WARMUP: {lr_config.warmup_lr}; OLD: {old_warmup_lr}"
        )
    return


def _get_rank() -> int:
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = int(os.environ.get("RANK", 0))
    return rank


def is_main_process() -> bool:
    """Check if the current process is the main process (rank 0)."""
    return _get_rank() == 0
