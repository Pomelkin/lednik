from collections.abc import Mapping

import torch.distributed as dist
from torch import Tensor
from torchmetrics import Metric
from torchmetrics import MetricCollection


def apply_suffix(
    metrics: Mapping[str, Metric | Tensor | int | float] | MetricCollection,
    suffix: str,
    add_dist_rank: bool = False,
) -> Mapping[str, Metric | Tensor | int | float] | MetricCollection:
    """Add stage prefix to metric names."""
    new_metrics_dict = {}
    for key, value in metrics.items():
        new_key = f"{suffix}/{key}"
        if add_dist_rank and dist.is_initialized():
            rank = dist.get_rank()
            new_key = f"{new_key}-rank:{rank}"
        new_metrics_dict[new_key] = value
    return new_metrics_dict


def format_per_class_metrics(
    metrics: Mapping[str, Tensor],
    num_classes: int,
    suffix: str = "per_class",
    idx_to_classname: Mapping[int, str] | None = None,
) -> Mapping[str, Tensor]:
    """Format per-class metrics by adding a suffix to each class metric."""
    new_metrics_dict = {}
    for key, value in metrics.items():
        for i in range(num_classes):
            if idx_to_classname is not None:
                class_name = idx_to_classname.get(i, i)
                new_key = f"{suffix}/{key}_{class_name}"
            else:
                new_key = f"{suffix}/{key}_class_{i}"
            new_metrics_dict[new_key] = value[i]
    return new_metrics_dict
