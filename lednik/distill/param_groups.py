from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .training_module import DistillationModule


def create_param_groups(
    model: DistillationModule,
    weight_decay: float,
    lr: float,
    freeze_student_embeddings: bool = False,
    embeddings_lr_multiplier: float | None = None,
    attn_proj_wd_multiplier: float | None = None,
    no_decay_keywords: set[str] | None = None,
) -> list[dict]:
    """Create optimizer parameter groups for a PyTorch model with fine-grained weight decay control."""
    no_decay_keywords_ = {
        "norm",
        "bias",
        "embedding",
        "emb",
    }
    if no_decay_keywords is not None:
        no_decay_keywords_ = no_decay_keywords_.union(no_decay_keywords)

    # TODO: Make if-checks FSDP/DDP Aware
    # if freeze_student_embeddings and not isinstance(model.student, LednikModel):
    #     raise ValueError(
    #         "freeze_student_embeddings is only supported for LednikModel student models."
    #     )

    # if attn_proj_wd_multiplier is not None and not isinstance(
    #     model.student, LednikModel
    # ):
    #     raise ValueError(
    #         "attn_proj_wd_multiplier is only supported for LednikModel student models."
    #     )

    param_groups = []
    for name, param in model.named_parameters():
        if param.requires_grad is False:
            continue

        if freeze_student_embeddings and "emb" in name:
            param_group = {
                "params": param,
                "lr": 0.0,
                "is_embedding": True,
                "lr_multiplier": embeddings_lr_multiplier or 1.0,
            }
        else:
            param_group = {"params": param, "lr": lr}

        if any(keyword in name for keyword in no_decay_keywords_):
            param_group["weight_decay"] = 0.0
        else:
            param_group["weight_decay"] = weight_decay

        if "q_proj" in name or "k_proj" in name:
            param_group["weight_decay"] = weight_decay * (
                attn_proj_wd_multiplier or 1.0
            )
        param_groups.append(param_group)

    fused_param_groups = _fuse_groups(param_groups)
    return fused_param_groups


def _fuse_groups(param_groups: list[dict]) -> list[dict]:
    fuse_dict: dict[str, dict[str, Any]] = {}
    for group in param_groups:
        group_key = ""
        for key, value in group.items():
            if key != "params":
                group_key += f"_{key}:{value}"

        if group_key not in fuse_dict:
            fuse_dict[group_key] = {"params": []}
            for k, v in group.items():
                if k != "params":
                    fuse_dict[group_key][k] = v
        fuse_dict[group_key]["params"].append(group["params"])
    return list(fuse_dict.values())
