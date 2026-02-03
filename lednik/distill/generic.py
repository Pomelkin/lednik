import torch


def get_torch_dtype(
    dtype: str | torch.dtype,
) -> torch.dtype:
    """Convert a string representation of a torch dtype to the actual torch dtype."""
    if isinstance(dtype, str):
        return getattr(torch, dtype)
    return dtype
