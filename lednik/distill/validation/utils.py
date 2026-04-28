from io import BytesIO

import torch
from torch import Tensor


def tensor_to_bytes(tensor: Tensor) -> bytes:
    """Serialize a tensor to bytes via `torch.save`."""
    tensor = tensor.detach().cpu()
    buf = BytesIO()
    torch.save(tensor, buf)
    return buf.getvalue()


def bytes_to_tensor(bytes_data: bytes) -> Tensor:
    """Deserialize a tensor from bytes produced by :func:`tensor_to_bytes`."""
    buf = BytesIO(bytes_data)
    return torch.load(buf)


def stratified_split(
    dataset: Tensor,
    labels: Tensor,
    test_size: float = 0.2,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Split a dataset into train/test subsets preserving label proportions.

    Performs a stratified split by sampling indices within each unique label in
    `labels`, using `test_size` fraction for the test subset. The resulting train
    and test indices are then independently shuffled.

    Args:
        dataset: Tensor of samples. The first dimension is treated as the sample axis.
        labels: Tensor of labels aligned with `dataset` along the first dimension.
        test_size: Fraction of samples per class to put into the test split.
        generator: Optional RNG generator for deterministic shuffling.

    Returns:
        A tuple `(train_dataset, train_labels, test_dataset, test_labels)`.

    Raises:
        ValueError: If `dataset` and `labels` have different lengths.
    """
    if len(labels) != len(dataset):
        raise ValueError("Labels and dataset must have the same length.")

    train_idx, test_idx = [], []
    for c in labels.unique():
        cls = (labels == c).nonzero(as_tuple=True)[0]
        perm = cls[torch.randperm(len(cls), generator=generator)]
        n_te = round(len(perm) * test_size)
        test_idx.append(perm[:n_te])
        train_idx.append(perm[n_te:])

    train_idx = torch.cat(train_idx)
    test_idx = torch.cat(test_idx)
    train_idx = train_idx[torch.randperm(len(train_idx), generator=generator)]
    test_idx = test_idx[torch.randperm(len(test_idx), generator=generator)]

    return dataset[train_idx], labels[train_idx], dataset[test_idx], labels[test_idx]
