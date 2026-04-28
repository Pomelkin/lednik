import torch
from torch import Tensor
from torchmetrics.functional import accuracy
from torchmetrics.functional import f1_score

from lednik.distill.validation.structs import KNNConfig


def _select_knn_chunk_size(batch_size: int, max_buffer_mb: int = 64) -> int:
    """Selects a row chunk size to cap temporary distance matrix memory usage."""
    if batch_size <= 0:
        return 1

    # Assume float32 distance buffer for conservative sizing.
    bytes_per_distance = torch.tensor([], dtype=torch.float32).element_size()
    max_distances = (max_buffer_mb * 1024 * 1024) // bytes_per_distance
    return max(1, min(batch_size, max_distances // batch_size))


@torch.no_grad()
def knn_predict(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_labels: int,
    k_neighbors: int = 5,
) -> torch.Tensor:
    """
    Predicts labels for each embedding using a self-excluding k-nearest neighbors vote.

    The function assumes a 2D tensor of embeddings and leverages Euclidean distance to
    determine the closest `k_neighbors` samples to each embedding (excluding itself),
    then assigns the label with the highest frequency among the neighbors. The number
    of candidate labels can be controlled via `num_labels`.

    Args:
        embeddings (torch.Tensor): A 2D tensor of shape (batch_size, embedding_dim)
            containing embedding vectors.
        labels (torch.Tensor): A 1D tensor of integer labels corresponding to the embeddings.
        num_labels (int): Total number of distinct labels; used to size `torch.bincount`.
        k_neighbors (int, optional): Number of nearest neighbors to consult; defaults to 5.
            Automatically capped at `batch_size - 1` to avoid empty neighbor sets.

    Returns:
        torch.Tensor: A 1D tensor containing the predicted label for each embedding.

    """
    if embeddings.dim() != 2:
        raise ValueError("Embeddings must be a 2D tensor.")
    if labels.dim() != 1:
        raise ValueError("Labels must be a 1D tensor.")
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError(
            "Number of embeddings must match number of labels."
            f" Got {embeddings.shape[0]} embeddings and {labels.shape[0]} labels."
        )

    if num_labels == 1:
        num_labels = 2  # Ensure at least two labels for bincount

    B, _ = embeddings.shape
    predicts = torch.zeros(B, dtype=labels.dtype, device=labels.device)

    k_neighbors = min(k_neighbors, B - 1)
    chunk_size = _select_knn_chunk_size(B)

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        chunk = embeddings[start:end]

        # Compute chunk-to-all distances to avoid materializing a full B x B matrix.
        chunk_distances = torch.cdist(chunk, embeddings)

        # Exclude self-neighbor only for rows represented in the current chunk.
        local_rows = torch.arange(end - start, device=embeddings.device)
        global_cols = torch.arange(start, end, device=embeddings.device)
        chunk_distances[local_rows, global_cols] = float("inf")

        top_k_neighbors = chunk_distances.topk(k_neighbors, largest=False).indices
        neighbors_labels = labels[top_k_neighbors].long()

        vote_counts = torch.zeros(
            (end - start, num_labels), dtype=torch.int32, device=labels.device
        )
        vote_counts.scatter_add_(
            1,
            neighbors_labels,
            torch.ones_like(neighbors_labels, dtype=vote_counts.dtype),
        )
        predicts[start:end] = vote_counts.argmax(dim=1).to(labels.dtype)

    return predicts


def _select_optimal_device() -> torch.device:
    """Moves the model to the optimal device and data type for training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def _select_dtype(device: torch.device) -> torch.dtype:
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    return dtype


def calculate_self_exc_knn_metrics(
    inputs: Tensor,
    targets: Tensor,
    knn_config: KNNConfig,
    num_classes: int,
    device: str = "auto",
) -> dict[str, float]:
    """
    Computes self-excluding k-NN F1 and accuracy metrics.

    Args:
        inputs (Tensor): Embedding tensor of shape ``(batch_size, embedding_dim)``.
        targets (Tensor): Ground-truth labels of shape ``(batch_size,)``.
        knn_config (KNNConfig): Configuration for k-NN evaluation.
        num_classes (int): The number of distinct classes in the dataset.
        device (str, optional): Device to perform computation on; defaults to "auto" for optimal selection.

    Returns:
        A dictionary with metric values (currently ``{"F1": ..., "Accuracy": ...}``).
    """
    device = _select_optimal_device() if device == "auto" else torch.device(device)  # type: ignore
    dtype = _select_dtype(device)  # type: ignore

    inputs = inputs.to(device=device, dtype=dtype, non_blocking=True)
    labels = targets.to(device=device, non_blocking=True)

    predicted_labels = knn_predict(inputs, labels, num_classes, knn_config.k)

    if num_classes <= 2:
        f1 = f1_score(predicted_labels, labels, task="binary").item()
        acc = accuracy(predicted_labels, labels, task="binary").item()
    else:
        f1 = f1_score(
            predicted_labels,
            labels,
            num_classes=num_classes,
            task="multiclass",
            average="macro",
        ).item()
        acc = accuracy(
            predicted_labels,
            labels,
            num_classes=num_classes,
            task="multiclass",
            average="macro",
        ).item()
    return {"F1": f1, "Accuracy": acc}
