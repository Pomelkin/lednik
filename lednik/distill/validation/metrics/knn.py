import torch
from torch import Tensor
from torchmetrics.functional import accuracy
from torchmetrics.functional import f1_score

from lednik.distill.validation.configs import KNNConfig


@torch.compile(mode="reduce-overhead")
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
        raise ValueError("Number of embeddings must match number of labels.")

    if num_labels == 1:
        num_labels = 2  # Ensure at least two labels for bincount

    k_neighbors = min(k_neighbors, embeddings.shape[0] - 1)
    B, _ = embeddings.shape
    predicts = torch.zeros(B, dtype=labels.dtype, device=labels.device)

    distances = (
        (embeddings.unsqueeze(1) - embeddings.unsqueeze(0)).pow(2).sum(-1).sqrt()
    )
    distances.fill_diagonal_(float("inf"))

    top_k_neighbors = distances.topk(k_neighbors, largest=False).indices
    neighbors_labels = labels[top_k_neighbors]

    for i in range(neighbors_labels.size(0)):
        bincount = neighbors_labels[i].bincount(minlength=num_labels)
        predicted_label = bincount.argmax()
        predicts[i] = predicted_label
    return predicts


def _select_optimal_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    """Moves the model to the optimal device and data type for training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    return device, dtype


def calculate_self_exc_knn_metrics(
    inputs: Tensor, targets: Tensor, knn_config: KNNConfig, num_classes: int
) -> dict[str, float]:
    """
    Computes self-excluding k-NN F1 and accuracy metrics.

    Args:
        inputs (Tensor): Embedding tensor of shape ``(batch_size, embedding_dim)``.
        targets (Tensor): Ground-truth labels of shape ``(batch_size,)``.
        knn_config (KNNConfig): Configuration for k-NN evaluation.
        num_classes (int): The number of distinct classes in the dataset.


    Returns:
        A dictionary with metric values (currently ``{"F1": ..., "Accuracy": ...}``).
    """
    device, dtype = _select_optimal_device_and_dtype()

    inputs = inputs.to(device=device, dtype=dtype, non_blocking=True)
    labels = targets.to(device=device, dtype=dtype, non_blocking=True)

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
