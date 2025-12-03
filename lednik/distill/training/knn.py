import torch


@torch.inference_mode()
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
    mask = torch.ones(B, dtype=torch.bool)
    predicts = torch.zeros(B, dtype=labels.dtype, device=labels.device)

    for i in range(B):
        batch_mask = mask.clone()
        batch_mask[i] = False
        embedding2pred = embeddings[i].unsqueeze(0)
        embeddings2index = embeddings[batch_mask]
        labels2index = labels[batch_mask]

        distances = (embeddings2index - embedding2pred).pow(2).sum(-1).sqrt()

        knn_indices = distances.topk(k_neighbors, largest=False).indices
        knn_labels = labels2index[knn_indices]

        counts = torch.bincount(knn_labels, minlength=num_labels)
        predicted_label = counts.argmax()
        predicts[i] = predicted_label
    return predicts
