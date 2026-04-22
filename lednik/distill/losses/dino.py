import torch
import torch.distributed as dist
import torch.nn.functional as F


@torch.no_grad()
def sinkhorn_knopp_teacher(
    teacher_output: torch.Tensor,
    teacher_temp: float,
    n_iterations: int = 3,
    process_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """
    Normalize teacher logits into a balanced assignment matrix using the Sinkhorn-Knopp algorithm.

    Args:
        teacher_output (torch.Tensor): Raw teacher logits of shape (batch_size, num_prototypes).
        teacher_temp (float): Temperature used to sharpen the teacher distribution before normalization.
        n_iterations (int, optional): Number of Sinkhorn iterations to perform. Defaults to 3.
        process_group (dist.ProcessGroup | None, optional): Distributed process group for synchronized
            reductions. Defaults to None.

    Returns:
        torch.Tensor: Doubly-stochastic assignments tensor of shape (batch_size, num_prototypes).

    """
    # teacher_output: [batch, prototypes]
    teacher_output = teacher_output.float()
    world_size = 1 if process_group is None else dist.get_world_size(process_group)
    Q = torch.exp(
        teacher_output / teacher_temp
    ).t()  # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * world_size  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    if dist.is_initialized():
        dist.all_reduce(sum_Q, group=process_group)
    Q /= sum_Q

    for _ in range(n_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if dist.is_initialized():
            dist.all_reduce(sum_of_rows, group=process_group)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the columns must sum to 1 so that Q is an assignment
    return Q.t()


def dino_ce_loss(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    student_temp: float,
) -> torch.Tensor:
    """
    Cross-entropy between softmax outputs of the teacher and student networks.
    student_logits: [batch * num tokens, prototypes]
    teacher_probs:  [batch * num tokens, prototypes].
    """  # noqa: D205
    if student_logits.ndim != 2 or teacher_probs.ndim != 2:
        raise ValueError(
            "student_logits and teacher_probs must be 2-dimensional tensors"
        )
    if student_logits.size(0) != teacher_probs.size(0):
        raise ValueError(
            "student_logits and teacher_probs must have the same batch size"
        )

    B_T, _ = student_logits.shape
    student_logprob = F.log_softmax(student_logits.float() / student_temp, dim=-1)
    loss = -torch.einsum("t e, t e ->", teacher_probs, student_logprob)
    return loss / B_T
