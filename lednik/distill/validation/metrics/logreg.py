from dataclasses import dataclass
from typing import Literal
from typing import cast

import torch
from kostyl.utils import setup_logger
from torch import Tensor
from torch import nn
from torch.optim import LBFGS
from torch.optim import Adam
from torch.optim import Muon
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchmetrics.functional import accuracy
from torchmetrics.functional import f1_score

from lednik.distill.validation.structs import LogRegConfig
from lednik.distill.validation.utils import stratified_split


logger = setup_logger()


@dataclass
class LogRegOutput:
    """Output of the logistic regression model."""

    logits: Tensor
    loss: Tensor | None


class LogisticRegression(nn.Module):
    """A logistic regression model implemented in PyTorch, supporting both binary and multiclass classification."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        solver: Literal["LBFGS", "Muon", "Adam"] = "LBFGS",
        tol: float = 1e-4,
        batch_size: int = -1,
        total_steps: int = 1000,
    ) -> None:
        """
        Initializes the Logistic Regression model.

        Args:
            num_features (int): Number of input features.
            num_classes (int): Number of target classes. If 1 or 2, treats as binary classification.
            lr (float, optional): Learning rate for the optimizer. Defaults to 3e-4.
            weight_decay (float, optional): L2 regularization penalty. Defaults to 0.01.
            solver (Literal["LBFGS", "Muon", "Adam"], optional): Optimizer to use. Defaults to "LBFGS".
            tol (float, optional): Tolerance for early stopping based on loss difference. Defaults to 1e-4.
            batch_size (int, optional): Batch size for training. If -1, uses full-batch training. Defaults to -1.
            total_steps (int, optional): Maximum number of training steps. Defaults to 1000.
        """
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be at least 1")

        if num_classes == 2:
            num_classes = 1  # Binary classification can be handled with a single output

        if batch_size not in (None, -1) and solver == "LBFGS":
            logger.warning(
                "LBFGS solver does not support mini-batch training. Ignoring batch_size and using full-batch instead."
            )
            batch_size = -1

        self.weight = nn.Parameter(torch.empty(num_features, num_classes))
        self.bias = nn.Parameter(torch.empty(1, num_classes))

        self.num_classes = num_classes
        self.lr = lr
        self.solver = solver
        self.tol = tol
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.total_steps = total_steps
        self._init_weights()
        return

    def _init_weights(self) -> None:
        nn.init.xavier_normal_(self.weight)
        nn.init.constant_(self.bias, 0)
        return

    @property
    def device(self) -> torch.device:
        """Returns the device of the model parameters."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Returns the data type of the model parameters."""
        return next(self.parameters()).dtype

    def _select_optimal_device(self) -> torch.device:
        """Moves the model to the optimal device and data type for training."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device

    def _select_dtype(self, device: torch.device) -> torch.dtype:
        dtype = (
            torch.bfloat16
            if device.type == "cuda"
            and torch.cuda.is_bf16_supported()
            and self.solver != "LBFGS"
            else torch.float32
        )
        return dtype

    @torch.no_grad()
    def forward(  # noqa: D102
        self,
        hidden_state: Tensor,
        labels: Tensor | None = None,
        loss_weights: Tensor | None = None,
    ) -> LogRegOutput:
        logits = hidden_state @ self.weight + self.bias

        if labels is not None:
            if loss_weights is not None:
                class_weights = loss_weights[labels]
            else:
                class_weights = torch.ones_like(labels)

            if logits.size(-1) == 1:
                probs = logits.sigmoid().squeeze(-1)
                inv_labels = 1 - labels
                loss = (
                    (probs * labels + (1 - probs) * inv_labels).log() * -1
                ) * class_weights

                dl_dprobs = (
                    (probs - 1 * labels) * class_weights / loss.size(0)
                ).unsqueeze(-1)
            else:
                probs = logits.softmax(dim=-1)
                loss = (
                    probs[torch.arange(probs.size(0)), labels].log() * -1
                ) * class_weights

                dl_dprobs = probs.clone()
                dl_dprobs[torch.arange(probs.size(0)), labels] -= 1
                dl_dprobs = dl_dprobs * class_weights.unsqueeze(-1) / loss.size(0)

            loss = loss.mean()

            if self.bias.grad is None:
                self.bias.grad = torch.zeros_like(self.bias)

            if self.weight.grad is None:
                self.weight.grad = torch.zeros_like(self.weight)

            self.bias.grad += dl_dprobs.sum(dim=0, keepdim=True)
            self.weight.grad += hidden_state.T @ dl_dprobs
            self.weight.grad += self.weight * self.weight_decay
        else:
            loss = None
        return LogRegOutput(logits=logits, loss=loss)

    def fit(  # noqa: C901
        self,
        inputs: Tensor,
        labels: Tensor,
        log_step: int | None = None,
        device: str = "auto",
    ) -> None:
        """Trains the logistic regression model using the specified optimization algorithm."""
        if inputs.shape[0] != labels.shape[0]:
            raise ValueError(
                "Number of inputs must match number of labels."
                f" Got {inputs.shape[0]} inputs and {labels.shape[0]} labels."
            )

        device = (
            self._select_optimal_device() if device == "auto" else torch.device(device)  # type: ignore
        )
        dtype = self._select_dtype(device)  # type: ignore
        self.to(device=device, dtype=dtype).train()

        match self.solver:
            case "LBFGS":
                optim = LBFGS([self.weight, self.bias], lr=self.lr)
            case "Muon":
                optim = Muon([self.weight, self.bias], weight_decay=0.0, lr=self.lr)
            case "Adam":
                optim = Adam([self.weight, self.bias], weight_decay=0.0, lr=self.lr)
            case _:
                raise ValueError(f"Unsupported solver: {self.solver}")

        labels_bincount = labels.bincount().float()
        labels_bincount /= labels_bincount.sum()
        labels_bincount = labels_bincount**-1
        labels_bincount *= 1 / labels_bincount.size(-1)
        loss_weights = labels_bincount

        loss_weights = loss_weights.to(dtype=self.dtype, device=self.device)

        if self.batch_size == -1 or self.batch_size >= inputs.size(0):
            dataloader = [(inputs, labels)]
        else:
            dataset = TensorDataset(inputs, labels)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        last_loss = None
        loss_diff = float("inf")
        step = 0
        while step < self.total_steps and loss_diff > self.tol:
            for batch_inputs, batch_labels in dataloader:
                if step >= self.total_steps or loss_diff <= self.tol:
                    break

                batch_inputs = batch_inputs.to(
                    self.device, self.dtype, non_blocking=True
                )
                batch_labels = batch_labels.to(
                    self.device, dtype=torch.long, non_blocking=True
                )

                def closure() -> Tensor:
                    optim.zero_grad()
                    out: LogRegOutput = self(batch_inputs, batch_labels, loss_weights)  # noqa: B023
                    return cast(Tensor, out.loss)

                loss_tensor = optim.step(closure)
                current_loss = cast(Tensor, loss_tensor).item()

                if log_step is not None and step % log_step == 0:
                    logger.info(f"Step {step}: {current_loss}")

                if last_loss is not None:
                    loss_diff = abs(last_loss - current_loss)
                last_loss = current_loss
                step += 1
        return

    def score(
        self, inputs: Tensor, labels: Tensor, device: str = "auto"
    ) -> dict[str, float]:
        """Evaluates the model on the given inputs and labels, returning F1 score and accuracy."""
        device = (
            self._select_optimal_device() if device == "auto" else torch.device(device)  # type: ignore
        )
        dtype = self._select_dtype(device)  # type: ignore
        self.to(device=device, dtype=dtype).eval()

        inputs = inputs.to(self.device, self.dtype, non_blocking=True)
        labels = labels.to(self.device, dtype=torch.long, non_blocking=True)

        output: LogRegOutput = self(inputs)
        if self.num_classes == 1:
            probs = output.logits.sigmoid().squeeze(-1)
            f1 = f1_score(probs, labels, task="binary").item()
            acc = accuracy(probs, labels, task="binary").item()
        else:
            probs = output.logits.softmax(dim=-1).argmax(dim=-1)
            f1 = f1_score(
                probs,
                labels,
                num_classes=self.num_classes,
                task="multiclass",
                average="macro",
            ).item()
            acc = accuracy(
                probs,
                labels,
                num_classes=self.num_classes,
                task="multiclass",
                average="macro",
            ).item()
        return {"F1": f1, "Accuracy": acc}


def calculate_logreg_metrics(
    inputs: Tensor,
    targets: Tensor,
    logreg_config: LogRegConfig,
    num_classes: int,
    log_step: int | None = None,
    device: str = "auto",
) -> dict[str, float]:
    """
    Train a logistic regression probe and compute classification metrics.

    The function performs a stratified train/test split of the provided features
    (`inputs`) and labels (`targets`), fits :class:`LogisticRegression` using the
    hyperparameters from `logreg_config`, and evaluates the probe on the test split.

    Args:
        inputs: Feature tensor of shape ``(N, D)``.
        targets: Label tensor of shape ``(N,)`` aligned with `inputs`.
        logreg_config: Configuration for optimizer and training parameters.
        num_classes: The number of classes for the classification task.
        log_step: Interval at which to log training progress.
            If ``None``, no intermediate logging is performed.
        device (str, optional): Device to perform computation on; defaults to "auto" for optimal selection.

    Returns:
        A dictionary with metric values (currently ``{"F1": ..., "Accuracy": ...}``).
    """
    train_inputs, train_labels, test_inputs, test_labels = stratified_split(
        dataset=inputs, labels=targets, test_size=0.2
    )
    logreg = LogisticRegression(
        num_features=inputs.size(1),
        num_classes=num_classes,
        lr=logreg_config.lr,
        weight_decay=logreg_config.weight_decay,
        solver=logreg_config.solver,
        tol=logreg_config.tol,
        batch_size=logreg_config.batch_size,
        total_steps=logreg_config.total_steps,
    )
    logreg.fit(train_inputs, train_labels, log_step=log_step, device=device)
    return logreg.score(test_inputs, test_labels, device=device)
