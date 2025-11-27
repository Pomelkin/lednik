from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DimReductionOutput:
    """Data class to hold the output of a dimensionality reduction process."""

    reduced_data: torch.Tensor
    reconstruction_loss: torch.Tensor | None = None


class Autoencoder(nn.Module):
    """
    A simple Autoencoder neural network for dimensionality reduction.

    This module implements a symmetric autoencoder architecture using RMSNorm normalization
    and GELU activation functions. It projects high-dimensional input data into a lower-dimensional
    latent space and attempts to reconstruct the original input from this latent representation.
    """

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        """Initialize Autoencoder with input and latent dimensions."""
        super().__init__()
        self.encoder_norm = nn.RMSNorm(input_dim)
        self.encoder = nn.Sequential(
            nn.GELU("tanh"),
            nn.Linear(input_dim, latent_dim, bias=False),
        )
        self.decoder_norm = nn.RMSNorm(latent_dim)
        self.decoder = nn.Sequential(
            nn.GELU("tanh"),
            nn.Linear(latent_dim, input_dim, bias=False),
        )
        self.apply(self._init_weights)
        return

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        match module:
            case nn.Linear():
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            case nn.RMSNorm():
                nn.init.ones_(module.weight)
            case _:
                pass
        return

    def forward(self, X: torch.Tensor) -> DimReductionOutput:
        """Forward pass to apply Autoencoder transformation."""
        output = self.transform(X)
        return output

    def transform(self, X: torch.Tensor) -> DimReductionOutput:
        """Apply the dimensionality reduction on X."""
        X = self.encoder_norm(X)
        latent = self.encoder(X)
        latent = self.decoder_norm(latent)
        X_rec = self.decoder(latent)
        loss = F.mse_loss(X, X_rec)
        output = DimReductionOutput(reduced_data=latent, reconstruction_loss=loss)
        return output


class PCA(nn.Module):
    """
    Principal Component Analysis (PCA) implementation as a PyTorch Module.

    This class performs linear dimensionality reduction using Singular Value Decomposition (SVD)
    of the data to project it to a lower dimensional space. It is designed to be compatible
    with PyTorch's neural network modules and supports optional compilation for performance.
    """

    def __init__(self, n_components: int, compile_transform: bool = False) -> None:
        """Initialize PCA with the number of components."""
        super().__init__()
        self.n_components = n_components
        self.compile_transform = compile_transform
        self.components_: torch.Tensor | None = None
        self.mean_: torch.Tensor | None = None
        self.explained_variance_: torch.Tensor | None = None
        self.explained_variance_ratio_: torch.Tensor | None = None
        return

    @property
    def mean(self) -> torch.Tensor:
        """Get mean of training data."""
        if self.mean_ is None:
            raise ValueError("Must call transform() first")
        return self.mean_

    @property
    def explained_variance(self) -> torch.Tensor:
        """Get explained variance."""
        if self.explained_variance_ is None:
            raise ValueError("Must call transform() first")
        return self.explained_variance_

    @property
    def explained_variance_ratio(self) -> torch.Tensor:
        """Get explained variance ratio."""
        if self.explained_variance_ratio_ is None:
            raise ValueError("Must call transform() first")
        return self.explained_variance_ratio_

    def forward(self, X: torch.Tensor) -> DimReductionOutput:
        """Forward pass to apply PCA transformation."""
        output = (
            self.transform(X)
            if not self.compile_transform
            else self.compiled_transform(X)
        )
        return output

    @torch.compile(dynamic=True)
    def compiled_transform(self, X: torch.Tensor) -> DimReductionOutput:
        """Apply the dimensionality reduction on X with compilation."""
        return self.transform(X)

    @torch.no_grad()
    def transform(self, X: torch.Tensor) -> DimReductionOutput:
        """Apply the dimensionality reduction on X."""
        if X.dim() != 2:
            raise ValueError("Input data must be a 2D tensor.")
        original_dtype = X.dtype
        X = X.to(torch.float64)
        n_samples, _ = X.shape

        self.mean_ = X.mean(dim=0, keepdim=True)
        X_centered = X - self.mean_

        U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)

        idx = torch.argmax(torch.abs(U), dim=0)
        signs = torch.sign(U[idx, torch.arange(U.size(1), device=U.device)])
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)

        U = U * signs.unsqueeze(0)
        Vh = Vh * signs.unsqueeze(1)

        U = U[:, : self.n_components]
        S = S[: self.n_components]
        self.components_ = Vh[: self.n_components, :]

        Z = U * S.unsqueeze(0)

        self.explained_variance_ = ((S**2) / (n_samples - 1)).to(original_dtype)
        total_var = X_centered.pow(2).sum() / (n_samples - 1)
        self.explained_variance_ratio_ = (self.explained_variance / total_var).to(
            original_dtype
        )

        Z = Z.to(original_dtype)
        output = DimReductionOutput(reduced_data=Z)
        return output
