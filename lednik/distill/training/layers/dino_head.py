import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


def _build_mlp(
    nlayers: int,
    in_dim: int,
    bottleneck_dim: int,
    hidden_dim: int,
    use_bn: bool = False,
    bias: bool = True,
) -> nn.Linear | nn.Sequential:
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)


class DINOHead(nn.Module):
    """
    Projection head module used in DINO-style self-supervised training.

    This module maps backbone feature vectors to a normalized embedding and then to
    output logits/features used by the distillation objective. It consists of an MLP
    ("projector") followed by L2-normalization and a final linear layer.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        use_bn: bool = False,
        nlayers: int = 3,
        bottleneck_dim: int = 256,
        mlp_bias: bool = True,
    ) -> None:
        """
        Initialize the DINO projection head.

        Args:
            in_dim: Dimensionality of the input features.
            out_dim: Dimensionality of the output logits/features.
            use_bn: If True, use BatchNorm layers in the MLP.
            nlayers: Number of layers in the MLP (minimum is 1).
            hidden_dim: Hidden dimension used for intermediate MLP layers.
            bottleneck_dim: Output dimension of the MLP before the final linear layer.
            mlp_bias: If True, include bias terms in MLP linear layers.

        Returns:
            None

        """
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(
            nlayers,
            in_dim,
            bottleneck_dim,
            hidden_dim=hidden_dim,
            use_bn=use_bn,
            bias=mlp_bias,
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self.init_weights()
        return

    def init_weights(self) -> None:
        """Initialize module parameters."""
        self.apply(self._init_weights)
        return None

    def _init_weights(self, m) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        return None

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)
        return x
