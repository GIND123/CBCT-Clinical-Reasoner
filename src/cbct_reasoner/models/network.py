"""CBCT encoders and the multi-label finding head.

With ~630 training volumes, training a 3D network from scratch is the wrong
default: the label signal is a few hundred bits per case and the input is tens of
millions of voxels. The ``slice2d`` backbone instead reuses an ImageNet-pretrained
2D encoder over multi-planar slices and pools them with gated attention, which
puts almost all of the capacity in weights that were learned elsewhere. A
from-scratch ``resnet3d`` is kept as an ensemble partner because its errors are
decorrelated from the 2D path.

The head's output bias is initialized to the training log-odds of each prototype.
The network therefore *starts* at the corpus prior - already a competitive report
- and spends its capacity learning where a given scan deviates from it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PLANES = ("axial", "coronal", "sagittal")


class GatedAttentionPool(nn.Module):
    """Attention MIL pooling (Ilse et al., 2018) over a variable slice sequence."""

    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.value = nn.Linear(dim, hidden)
        self.gate = nn.Linear(dim, hidden)
        self.score = nn.Linear(hidden, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (batch, slices, dim)
        weights = self.score(torch.tanh(self.value(features)) * torch.sigmoid(self.gate(features)))
        attention = torch.softmax(weights, dim=1)
        return (attention * features).sum(dim=1)


class Slice2dBackbone(nn.Module):
    """Multi-planar 2.5D encoder built on a pretrained timm classifier."""

    def __init__(
        self,
        timm_model: str = "convnext_tiny",
        *,
        planes: Sequence[str] = ("axial",),
        slices_per_plane: int = 24,
        pretrained: bool = True,
        in_chans: int = 3,
    ) -> None:
        super().__init__()
        unknown = set(planes) - set(PLANES)
        if unknown:
            raise ValueError(f"Unknown planes: {sorted(unknown)}")
        self.planes = tuple(planes)
        self.slices_per_plane = slices_per_plane
        self.in_chans = in_chans

        try:
            import timm
        except ImportError as error:  # pragma: no cover - environment specific
            raise RuntimeError(
                "timm is required for the slice2d backbone; install '.[train]'"
            ) from error

        self.encoder = timm.create_model(
            timm_model, pretrained=pretrained, num_classes=0, in_chans=in_chans
        )
        self.feature_dim = int(self.encoder.num_features)
        self.pools = nn.ModuleList([GatedAttentionPool(self.feature_dim) for _ in self.planes])

    @property
    def output_dim(self) -> int:
        return self.feature_dim * len(self.planes)

    def _slices(self, volume: torch.Tensor, plane: str) -> torch.Tensor:
        """Return ``(batch, slices, in_chans, height, width)`` for one plane."""
        # volume: (batch, 1, D, H, W)
        if plane == "axial":
            stack = volume[:, 0]  # (B, D, H, W)
        elif plane == "coronal":
            stack = volume[:, 0].permute(0, 2, 1, 3)  # (B, H, D, W)
        else:
            stack = volume[:, 0].permute(0, 3, 1, 2)  # (B, W, D, H)

        depth = stack.shape[1]
        count = min(self.slices_per_plane, depth)
        centres = (
            torch.linspace(
                self.in_chans // 2, depth - 1 - self.in_chans // 2, count, device=stack.device
            )
            .round()
            .long()
            .clamp(0, depth - 1)
        )

        offsets = torch.arange(self.in_chans, device=stack.device) - self.in_chans // 2
        indices = (centres[:, None] + offsets[None, :]).clamp(0, depth - 1)  # (count, in_chans)
        gathered = stack[:, indices.reshape(-1)]  # (B, count*C, H, W)
        batch, _, height, width = gathered.shape
        return gathered.reshape(batch, count, self.in_chans, height, width)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        embeddings = []
        for pool, plane in zip(self.pools, self.planes, strict=True):
            slices = self._slices(volume, plane)
            batch, count = slices.shape[:2]
            flat = slices.reshape(batch * count, *slices.shape[2:])
            features = self.encoder(flat).reshape(batch, count, self.feature_dim)
            embeddings.append(pool(features))
        return torch.cat(embeddings, dim=1)


def _conv_block(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.InstanceNorm3d(out_channels, affine=True),
        nn.SiLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm3d(out_channels, affine=True),
        nn.SiLU(inplace=True),
    )


class ResNet3dBackbone(nn.Module):
    """Compact from-scratch 3D CNN used for ensemble diversity."""

    def __init__(self, width: int = 32, depth: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        channels = [1] + [width * (2**index) for index in range(depth)]
        self.blocks = nn.ModuleList(
            [_conv_block(channels[index], channels[index + 1], stride=2) for index in range(depth)]
        )
        self.dropout = nn.Dropout3d(dropout)
        self.feature_dim = channels[-1] * 2

    @property
    def output_dim(self) -> int:
        return self.feature_dim

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        features = volume
        for block in self.blocks:
            features = block(features)
        features = self.dropout(features)
        pooled = torch.cat(
            [
                F.adaptive_avg_pool3d(features, 1).flatten(1),
                F.adaptive_max_pool3d(features, 1).flatten(1),
            ],
            dim=1,
        )
        return pooled


class FindingNet(nn.Module):
    """Backbone + geometry features -> one logit per sentence prototype."""

    def __init__(
        self,
        backbone: nn.Module,
        num_prototypes: int,
        *,
        meta_dim: int = 10,
        hidden: int = 512,
        dropout: float = 0.2,
        prior: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        input_dim = int(backbone.output_dim) + meta_dim  # type: ignore[attr-defined]
        self.neck = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden, num_prototypes)
        self.embedding_dim = hidden
        self._init_prior(prior, num_prototypes)

    def _init_prior(self, prior: Sequence[float] | np.ndarray | None, num_prototypes: int) -> None:
        """Start every logit at the corpus base rate for that prototype."""
        nn.init.zeros_(self.classifier.weight)
        if prior is None:
            nn.init.zeros_(self.classifier.bias)
            return
        values = np.clip(np.asarray(prior, dtype=np.float64), 1e-4, 1 - 1e-4)
        if values.shape != (num_prototypes,):
            raise ValueError(f"prior must have shape ({num_prototypes},), got {values.shape}")
        with torch.no_grad():
            self.classifier.bias.copy_(torch.from_numpy(np.log(values / (1 - values))).float())

    def embed(self, volume: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        features = self.backbone(volume)
        return self.neck(torch.cat([features, meta], dim=1))

    def forward(self, volume: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(volume, meta))


def build_network(
    *,
    backbone: str,
    num_prototypes: int,
    prior: np.ndarray | None,
    meta_dim: int = 10,
    timm_model: str = "convnext_tiny",
    width: int = 32,
    dropout: float = 0.2,
    planes: Sequence[str] = ("axial",),
    slices_per_plane: int = 24,
    pretrained: bool = True,
) -> FindingNet:
    if backbone == "slice2d":
        encoder: nn.Module = Slice2dBackbone(
            timm_model,
            planes=planes,
            slices_per_plane=slices_per_plane,
            pretrained=pretrained,
        )
    elif backbone == "resnet3d":
        encoder = ResNet3dBackbone(width=width, dropout=dropout)
    else:
        raise ValueError(f"Unknown backbone {backbone!r}; expected 'slice2d' or 'resnet3d'")
    return FindingNet(
        encoder,
        num_prototypes,
        meta_dim=meta_dim,
        dropout=dropout,
        prior=prior,
    )
