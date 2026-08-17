"""Losses and weight averaging for long-tailed multi-label finding prediction.

Prototype prevalence spans two orders of magnitude - some statements appear in
almost every report, others in a handful of cases. Plain BCE lets the abundant
negatives of the rare prototypes dominate the gradient, and the model collapses
to the prior. Asymmetric loss decouples the positive and negative focusing rates
and discards easy negatives outright, which is what keeps the rare-but-decisive
findings learnable.
"""

from __future__ import annotations

import copy

import torch
from torch import nn


class AsymmetricLoss(nn.Module):
    """Asymmetric loss for multi-label classification (Ridnik et al., 2021)."""

    def __init__(
        self,
        gamma_negative: float = 2.0,
        gamma_positive: float = 0.0,
        clip: float = 0.05,
        label_smoothing: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.gamma_negative = gamma_negative
        self.gamma_positive = gamma_positive
        self.clip = clip
        self.label_smoothing = label_smoothing
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        column_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        positive_prob = torch.sigmoid(logits)
        negative_prob = 1.0 - positive_prob
        if self.clip > 0:
            # Probability shifting: a negative already predicted below `clip` is
            # treated as fully solved and contributes no gradient at all.
            negative_prob = (negative_prob + self.clip).clamp(max=1.0)

        loss_positive = targets * torch.log(positive_prob.clamp(min=self.eps))
        loss_negative = (1 - targets) * torch.log(negative_prob.clamp(min=self.eps))

        with torch.no_grad():
            weight = torch.pow(
                1
                - positive_prob * targets
                - (1.0 - self.clip) * (1 - targets) * (1 - negative_prob),
                self.gamma_positive * targets + self.gamma_negative * (1 - targets),
            )
        per_column = -(loss_positive + loss_negative) * weight
        if column_mask is not None:
            # Statements with too few positives to learn are left to their
            # prior-initialized bias: no gradient, so no noise added to the
            # backbone from columns that cannot be predicted anyway.
            per_column = per_column * column_mask
        return per_column.sum(dim=1).mean()


class ModelEma:
    """Exponential moving average of the weights.

    Fold sizes here are ~120 validation cases, so single-epoch validation scores
    are noisy enough to select the wrong checkpoint. Averaging removes most of
    that variance for free.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_value, value in zip(
            self.module.state_dict().values(), model.state_dict().values(), strict=True
        ):
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(value.detach(), alpha=1 - self.decay)
            else:
                ema_value.copy_(value)


def average_precision(
    scores: torch.Tensor, targets: torch.Tensor, column_mask: torch.Tensor | None = None
) -> float:
    """Per-column average precision, averaged over columns with a positive.

    ``column_mask`` restricts the average to learnable columns. Over a
    thousand-statement label space the unrestricted mean is dominated by
    prototypes seen in two or three cases, where average precision is noise,
    and it barely moves however good the model is.
    """
    if scores.shape != targets.shape:
        raise ValueError("scores and targets must have the same shape")
    values: list[float] = []
    for column in range(scores.shape[1]):
        if column_mask is not None and not bool(column_mask[column]):
            continue
        target = targets[:, column]
        positives = int(target.sum().item())
        if positives == 0 or positives == target.numel():
            continue
        order = torch.argsort(scores[:, column], descending=True)
        ranked = target[order]
        cumulative = torch.cumsum(ranked, dim=0)
        precision = cumulative / torch.arange(1, len(ranked) + 1, device=ranked.device)
        values.append(float((precision * ranked).sum().item() / positives))
    return float(sum(values) / len(values)) if values else 0.0
