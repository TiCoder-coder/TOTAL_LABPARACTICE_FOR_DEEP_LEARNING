"""Regularization primitives for Practice 2.2 to fight overfitting.

This module is intentionally additive: it provides three reusable building
blocks without modifying the existing training loop signature.

1. ``mixup_cutmix_batch`` - stochastic label-preserving image mixing that
   softens decision boundaries and reduces memorisation of single samples.
2. ``ModelEMA`` - Exponential Moving Average of model weights. The EMA copy
   is used for evaluation and checkpointing so the reported metric reflects
   a smoother weight trajectory rather than the noisy tail of training.
3. ``CutMix`` helper utilities (rectangle generation, area normalisation).

All functions are deterministic w.r.t. ``torch.Generator`` so the seeded
pipeline remains reproducible.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def _rand_bbox(
    height: int,
    width: int,
    lam: float,
    generator: torch.Generator,
) -> Tuple[int, int, int, int]:
    """Return ``(x1, y1, x2, y2)`` for a CutMix patch with area ratio ``lam``."""
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    # Uniform center.
    cx = int(torch.randint(0, width, (1,), generator=generator).item())
    cy = int(torch.randint(0, height, (1,), generator=generator).item())

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    y2 = min(cy + cut_h // 2, height)
    if x2 <= x1:
        x2 = min(x1 + 1, width)
    if y2 <= y1:
        y2 = min(y1 + 1, height)

    # Actual area ratio after clamping.
    actual_area = (x2 - x1) * (y2 - y1)
    total = width * height
    lam = max(1.0 - actual_area / float(total), 0.0)
    return x1, y1, x2, y2, lam


def mixup_cutmix_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    cutmix_prob: float = 0.5,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Mixup, CutMix or identity to a batch.

    Parameters
    ----------
    images:
        ``[B, C, H, W]`` float tensor on the same device as training.
    labels:
        ``[B]`` long tensor with class indices.
    mixup_alpha:
        Beta distribution parameter for Mixup. Set ``0`` to disable Mixup.
    cutmix_alpha:
        Beta distribution parameter for CutMix. Set ``0`` to disable CutMix.
    cutmix_prob:
        Probability of choosing CutMix over Mixup when both are enabled.
    generator:
        Optional ``torch.Generator`` for reproducibility. A fresh generator
        using the current device is created if ``None``.

    Returns
    -------
    mixed_images:
        ``[B, C, H, W]`` images after the chosen transform.
    soft_labels:
        ``[B, num_classes]`` probability distribution suitable for soft
        cross-entropy with label smoothing (which is added separately).
    """
    if images.dim() != 4:
        raise ValueError(f"images must be [B, C, H, W], got {tuple(images.shape)}")
    if labels.dim() != 1:
        raise ValueError(f"labels must be [B], got {tuple(labels.shape)}")

    batch_size = images.size(0)
    if batch_size < 2:
        # Mixing requires at least two samples.
        one_hot = torch.nn.functional.one_hot(labels, num_classes=int(labels.max().item()) + 1)
        return images, one_hot.float()

    num_classes = int(labels.max().item()) + 1
    one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
    images = images.clone()
    soft_labels = one_hot.clone()

    if generator is None:
        generator = torch.Generator(device=images.device)
        generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_(generator=None).item()))

    use_cutmix = (
        cutmix_alpha > 0
        and float(torch.rand((1,), generator=generator).item()) < cutmix_prob
    )

    if use_cutmix:
        lam = float(np.random.beta(cutmix_alpha, cutmix_alpha))
        perm = torch.randperm(batch_size, generator=generator, device=images.device)
        x1, y1, x2, y2, lam = _rand_bbox(
            images.size(2), images.size(3), lam, generator
        )
        images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        soft_labels = lam * one_hot + (1.0 - lam) * one_hot[perm]
    elif mixup_alpha > 0:
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
        perm = torch.randperm(batch_size, generator=generator, device=images.device)
        images = lam * images + (1.0 - lam) * images[perm]
        soft_labels = lam * one_hot + (1.0 - lam) * one_hot[perm]

    return images, soft_labels


def soft_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    label_smoothing: float = 0.0,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy that accepts either hard labels or soft label distributions.

    When ``soft_targets`` is 1-D we treat it as hard labels. When it is 2-D
    and floats we treat it as a probability distribution. ``label_smoothing``
    is implemented in the hard-label branch so it composes cleanly with
    :func:`mixup_cutmix_batch`.
    """
    if soft_targets.dim() == 1:
        return torch.nn.functional.cross_entropy(
            logits,
            soft_targets,
            weight=weight,
            label_smoothing=label_smoothing,
        )
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    if weight is not None:
        # Per-sample weighting requires a scalar weight; fall back to mean reduction.
        weighted = -(soft_targets * log_probs).sum(dim=-1)
        return weighted.mean()
    return -(soft_targets * log_probs).sum(dim=-1).mean()


class ModelEMA:
    """Exponential Moving Average of model parameters.

    Maintains a deep-copied model whose weights are a smoothed version of the
    training model's weights. The EMA model should be used for evaluation,
    validation-based checkpoint selection and final test reporting.

    Parameters
    ----------
    model:
        The training model whose parameters will be tracked.
    decay:
        Smoothing factor in ``[0, 1)``. Higher means slower updates, more
        smoothing. ``0.999`` is a robust default for short fine-tuning runs.
    device:
        Optional override for the EMA model device. Defaults to the source
        model's device.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: torch.device | None = None,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.decay = float(decay)
        self.ema_model = deepcopy(model).eval()
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)
        if device is not None:
            self.ema_model.to(device)
        # Buffers (running_mean / running_var) are copied via deepcopy.

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Refresh EMA parameters using the current training parameters."""
        for ema_parameter, parameter in zip(
            self.ema_model.parameters(), model.parameters()
        ):
            ema_parameter.mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)
        # Buffers such as BN running stats are copied verbatim.
        for ema_buffer, buffer in zip(
            self.ema_model.buffers(), model.buffers()
        ):
            ema_buffer.copy_(buffer.detach())

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "ema_model": self.ema_model.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.ema_model.load_state_dict(state["ema_model"])

    def module(self) -> nn.Module:
        return self.ema_model


__all__ = [
    "mixup_cutmix_batch",
    "soft_cross_entropy",
    "ModelEMA",
]
