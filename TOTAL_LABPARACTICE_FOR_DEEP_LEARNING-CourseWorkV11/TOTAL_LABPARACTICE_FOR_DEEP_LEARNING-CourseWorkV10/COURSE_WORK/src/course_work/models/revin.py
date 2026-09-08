"""
Phase 40 — S18 RevIN Adapter

Target-selective reversible adapter for multivariate sequence-to-one regression.
Applies per-window per-feature normalization across the time dimension only on
eligible signal channels; pass-through deterministic time features.

Frozen project contract:
- revin_enabled      = True (RN1)
- eps                = 1e-5
- affine             = True
- affine_weight_init = 1
- affine_bias_init   = 0
- centering          = mean
- variance           = population variance
- unbiased             = False
- statistics_detached= True
- running_stats      = False
- subtract_last      = False

External contract:
    forward(x) -> [B, 1]  y_model
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


REVIN_EPS = 1e-5
REVIN_AFFINE = True
AFFINE_WEIGHT_INIT = 1.0
AFFINE_BIAS_INIT = 0.0
TIME_FEATURE_NAMES = ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "weekend")


@dataclass(frozen=True)
class RevINScope:
    revin_channel_names: list[str]
    revin_channel_indices: list[int]
    passthrough_channel_names: list[str]
    passthrough_channel_indices: list[int]
    target_channel_name: str
    target_original_index: int
    target_revin_subset_index: int

    @property
    def revin_channel_count(self) -> int:
        return len(self.revin_channel_names)


class TargetSelectiveRevIN(nn.Module):
    """Per-window per-channel instance normalization with target-selective
    reversible denormalization.

    RN1 contract:
        z = (x - mu) / sigma
        y_norm = z * gamma + beta
        [time features pass through unchanged]
    """

    def __init__(
        self,
        revin_indices: list[int],
        passthrough_indices: list[int],
        target_revin_subset_index: int,
        eps: float = REVIN_EPS,
        affine: bool = REVIN_AFFINE,
        affine_weight_init: float = AFFINE_WEIGHT_INIT,
        affine_bias_init: float = AFFINE_BIAS_INIT,
    ) -> None:
        super().__init__()
        if eps <= 0 or not np.isfinite(eps):
            raise ValueError(f"RevIN eps must be positive finite, got {eps}")
        if not revin_indices:
            raise ValueError("RevIN eligible channel list is empty")
        if target_revin_subset_index < 0 or target_revin_subset_index >= len(revin_indices):
            raise ValueError("target_revin_subset_index out of range")

        self.revin_indices = list(revin_indices)
        self.passthrough_indices = list(passthrough_indices)
        self.target_revin_subset_index = int(target_revin_subset_index)
        self.eps = float(eps)
        self.affine = bool(affine)
        self._total_channels = len(self.revin_indices) + len(self.passthrough_indices)

        if self.affine:
            self.gamma = nn.Parameter(
                torch.full((len(self.revin_indices),), float(affine_weight_init))
            )
            self.beta = nn.Parameter(
                torch.full((len(self.revin_indices),), float(affine_bias_init))
            )

    def normalize(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Apply normalization.  Returns normalized tensor + forward-local context.

        x: [B, L, F]
        Returns: x_norm [B, L, F]  with same feature order, context dict with
            detached mean/stdev/target_subset_idx.
        """
        if x.ndim != 3:
            raise ValueError(f"x must be [B, L, F], got shape {tuple(x.shape)}")
        batch, lookback, feat = x.shape
        if feat != self._total_channels:
            raise ValueError(
                f"Feature dim mismatch: x has {feat}, expected {self._total_channels}"
            )

        x_rev = x[:, :, self.revin_indices]
        mu = x_rev.mean(dim=1, keepdim=True)
        var = ((x_rev - mu) ** 2).mean(dim=1, keepdim=True)
        sigma = torch.sqrt(var + self.eps)

        mu_det = mu.detach()
        sigma_det = sigma.detach()

        z_rev = (x_rev - mu_det) / sigma_det

        if self.affine:
            gamma = self.gamma.view(1, 1, -1)
            beta = self.beta.view(1, 1, -1)
            y_rev = z_rev * gamma + beta
        else:
            y_rev = z_rev

        x_norm = torch.empty_like(x)
        x_norm[:, :, self.revin_indices] = y_rev
        x_norm[:, :, self.passthrough_indices] = x[:, :, self.passthrough_indices]

        context = {
            "mean": mu_det,
            "stdev": sigma_det,
            "target_revin_subset_index": torch.tensor(
                self.target_revin_subset_index, dtype=torch.long
            ),
        }
        return x_norm, context

    def denormalize_target(self, pred_norm: Tensor, context: dict[str, Tensor]) -> Tensor:
        """Reverse normalization for target channel only.

        pred_norm: [B, output_size]
        Returns: pred_x_coord [B, output_size]
        """
        mean = context["mean"][:, 0, self.target_revin_subset_index].unsqueeze(-1)
        stdev = context["stdev"][:, 0, self.target_revin_subset_index].unsqueeze(-1)
        if self.affine:
            gamma = self.gamma[self.target_revin_subset_index]
            beta = self.beta[self.target_revin_subset_index]
        else:
            gamma = torch.tensor(1.0, device=pred_norm.device, dtype=pred_norm.dtype)
            beta = torch.tensor(0.0, device=pred_norm.device, dtype=pred_norm.dtype)

        z = (pred_norm - beta) / gamma
        x_target = z * stdev + mean
        return x_target

    def forward(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        return self.normalize(x)


class RevINWrappedTransformerRegressor(nn.Module):
    """Wraps the canonical TransformerRegressor with TargetSelectiveRevIN.

    External contract preserved:
        forward(x) -> [B, 1]  y_model
        forward_with_attention(x) -> (prediction [B,1], attention list)
    """

    def __init__(
        self,
        backbone: nn.Module,
        revin: TargetSelectiveRevIN,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.revin = revin

    def forward(self, x: Tensor) -> Tensor:
        x_norm, context = self.revin.normalize(x)
        pooled, _ = self.backbone._encode(x_norm, return_attention=False)
        pred_norm = self.backbone.head(pooled)
        pred_x_coord = self.revin.denormalize_target(pred_norm, context)
        return pred_x_coord

    def forward_with_attention(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        x_norm, context = self.revin.normalize(x)
        pooled, attention_maps = self.backbone._encode(x_norm, return_attention=True)
        pred_norm = self.backbone.head(pooled)
        pred_x_coord = self.revin.denormalize_target(pred_norm, context)
        return pred_x_coord, attention_maps


def resolve_revin_scope_from_feature_order(
    feature_order: list[str],
    target_channel_name: str = "Appliances",
    time_feature_names: tuple[str, ...] = TIME_FEATURE_NAMES,
) -> tuple[bool, RevINScope | None, str]:
    """Resolve RevIN channel scope from canonical feature order.

    Returns (success, scope_or_none, reason).
    Fails if target_channel_name absent OR present multiple times.
    """
    if feature_order is None:
        return False, None, "feature_order is None"

    target_count = sum(1 for n in feature_order if n == target_channel_name)
    if target_count == 0:
        return False, None, f"target_channel '{target_channel_name}' not in feature_order"
    if target_count > 1:
        raise ValueError(
            f"target_channel '{target_channel_name}' appears {target_count} times in feature_order"
        )

    target_idx = feature_order.index(target_channel_name)

    time_set = set(time_feature_names)
    passthrough_names: list[str] = []
    passthrough_indices: list[int] = []
    revin_names: list[str] = []
    revin_indices: list[int] = []
    for i, name in enumerate(feature_order):
        if name in time_set:
            passthrough_names.append(name)
            passthrough_indices.append(i)
        else:
            revin_names.append(name)
            revin_indices.append(i)

    if target_channel_name not in revin_names:
        raise ValueError(
            f"target '{target_channel_name}' must be in eligible RevIN channel set, "
            f"but ended up in passthrough"
        )

    target_subset_idx = revin_names.index(target_channel_name)

    scope = RevINScope(
        revin_channel_names=revin_names,
        revin_channel_indices=revin_indices,
        passthrough_channel_names=passthrough_names,
        passthrough_channel_indices=passthrough_indices,
        target_channel_name=target_channel_name,
        target_original_index=target_idx,
        target_revin_subset_index=target_subset_idx,
    )
    return True, scope, "OK"


def build_revin_wrapped_model(
    revin_scope: RevINScope,
    backbone: nn.Module,
    eps: float = REVIN_EPS,
    affine_weight_init: float = AFFINE_WEIGHT_INIT,
    affine_bias_init: float = AFFINE_BIAS_INIT,
) -> RevINWrappedTransformerRegressor:
    revin = TargetSelectiveRevIN(
        revin_indices=revin_scope.revin_channel_indices,
        passthrough_indices=revin_scope.passthrough_channel_indices,
        target_revin_subset_index=revin_scope.target_revin_subset_index,
        eps=eps,
        affine=REVIN_AFFINE,
        affine_weight_init=affine_weight_init,
        affine_bias_init=affine_bias_init,
    )
    return RevINWrappedTransformerRegressor(backbone=backbone, revin=revin)