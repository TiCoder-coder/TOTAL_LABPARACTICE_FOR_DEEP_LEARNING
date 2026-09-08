"""Phase 54 — layer head-mean profile and seed overall profile.

* Layer head-mean profile: for each seed/layer/lag, head_mean_weight = mean over
  heads of mean_weight (which is equivalent to mean over all targets and heads
  for that lag). Permutation-invariant to head ordering.
* Seed overall profile: for each seed/lag, mean_weight = mean over (layer, head,
  target) for that lag. Descriptive only.
"""

from __future__ import annotations

import numpy as np

from .sources import LOOKBACK, NUM_HEADS, NUM_LAYERS, SEEDS


def build_layer_head_mean_profile(raw: dict[int, np.ndarray]) -> list[dict]:
    """Compute layer-level head-mean profile per seed/layer/lag."""
    out: list[dict] = []
    for seed, arr in sorted(raw.items()):
        # arr: [N, L_layers, H_heads, L]
        # mean over targets: [L_layers, H_heads, L]
        mean_block = arr.mean(axis=0)
        # head mean over heads: [L_layers, L]
        head_mean = mean_block.mean(axis=1)
        for li in range(NUM_LAYERS):
            profile_sum = float(head_mean[li].sum())
            for p in range(LOOKBACK):
                lag_steps = LOOKBACK - p
                lag_minutes = lag_steps * 10
                out.append({
                    "seed": int(seed),
                    "layer_idx0": int(li),
                    "lag_steps": int(lag_steps),
                    "lag_minutes": int(lag_minutes),
                    "head_mean_weight": float(head_mean[li, p]),
                    "profile_sum": profile_sum,
                    "status": "PASS" if abs(profile_sum - 1.0) <= 1e-4 else "FAIL",
                })
    return out


def build_seed_overall_profile(raw: dict[int, np.ndarray]) -> list[dict]:
    """Compute seed overall descriptive profile per seed/lag.

    mean_weight_across_layers_heads_targets = mean over (L, H, N) for that lag.
    """
    out: list[dict] = []
    for seed, arr in sorted(raw.items()):
        # arr: [N, L, H, L]
        # mean over (N, L, H) -> per lag scalar
        mean_per_lag = arr.mean(axis=(0, 1, 2))
        profile_sum = float(mean_per_lag.sum())
        for p in range(LOOKBACK):
            lag_steps = LOOKBACK - p
            lag_minutes = lag_steps * 10
            out.append({
                "seed": int(seed),
                "lag_steps": int(lag_steps),
                "lag_minutes": int(lag_minutes),
                "mean_weight_across_layers_heads_targets": float(mean_per_lag[p]),
                "profile_sum": profile_sum,
                "status": "PASS" if abs(profile_sum - 1.0) <= 1e-4 else "FAIL",
            })
    return out
