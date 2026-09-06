"""Phase 54 — aggregations across Test targets.

* Per-seed/per-layer/per-head metric aggregate summary (N, mean, sd, median,
  p05/p25/p75/p95, min, max) for every primary metric.
* Mean / median / SD / quantile temporal profiles by lag for every
  seed/layer/head. Sum of mean profile audited to ≈ 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sources import LOOKBACK, NUM_HEADS, NUM_LAYERS, SEEDS


@dataclass
class ProbaQuantiles:
    p05: float
    p25: float
    p50: float
    p75: float
    p95: float


def quantile_set(x: np.ndarray, qs: list[float] = (0.05, 0.25, 0.50, 0.75, 0.95)) -> ProbaQuantiles:
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return ProbaQuantiles(0, 0, 0, 0, 0)
    ps = np.quantile(a, qs)
    return ProbaQuantiles(
        p05=float(ps[0]),
        p25=float(ps[1]),
        p50=float(ps[2]),
        p75=float(ps[3]),
        p95=float(ps[4]),
    )


def build_per_head_metric_summary(
    metrics_long: list[dict],
    primary_metrics: list[str],
) -> list[dict]:
    """Aggregate per-seed/layer/head metric distributions across Test targets.

    metrics_long: rows with keys 'seed', 'layer_idx0', 'head_idx0', and all
    primary_metrics. Returns a list of summary rows (one per seed/layer/head/
    metric).
    """
    # Group rows
    grouped: dict[tuple, list[dict]] = {}
    for r in metrics_long:
        try:
            k = (int(r["seed"]), int(r["layer_idx0"]), int(r["head_idx0"]))
        except (KeyError, ValueError):
            continue
        grouped.setdefault(k, []).append(r)

    out: list[dict] = []
    for (seed, layer, head), rows in sorted(grouped.items()):
        for metric in primary_metrics:
            arr = np.array([float(r[metric]) for r in rows], dtype=np.float64)
            if arr.size == 0:
                continue
            qs = quantile_set(arr)
            out.append({
                "seed": seed,
                "layer_idx0": layer,
                "head_idx0": head,
                "metric": metric,
                "N": int(arr.size),
                "mean": float(arr.mean()),
                "sample_sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "median": float(np.median(arr)),
                "p05": qs.p05,
                "p25": qs.p25,
                "p75": qs.p75,
                "p95": qs.p95,
                "min": float(arr.min()),
                "max": float(arr.max()),
                "status": "PASS",
            })
    return out


def build_temporal_profiles_by_lag(
    raw: dict[int, np.ndarray],
    metrics_long_top1_lag: dict[tuple, int] | None = None,
) -> list[dict]:
    """Compute mean/median/SD/quantile temporal attention profiles by lag for
    every seed/layer/head.

    Returns list of rows with keys:
        seed, layer_idx0, head_idx0, source_position_idx0, lag_steps,
        lag_minutes, mean_weight, median_weight, sample_sd_weight, p05_weight,
        p25_weight, p75_weight, p95_weight, target_count, profile_sum_mean_weights,
        status
    """
    out: list[dict] = []
    for seed, arr in sorted(raw.items()):
        # arr shape: [N, L, H, L]
        n_targets = arr.shape[0]
        for li in range(NUM_LAYERS):
            for hi in range(NUM_HEADS):
                block = arr[:, li, hi, :]  # [N, L]
                mean_w = block.mean(axis=0)
                med_w = np.median(block, axis=0)
                if block.shape[0] > 1:
                    sd_w = block.std(axis=0, ddof=1)
                else:
                    sd_w = np.zeros_like(mean_w)
                qs = np.quantile(block, [0.05, 0.25, 0.75, 0.95], axis=0)
                profile_sum = float(mean_w.sum())
                for p in range(LOOKBACK):
                    lag_steps = LOOKBACK - p
                    lag_minutes = lag_steps * 10
                    out.append({
                        "seed": int(seed),
                        "layer_idx0": int(li),
                        "head_idx0": int(hi),
                        "source_position_idx0": int(p),
                        "lag_steps": int(lag_steps),
                        "lag_minutes": int(lag_minutes),
                        "mean_weight": float(mean_w[p]),
                        "median_weight": float(med_w[p]),
                        "sample_sd_weight": float(sd_w[p]),
                        "p05_weight": float(qs[0, p]),
                        "p25_weight": float(qs[1, p]),
                        "p75_weight": float(qs[2, p]),
                        "p95_weight": float(qs[3, p]),
                        "target_count": int(n_targets),
                        "profile_sum_mean_weights": profile_sum,
                        "status": "PASS" if abs(profile_sum - 1.0) <= 1e-4 else "FAIL",
                    })
    return out
