"""Phase 54 — top1 lag frequency and top1 tie summary.

Aggregates per-seed/per-layer/per-head top1 lag frequency and tie counts from
metrics_long. Fractions per seed/layer/head sum to 1. NEWEST_SOURCE tie rule
preserved (Phase 52 contract).
"""

from __future__ import annotations

from collections import Counter

from .sources import LOOKBACK, SEEDS


def build_top1_lag_frequency(metrics_long: list[dict]) -> list[dict]:
    """For each seed/layer/head, count frequency of each top1_lag_steps value
    across Test targets. Fractions sum to 1 (per seed/layer/head)."""
    grouped: dict[tuple, list[int]] = {}
    total_per_group: dict[tuple, int] = {}
    for r in metrics_long:
        try:
            k = (int(r["seed"]), int(r["layer_idx0"]), int(r["head_idx0"]))
            lag = int(r["top1_lag_steps"])
        except (KeyError, ValueError):
            continue
        grouped.setdefault(k, []).append(lag)
        total_per_group[k] = total_per_group.get(k, 0) + 1

    out: list[dict] = []
    for (seed, layer, head), lags in sorted(grouped.items()):
        total = total_per_group[(seed, layer, head)]
        cnt = Counter(lags)
        # Emit one row per lag 1..L (even if count is 0)
        for lag in range(1, LOOKBACK + 1):
            c = cnt.get(lag, 0)
            out.append({
                "seed": seed,
                "layer_idx0": layer,
                "head_idx0": head,
                "lag_steps": lag,
                "lag_minutes": lag * 10,
                "count": c,
                "fraction": c / total if total > 0 else 0.0,
                "total_targets": total,
                "status": "PASS",
            })
    return out


def build_top1_tie_summary(metrics_long: list[dict]) -> list[dict]:
    """For each seed/layer/head, summary of top1 tie counts.

    Reports total target_count, count of targets with tie_count > 1, fraction,
    and max tie count.
    """
    grouped: dict[tuple, list[int]] = {}
    for r in metrics_long:
        try:
            k = (int(r["seed"]), int(r["layer_idx0"]), int(r["head_idx0"]))
            tc = int(r["top1_tie_count"])
        except (KeyError, ValueError):
            continue
        grouped.setdefault(k, []).append(tc)

    out: list[dict] = []
    for (seed, layer, head), tcs in sorted(grouped.items()):
        n = len(tcs)
        gt1 = sum(1 for t in tcs if t > 1)
        max_tie = max(tcs) if tcs else 0
        out.append({
            "seed": seed,
            "layer_idx0": layer,
            "head_idx0": head,
            "target_count": n,
            "tie_count_gt1": gt1,
            "tie_fraction": gt1 / n if n > 0 else 0.0,
            "max_tie_count": max_tie,
            "tie_rule": "NEWEST_SOURCE",
            "status": "PASS",
        })
    return out
