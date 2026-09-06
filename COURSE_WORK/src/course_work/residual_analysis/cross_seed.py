from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from .magnitude_associations import pearson, spearman


SEED_PAIRS: tuple[tuple[str, str], ...] = (
    ("42", "123"),
    ("42", "2026"),
    ("123", "2026"),
)


def compute_pair_residual_agreement(
    residuals_a: np.ndarray,
    residuals_b: np.ndarray,
    seed_a: str,
    seed_b: str,
) -> dict[str, Any]:
    a = np.asarray(residuals_a, dtype=np.float64)
    b = np.asarray(residuals_b, dtype=np.float64)
    if a.size != b.size:
        raise ValueError(f"residual arrays must have equal length; got {a.size} vs {b.size}")
    return {
        "seed_a": seed_a,
        "seed_b": seed_b,
        "N": int(a.size),
        "pearson_residual_correlation": pearson(a, b),
        "spearman_residual_correlation": spearman(a, b),
        "mean_absolute_residual_difference": float(np.mean(np.abs(a - b))),
        "rmse_between_residuals": float(np.sqrt(np.mean((a - b) ** 2))),
        "max_absolute_residual_difference": float(np.max(np.abs(a - b))),
        "scope": "CROSS_SEED_RESIDUAL_AGREEMENT_DIAGNOSTIC",
        "purpose": "NOT model-performance comparison",
    }


def build_residual_vectors_by_target_id(
    long_table: list[dict[str, Any]],
    seed: str,
) -> dict[str, dict[str, float]]:
    """Return target_id -> {residual_wh, y_pred_wh, sign, y_true_wh} for one seed.
    Target IDs are aligned because all seeds share the same Test population."""
    return {
        r["target_id"]: {
            "residual_wh": float(r["residual_wh"]),
            "y_pred_wh": float(r["y_pred_wh"]),
            "sign": r["residual_sign"],
            "y_true_wh": float(r["y_true_wh"]),
            "target_timestamp": r["target_timestamp"],
        }
        for r in long_table
        if r["seed"] == seed
    }


def compute_cross_seed_sign_consensus(
    long_table: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute per-target sign consensus across 3 seeds."""
    seed_data = {
        seed: build_residual_vectors_by_target_id(long_table, seed) for seed in ("42", "123", "2026")
    }
    target_ids = sorted(seed_data["42"].keys())
    if (
        set(seed_data["123"].keys()) != set(target_ids)
        or set(seed_data["2026"].keys()) != set(target_ids)
    ):
        raise ValueError("Cross-seed target_id misalignment")

    consensus_counts: dict[str, int] = {
        "ALL_UNDER": 0,
        "ALL_OVER": 0,
        "ALL_EXACT": 0,
        "MIXED": 0,
        "TWO_UNDER_ONE_OVER": 0,
        "TWO_OVER_ONE_UNDER": 0,
    }
    per_target_class: list[dict[str, Any]] = []
    for tid in target_ids:
        s42 = seed_data["42"][tid]["sign"]
        s123 = seed_data["123"][tid]["sign"]
        s2026 = seed_data["2026"][tid]["sign"]
        u = sum(1 for s in (s42, s123, s2026) if s == "UNDERPREDICTION")
        o = sum(1 for s in (s42, s123, s2026) if s == "OVERPREDICTION")
        e = sum(1 for s in (s42, s123, s2026) if s == "EXACT")
        if u == 3:
            cls = "ALL_UNDER"
        elif o == 3:
            cls = "ALL_OVER"
        elif e == 3:
            cls = "ALL_EXACT"
        elif u == 2 and o == 1:
            cls = "TWO_UNDER_ONE_OVER"
        elif o == 2 and u == 1:
            cls = "TWO_OVER_ONE_UNDER"
        else:
            cls = "MIXED"
        consensus_counts[cls] += 1
        per_target_class.append(
            {
                "target_id": tid,
                "sign_seed42": s42,
                "sign_seed123": s123,
                "sign_seed2026": s2026,
                "consensus_class": cls,
            }
        )

    n = len(target_ids)
    consensus_rows: list[dict[str, Any]] = []
    for cls in (
        "ALL_UNDER",
        "ALL_OVER",
        "ALL_EXACT",
        "TWO_UNDER_ONE_OVER",
        "TWO_OVER_ONE_UNDER",
        "MIXED",
    ):
        consensus_rows.append(
            {
                "consensus_class": cls,
                "count": consensus_counts[cls],
                "fraction": consensus_counts[cls] / n if n > 0 else float("nan"),
            }
        )
    return {
        "n_targets": n,
        "rows": consensus_rows,
        "per_target_class": per_target_class,
        "fractions_sum_to_one": bool(
            abs(sum(r["fraction"] for r in consensus_rows) - 1.0) < 1e-12
        ),
        "zero_policy": "EXACT_ZERO",
        "epsilon_used": False,
    }
