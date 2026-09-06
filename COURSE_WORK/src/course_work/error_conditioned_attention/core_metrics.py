"""Phase 56 - core attention metrics v1 computation.

Implements the 6 CORE_ATTENTION_METRICS-v1 metrics:
- normalized_entropy
- expected_lag_minutes
- recent_1h_mass
- recent_6h_mass
- top5_mass
- lag80_minutes

Used both for verifying Phase 54 values and for recomputing metrics on
layer head-mean vectors (per the plan: mean vector FIRST, then metrics).
"""

from __future__ import annotations

import numpy as np


EPSILON_H = 1e-12
CADENCE_MINUTES = 10
LOOKBACK = 72

# Lag support in minutes (recency-ordered: pos L-1 -> 10 min, pos 0 -> 720 min)
LAG_MINUTES = (LOOKBACK - np.arange(LOOKBACK)) * CADENCE_MINUTES


def _normalize(a: np.ndarray) -> np.ndarray:
    s = float(a.sum())
    if s > 0:
        return a.astype(np.float64) / s
    return a.astype(np.float64)


def normalized_entropy(a: np.ndarray) -> float:
    p = _normalize(a)
    L = len(p)
    if L <= 1:
        return 0.0
    q = np.clip(p, EPSILON_H, 1.0)
    H = float(-np.sum(q * np.log(q)))
    return H / float(np.log(L))


def expected_lag_minutes(a: np.ndarray) -> float:
    p = _normalize(a)
    L = len(p)
    # lag_steps_p = L - position_idx
    lag_steps = np.array([L - idx for idx in range(L)], dtype=np.float64)
    return float(np.sum(p * lag_steps)) * CADENCE_MINUTES


def recent_1h_mass(a: np.ndarray) -> float:
    """1h = 6 steps at 10-min cadence."""
    L = len(a)
    eff = min(6, L)
    return float(_normalize(a)[L - eff:].sum())


def recent_6h_mass(a: np.ndarray) -> float:
    """6h = 36 steps."""
    L = len(a)
    eff = min(36, L)
    return float(_normalize(a)[L - eff:].sum())


def top5_mass(a: np.ndarray, k: int = 5) -> float:
    arr = np.asarray(a, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    k = min(k, arr.size)
    return float(np.partition(arr, -k)[-k:].sum())


def lag80_minutes(a: np.ndarray) -> float:
    """Lag80: smallest lag (in minutes) such that cumulative mass >= 80%."""
    p = _normalize(a)
    L = len(p)
    if p.sum() <= 0:
        return 0.0
    # Cumulative mass from newest to oldest (i.e. position L-1, L-2, ..., 0)
    # Lag is the lag value at the 80% threshold
    cum = 0.0
    for i in range(L - 1, -1, -1):
        cum += p[i]
        if cum >= 0.80:
            lag_steps = L - i
            return float(lag_steps) * CADENCE_MINUTES
    # If never reaches 80% (rare), return max lag
    return float(L) * CADENCE_MINUTES


METRIC_FUNCS = {
    "normalized_entropy": normalized_entropy,
    "expected_lag_minutes": expected_lag_minutes,
    "recent_1h_mass": recent_1h_mass,
    "recent_6h_mass": recent_6h_mass,
    "top5_mass": top5_mass,
    "lag80_minutes": lag80_minutes,
}


def compute_core_metrics_vector(a: np.ndarray) -> dict[str, float]:
    """Compute all 6 CORE_ATTENTION_METRICS-v1 on a single attention vector."""
    return {name: float(fn(a)) for name, fn in METRIC_FUNCS.items()}
