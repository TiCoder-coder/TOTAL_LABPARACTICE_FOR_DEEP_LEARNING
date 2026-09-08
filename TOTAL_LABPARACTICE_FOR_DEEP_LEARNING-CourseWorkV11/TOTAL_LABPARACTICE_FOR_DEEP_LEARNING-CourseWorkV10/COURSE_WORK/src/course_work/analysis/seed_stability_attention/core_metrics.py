"""Phase 57 - core attention metric implementations (CORE_ATTENTION_METRICS-v1).

Reuses the canonical Phase 56 metric set:
  normalized_entropy, expected_lag_minutes, recent_1h_mass, recent_6h_mass,
  top5_mass, lag80_minutes.

These functions take a single last-query attention vector (sum ~= 1, lookback=72)
and return a dict of metric values.
"""

from __future__ import annotations

import numpy as np


LOOKBACK = 72
LAG_STEP_MINUTES = 10
LAG_MINUTES = (LOOKBACK - np.arange(LOOKBACK)) * LAG_STEP_MINUTES
EPS = 1e-12


def _normalize(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    s = float(a.sum())
    if s <= 0:
        return a
    return a / s


def normalized_entropy(a: np.ndarray) -> float:
    a = _normalize(a)
    H = -float(np.sum(a * np.log(np.clip(a, EPS, None))))
    Hmax = float(np.log(len(a)))
    if Hmax <= 0:
        return 0.0
    return H / Hmax


def expected_lag_minutes(a: np.ndarray) -> float:
    a = _normalize(a)
    return float(np.sum(a * LAG_MINUTES))


def recent_1h_mass(a: np.ndarray) -> float:
    """Lags <= 60 minutes = positions where LAG_MINUTES <= 60 (newest 6 positions)."""
    a = _normalize(a)
    mask = LAG_MINUTES <= 60
    return float(np.sum(a[mask]))


def recent_6h_mass(a: np.ndarray) -> float:
    a = _normalize(a)
    mask = LAG_MINUTES <= 360
    return float(np.sum(a[mask]))


def top5_mass(a: np.ndarray, k: int = 5) -> float:
    a = _normalize(a)
    return float(np.sum(np.sort(a)[::-1][:k]))


def lag80_minutes(a: np.ndarray) -> float:
    """Lag (in minutes) at which cumulative mass first exceeds 0.80.
    Returns 720 if never reaches 0.80 (entire lookback consumed)."""
    a = _normalize(a)
    cs = np.cumsum(a)
    idx = int(np.searchsorted(cs, 0.80))
    if idx >= len(cs):
        return float(LAG_MINUTES[0])  # = 720
    return float(LAG_MINUTES[idx])


METRIC_FUNCS = {
    "normalized_entropy": normalized_entropy,
    "expected_lag_minutes": expected_lag_minutes,
    "recent_1h_mass": recent_1h_mass,
    "recent_6h_mass": recent_6h_mass,
    "top5_mass": top5_mass,
    "lag80_minutes": lag80_minutes,
}


def compute_core_metrics_vector(a: np.ndarray) -> dict:
    """Return {metric_name: value} for the 6 CORE_ATTENTION_METRICS-v1."""
    return {name: float(fn(a)) for name, fn in METRIC_FUNCS.items()}
