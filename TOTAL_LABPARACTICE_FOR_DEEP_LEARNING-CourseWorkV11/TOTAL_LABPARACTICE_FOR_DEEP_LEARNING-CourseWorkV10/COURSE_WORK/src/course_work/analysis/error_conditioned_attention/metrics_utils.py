"""Phase 56 - profile metrics (JSD, L1, Cosine, Wasserstein, Cliff's delta).

Mirrors Phase 55 conventions: natural-log JSD, Wasserstein in minutes,
Cliff's delta via fast O(N log N) sorting.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import wasserstein_distance, spearmanr


def _safe_log(x: np.ndarray) -> np.ndarray:
    """Safe natural log; masks x<=0 to 0 (since 0*log(0) := 0)."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x > 0, np.log(x), 0.0)


def jsd_natural_log(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence with natural log.

    p, q must be non-negative probability vectors summing to ~1.
    Returns scalar in [0, ln(2)].
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.where(p < 0, 0.0, p)
    q = np.where(q < 0, 0.0, q)
    s_p = p.sum()
    s_q = q.sum()
    if s_p > 0:
        p = p / s_p
    if s_q > 0:
        q = q / s_q
    m = 0.5 * (p + q)
    pm = np.where(p > 0, p * _safe_log(p / m), 0.0)
    qm = np.where(q > 0, q * _safe_log(q / m), 0.0)
    return float(0.5 * (pm.sum() + qm.sum()))


def cosine_similarity(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    np_ = np.linalg.norm(p)
    nq_ = np.linalg.norm(q)
    if np_ == 0 or nq_ == 0:
        return 0.0
    return float(np.dot(p, q) / (np_ * nq_))


def l1_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(np.abs(p - q).sum())


def wasserstein_minutes(p: np.ndarray, q: np.ndarray, support_minutes: np.ndarray) -> float:
    """Wasserstein-1 distance using lag minutes as support."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    s_p = p.sum()
    s_q = q.sum()
    if s_p > 0:
        p = p / s_p
    if s_q > 0:
        q = q / s_q
    return float(wasserstein_distance(support_minutes, support_minutes, p, q))



def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's delta via fast sorted rank-counting.

    delta = (#x > y + 0.5*#x == y - 0.5*#x > y - 0.5*#x == y - #x < y) / (n_x * n_y)
    Range: [-1, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    n_x = len(x)
    n_y = len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    a = np.concatenate([x, y])
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    a_sorted = a[order]
    n = len(a_sorted)
    i = 0
    while i < n:
        j = i
        while j < n and a_sorted[j] == a_sorted[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)  # 1-indexed
        ranks[order[i:j]] = avg_rank
        i = j

    R_y = ranks[n_x:].sum()
    delta = (2.0 * R_y - n_y * (n + 1)) / (n_x * n_y)
    return float(max(-1.0, min(1.0, delta)))


def spearman_rho_safe(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
    """Spearman rho with NOT_DEFINED detection for constant input."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return 0.0, "NOT_DEFINED_INSUFFICIENT_DATA"
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0, "NOT_DEFINED_CONSTANT_INPUT"
    rho, _ = spearmanr(x, y)
    if np.isnan(rho):
        return 0.0, "NOT_DEFINED"
    return float(rho), "OK"
