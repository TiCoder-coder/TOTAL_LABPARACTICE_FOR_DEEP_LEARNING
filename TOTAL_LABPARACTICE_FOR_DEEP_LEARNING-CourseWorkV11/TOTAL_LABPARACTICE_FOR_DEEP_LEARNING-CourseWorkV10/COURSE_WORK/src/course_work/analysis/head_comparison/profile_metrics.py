"""Phase 55 - pairwise profile similarity and distance metrics.

All metrics computed on the same frozen lag_minutes support (72, 10..720).

Canonical Phase 55 §14, §15-§17, §72-§74.

Phase 55 does NOT silently renormalize profiles; profiles are required to sum
to 1 within the frozen tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .sources import LAG_MINUTES, JS_METRIC_BOUNDS


# ---------------------------------------------------------------------------
# Math primitives
# ---------------------------------------------------------------------------

def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Natural log with epsilon safe for zero."""
    return np.log(np.maximum(np.asarray(x, dtype=np.float64), eps))


def pearson_similarity(p: np.ndarray, q: np.ndarray) -> float:
    """Pearson correlation between two vectors (linear shape association)."""
    if p.size < 2:
        return float("nan")
    pv = p - p.mean()
    qv = q - q.mean()
    denom = float(np.linalg.norm(pv) * np.linalg.norm(qv))
    if denom < 1e-15:
        return float("nan")
    return float(np.dot(pv, qv) / denom)


def spearman_similarity(p: np.ndarray, q: np.ndarray) -> float:
    """Spearman rank correlation (monotonic similarity)."""
    if p.size < 2:
        return float("nan")
    res = stats.spearmanr(p, q)
    return float(res.statistic)


def cosine_similarity(p: np.ndarray, q: np.ndarray) -> float:
    """Cosine similarity in [0, 1] for nonnegative vectors."""
    denom = float(np.linalg.norm(p) * np.linalg.norm(q))
    if denom < 1e-15:
        return float("nan")
    return float(np.dot(p, q) / denom)


def jsd_natural_log(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence with natural log; bounded in [0, ln(2)]."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.size == 0 or q.size == 0:
        return float("nan")
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum <= 0 or q_sum <= 0:
        return float("nan")
    p_n = p / p_sum
    q_n = q / q_sum
    m = 0.5 * (p_n + q_n)
    # Mask p=0 and q=0 terms to avoid 0*log(0) and 0/0
    pmask = p_n > 0
    qmask = q_n > 0
    kl_pm = float(np.sum(p_n[pmask] * (np.log(p_n[pmask]) - np.log(m[pmask]))))
    kl_qm = float(np.sum(q_n[qmask] * (np.log(q_n[qmask]) - np.log(m[qmask]))))
    return 0.5 * (kl_pm + kl_qm)


def l1_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.abs(p - q)))


def l2_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(np.sum((p - q) ** 2)))


def wasserstein_1d_minutes(p: np.ndarray, q: np.ndarray, support_minutes: np.ndarray) -> float:
    """Wasserstein-1 distance with cumulative-distribution on support minutes.

    Using the standard 1D Wasserstein-1:
        W1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
    For discrete supports with weights w_i, the closed form over the CDF
    differences along the sorted support equals the area between CDFs.
    """
    if p.size == 0 or q.size == 0:
        return float("nan")
    if p.size != support_minutes.size or q.size != support_minutes.size:
        raise ValueError("support_minutes size mismatch with profile")
    p_sum = float(p.sum())
    q_sum = float(q.sum())
    if p_sum <= 0 or q_sum <= 0:
        return float("nan")
    p_n = p / p_sum
    q_n = q / q_sum
    cdf_p = np.cumsum(p_n)
    cdf_q = np.cumsum(q_n)
    # Support sorted newest->oldest: LAG_MINUTES = (LOOKBACK - arange)*10 -> desc.
    # Take absolute value of support spacing; use absolute delta in minutes.
    diff = np.abs(cdf_p - cdf_q)
    # Trapezoidal integration over sorted support minutes (descending).
    sorted_idx = np.argsort(support_minutes)
    x_sorted = support_minutes[sorted_idx]
    diff_sorted = diff[sorted_idx]
    return float(np.trapezoid(diff_sorted, x_sorted))


def top1_tvd(a: np.ndarray, b: np.ndarray) -> float:
    """Total Variation Distance for two probability distributions on identical support."""
    if a.size != b.size or a.size == 0:
        return float("nan")
    a_sum = a.sum()
    b_sum = b.sum()
    if a_sum <= 0 or b_sum <= 0:
        return float("nan")
    a_n = a / a_sum
    b_n = b / b_sum
    return float(0.5 * np.sum(np.abs(a_n - b_n)))


# ---------------------------------------------------------------------------
# Convenience container
# ---------------------------------------------------------------------------

@dataclass
class PairProfileMetrics:
    seed: int
    layer_idx0: int
    head_a_idx0: int
    head_b_idx0: int
    head_a_display: str
    head_b_display: str
    pearson_profile: float
    spearman_profile: float
    cosine_profile: float
    jsd_profile: float
    l1_profile: float
    l2_profile: float
    wasserstein_minutes: float


def compute_pair_profile_metrics(
    seed: int,
    layer_idx0: int,
    head_a_idx0: int,
    head_b_idx0: int,
    profile_a: np.ndarray,
    profile_b: np.ndarray,
    support_minutes: np.ndarray,
) -> PairProfileMetrics:
    return PairProfileMetrics(
        seed=seed,
        layer_idx0=layer_idx0,
        head_a_idx0=head_a_idx0,
        head_b_idx0=head_b_idx0,
        head_a_display=f"H{head_a_idx0 + 1}",
        head_b_display=f"H{head_b_idx0 + 1}",
        pearson_profile=pearson_similarity(profile_a, profile_b),
        spearman_profile=spearman_similarity(profile_a, profile_b),
        cosine_profile=cosine_similarity(profile_a, profile_b),
        jsd_profile=jsd_natural_log(profile_a, profile_b),
        l1_profile=l1_distance(profile_a, profile_b),
        l2_profile=l2_distance(profile_a, profile_b),
        wasserstein_minutes=wasserstein_1d_minutes(profile_a, profile_b, support_minutes),
    )


def validate_profile(p: np.ndarray, tol: float) -> tuple[bool, float]:
    """Returns (is_valid, sum)."""
    if p.size == 0:
        return (False, 0.0)
    s = float(p.sum())
    return (abs(s - 1.0) < tol, s)
