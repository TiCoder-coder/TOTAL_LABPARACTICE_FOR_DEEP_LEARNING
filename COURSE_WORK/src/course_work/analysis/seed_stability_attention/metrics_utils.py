"""Phase 57 - metrics utilities (probability-profile and distance metrics).

Implements JSD (natural log), L1, L2, cosine, Pearson, Spearman, Wasserstein
(in minutes), Cliff's delta. All pure functions. No training, no model.

Conventions:
- `p` and `q` are probability vectors that sum to ~1.
- Wasserstein operates on lag-minutes support as provided.
- Spearman handles constant inputs by returning (0.0, "UNDEFINED").
- Cliff's delta = P(x>q) - P(x<q); range [-1, 1].
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import spearmanr, wasserstein_distance


def jsd_natural_log(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence using natural log. Range [0, ln(2)]."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    # Defensive renormalization (upstream should already be normalized)
    p = p / max(p.sum(), 1e-300)
    q = q / max(q.sum(), 1e-300)
    p = np.clip(p, 1e-300, 1.0)
    q = np.clip(q, 1e-300, 1.0)
    m = 0.5 * (p + q)
    # natural log JSD
    jsd = 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))
    return float(max(jsd, 0.0))


def l1_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(np.sum(np.abs(p - q)))


def l2_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(np.sqrt(np.sum((p - q) ** 2)))


def cosine_similarity(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    pn = np.linalg.norm(p)
    qn = np.linalg.norm(q)
    if pn == 0 or qn == 0:
        return 0.0
    return float(np.dot(p, q) / (pn * qn))


def pearson_corr(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if len(p) < 2:
        return 0.0
    if np.std(p) == 0 or np.std(q) == 0:
        return 0.0
    return float(np.corrcoef(p, q)[0, 1])


def spearman_rho_safe(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return (0.0, "UNDEFINED")
    if np.std(x) == 0 or np.std(y) == 0:
        return (0.0, "UNDEFINED")
    rho, _ = spearmanr(x, y)
    if np.isnan(rho):
        return (0.0, "UNDEFINED")
    return (float(rho), "OK")


def wasserstein_minutes(p: np.ndarray, q: np.ndarray, support_minutes: np.ndarray) -> float:
    """Wasserstein-1 distance on a known support, expressed in minutes."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    s = np.asarray(support_minutes, dtype=np.float64)
    # scipy wasserstein_distance takes values + weights; we want 1-D Wasserstein
    # between two discrete distributions on the same support.
    if len(s) != len(p) or len(s) != len(q):
        raise ValueError("support length must match distribution length")
    ps = p / max(p.sum(), 1e-300)
    qs = q / max(q.sum(), 1e-300)
    return float(wasserstein_distance(s, s, u_weights=ps, v_weights=qs))


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) == 0 or len(y) == 0:
        return 0.0
    gt = 0
    lt = 0
    for xi in x:
        gt += int(np.sum(xi > y))
        lt += int(np.sum(xi < y))
    denom = float(len(x) * len(y))
    if denom == 0:
        return 0.0
    return float((gt - lt) / denom)


def profile_sum(p: np.ndarray) -> float:
    return float(np.sum(p))


def verify_probability_profile(p: np.ndarray, atol: float = 1e-5) -> Tuple[bool, float]:
    p = np.asarray(p, dtype=np.float64)
    s = float(np.sum(p))
    is_nonneg = bool(np.all(p >= 0))
    finite = bool(np.all(np.isfinite(p)))
    ok = abs(s - 1.0) <= atol and is_nonneg and finite
    return ok, s


def hash_vector(v: np.ndarray) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(np.asarray(v, dtype=np.float64).tobytes())
    return h.hexdigest()
