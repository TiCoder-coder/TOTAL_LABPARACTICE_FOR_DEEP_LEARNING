"""Phase 54 — per-vector metrics.

Implements all per-vector last-query metrics:
* entropy (Shannon, with epsilon_H)
* normalized entropy
* effective source count
* expected lag steps / minutes
* lag SD steps / minutes
* top1 lag / weight / tie_count (NEWEST_SOURCE tie rule)
* top5 mass
* recent 1h/6h/12h/24h mass (with truncation flag)

Inputs: a single last-query vector of shape (L,) and the lag_steps array.
All metrics are deterministic and use frozen Phase 52 tolerances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sources import CADENCE_MINUTES, EPSILON_H, RECENT_WINDOWS_STEPS


@dataclass
class VectorMetrics:
    entropy: float
    normalized_entropy: float
    effective_source_count: float
    expected_lag_steps: float
    expected_lag_minutes: float
    lag_sd_steps: float
    lag_sd_minutes: float
    top1_source_position_idx0: int
    top1_lag_steps: int
    top1_lag_minutes: int
    top1_weight: float
    top1_tie_count: int
    top5_mass: float
    recent_1h_mass: float
    recent_1h_effective_steps: int
    recent_1h_coverage_truncated: bool
    recent_6h_mass: float
    recent_6h_effective_steps: int
    recent_6h_coverage_truncated: bool
    recent_12h_mass: float
    recent_12h_effective_steps: int
    recent_12h_coverage_truncated: bool
    recent_24h_mass: float
    recent_24h_effective_steps: int
    recent_24h_coverage_truncated: bool


def shannon_entropy(a: np.ndarray, eps: float = EPSILON_H) -> float:
    """Shannon entropy in nats. Returns 0 if vector is empty/zero."""
    p = np.asarray(a, dtype=np.float64)
    s = float(p.sum())
    if s <= 0:
        return 0.0
    q = p / s
    q = np.clip(q, eps, 1.0)
    return float(-np.sum(q * np.log(q)))


def normalized_entropy(a: np.ndarray, eps: float = EPSILON_H) -> float:
    H = shannon_entropy(a, eps)
    L = a.shape[0]
    if L <= 1:
        return 0.0
    return float(H / np.log(L))


def effective_source_count(a: np.ndarray) -> float:
    """N_eff = exp(H)."""
    H = shannon_entropy(a)
    return float(np.exp(H))


def compute_top1_with_tie_rule(a: np.ndarray) -> tuple[int, float, int]:
    """Top1 source position (idx0), weight, and tie count.

    Tie rule: NEWEST_SOURCE. Among exact maxima (within machine epsilon),
    choose the position with the LARGEST INDEX (newest).

    Uses np.isclose for tie detection to match Phase 52 exactly.
    """
    arr = np.asarray(a, dtype=np.float64)
    max_val = float(arr.max())
    if max_val <= 0:
        return int(arr.argmax()), 0.0, int((arr == max_val).sum())
    # Use isclose for tie detection (matches Phase 52)
    tied = np.flatnonzero(np.isclose(arr, max_val))
    n_ties = int(tied.size)
    chosen = int(tied.max())
    return chosen, float(arr[chosen]), n_ties


def compute_top5_mass(a: np.ndarray, k: int = 5) -> float:
    """Sum of K largest values."""
    arr = np.asarray(a, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    k = min(k, arr.size)
    # Use partition for O(N)
    partitioned = np.partition(arr, -k)[-k:]
    return float(partitioned.sum())


def expected_lag_steps(a: np.ndarray, lag_steps: np.ndarray) -> float:
    p = np.asarray(a, dtype=np.float64)
    ls = np.asarray(lag_steps, dtype=np.float64)
    s = float(p.sum())
    if s <= 0:
        return 0.0
    return float(np.sum((p / s) * ls))


def lag_sd_steps(a: np.ndarray, lag_steps: np.ndarray) -> float:
    p = np.asarray(a, dtype=np.float64)
    ls = np.asarray(lag_steps, dtype=np.float64)
    s = float(p.sum())
    if s <= 0:
        return 0.0
    q = p / s
    mu = float(np.sum(q * ls))
    var = float(np.sum(q * (ls - mu) ** 2))
    if var < 0:
        return 0.0
    return float(np.sqrt(var))


def compute_recent_masses(a: np.ndarray, L: int) -> dict[str, tuple[float, int, bool]]:
    """Compute recent-window masses for {1h,6h,12h,24h}.

    Returns dict window_name -> (mass, effective_steps, truncated).
    """
    p = np.asarray(a, dtype=np.float64)
    s = float(p.sum())
    if s > 0:
        p = p / s
    # Lag steps array (raw positions p_index -> lag_steps = L - p_index)
    lag_steps_p = np.array([L - idx for idx in range(L)], dtype=np.int64)
    out: dict[str, tuple[float, int, bool]] = {}
    for win_name, requested_steps in RECENT_WINDOWS_STEPS.items():
        eff = min(requested_steps, L)
        truncated = eff < requested_steps
        # Source positions where lag_steps_p <= eff
        # Equivalently: positions p where (L - p) <= eff, i.e. p >= L - eff
        if eff <= 0:
            mass = 0.0
        else:
            idx_start = L - eff  # inclusive lower bound on positions
            mass = float(p[idx_start:].sum())
        out[win_name] = (mass, eff, truncated)
    return out


def compute_vector_metrics(a: np.ndarray, lag_steps: np.ndarray) -> VectorMetrics:
    """Compute all per-vector metrics in one call."""
    L = a.shape[0]
    H = shannon_entropy(a)
    H_norm = normalized_entropy(a)  # CORRECTED (v2): no lag_steps arg
    N_eff = effective_source_count(a)
    E_lag_s = expected_lag_steps(a, lag_steps)
    E_lag_m = E_lag_s * CADENCE_MINUTES
    sd_s = lag_sd_steps(a, lag_steps)
    sd_m = sd_s * CADENCE_MINUTES
    top1_pos, top1_w, top1_tie = compute_top1_with_tie_rule(a)
    # top1 lag
    top1_lag_s = int(L - top1_pos)  # lag_steps_p = L - p
    top1_lag_m = top1_lag_s * CADENCE_MINUTES
    top5 = compute_top5_mass(a, k=5)
    recents = compute_recent_masses(a, L)

    return VectorMetrics(
        entropy=H,
        normalized_entropy=H_norm,
        effective_source_count=N_eff,
        expected_lag_steps=E_lag_s,
        expected_lag_minutes=E_lag_m,
        lag_sd_steps=sd_s,
        lag_sd_minutes=sd_m,
        top1_source_position_idx0=top1_pos,
        top1_lag_steps=top1_lag_s,
        top1_lag_minutes=top1_lag_m,
        top1_weight=top1_w,
        top1_tie_count=top1_tie,
        top5_mass=top5,
        recent_1h_mass=recents["1h"][0],
        recent_1h_effective_steps=recents["1h"][1],
        recent_1h_coverage_truncated=recents["1h"][2],
        recent_6h_mass=recents["6h"][0],
        recent_6h_effective_steps=recents["6h"][1],
        recent_6h_coverage_truncated=recents["6h"][2],
        recent_12h_mass=recents["12h"][0],
        recent_12h_effective_steps=recents["12h"][1],
        recent_12h_coverage_truncated=recents["12h"][2],
        recent_24h_mass=recents["24h"][0],
        recent_24h_effective_steps=recents["24h"][1],
        recent_24h_coverage_truncated=recents["24h"][2],
    )
