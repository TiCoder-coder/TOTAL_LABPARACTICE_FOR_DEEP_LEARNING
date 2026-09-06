"""Phase 55 - square pairwise similarity / distance matrices (architectural order).

Phase 55 §32-§33, §66-§67: matrices in architectural head order with audits.

Canonical metric matrices:
  - JSD (diagonal=0, symmetric, [0, ln2])
  - Cosine (diagonal=1, symmetric, [0, 1])
  - Pearson (diagonal=1, symmetric, [-1, 1])
  - Spearman (diagonal=1, symmetric, [-1, 1])
  - L1 (diagonal=0, symmetric, >=0)
  - Wasserstein (diagonal=0, symmetric, units=minutes)
  - Expected-lag absolute difference (diagonal=0, symmetric, >=0)
  - Recent1h absolute difference (diagonal=0, symmetric, >=0)
  - Top1 TVD (diagonal=0, symmetric, [0, 1])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .sources import NUM_HEADS, NUM_LAYERS, SEEDS
from .profile_metrics import (
    pearson_similarity,
    spearman_similarity,
    cosine_similarity,
    jsd_natural_log,
    l1_distance,
    wasserstein_1d_minutes,
    top1_tvd,
)
from .behavior_cards import (
    pivot_profiles_by_head,
    pivot_layer_head_mean_profiles,
)
from .metric_diffs import _pivot_top1_frequency


HEAD_LABELS: tuple[str, ...] = tuple(f"H{i + 1}" for i in range(NUM_HEADS))


@dataclass
class MatrixAuditResult:
    metric: str
    seed: int
    layer_idx0: int
    diagonal_pass: bool
    symmetric_pass: bool
    range_pass: bool
    status: str


def _empty_matrix(metric: str) -> np.ndarray:
    return np.zeros((NUM_HEADS, NUM_HEADS), dtype=np.float64)


def build_jsd_matrix(
    by_head: dict[tuple[int, int, int], np.ndarray],
    seed: int,
    layer_idx0: int,
    support_minutes: np.ndarray,
) -> tuple[np.ndarray, MatrixAuditResult]:
    m = _empty_matrix("jsd")
    for i in range(NUM_HEADS):
        for j in range(NUM_HEADS):
            if i == j:
                m[i, j] = 0.0
            else:
                pi = by_head.get((seed, layer_idx0, i))
                pj = by_head.get((seed, layer_idx0, j))
                if pi is None or pj is None:
                    m[i, j] = float("nan")
                else:
                    m[i, j] = jsd_natural_log(pi, pj)
    audit = audit_matrix(m, "jsd", max_val=float(np.log(2.0)) + 1e-6)
    return m, audit


def build_cosine_matrix(
    by_head: dict[tuple[int, int, int], np.ndarray],
    seed: int,
    layer_idx0: int,
) -> tuple[np.ndarray, MatrixAuditResult]:
    m = _empty_matrix("cosine")
    for i in range(NUM_HEADS):
        for j in range(NUM_HEADS):
            if i == j:
                m[i, j] = 1.0
            else:
                pi = by_head.get((seed, layer_idx0, i))
                pj = by_head.get((seed, layer_idx0, j))
                if pi is None or pj is None:
                    m[i, j] = float("nan")
                else:
                    m[i, j] = cosine_similarity(pi, pj)
    audit = audit_matrix(m, "cosine", max_val=1.0 + 1e-6, diagonal_expected=1.0)
    return m, audit


def build_pearson_matrix(
    by_head: dict[tuple[int, int, int], np.ndarray],
    seed: int,
    layer_idx0: int,
) -> tuple[np.ndarray, MatrixAuditResult]:
    m = _empty_matrix("pearson")
    for i in range(NUM_HEADS):
        for j in range(NUM_HEADS):
            if i == j:
                m[i, j] = 1.0
            else:
                pi = by_head.get((seed, layer_idx0, i))
                pj = by_head.get((seed, layer_idx0, j))
                if pi is None or pj is None:
                    m[i, j] = float("nan")
                else:
                    m[i, j] = pearson_similarity(pi, pj)
    audit = audit_matrix(m, "pearson", min_val=-1.0 - 1e-6, max_val=1.0 + 1e-6, diagonal_expected=1.0)
    return m, audit


def build_spearman_matrix(
    by_head: dict[tuple[int, int, int], np.ndarray],
    seed: int,
    layer_idx0: int,
) -> tuple[np.ndarray, MatrixAuditResult]:
    m = _empty_matrix("spearman")
    for i in range(NUM_HEADS):
        for j in range(NUM_HEADS):
            if i == j:
                m[i, j] = 1.0
            else:
                pi = by_head.get((seed, layer_idx0, i))
                pj = by_head.get((seed, layer_idx0, j))
                if pi is None or pj is None:
                    m[i, j] = float("nan")
                else:
                    m[i, j] = spearman_similarity(pi, pj)
    audit = audit_matrix(m, "spearman", min_val=-1.0 - 1e-6, max_val=1.0 + 1e-6, diagonal_expected=1.0)
    return m, audit


def build_l1_matrix(
    by_head: dict[tuple[int, int, int], np.ndarray],
    seed: int,
    layer_idx0: int,
) -> tuple[np.ndarray, MatrixAuditResult]:
    m = _empty_matrix("l1")
    for i in range(NUM_HEADS):
        for j in range(NUM_HEADS):
            if i == j:
                m[i, j] = 0.0
            else:
                pi = by_head.get((seed, layer_idx0, i))
                pj = by_head.get((seed, layer_idx0, j))
                if pi is None or pj is None:
                    m[i, j] = float("nan")
                else:
                    m[i, j] = l1_distance(pi, pj)
    audit = audit_matrix(m, "l1", min_val=-1e-6, max_val=None, diagonal_expected=0.0)
    return m, audit


def build_wasserstein_matrix(
    by_head: dict[tuple[int, int, int], np.ndarray],
    seed: int,
    layer_idx0: int,
    support_minutes: np.ndarray,
) -> tuple[np.ndarray, MatrixAuditResult]:
    m = _empty_matrix("wasserstein_minutes")
    for i in range(NUM_HEADS):
        for j in range(NUM_HEADS):
            if i == j:
                m[i, j] = 0.0
            else:
                pi = by_head.get((seed, layer_idx0, i))
                pj = by_head.get((seed, layer_idx0, j))
                if pi is None or pj is None:
                    m[i, j] = float("nan")
                else:
                    m[i, j] = wasserstein_1d_minutes(pi, pj, support_minutes)
    audit = audit_matrix(m, "wasserstein_minutes", min_val=-1e-6, max_val=None, diagonal_expected=0.0)
    return m, audit


def build_expected_lag_diff_matrix(
    metric_diff_rows: list[Any],
    seed: int,
    layer_idx0: int,
) -> tuple[np.ndarray, MatrixAuditResult]:
    m = _empty_matrix("abs_expected_lag_diff")
    m[:] = float("nan")
    for r in metric_diff_rows:
        if r.seed != seed or r.layer_idx0 != layer_idx0:
            continue
        if r.metric != "expected_lag_minutes":
            continue
        if np.isnan(r.abs_delta):
            continue
        m[r.head_a_idx0, r.head_b_idx0] = r.abs_delta
        m[r.head_b_idx0, r.head_a_idx0] = r.abs_delta
    np.fill_diagonal(m, 0.0)
    audit = audit_matrix(m, "abs_expected_lag_diff", min_val=-1e-6, max_val=None, diagonal_expected=0.0)
    return m, audit


def build_recent1h_diff_matrix(
    metric_diff_rows: list[Any],
    seed: int,
    layer_idx0: int,
) -> tuple[np.ndarray, MatrixAuditResult]:
    m = _empty_matrix("abs_recent1h_diff")
    m[:] = float("nan")
    for r in metric_diff_rows:
        if r.seed != seed or r.layer_idx0 != layer_idx0:
            continue
        if r.metric != "recent_1h_mass":
            continue
        if np.isnan(r.abs_delta):
            continue
        m[r.head_a_idx0, r.head_b_idx0] = r.abs_delta
        m[r.head_b_idx0, r.head_a_idx0] = r.abs_delta
    np.fill_diagonal(m, 0.0)
    audit = audit_matrix(m, "abs_recent1h_diff", min_val=-1e-6, max_val=None, diagonal_expected=0.0)
    return m, audit


def build_top1_tvd_matrix(
    top1_dist_rows: list[dict[str, Any]],
    seed: int,
    layer_idx0: int,
) -> tuple[np.ndarray, MatrixAuditResult]:
    m = _empty_matrix("top1_tvd")
    m[:] = float("nan")
    for r in top1_dist_rows:
        if r["seed"] != seed or r["layer_idx0"] != layer_idx0:
            continue
        if isinstance(r["tvd"], float) and np.isnan(r["tvd"]):
            continue
        m[r["head_a"], r["head_b"]] = float(r["tvd"])
        m[r["head_b"], r["head_a"]] = float(r["tvd"])
    np.fill_diagonal(m, 0.0)
    audit = audit_matrix(m, "top1_tvd", min_val=-1e-6, max_val=1.0 + 1e-6, diagonal_expected=0.0)
    return m, audit


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def audit_matrix(
    m: np.ndarray,
    metric: str,
    min_val: float | None = None,
    max_val: float | None = None,
    diagonal_expected: float | None = None,
    tol: float = 1e-6,
) -> MatrixAuditResult:
    """Audit: diagonal value, symmetry, range."""
    # Diagonal
    diag_pass = True
    if diagonal_expected is not None:
        diag = np.diag(m)
        diag_pass = bool(np.all(np.abs(diag - diagonal_expected) < tol))
    # Symmetry
    sym_pass = bool(np.allclose(m, m.T, atol=tol, equal_nan=True))
    # Range
    range_pass = True
    finite = m[np.isfinite(m)]
    if finite.size:
        if min_val is not None:
            range_pass = range_pass and bool(finite.min() >= min_val - tol)
        if max_val is not None:
            range_pass = range_pass and bool(finite.max() <= max_val + tol)
    status = "PASS"
    if not (diag_pass and sym_pass and range_pass):
        status = "FAIL"
    return MatrixAuditResult(
        metric=metric,
        seed=-1,
        layer_idx0=-1,
        diagonal_pass=diag_pass,
        symmetric_pass=sym_pass,
        range_pass=range_pass,
        status=status,
    )


def matrix_to_long_rows(
    m: np.ndarray,
    metric: str,
    seed: int,
    layer_idx0: int,
) -> list[dict[str, Any]]:
    """Convert matrix to long form (per (seed, layer, row_head, col_head, value))."""
    out: list[dict[str, Any]] = []
    for i in range(NUM_HEADS):
        for j in range(NUM_HEADS):
            out.append({
                "seed": seed,
                "layer_idx0": layer_idx0,
                "row_head": HEAD_LABELS[i],
                "col_head": HEAD_LABELS[j],
                "value": float(m[i, j]),
                "metric": metric,
                "status": "OK" if np.isfinite(m[i, j]) else "MISSING",
            })
    return out


def matrix_to_wide_pivot(
    m: np.ndarray,
    metric: str,
    seed: int,
    layer_idx0: int,
) -> list[dict[str, Any]]:
    """Wide pivot: row_head + one column per col_head."""
    out: list[dict[str, Any]] = []
    for i in range(NUM_HEADS):
        row: dict[str, Any] = {"seed": seed, "layer_idx0": layer_idx0, "row_head": HEAD_LABELS[i]}
        for j in range(NUM_HEADS):
            row[HEAD_LABELS[j]] = float(m[i, j])
        out.append(row)
    return out


def collect_matrix_audits() -> list[MatrixAuditResult]:
    return []
