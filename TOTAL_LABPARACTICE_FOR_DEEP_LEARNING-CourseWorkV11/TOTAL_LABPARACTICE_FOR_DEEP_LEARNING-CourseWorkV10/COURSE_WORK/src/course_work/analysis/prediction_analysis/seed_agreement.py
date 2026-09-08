"""Phase 48-D1/D2/D3 — Pairwise seed agreement + per-target seed spread + top-K disagreement.

Strictly descriptive cross-seed diagnostics.
* No model performance metric (Pearson/Spearman/RMSE_between_predictions compare seed outputs to each other, NOT to truth).
* Seed SD uses ddof=1.
* Top-K = 20, ranked ONLY by seed_range_prediction descending, deterministic tie-break by target_id ascending.
* Seed spread labelled "CROSS-SEED PREDICTION SPREAD" — never confidence interval.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

from course_work.analysis.prediction_analysis.contract import (
    OUTPUT_DIR,
    SEED_STD_DDOF,
    TOP_DISAGREEMENT_K,
)
from course_work.analysis.prediction_analysis.writers import write_csv


SEED_COLUMNS = [("42", "y_pred_seed42"),
                ("123", "y_pred_seed123"),
                ("2026", "y_pred_seed2026")]

SEED_PAIR_LABELS = ["42-123", "42-2026", "123-2026"]


# ---------------------------------------------------------------------------
# 1. Pairwise seed agreement
# ---------------------------------------------------------------------------

def _safe_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    if x.size < 2:
        return 0.0, "NOT_AVAILABLE"
    if np.std(x, ddof=1) == 0 or np.std(y, ddof=1) == 0:
        return 0.0, "NOT_AVAILABLE"
    r, _ = pearsonr(x, y)
    return float(r), "PASS"


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    if x.size < 2:
        return 0.0, "NOT_AVAILABLE"
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    if np.std(rx, ddof=1) == 0 or np.std(ry, ddof=1) == 0:
        return 0.0, "NOT_AVAILABLE"
    rho, _ = spearmanr(rx, ry)
    if rho != rho:  # NaN guard
        return 0.0, "NOT_AVAILABLE"
    return float(rho), "PASS"


def compute_pairwise_agreement(wide_rows: list[dict]) -> list[dict]:
    """Per plan §105. Three pairs (42-123, 42-2026, 123-2026)."""
    arrays = {label: np.array([float(r[col]) for r in wide_rows], dtype=float)
              for label, col in SEED_COLUMNS}
    n = len(wide_rows)
    rows: list[dict] = []
    pairs = [("42", "123"), ("42", "2026"), ("123", "2026")]
    for a, b in pairs:
        x, y = arrays[a], arrays[b]
        diff = x - y
        pear, pear_st = _safe_pearson(x, y)
        spear, spear_st = _safe_spearman(x, y)
        rmse_pair = float(np.sqrt(np.mean(diff ** 2)))
        rows.append({
            "seed_a": a,
            "seed_b": b,
            "pearson_correlation": pear,
            "spearman_correlation": spear,
            "mean_absolute_prediction_difference": float(np.mean(np.abs(diff))),
            "rmse_between_predictions": rmse_pair,
            "max_absolute_prediction_difference": float(np.max(np.abs(diff))),
            "N": int(n),
            "pearson_status": pear_st,
            "spearman_status": spear_st,
            "label": "SEED_AGREEMENT_DIAGNOSTIC",
            "status": "PASS",
        })
    return rows


def write_pairwise_agreement(wide_rows: list[dict]) -> Path:
    rows = compute_pairwise_agreement(wide_rows)
    return write_csv(
        OUTPUT_DIR / "prediction_seed_pairwise_agreement.csv",
        ["seed_a", "seed_b", "pearson_correlation", "spearman_correlation",
         "mean_absolute_prediction_difference", "rmse_between_predictions",
         "max_absolute_prediction_difference", "N", "pearson_status",
         "spearman_status", "label", "status"],
        rows,
    )


# ---------------------------------------------------------------------------
# 2. Per-target seed spread (re-use wide table values: ddof=1 already enforced)
# ---------------------------------------------------------------------------

def build_seed_spread(wide_rows: list[dict]) -> list[dict]:
    """Per plan §106 — re-use canonical wide-table spread columns (ddof=1)."""
    rows: list[dict] = []
    for r in wide_rows:
        rows.append({
            "target_id": r["target_id"],
            "target_timestamp": r["target_timestamp"],
            "y_true_wh": float(r["y_true_wh"]),
            "seed_mean_prediction": float(r["seed_mean_prediction"]),
            "seed_std_prediction": float(r["seed_std_prediction"]),
            "seed_min_prediction": float(r["seed_min_prediction"]),
            "seed_max_prediction": float(r["seed_max_prediction"]),
            "seed_range_prediction": float(r["seed_range_prediction"]),
            "spread_semantics": "CROSS_SEED_PREDICTION_SPREAD",
            "status": "PASS",
        })
    return rows


def write_seed_spread(wide_rows: list[dict]) -> Path:
    rows = build_seed_spread(wide_rows)
    return write_csv(
        OUTPUT_DIR / "prediction_seed_spread.csv",
        ["target_id", "target_timestamp", "y_true_wh",
         "seed_mean_prediction", "seed_std_prediction",
         "seed_min_prediction", "seed_max_prediction",
         "seed_range_prediction", "spread_semantics", "status"],
        rows,
    )


# ---------------------------------------------------------------------------
# 3. Top-20 seed disagreement
# ---------------------------------------------------------------------------

def compute_top_seed_disagreement(wide_rows: list[dict], k: int = TOP_DISAGREEMENT_K) -> list[dict]:
    """Per plan §107.

    K=20 exactly. Rank ONLY by seed_range_prediction descending.
    Deterministic tie-break: target_id ascending.
    """
    candidates = []
    for r in wide_rows:
        candidates.append({
            "target_id": r["target_id"],
            "target_timestamp": r["target_timestamp"],
            "y_true_wh": float(r["y_true_wh"]),
            "seed42": float(r["y_pred_seed42"]),
            "seed123": float(r["y_pred_seed123"]),
            "seed2026": float(r["y_pred_seed2026"]),
            "seed_mean": float(r["seed_mean_prediction"]),
            "seed_std": float(r["seed_std_prediction"]),
            "seed_range": float(r["seed_range_prediction"]),
        })
    candidates.sort(key=lambda c: (-c["seed_range"], c["target_id"]))
    rows: list[dict] = []
    for i, c in enumerate(candidates[:k], start=1):
        rows.append({
            "rank": i,
            "target_id": c["target_id"],
            "target_timestamp": c["target_timestamp"],
            "y_true_wh": c["y_true_wh"],
            "seed42": c["seed42"],
            "seed123": c["seed123"],
            "seed2026": c["seed2026"],
            "seed_mean": c["seed_mean"],
            "seed_std": c["seed_std"],
            "seed_range": c["seed_range"],
            "ranking_basis": "seed_range_prediction_descending_target_id_ascending_tiebreak",
            "status": "PASS",
        })
    return rows


def write_top_seed_disagreement(wide_rows: list[dict]) -> Path:
    rows = compute_top_seed_disagreement(wide_rows)
    return write_csv(
        OUTPUT_DIR / "prediction_top_seed_disagreement.csv",
        ["rank", "target_id", "target_timestamp", "y_true_wh",
         "seed42", "seed123", "seed2026", "seed_mean", "seed_std",
         "seed_range", "ranking_basis", "status"],
        rows,
    )
