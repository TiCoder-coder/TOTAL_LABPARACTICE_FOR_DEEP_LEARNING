"""Phase 48-C1 — Distribution summary + Range/compression.

Reads the canonical Phase 48-B wide table.
Computes descriptive distribution stats per series and range/variance
compression descriptors per Transformer seed. No prediction modification.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from course_work.prediction_analysis.contract import OUTPUT_DIR
from course_work.prediction_analysis.writers import write_csv, utc_now_iso


SERIES_LABELS = ["ACTUAL", "SEED42", "SEED123", "SEED2026", "SEED_MEAN_DESCRIPTIVE"]
SEED_COLUMNS = ["y_pred_seed42", "y_pred_seed123", "y_pred_seed2026"]


def _percentile(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q))


def _distribution_row(series_id: str, values: np.ndarray) -> dict[str, Any]:
    """Per plan §97: N, mean, std (ddof=1), min, p05, q1, median, q3, p95, max, iqr."""
    return {
        "series_id": series_id,
        "N": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size >= 2 else 0.0,
        "min": float(np.min(values)),
        "p05": _percentile(values, 5),
        "q1": _percentile(values, 25),
        "median": float(np.median(values)),
        "q3": _percentile(values, 75),
        "p95": _percentile(values, 95),
        "max": float(np.max(values)),
        "iqr": _percentile(values, 75) - _percentile(values, 25),
        "status": "PASS",
    }


def compute_distribution_summary(wide_rows: list[dict]) -> list[dict]:
    """Per plan §97 — 5 series x 12 stats."""
    y_true = np.array([float(r["y_true_wh"]) for r in wide_rows], dtype=float)
    y42 = np.array([float(r["y_pred_seed42"]) for r in wide_rows], dtype=float)
    y123 = np.array([float(r["y_pred_seed123"]) for r in wide_rows], dtype=float)
    y2026 = np.array([float(r["y_pred_seed2026"]) for r in wide_rows], dtype=float)
    ymean = np.array([float(r["seed_mean_prediction"]) for r in wide_rows], dtype=float)

    arrays = [
        ("ACTUAL", y_true),
        ("SEED42", y42),
        ("SEED123", y123),
        ("SEED2026", y2026),
        ("SEED_MEAN_DESCRIPTIVE", ymean),
    ]
    return [_distribution_row(label, arr) for label, arr in arrays]


def compute_range_compression(wide_rows: list[dict]) -> list[dict]:
    """Per plan §98 — per-Transformer-seed compression ratios + mean shift."""
    y_true = np.array([float(r["y_true_wh"]) for r in wide_rows], dtype=float)
    true_std = float(np.std(y_true, ddof=1)) if y_true.size >= 2 else 0.0
    true_q1 = _percentile(y_true, 25)
    true_q3 = _percentile(y_true, 75)
    true_iqr = true_q3 - true_q1
    true_min = float(np.min(y_true))
    true_max = float(np.max(y_true))
    true_range = true_max - true_min
    true_mean = float(np.mean(y_true))

    seed_map = {"SEED42": "y_pred_seed42", "SEED123": "y_pred_seed123", "SEED2026": "y_pred_seed2026"}
    rows: list[dict] = []
    for label, col in seed_map.items():
        y = np.array([float(r[col]) for r in wide_rows], dtype=float)
        pred_std = float(np.std(y, ddof=1))
        pred_q1 = _percentile(y, 25)
        pred_q3 = _percentile(y, 75)
        pred_iqr = pred_q3 - pred_q1
        pred_min = float(np.min(y))
        pred_max = float(np.max(y))
        pred_range = pred_max - pred_min
        rows.append({
            "seed": label,
            "pred_std": pred_std,
            "true_std": true_std,
            "std_ratio": (pred_std / true_std) if true_std > 0 else 0.0,
            "pred_iqr": pred_iqr,
            "true_iqr": true_iqr,
            "iqr_ratio": (pred_iqr / true_iqr) if true_iqr > 0 else 0.0,
            "pred_range": pred_range,
            "true_range": true_range,
            "range_ratio": (pred_range / true_range) if true_range > 0 else 0.0,
            "mean_shift_pred_minus_true": float(np.mean(y)) - true_mean,
            "status": "PASS",
        })
    return rows


def write_distribution_summary(wide_rows: list[dict]) -> Path:
    rows = compute_distribution_summary(wide_rows)
    return write_csv(
        OUTPUT_DIR / "prediction_distribution_summary.csv",
        ["series_id", "N", "mean", "std", "min", "p05", "q1", "median", "q3", "p95", "max", "iqr", "status"],
        rows,
    )


def write_range_compression(wide_rows: list[dict]) -> Path:
    rows = compute_range_compression(wide_rows)
    return write_csv(
        OUTPUT_DIR / "prediction_range_compression.csv",
        ["seed", "pred_std", "true_std", "std_ratio", "pred_iqr", "true_iqr", "iqr_ratio",
         "pred_range", "true_range", "range_ratio", "mean_shift_pred_minus_true", "status"],
        rows,
    )
