"""Phase 48-C7/C8 — Local extrema + peak timing.

Per plan §43-§48:

Local maximum at t (with both neighbors exactly 10 minutes apart):
    y_t > y_{t-1}  AND  y_t >= y_{t+1}

Local minimum at t:
    y_t < y_{t-1}  AND  y_t <= y_{t+1}

Edges (first / last sample of a contiguous segment) are NOT eligible.

Peak timing: at each true local maximum, search the prediction series within
±1 step (i.e. t-1, t, t+1) for the nearest local prediction maximum and report
same-step peak rate and within-±1-step peak rate.

Window is FROZEN at ±1 step = ±10 minutes. No widening after results.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np

from course_work.phase48.contract import PEAK_TIMING_WINDOW_STEPS, OUTPUT_DIR
from course_work.phase48.writers import write_csv


CADENCE_MINUTES = 10
SEED_COLUMNS = [("SEED42", "y_pred_seed42"),
                ("SEED123", "y_pred_seed123"),
                ("SEED2026", "y_pred_seed2026")]


def _parse_ts(s: str):
    from datetime import datetime
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def _is_local_extremum_at(values: np.ndarray, t: int, neighbor_mask: np.ndarray) -> str:
    """Return 'MAX' | 'MIN' | None for index t using neighbor_mask."""
    if not neighbor_mask[t]:
        return None
    if t - 1 < 0 or t + 1 >= len(values):
        return None
    if not neighbor_mask[t - 1] or not neighbor_mask[t + 1]:
        return None
    a, b, c = values[t - 1], values[t], values[t + 1]
    if b > a and b >= c:
        return "MAX"
    if b < a and b <= c:
        return "MIN"
    return None


def build_neighbor_mask(timestamps: list[str]) -> np.ndarray:
    """True at index t iff both transitions (t-1, t) and (t, t+1) are exactly 10 minutes."""
    n = len(timestamps)
    out = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        d_back = _parse_ts(timestamps[i]) - _parse_ts(timestamps[i - 1])
        d_fwd = _parse_ts(timestamps[i + 1]) - _parse_ts(timestamps[i])
        if d_back == timedelta(minutes=CADENCE_MINUTES) and d_fwd == timedelta(minutes=CADENCE_MINUTES):
            out[i] = True
    return out


def detect_local_extrema(values: np.ndarray, neighbor_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (is_max_boolean, is_min_boolean) arrays of length N."""
    n = values.size
    is_max = np.zeros(n, dtype=bool)
    is_min = np.zeros(n, dtype=bool)
    for t in range(n):
        kind = _is_local_extremum_at(values, t, neighbor_mask)
        if kind == "MAX":
            is_max[t] = True
        elif kind == "MIN":
            is_min[t] = True
    return is_max, is_min


def compute_local_extrema_summary(wide_rows: list[dict]) -> tuple[list[dict], dict]:
    """Per plan §103."""
    timestamps = [r["target_timestamp"] for r in wide_rows]
    neighbor_mask = build_neighbor_mask(timestamps)
    y_true = np.array([float(r["y_true_wh"]) for r in wide_rows], dtype=float)
    is_max, is_min = detect_local_extrema(y_true, neighbor_mask)

    max_idx = np.where(is_max)[0]
    min_idx = np.where(is_min)[0]

    rows: list[dict] = []
    for seed_label, col in SEED_COLUMNS:
        y_pred = np.array([float(r[col]) for r in wide_rows], dtype=float)

        if max_idx.size > 0:
            actual_at_peak = y_true[max_idx]
            pred_at_peak = y_pred[max_idx]
            actual_peak_mean = float(np.mean(actual_at_peak))
            actual_peak_median = float(np.median(actual_at_peak))
            pred_at_peak_mean = float(np.mean(pred_at_peak))
            pred_at_peak_median = float(np.median(pred_at_peak))
            peak_level_ratio = (pred_at_peak_mean / actual_peak_mean) if actual_peak_mean > 0 else 0.0
        else:
            actual_peak_mean = actual_peak_median = pred_at_peak_mean = pred_at_peak_median = 0.0
            peak_level_ratio = 0.0

        if min_idx.size > 0:
            actual_at_trough = y_true[min_idx]
            pred_at_trough = y_pred[min_idx]
            actual_trough_mean = float(np.mean(actual_at_trough))
            actual_trough_median = float(np.median(actual_at_trough))
            pred_at_trough_mean = float(np.mean(pred_at_trough))
            pred_at_trough_median = float(np.median(pred_at_trough))
        else:
            actual_trough_mean = actual_trough_median = pred_at_trough_mean = pred_at_trough_median = 0.0

        rows.append({
            "seed": seed_label,
            "true_local_max_count": int(max_idx.size),
            "actual_peak_mean": actual_peak_mean,
            "actual_peak_median": actual_peak_median,
            "pred_at_peak_mean": pred_at_peak_mean,
            "pred_at_peak_median": pred_at_peak_median,
            "peak_level_ratio": peak_level_ratio,
            "true_local_min_count": int(min_idx.size),
            "actual_trough_mean": actual_trough_mean,
            "actual_trough_median": actual_trough_median,
            "pred_at_trough_mean": pred_at_trough_mean,
            "pred_at_trough_median": pred_at_trough_median,
            "status": "PASS",
        })

    extras = {
        "true_local_max_indices": max_idx.tolist(),
        "true_local_min_indices": min_idx.tolist(),
        "neighbor_mask_count": int(neighbor_mask.sum()),
    }
    return rows, extras


def compute_peak_timing_summary(wide_rows: list[dict], extras: dict) -> list[dict]:
    """Per plan §104.

    Window = ±PEAK_TIMING_WINDOW_STEPS (FROZEN to 1 = ±10 minutes).

    For each true local maximum at t, check the prediction series:
      - same_step: y_pred[t] is a local maximum in y_pred
      - within ±1: y_pred has a local maximum in window [t-1, t+1]
    """
    timestamps = [r["target_timestamp"] for r in wide_rows]
    neighbor_mask = build_neighbor_mask(timestamps)
    y_pred_arrays = {
        "SEED42": np.array([float(r["y_pred_seed42"]) for r in wide_rows], dtype=float),
        "SEED123": np.array([float(r["y_pred_seed123"]) for r in wide_rows], dtype=float),
        "SEED2026": np.array([float(r["y_pred_seed2026"]) for r in wide_rows], dtype=float),
    }
    pred_is_max = {sid: detect_local_extrema(arr, neighbor_mask)[0] for sid, arr in y_pred_arrays.items()}
    pred_neighbor_ok = neighbor_mask

    max_idx = np.array(extras["true_local_max_indices"], dtype=int)
    window = PEAK_TIMING_WINDOW_STEPS

    rows: list[dict] = []
    for seed_label, _ in SEED_COLUMNS:
        if max_idx.size == 0:
            rows.append({
                "seed": seed_label,
                "eligible_true_peaks": 0,
                "same_step_pred_peak_count": 0,
                "same_step_rate": 0.0,
                "within_plus_minus_1_step_count": 0,
                "within_plus_minus_1_step_rate": 0.0,
                "status": "PASS",
            })
            continue

        same_step_count = 0
        window_count = 0
        for t in max_idx:
            if pred_is_max[seed_label][t]:
                same_step_count += 1
                window_count += 1
            else:
                lo = max(0, t - window)
                hi = min(len(timestamps) - 1, t + window)
                hit = False
                for k in range(lo, hi + 1):
                    if k == t:
                        continue
                    if pred_neighbor_ok[k] and pred_is_max[seed_label][k]:
                        hit = True
                        break
                if hit:
                    window_count += 1

        rows.append({
            "seed": seed_label,
            "eligible_true_peaks": int(max_idx.size),
            "same_step_pred_peak_count": int(same_step_count),
            "same_step_rate": float(same_step_count / max_idx.size),
            "within_plus_minus_1_step_count": int(window_count),
            "within_plus_minus_1_step_rate": float(window_count / max_idx.size),
            "status": "PASS",
        })
    return rows


def write_local_extrema_summary(wide_rows: list[dict]) -> tuple[Path, dict]:
    rows, extras = compute_local_extrema_summary(wide_rows)
    path = write_csv(
        OUTPUT_DIR / "prediction_local_extrema_summary.csv",
        ["seed", "true_local_max_count", "actual_peak_mean", "actual_peak_median",
         "pred_at_peak_mean", "pred_at_peak_median", "peak_level_ratio",
         "true_local_min_count", "actual_trough_mean", "actual_trough_median",
         "pred_at_trough_mean", "pred_at_trough_median", "status"],
        rows,
    )
    return path, extras


def write_peak_timing_summary(wide_rows: list[dict], extras: dict) -> Path:
    rows = compute_peak_timing_summary(wide_rows, extras)
    return write_csv(
        OUTPUT_DIR / "prediction_peak_timing_summary.csv",
        ["seed", "eligible_true_peaks", "same_step_pred_peak_count", "same_step_rate",
         "within_plus_minus_1_step_count", "within_plus_minus_1_step_rate", "status"],
        rows,
    )
