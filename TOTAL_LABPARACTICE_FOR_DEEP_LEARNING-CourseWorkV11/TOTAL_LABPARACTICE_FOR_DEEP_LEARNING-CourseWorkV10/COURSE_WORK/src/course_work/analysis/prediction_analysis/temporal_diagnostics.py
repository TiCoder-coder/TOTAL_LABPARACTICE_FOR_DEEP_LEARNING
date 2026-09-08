"""Phase 48-C5/C6 — Fixed lag diagnostics + prediction ACF.

Lag convention (frozen in contract):
  lag k > 0 means prediction series is compared to truth shifted k future steps
  i.e. corr( y_pred[t], y_true[t + k] ) when k > 0
  equivalently corr( y_pred[t - k], y_true[t] ) when k > 0

For negative k: corr( y_pred[t], y_true[t + k] ) with k < 0.
Both interpretations produce the same numerical result via shifting.

Implementation:
  Define shifted_actual = y_true[k:] for k>0 and shifted_pred = y_pred[:N-k]
  Pair them via direct indexing; only positions where continuity is preserved.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np

from course_work.analysis.prediction_analysis.contract import (
    ACF_REGISTERED_LAGS,
    LAG_RANGE,
    OUTPUT_DIR,
)
from course_work.analysis.prediction_analysis.writers import write_csv


CADENCE_MINUTES = 10
LAG_CONVENTION = "lag_k_positive_means_prediction_compared_to_truth_shifted_k_future_steps"
SEED_COLUMNS = [("SEED42", "y_pred_seed42"),
                ("SEED123", "y_pred_seed123"),
                ("SEED2026", "y_pred_seed2026")]


def _parse_ts(s: str):
    from datetime import datetime
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def _build_valid_pairs_for_lag(timestamps: list[str], lag: int) -> np.ndarray:
    """Boolean mask of length N-|lag|: True if ts[i+|lag|] - ts[i] == 10*|lag| minutes.

    For lag > 0: pair is (pred[i], true[i+lag]) over i = 0..N-lag-1
                 continuity: ts[i+lag] - ts[i] == 10*lag minutes.
    For lag < 0: pair is (pred[i+abs_lag], true[i]) over i = 0..N-|lag|-1
                 continuity: ts[i+|lag|] - ts[i] == 10*|lag| minutes.
                 Equivalent to ts[i] - ts[i+abs_lag] == 10*abs_lag minutes.
    For lag = 0: every index is trivially valid (length N).
    """
    n = len(timestamps)
    if lag == 0:
        return np.ones(n, dtype=bool)
    abs_lag = abs(lag)
    out = np.zeros(n - abs_lag, dtype=bool)
    expected_step = timedelta(minutes=CADENCE_MINUTES * abs_lag)
    for i in range(n - abs_lag):
        if lag > 0:
            d = _parse_ts(timestamps[i + abs_lag]) - _parse_ts(timestamps[i])
        else:
            d = _parse_ts(timestamps[i + abs_lag]) - _parse_ts(timestamps[i])
        out[i] = (d == expected_step)
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    if np.std(x, ddof=1) == 0 or np.std(y, ddof=1) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def compute_lag_diagnostics(wide_rows: list[dict], lags: list[int] = None) -> list[dict]:
    """Per plan §101. Schema: seed, lag_steps, lag_minutes, valid_pair_count,
    pearson_correlation, lag_convention, status."""
    if lags is None:
        lags = list(LAG_RANGE)
    timestamps = [r["target_timestamp"] for r in wide_rows]
    y_true = np.array([float(r["y_true_wh"]) for r in wide_rows], dtype=float)

    rows: list[dict] = []
    for seed_label, col in SEED_COLUMNS:
        y_pred = np.array([float(r[col]) for r in wide_rows], dtype=float)
        for k in lags:
            valid = _build_valid_pairs_for_lag(timestamps, k)
            if k >= 0:
                if k == 0:
                    x = y_pred
                    y = y_true
                else:
                    x = y_pred[: len(y_pred) - k]
                    y = y_true[k:]
                m = valid
            else:
                kk = -k
                x = y_pred[kk:]
                y = y_true[: len(y_true) - kk]
                m = valid

            n_valid = int(m.sum())
            if n_valid >= 2:
                corr = _pearson(x[m], y[m])
            else:
                corr = 0.0
            rows.append({
                "seed": seed_label,
                "lag_steps": int(k),
                "lag_minutes": int(k * CADENCE_MINUTES),
                "valid_pair_count": n_valid,
                "pearson_correlation": corr,
                "lag_convention": LAG_CONVENTION,
                "status": "PASS",
            })
    return rows


def compute_acf_diagnostics(wide_rows: list[dict], lags: list[int] = None) -> list[dict]:
    """Per plan §102. Prediction-series ACF only. NO residual ACF.

    Series: ACTUAL, SEED42, SEED123, SEED2026, SEED_MEAN_DESCRIPTIVE.
    For lag k, compute corr(x[t], x[t+k]) only on contiguous-10-min transitions.
    If lag+1 > N of contiguous segment: mark NOT_AVAILABLE.
    """
    if lags is None:
        lags = list(ACF_REGISTERED_LAGS)
    timestamps = [r["target_timestamp"] for r in wide_rows]
    y_true = np.array([float(r["y_true_wh"]) for r in wide_rows], dtype=float)
    ymean = np.array([float(r["seed_mean_prediction"]) for r in wide_rows], dtype=float)
    seed_arrays = {
        "SEED42": np.array([float(r["y_pred_seed42"]) for r in wide_rows], dtype=float),
        "SEED123": np.array([float(r["y_pred_seed123"]) for r in wide_rows], dtype=float),
        "SEED2026": np.array([float(r["y_pred_seed2026"]) for r in wide_rows], dtype=float),
    }

    series = [("ACTUAL", y_true), ("SEED42", seed_arrays["SEED42"]),
              ("SEED123", seed_arrays["SEED123"]), ("SEED2026", seed_arrays["SEED2026"]),
              ("SEED_MEAN_DESCRIPTIVE", ymean)]

    rows: list[dict] = []
    for sid, x in series:
        for k in lags:
            valid = _build_valid_pairs_for_lag(timestamps, k)
            x_head = x[: len(x) - k] if k > 0 else x
            y_tail = x[k:] if k > 0 else x
            m = valid
            n_valid = int(m.sum())
            if n_valid >= 2 and np.std(x_head[m], ddof=1) > 0 and np.std(y_tail[m], ddof=1) > 0:
                acf = float(np.corrcoef(x_head[m], y_tail[m])[0, 1])
                status = "PASS"
            else:
                acf = None
                status = "NOT_AVAILABLE"
            rows.append({
                "series_id": sid,
                "lag_steps": int(k),
                "lag_minutes": int(k * CADENCE_MINUTES),
                "acf": acf,
                "valid_pair_count": n_valid,
                "status": status,
            })
    return rows


def write_lag_diagnostics(wide_rows: list[dict]) -> Path:
    rows = compute_lag_diagnostics(wide_rows)
    return write_csv(
        OUTPUT_DIR / "prediction_lag_diagnostics.csv",
        ["seed", "lag_steps", "lag_minutes", "valid_pair_count",
         "pearson_correlation", "lag_convention", "status"],
        rows,
    )


def write_acf_diagnostics(wide_rows: list[dict]) -> Path:
    rows = compute_acf_diagnostics(wide_rows)
    return write_csv(
        OUTPUT_DIR / "prediction_acf_diagnostics.csv",
        ["series_id", "lag_steps", "lag_minutes", "acf", "valid_pair_count", "status"],
        rows,
    )
