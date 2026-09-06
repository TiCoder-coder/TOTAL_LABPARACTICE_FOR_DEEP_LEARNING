"""Phase 48-D4 — Gap-safe 144-point rolling tracking diagnostics.

Per plan §59-§63:

* Canonical cadence = 10 minutes.
* Window = exactly 144 contiguous observations.
* No forward filling.
* No partial-window substitution.
* If any of the 143 within-window transitions is NOT exactly 10 minutes, the window is INVALID (status=INVALID).
* Only valid windows are emitted.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from course_work.prediction_analysis.contract import OUTPUT_DIR, ROLLING_WINDOW
from course_work.prediction_analysis.writers import write_csv


CADENCE_MINUTES = 10
SEED_COLUMNS = [("SEED42", "y_pred_seed42"),
                ("SEED123", "y_pred_seed123"),
                ("SEED2026", "y_pred_seed2026")]


def _parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def _is_full_window_contiguous(timestamps: list[str], start: int, window: int) -> bool:
    """True iff all (window - 1) transitions are exactly 10 minutes."""
    end = start + window
    for i in range(start, end - 1):
        d = _parse_ts(timestamps[i + 1]) - _parse_ts(timestamps[i])
        if d != timedelta(minutes=CADENCE_MINUTES):
            return False
    return True


def compute_rolling_tracking(wide_rows: list[dict], window: int = ROLLING_WINDOW) -> list[dict]:
    """Per plan §108.

    Only valid contiguous windows are emitted. Invalid windows are NOT included.
    """
    timestamps = [r["target_timestamp"] for r in wide_rows]
    y_true = np.array([float(r["y_true_wh"]) for r in wide_rows], dtype=float)
    n = len(wide_rows)
    rows: list[dict] = []
    if window < 2 or window > n:
        return rows

    for start in range(0, n - window + 1):
        end = start + window
        ts_window = timestamps[start:end]
        if not _is_full_window_contiguous(ts_window, 0, window):
            continue  # invalid window -> skip entirely (no partial substitute)

        anchor_ts = timestamps[start]
        window_valid_count = window
        true_slice = y_true[start:end]
        true_mean = float(np.mean(true_slice))
        true_std = float(np.std(true_slice, ddof=1)) if window >= 2 else 0.0

        for seed_label, col in SEED_COLUMNS:
            pred_slice = np.array([float(r[col]) for r in wide_rows[start:end]], dtype=float)
            pred_mean = float(np.mean(pred_slice))
            pred_std = float(np.std(pred_slice, ddof=1)) if window >= 2 else 0.0

            if true_std > 0 and pred_std > 0:
                r = float(np.corrcoef(true_slice, pred_slice)[0, 1])
            else:
                r = 0.0

            rows.append({
                "timestamp": anchor_ts,
                "seed": seed_label,
                "rolling_24h_true_mean": true_mean,
                "rolling_24h_pred_mean": pred_mean,
                "rolling_24h_true_std": true_std,
                "rolling_24h_pred_std": pred_std,
                "rolling_24h_corr": r,
                "window_valid_count": window_valid_count,
                "status": "PASS",
            })

    return rows


def write_rolling_tracking(wide_rows: list[dict]) -> Path:
    rows = compute_rolling_tracking(wide_rows)
    return write_csv(
        OUTPUT_DIR / "prediction_rolling_tracking.csv",
        ["timestamp", "seed", "rolling_24h_true_mean", "rolling_24h_pred_mean",
         "rolling_24h_true_std", "rolling_24h_pred_std", "rolling_24h_corr",
         "window_valid_count", "status"],
        rows,
    )
