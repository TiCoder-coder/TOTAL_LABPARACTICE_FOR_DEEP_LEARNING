"""Phase 48-C3/C4 — Gap-safe first-differences + direction-of-change agreement.

Rules:
* Cadence = 10 minutes; only difference across exact-10-minute transitions.
* Direction classes: NEGATIVE / ZERO / POSITIVE — NO epsilon threshold.
* delta < 0 -> NEGATIVE, delta == 0 -> ZERO, delta > 0 -> POSITIVE.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from course_work.prediction_analysis.contract import OUTPUT_DIR
from course_work.prediction_analysis.writers import write_csv


CADENCE_MINUTES = 10
SEED_COLUMNS = ["y_pred_seed42", "y_pred_seed123", "y_pred_seed2026"]
SERIES_LABELS_CHANGE = ["ACTUAL", "SEED42", "SEED123", "SEED2026"]


def _parse_ts(s: str):
    from datetime import datetime
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def build_continuity_mask(wide_rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return (continuity_mask_of_length_N, transitions_mask_of_length_N-1).

    continuity_mask[i] = True iff ts[i] - ts[i-1] == 10 minutes (for i >= 1).
    For i = 0 the continuity is N/A (always True for transition validity).
    transitions_mask[t] = True iff ts[t+1] - ts[t] == 10 minutes.
    """
    from datetime import timedelta

    n = len(wide_rows)
    transitions = np.zeros(n - 1, dtype=bool)
    for i in range(n - 1):
        delta = _parse_ts(wide_rows[i + 1]["target_timestamp"]) - _parse_ts(wide_rows[i]["target_timestamp"])
        transitions[i] = (delta == timedelta(minutes=CADENCE_MINUTES))

    # Continuity array of length n: index i refers to whether index i-1 -> i is valid.
    # transitions[i-1] tells us if (i-1, i) is a valid 10-min transition.
    # continuity[i] = True if i == 0 or transitions[i-1] is True.
    continuity = np.zeros(n, dtype=bool)
    continuity[0] = True
    for i in range(1, n):
        continuity[i] = bool(transitions[i - 1])
    return continuity, transitions


def compute_change_summary(wide_rows: list[dict]) -> tuple[list[dict], dict]:
    """Per plan §99: 4 series (ACTUAL + 3 seeds) x {N, mean|delta|, median|delta|, p90, p95, std delta}."""
    _, transitions = build_continuity_mask(wide_rows)
    valid_mask = transitions  # length N-1

    y_true = np.array([float(r["y_true_wh"]) for r in wide_rows], dtype=float)
    series_arrays = {
        "ACTUAL": y_true,
        "SEED42": np.array([float(r["y_pred_seed42"]) for r in wide_rows], dtype=float),
        "SEED123": np.array([float(r["y_pred_seed123"]) for r in wide_rows], dtype=float),
        "SEED2026": np.array([float(r["y_pred_seed2026"]) for r in wide_rows], dtype=float),
    }

    rows: list[dict] = []
    for series_id, arr in series_arrays.items():
        deltas = np.diff(arr)
        valid_deltas = deltas[valid_mask]
        abs_valid = np.abs(valid_deltas)

        n_valid = int(valid_mask.sum())
        n_total = int(transitions.size)
        n_excluded = int(n_total - n_valid)

        if n_valid > 0:
            mean_abs = float(np.mean(abs_valid))
            median_abs = float(np.median(abs_valid))
            p90 = float(np.percentile(abs_valid, 90))
            p95 = float(np.percentile(abs_valid, 95))
            std_delta = float(np.std(valid_deltas, ddof=1)) if n_valid >= 2 else 0.0
        else:
            mean_abs = median_abs = p90 = p95 = std_delta = 0.0

        rows.append({
            "series_id": series_id,
            "valid_transition_count": n_valid,
            "gap_excluded_count": n_excluded,
            "mean_abs_delta": mean_abs,
            "median_abs_delta": median_abs,
            "p90_abs_delta": p90,
            "p95_abs_delta": p95,
            "std_delta": std_delta,
            "status": "PASS",
        })

    extras = {"valid_transition_count": int(valid_mask.sum()),
              "gap_excluded_count": int(transitions.size - valid_mask.sum())}
    return rows, extras


def _sign_class(x: float) -> str:
    """Exact three-class sign: NEGATIVE / ZERO / POSITIVE. NO epsilon."""
    if x < 0:
        return "NEGATIVE"
    if x > 0:
        return "POSITIVE"
    return "ZERO"


def compute_direction_agreement(wide_rows: list[dict]) -> list[dict]:
    """Per plan §100.

    For each seed: report
      - valid_adjacent_count
      - exact_three_class_agreement_count / rate (NEGATIVE/ZERO/POSITIVE)
      - actual_nonzero_count
      - nonzero_direction_agreement_count / rate (excludes actual ZERO)
    """
    _, transitions = build_continuity_mask(wide_rows)
    valid_mask = transitions

    y_true = np.array([float(r["y_true_wh"]) for r in wide_rows], dtype=float)
    actual_deltas = np.diff(y_true)
    actual_classes = np.array([_sign_class(float(d)) for d in actual_deltas])
    nonzero_actual_mask = (actual_classes != "ZERO")

    seed_columns = [("SEED42", "y_pred_seed42"),
                    ("SEED123", "y_pred_seed123"),
                    ("SEED2026", "y_pred_seed2026")]

    rows: list[dict] = []
    n_valid = int(valid_mask.sum())
    for label, col in seed_columns:
        y_pred = np.array([float(r[col]) for r in wide_rows], dtype=float)
        pred_deltas = np.diff(y_pred)
        pred_classes = np.array([_sign_class(float(d)) for d in pred_deltas])

        valid_actual = actual_classes[valid_mask]
        valid_pred = pred_classes[valid_mask]

        exact_match = (valid_actual == valid_pred)
        exact_count = int(exact_match.sum())
        exact_rate = exact_count / n_valid if n_valid > 0 else 0.0

        nz_actual_mask_v = nonzero_actual_mask & valid_mask
        nz_count = int(nz_actual_mask_v.sum())
        if nz_count > 0:
            nz_match = (pred_classes[nz_actual_mask_v] == actual_classes[nz_actual_mask_v])
            nz_match_count = int(nz_match.sum())
            nz_rate = nz_match_count / nz_count
        else:
            nz_match_count = 0
            nz_rate = 0.0

        rows.append({
            "seed": label,
            "valid_adjacent_count": n_valid,
            "exact_three_class_agreement_count": exact_count,
            "exact_three_class_agreement_rate": float(exact_rate),
            "actual_nonzero_count": nz_count,
            "nonzero_direction_agreement_count": nz_match_count,
            "nonzero_direction_agreement_rate": float(nz_rate),
            "status": "PASS",
        })

    return rows


def write_change_summary(wide_rows: list[dict]) -> tuple[Path, dict]:
    rows, extras = compute_change_summary(wide_rows)
    path = write_csv(
        OUTPUT_DIR / "prediction_change_summary.csv",
        ["series_id", "valid_transition_count", "gap_excluded_count",
         "mean_abs_delta", "median_abs_delta", "p90_abs_delta",
         "p95_abs_delta", "std_delta", "status"],
        rows,
    )
    return path, extras


def write_direction_agreement(wide_rows: list[dict]) -> Path:
    rows = compute_direction_agreement(wide_rows)
    return write_csv(
        OUTPUT_DIR / "prediction_direction_agreement.csv",
        ["seed", "valid_adjacent_count", "exact_three_class_agreement_count",
         "exact_three_class_agreement_rate", "actual_nonzero_count",
         "nonzero_direction_agreement_count", "nonzero_direction_agreement_rate", "status"],
        rows,
    )
