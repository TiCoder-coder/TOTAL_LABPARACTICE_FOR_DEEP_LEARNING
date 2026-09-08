from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np


PHASE49_CADENCE_MINUTES = 10
PHASE49_ACF_LAG_RANGE = (1, 144)
PHASE49_ACF_KEY_LAGS = (1, 6, 12, 36, 72, 144)


def parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def detect_contiguous_segments(
    timestamps: list[datetime],
    cadence_minutes: int = PHASE49_CADENCE_MINUTES,
) -> list[tuple[int, int]]:
    """Return list of (start_idx, end_idx_exclusive) for each contiguous segment.

    A new segment starts when the delta between consecutive timestamps is not
    exactly cadence_minutes.
    """
    if not timestamps:
        return []
    segments: list[tuple[int, int]] = []
    seg_start = 0
    for i in range(1, len(timestamps)):
        delta_minutes = (timestamps[i] - timestamps[i - 1]).total_seconds() / 60.0
        if delta_minutes != cadence_minutes:
            segments.append((seg_start, i))
            seg_start = i
    segments.append((seg_start, len(timestamps)))
    return segments


def compute_acf_for_segment(
    residuals: np.ndarray,
    lag: int,
) -> tuple[float, int]:
    """Compute lag-k autocorrelation for a single contiguous segment.

    Returns (acf_value, valid_pair_count).
    acf_k = sum_{i=0..n-k-1}((r_i - mean)(r_{i+k} - mean)) / sum((r_i - mean)^2)
    """
    n = residuals.size
    if n - lag < 1:
        return float("nan"), 0
    r = residuals - float(np.mean(residuals))
    denom = float(np.sum(r * r))
    if denom == 0.0:
        return float("nan"), 0
    numer = float(np.sum(r[: n - lag] * r[lag:]))
    return numer / denom, n - lag


def compute_acf_for_seed(
    timestamps: list[datetime],
    residuals: np.ndarray,
    max_lag: int = PHASE49_ACF_LAG_RANGE[1],
) -> list[dict[str, Any]]:
    """Compute gap-safe ACF for lags 1..max_lag using per-segment aggregation.

    If the seed series is one fully contiguous segment, ACF is computed on that
    single segment. If multiple segments exist, each lag uses the combined
    pair count across segments (still gap-safe: no pair crosses a gap).
    """
    segments = detect_contiguous_segments(timestamps)
    rows: list[dict[str, Any]] = []
    for lag in range(1, max_lag + 1):
        total_numer = 0.0
        total_denom = 0.0
        total_pairs = 0
        for start, end in segments:
            seg_residuals = residuals[start:end]
            seg_n = seg_residuals.size
            if seg_n - lag < 1:
                continue
            seg_mean = float(np.mean(seg_residuals))
            r_centered = seg_residuals - seg_mean
            seg_denom = float(np.sum(r_centered * r_centered))
            if seg_denom == 0.0:
                continue
            seg_numer = float(np.sum(r_centered[: seg_n - lag] * r_centered[lag:]))
            total_numer += seg_numer
            total_denom += seg_denom
            total_pairs += seg_n - lag

        if total_denom == 0.0 or total_pairs == 0:
            acf_value = float("nan")
            status = "NOT_COMPUTABLE"
        else:
            acf_value = total_numer / total_denom
            status = "PASS"

        rows.append(
            {
                "lag_steps": lag,
                "lag_minutes": lag * PHASE49_CADENCE_MINUTES,
                "valid_pair_count": total_pairs,
                "acf_value": float(acf_value),
                "segment_count": len(segments),
                "status": status,
            }
        )
    return rows


def contiguous_test_segments_per_seed(
    timestamps: list[datetime],
) -> list[dict[str, Any]]:
    segments = detect_contiguous_segments(timestamps)
    out: list[dict[str, Any]] = []
    for i, (start, end) in enumerate(segments):
        out.append(
            {
                "segment_index": i,
                "start_index": start,
                "end_index_exclusive": end,
                "length": end - start,
                "start_timestamp": timestamps[start].isoformat(),
                "end_timestamp_last_inclusive": timestamps[end - 1].isoformat(),
            }
        )
    return out


def acf_audit_summary(
    acf_rows: list[dict[str, Any]],
    seed: str,
) -> dict[str, Any]:
    segments = {r["segment_count"] for r in acf_rows}
    return {
        "seed": seed,
        "n_lags_emitted": len(acf_rows),
        "lag_range": (acf_rows[0]["lag_steps"], acf_rows[-1]["lag_steps"]),
        "key_lags_reported": [lag for lag in PHASE49_ACF_KEY_LAGS],
        "segment_count_per_lag_consistent": len(segments) == 1,
        "segment_count_value": next(iter(segments)) if len(segments) == 1 else None,
        "gap_safe": True,
    }
