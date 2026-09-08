from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from .autocorrelation import PHASE49_CADENCE_MINUTES, detect_contiguous_segments


PHASE49_ROLLING_WINDOW_SAMPLES = 144


def rolling_residual_diagnostics(
    residuals: np.ndarray,
    signs: list[str],
    timestamps: list[datetime],
    window_size: int = PHASE49_ROLLING_WINDOW_SAMPLES,
    cadence_minutes: int = PHASE49_CADENCE_MINUTES,
) -> list[dict[str, Any]]:
    """Rolling residual diagnostics.

    Rules:
    - exact window_size (144) contiguous observations per window
    - no partial windows; no interpolation; no padding; no forward fill
    - rolling std uses ddof=1 (consistent with Phase49-B metric contract)
    """
    if residuals.size != signs.__len__() != timestamps.__len__():
        raise ValueError("residuals, signs, timestamps length mismatch")
    if window_size != PHASE49_ROLLING_WINDOW_SAMPLES:
        raise ValueError(
            f"Phase49-D rolling window must be exactly {PHASE49_ROLLING_WINDOW_SAMPLES}; got {window_size}"
        )

    segments = detect_contiguous_segments(timestamps, cadence_minutes=cadence_minutes)
    out: list[dict[str, Any]] = []
    window_index = 0
    for seg_start, seg_end in segments:
        seg_residuals = residuals[seg_start:seg_end]
        seg_signs = signs[seg_start:seg_end]
        seg_timestamps = timestamps[seg_start:seg_end]
        seg_n = seg_residuals.size
        if seg_n < window_size:
            continue
        n_windows_in_segment = seg_n - window_size + 1
        for w in range(n_windows_in_segment):
            window_residuals = seg_residuals[w : w + window_size]
            window_signs = seg_signs[w : w + window_size]
            start_ts = seg_timestamps[w]
            end_ts = seg_timestamps[w + window_size - 1]

            n_under = int(sum(1 for s in window_signs if s == "UNDERPREDICTION"))
            n_over = int(sum(1 for s in window_signs if s == "OVERPREDICTION"))
            n_exact = int(sum(1 for s in window_signs if s == "EXACT"))

            out.append(
                {
                    "window_index": window_index,
                    "segment_index": len(segments),
                    "window_valid_count": int(window_residuals.size),
                    "window_start_timestamp": start_ts.isoformat(),
                    "window_end_timestamp": end_ts.isoformat(),
                    "rolling_residual_mean_wh": float(np.mean(window_residuals)),
                    "rolling_residual_std_wh": float(np.std(window_residuals, ddof=1)),
                    "rolling_mae_wh": float(np.mean(np.abs(window_residuals))),
                    "rolling_rmse_wh": float(np.sqrt(np.mean(window_residuals * window_residuals))),
                    "rolling_underprediction_fraction": n_under / window_size,
                    "rolling_overprediction_fraction": n_over / window_size,
                    "rolling_exact_fraction": n_exact / window_size,
                    "status": "PASS",
                }
            )
            window_index += 1
    return out
