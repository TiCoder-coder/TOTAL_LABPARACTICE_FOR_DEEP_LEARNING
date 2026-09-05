from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


PHASE49_LJUNG_BOX_LAGS: tuple[int, ...] = (6, 36, 144)
PHASE49_LJUNG_BOX_POLICY = "SECONDARY_DIAGNOSTIC"


def _ljung_box_for_segment(
    residuals_segment: np.ndarray,
    lags: list[int],
) -> dict[int, dict[str, float]]:
    """Compute Ljung-Box for a single contiguous segment.

    For each requested lag k:
      Q_k = n(n+2) sum_{i=1..k} acf_i^2 / (n-i)
      p_k = 1 - chi2.cdf(Q_k, df=k)
    Each lag gets its own Q statistic (partial sum up to that lag).
    """
    n = residuals_segment.size
    if n < 2:
        return {k: {"Q": float("nan"), "p_value": float("nan"), "n": int(n)} for k in lags}
    r = residuals_segment - float(np.mean(residuals_segment))
    denom = float(np.sum(r * r))
    out: dict[int, dict[str, float]] = {}
    if denom == 0.0:
        return {k: {"Q": float("nan"), "p_value": float("nan"), "n": int(n)} for k in lags}

    sorted_lags = sorted(lags)
    q_acc = 0.0
    prev_max = 0
    for k in sorted_lags:
        if n - k < 1:
            out[k] = {"Q": float("nan"), "p_value": float("nan"), "n": int(n)}
            continue
        for i in range(prev_max + 1, k + 1):
            if n - i < 1:
                continue
            acf_i = float(np.sum(r[: n - i] * r[i:]) / denom)
            q_acc += (acf_i * acf_i) / (n - i)
        prev_max = max(prev_max, k)
        q = float(n * (n + 2) * q_acc)
        df = k
        p_value = float(1.0 - stats.chi2.cdf(q, df=df))
        out[k] = {"Q": q, "p_value": p_value, "n": int(n)}
    return out


def ljung_box_for_seed(
    timestamps_segments: list[tuple[int, int]],
    residuals: np.ndarray,
    lags: list[int] | None = None,
) -> dict[str, Any]:
    """Compute Ljung-Box per seed using contiguous segments only.

    - If the seed has multiple contiguous segments, Ljung-Box is run per
      segment AND aggregated descriptively.
    - status is NEVER derived from p_value.
    """
    if lags is None:
        lags = list(PHASE49_LJUNG_BOX_LAGS)
    else:
        lags = sorted(set(lags))
    if tuple(lags) != PHASE49_LJUNG_BOX_LAGS:
        raise ValueError(
            f"Phase49-D Ljung-Box lags must be exactly {PHASE49_LJUNG_BOX_LAGS}; got {lags}"
        )

    if not timestamps_segments:
        return {
            "lags": list(lags),
            "policy": PHASE49_LJUNG_BOX_POLICY,
            "used_for_pass_fail": False,
            "n_segments": 0,
            "per_segment": [],
            "status": "NOT_APPLICABLE",
        }

    per_segment_rows: list[dict[str, Any]] = []
    any_computed = False
    for seg_index, (start, end) in enumerate(timestamps_segments):
        seg_residuals = residuals[start:end]
        n = int(seg_residuals.size)
        if n <= max(lags):
            per_segment_rows.append(
                {
                    "segment_index": seg_index,
                    "n": n,
                    "status": "NOT_APPLICABLE_SEGMENT_TOO_SHORT",
                    **{f"Q_lag{k}": float("nan") for k in lags},
                    **{f"p_value_lag{k}": float("nan") for k in lags},
                }
            )
            continue
        ljung = _ljung_box_for_segment(seg_residuals, lags)
        row: dict[str, Any] = {
            "segment_index": seg_index,
            "n": n,
            "status": "SECONDARY_DIAGNOSTIC",
        }
        for k in lags:
            row[f"Q_lag{k}"] = float(ljung[k]["Q"])
            row[f"p_value_lag{k}"] = float(ljung[k]["p_value"])
        per_segment_rows.append(row)
        any_computed = True

    return {
        "lags": list(lags),
        "policy": PHASE49_LJUNG_BOX_POLICY,
        "used_for_pass_fail": False,
        "n_segments": len(timestamps_segments),
        "per_segment": per_segment_rows,
        "status": "SECONDARY_DIAGNOSTIC" if any_computed else "NOT_APPLICABLE",
    }
