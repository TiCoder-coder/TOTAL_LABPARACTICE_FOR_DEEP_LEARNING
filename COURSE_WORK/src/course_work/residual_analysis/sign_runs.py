from __future__ import annotations

from typing import Any

import numpy as np

from .autocorrelation import PHASE49_CADENCE_MINUTES, detect_contiguous_segments


def sign_runs_for_seed(
    signs: list[str],
    timestamps: list,
    cadence_minutes: int = PHASE49_CADENCE_MINUTES,
) -> dict[str, Any]:
    """Compute sign runs.

    A run breaks on:
    - sign change
    - exact ZERO
    - temporal gap (delta != cadence_minutes)
    """
    valid_signs = {"UNDERPREDICTION", "OVERPREDICTION", "EXACT"}
    for s in signs:
        if s not in valid_signs:
            raise ValueError(f"Invalid sign label: {s!r}")

    segments = detect_contiguous_segments(timestamps, cadence_minutes=cadence_minutes)

    runs: list[dict[str, Any]] = []
    run_lengths: list[int] = []
    under_run_lengths: list[int] = []
    over_run_lengths: list[int] = []
    exact_zero_total = 0

    for seg_start, seg_end in segments:
        seg_signs = signs[seg_start:seg_end]
        if not seg_signs:
            continue
        current_sign = seg_signs[0]
        current_length = 1
        for s in seg_signs[1:]:
            if s != current_sign:
                runs.append(
                    {
                        "sign": current_sign,
                        "length": current_length,
                        "segment_index": len(runs),
                    }
                )
                run_lengths.append(current_length)
                if current_sign == "UNDERPREDICTION":
                    under_run_lengths.append(current_length)
                elif current_sign == "OVERPREDICTION":
                    over_run_lengths.append(current_length)
                elif current_sign == "EXACT":
                    exact_zero_total += current_length
                current_sign = s
                current_length = 1
            else:
                current_length += 1
        runs.append(
            {
                "sign": current_sign,
                "length": current_length,
                "segment_index": len(runs),
            }
        )
        run_lengths.append(current_length)
        if current_sign == "UNDERPREDICTION":
            under_run_lengths.append(current_length)
        elif current_sign == "OVERPREDICTION":
            over_run_lengths.append(current_length)
        elif current_sign == "EXACT":
            exact_zero_total += current_length

    return {
        "run_count": len(runs),
        "underprediction_run_count": sum(1 for r in runs if r["sign"] == "UNDERPREDICTION"),
        "overprediction_run_count": sum(1 for runs_ in [runs] for r in runs_ if r["sign"] == "OVERPREDICTION"),
        "exact_zero_count": exact_zero_total,
        "underprediction_run_lengths": under_run_lengths,
        "overprediction_run_lengths": over_run_lengths,
        "run_lengths": run_lengths,
        "mean_run_length": float(np.mean(run_lengths)) if run_lengths else float("nan"),
        "median_run_length": float(np.median(run_lengths)) if run_lengths else float("nan"),
        "max_run_length": int(np.max(run_lengths)) if run_lengths else 0,
        "p95_run_length": float(np.quantile(run_lengths, 0.95)) if run_lengths else float("nan"),
        "n_sign_segments": len(segments),
        "run_table": runs,
    }


def sign_run_summary_row(
    seed: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "seed": seed,
        "n_sign_segments": result["n_sign_segments"],
        "run_count": result["run_count"],
        "underprediction_run_count": result["underprediction_run_count"],
        "overprediction_run_count": result["overprediction_run_count"],
        "exact_zero_count": result["exact_zero_count"],
        "mean_run_length": result["mean_run_length"],
        "median_run_length": result["median_run_length"],
        "max_run_length": result["max_run_length"],
        "p95_run_length": result["p95_run_length"],
        "breaks_at_zero": True,
        "breaks_at_gap": True,
        "breaks_at_sign_change": True,
    }
