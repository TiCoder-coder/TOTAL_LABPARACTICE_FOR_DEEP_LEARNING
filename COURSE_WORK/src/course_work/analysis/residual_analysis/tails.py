from __future__ import annotations

from typing import Any

import numpy as np


def compute_tail_diagnostics(abs_errors: np.ndarray) -> dict[str, float]:
    if abs_errors.size != 2961:
        raise ValueError(
            f"Phase49-C tail diagnostics expects N=2961; got {abs_errors.size}"
        )
    p90 = float(np.quantile(abs_errors, 0.90))
    p95 = float(np.quantile(abs_errors, 0.95))
    p99 = float(np.quantile(abs_errors, 0.99))
    return {
        "n": int(abs_errors.size),
        "abs_error_p90": p90,
        "abs_error_p95": p95,
        "abs_error_p99": p99,
        "max_absolute_error": float(np.max(abs_errors)),
        "fraction_above_p90": float(np.mean(abs_errors > p90)),
        "fraction_above_p95": float(np.mean(abs_errors > p95)),
        "fraction_above_p99": float(np.mean(abs_errors > p99)),
    }


TAIL_FIELDS: list[str] = [
    "seed",
    "n",
    "abs_error_p90",
    "abs_error_p95",
    "abs_error_p99",
    "max_absolute_error",
    "fraction_above_p90",
    "fraction_above_p95",
    "fraction_above_p99",
]
