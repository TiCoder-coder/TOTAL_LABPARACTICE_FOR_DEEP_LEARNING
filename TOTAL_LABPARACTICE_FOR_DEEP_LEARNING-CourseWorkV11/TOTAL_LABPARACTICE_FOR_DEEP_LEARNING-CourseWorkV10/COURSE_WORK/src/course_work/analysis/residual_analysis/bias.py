from __future__ import annotations

import csv
from typing import Any

import numpy as np


def _sign_label_exact(residual: float) -> str:
    if residual > 0.0:
        return "UNDERPREDICTION"
    if residual < 0.0:
        return "OVERPREDICTION"
    return "EXACT"


def signed_bias_for_residuals(residuals: np.ndarray) -> dict[str, float]:
    if residuals.size != 2961:
        raise ValueError(
            f"Phase49-C bias expects N=2961; got {residuals.size}"
        )

    under = residuals[residuals > 0.0]
    over = residuals[residuals < 0.0]
    exact = residuals[residuals == 0.0]

    mean_residual = float(np.mean(residuals))
    median_residual = float(np.median(residuals))
    mean_abs_residual = float(np.mean(np.abs(residuals)))

    mean_under = float(np.mean(under)) if under.size > 0 else 0.0
    mean_over = float(np.mean(over)) if over.size > 0 else 0.0

    n = int(residuals.size)
    under_count = int(under.size)
    over_count = int(over.size)
    exact_count = int(exact.size)

    under_frac = under_count / n
    over_frac = over_count / n
    exact_frac = exact_count / n
    fractions_sum_to_one = bool(abs((under_frac + over_frac + exact_frac) - 1.0) < 1e-12)

    return {
        "n": n,
        "mean_residual_wh": mean_residual,
        "median_residual_wh": median_residual,
        "mean_absolute_residual_wh": mean_abs_residual,
        "mean_underprediction_residual_wh": mean_under,
        "mean_overprediction_residual_wh": mean_over,
        "underprediction_count": under_count,
        "overprediction_count": over_count,
        "exact_count": exact_count,
        "underprediction_fraction": under_frac,
        "overprediction_fraction": over_frac,
        "exact_fraction": exact_frac,
        "fractions_sum_to_one": fractions_sum_to_one,
    }


def _sign_label_for_row(row: dict[str, Any]) -> str:
    residual = float(row["residual_wh"])
    return _sign_label_exact(residual)


def recompute_sign_class(long_table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in long_table:
        row = dict(r)
        row["residual_sign_recomputed"] = _sign_label_for_row(r)
        out.append(row)
    return out


def sign_balance_rows(
    long_table: list[dict[str, Any]],
    seeds: tuple[str, ...] = ("42", "123", "2026"),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        seed_rows = [r for r in long_table if r["seed"] == seed]
        residuals = np.asarray(
            [float(r["residual_wh"]) for r in seed_rows], dtype=np.float64
        )
        rows.append(
            {
                "seed": seed,
                "n": int(residuals.size),
                "underprediction_count": int(np.sum(residuals > 0.0)),
                "underprediction_fraction": float(np.mean(residuals > 0.0)),
                "overprediction_count": int(np.sum(residuals < 0.0)),
                "overprediction_fraction": float(np.mean(residuals < 0.0)),
                "exact_count": int(np.sum(residuals == 0.0)),
                "exact_fraction": float(np.mean(residuals == 0.0)),
                "fractions_sum_to_one": bool(
                    abs(
                        (
                            float(np.mean(residuals > 0.0))
                            + float(np.mean(residuals < 0.0))
                            + float(np.mean(residuals == 0.0))
                        )
                        - 1.0
                    )
                    < 1e-12
                ),
            }
        )
    return rows


BIAS_FIELDS: list[str] = [
    "seed",
    "n",
    "mean_residual_wh",
    "median_residual_wh",
    "mean_absolute_residual_wh",
    "mean_underprediction_residual_wh",
    "mean_overprediction_residual_wh",
    "underprediction_count",
    "overprediction_count",
    "exact_count",
    "underprediction_fraction",
    "overprediction_fraction",
    "exact_fraction",
    "fractions_sum_to_one",
]


SIGN_BALANCE_FIELDS: list[str] = [
    "seed",
    "n",
    "underprediction_count",
    "underprediction_fraction",
    "overprediction_count",
    "overprediction_fraction",
    "exact_count",
    "exact_fraction",
    "fractions_sum_to_one",
]
