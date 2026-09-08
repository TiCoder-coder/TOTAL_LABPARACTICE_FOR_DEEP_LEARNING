from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from course_workutils.artifacts import get_project_root


PHASE49_B_LONG_TABLE_REL = Path("artifacts/residual_analysis/residual_long_table.csv")


def load_phase49_b_long_table(
    project_root: Path | None = None,
) -> list[dict[str, Any]]:
    root = project_root if project_root is not None else get_project_root()
    path = root / PHASE49_B_LONG_TABLE_REL
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def residuals_for_seed(
    long_table: list[dict[str, Any]],
    seed: str,
) -> np.ndarray:
    rows = [r for r in long_table if r["seed"] == seed]
    if len(rows) != 2961:
        raise ValueError(
            f"Phase49-C expects N=2961 residuals for seed {seed}; got {len(rows)}"
        )
    return np.asarray([float(r["residual_wh"]) for r in rows], dtype=np.float64)


def absolute_errors_for_seed(
    long_table: list[dict[str, Any]],
    seed: str,
) -> np.ndarray:
    rows = [r for r in long_table if r["seed"] == seed]
    if len(rows) != 2961:
        raise ValueError(
            f"Phase49-C expects N=2961 abs errors for seed {seed}; got {len(rows)}"
        )
    return np.asarray([float(r["absolute_error_wh"]) for r in rows], dtype=np.float64)


def compute_distribution_summary(
    residuals: np.ndarray,
    ddof: int = 1,
) -> dict[str, float]:
    if residuals.size != 2961:
        raise ValueError(
            f"Phase49-C distribution expects N=2961; got {residuals.size}"
        )
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    return {
        "n": int(residuals.size),
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals, ddof=ddof)),
        "min": float(np.min(residuals)),
        "p01": float(np.quantile(residuals, 0.01)),
        "p05": float(np.quantile(residuals, 0.05)),
        "q1": float(np.quantile(residuals, 0.25)),
        "median": median,
        "q3": float(np.quantile(residuals, 0.75)),
        "p95": float(np.quantile(residuals, 0.95)),
        "p99": float(np.quantile(residuals, 0.99)),
        "max": float(np.max(residuals)),
        "iqr": float(np.quantile(residuals, 0.75) - np.quantile(residuals, 0.25)),
        "mad": mad,
        "skewness": float(stats.skew(residuals, bias=False)),
        "kurtosis_excess": float(stats.kurtosis(residuals, fisher=True, bias=False)),
    }


DISTRIBUTION_FIELDS: list[str] = [
    "n",
    "mean",
    "std",
    "min",
    "p01",
    "p05",
    "q1",
    "median",
    "q3",
    "p95",
    "p99",
    "max",
    "iqr",
    "mad",
    "skewness",
    "kurtosis_excess",
]


def distribution_summary_rows(
    long_table: list[dict[str, Any]],
    seeds: tuple[str, ...] = ("42", "123", "2026"),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        residuals = residuals_for_seed(long_table, seed)
        summary = compute_distribution_summary(residuals)
        summary_row: dict[str, Any] = {"seed": seed}
        summary_row.update(summary)
        rows.append(summary_row)
    return rows
