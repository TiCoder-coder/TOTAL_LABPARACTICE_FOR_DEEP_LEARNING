from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size or x.size < 2:
        return float("nan")
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(stats.pearsonr(x, y)[0])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size or x.size < 2:
        return float("nan")
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(stats.spearmanr(x, y)[0])


def compute_magnitude_associations(
    residuals: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, Any]]:
    """Compute descriptive associations between residual magnitude and
    plan-authorized variables.

    Pair list:
    1. |residual| vs y_true
    2. |residual| vs y_pred
    3. residual  vs y_true
    4. residual  vs y_pred

    Both Pearson and Spearman are computed.
    NO causal interpretation. NO correction rules.
    """
    abs_resid = np.abs(residuals)
    pairs = [
        ("|residual|", "y_true_wh", abs_resid, y_true),
        ("|residual|", "y_pred_wh", abs_resid, y_pred),
        ("residual", "y_true_wh", residuals, y_true),
        ("residual", "y_pred_wh", residuals, y_pred),
    ]
    rows: list[dict[str, Any]] = []
    for x_label, y_label, x_arr, y_arr in pairs:
        for assoc_type in ("pearson", "spearman"):
            if assoc_type == "pearson":
                value = pearson(x_arr, y_arr)
            else:
                value = spearman(x_arr, y_arr)
            rows.append(
                {
                    "x_variable": x_label,
                    "y_variable": y_label,
                    "association_type": assoc_type,
                    "value": value,
                    "N": int(residuals.size),
                    "status": "DESCRIPTIVE",
                    "causal_interpretation": "NONE",
                }
            )
    return rows
