from __future__ import annotations

from typing import Any

import numpy as np


def _as_float64_array(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def mae_from_residuals(residuals: list[float]) -> float:
    arr = _as_float64_array(residuals)
    return float(np.mean(np.abs(arr)))


def rmse_from_residuals(residuals: list[float]) -> float:
    arr = _as_float64_array(residuals)
    return float(np.sqrt(np.mean(arr * arr)))


def r2_from_residuals_and_y_true(
    residuals: list[float],
    y_true: list[float],
) -> float:
    if len(residuals) != len(y_true):
        raise ValueError("residuals and y_true length mismatch")
    e = _as_float64_array(residuals)
    y = _as_float64_array(y_true)
    n = y.size
    if n < 2:
        return float("nan")
    y_mean = float(np.mean(y))
    ss_res = float(np.sum(e * e))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def reconstruct_seed_metrics(
    residuals: list[float],
    y_true: list[float],
) -> dict[str, float]:
    return {
        "n": len(residuals),
        "mae_wh": mae_from_residuals(residuals),
        "rmse_wh": rmse_from_residuals(residuals),
        "r2": r2_from_residuals_and_y_true(residuals, y_true),
    }


def compare_recomputed_to_phase47(
    recomputed: dict[str, float],
    phase47: dict[str, float],
    tolerance: float = 0.0,
) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    overall_pass = True
    for metric in ("mae_wh", "rmse_wh", "r2"):
        rec = float(recomputed[metric])
        sto = float(phase47[metric])
        abs_diff = abs(rec - sto)
        status = "PASS" if abs_diff <= tolerance else "FAIL"
        if status != "PASS":
            overall_pass = False
        rows[metric] = {
            "recomputed_value": rec,
            "stored_phase47_value": sto,
            "absolute_difference": abs_diff,
            "tolerance": tolerance,
            "status": status,
        }
    rows["n"] = {
        "recomputed_value": int(recomputed["n"]),
        "stored_phase47_value": int(phase47["n"]),
        "absolute_difference": int(abs(int(recomputed["n"]) - int(phase47["n"]))),
        "tolerance": 0,
        "status": "PASS" if int(recomputed["n"]) == int(phase47["n"]) else "FAIL",
    }
    return {"per_metric": rows, "overall_pass": overall_pass}
