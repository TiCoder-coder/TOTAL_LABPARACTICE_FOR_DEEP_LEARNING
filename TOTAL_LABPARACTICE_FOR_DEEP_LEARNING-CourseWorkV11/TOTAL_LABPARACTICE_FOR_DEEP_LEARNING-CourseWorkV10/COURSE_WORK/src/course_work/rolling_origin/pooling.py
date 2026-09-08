"""Phase 44 — True pooled metrics.

Pooled MAE / RMSE / R^2 are computed from concatenated raw residuals
across all 3 folds. NOT from mean(fold_RMSE).

Macro fold metrics are reported alongside but DO NOT participate in
primary ranking.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PooledMetrics:
    candidate_id: str
    fold_count: int
    pooled_mae_wh: float
    pooled_rmse_wh: float
    pooled_r2: float

    def as_row(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "fold_count": self.fold_count,
            "pooled_mae_wh": self.pooled_mae_wh,
            "pooled_rmse_wh": self.pooled_rmse_wh,
            "pooled_r2": self.pooled_r2,
        }


@dataclass(frozen=True)
class MacroRobustnessMetrics:
    candidate_id: str
    fold_count: int
    macro_mae_wh: float
    macro_rmse_wh: float
    worst_fold_rmse_wh: float
    best_fold_rmse_wh: float
    fold_rmse_sd_wh: float

    def as_row(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "fold_count": self.fold_count,
            "macro_mae_wh": self.macro_mae_wh,
            "macro_rmse_wh": self.macro_rmse_wh,
            "worst_fold_rmse_wh": self.worst_fold_rmse_wh,
            "best_fold_rmse_wh": self.best_fold_rmse_wh,
            "fold_rmse_sd_wh": self.fold_rmse_sd_wh,
        }


def compute_pooled_metrics(
    *,
    candidate_id: str,
    y_true_per_fold: list[np.ndarray],
    y_pred_per_fold: list[np.ndarray],
) -> PooledMetrics:
    """Pool residuals from all folds and compute true pooled metrics.

    Pre-conditions:
      - each (y_true, y_pred) pair has the same length
      - target_ids across folds are disjoint (asserted by caller via fold
        population audit)
    """
    if len(y_true_per_fold) != len(y_pred_per_fold):
        raise ValueError("y_true and y_pred fold lists must have same length")
    if len(y_true_per_fold) == 0:
        raise ValueError("At least one fold required")

    y_true = np.concatenate([np.asarray(a, dtype=np.float64) for a in y_true_per_fold])
    y_pred = np.concatenate([np.asarray(a, dtype=np.float64) for a in y_pred_per_fold])
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Pooled shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("Pooled arrays are empty")

    resid = y_true - y_pred
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid * resid)))
    ss_res = float(np.sum(resid * resid))
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    if ss_tot < 1e-12:
        r2 = float("nan")
    else:
        r2 = float(1.0 - ss_res / ss_tot)

    return PooledMetrics(
        candidate_id=candidate_id,
        fold_count=len(y_true_per_fold),
        pooled_mae_wh=mae,
        pooled_rmse_wh=rmse,
        pooled_r2=r2,
    )


def compute_macro_metrics(
    *,
    candidate_id: str,
    fold_rmse_wh: list[float],
    fold_mae_wh: list[float],
) -> MacroRobustnessMetrics:
    """Macro-level summary across folds.

    `macro_rmse_wh` is sqrt(mean(fold_rmse^2)) for completeness; it is
    REPORTED but does NOT participate in primary ranking.
    """
    if len(fold_rmse_wh) != len(fold_mae_wh):
        raise ValueError("fold_rmse_wh and fold_mae_wh must be same length")
    if not fold_rmse_wh:
        raise ValueError("At least one fold required")

    fold_rmse = np.asarray(fold_rmse_wh, dtype=np.float64)
    fold_mae = np.asarray(fold_mae_wh, dtype=np.float64)

    return MacroRobustnessMetrics(
        candidate_id=candidate_id,
        fold_count=len(fold_rmse_wh),
        macro_mae_wh=float(np.mean(fold_mae)),
        macro_rmse_wh=float(np.sqrt(np.mean(fold_rmse * fold_rmse))),
        worst_fold_rmse_wh=float(np.max(fold_rmse)),
        best_fold_rmse_wh=float(np.min(fold_rmse)),
        fold_rmse_sd_wh=float(np.std(fold_rmse)) if len(fold_rmse) >= 2 else 0.0,
    )