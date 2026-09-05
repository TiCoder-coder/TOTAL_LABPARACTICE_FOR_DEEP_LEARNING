from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn


HUBER_DELTA_MODEL_SPACE = 1.0
LOSS_REDUCTION = "mean"


def criterion_config(training_config: dict[str, Any]) -> dict[str, Any]:
    loss_name = str(training_config.get("loss_name", "")).upper()
    if loss_name == "MSE":
        if training_config.get("huber_delta") is not None:
            raise ValueError("MSE must not carry a Huber delta")
        return {
            "loss_id": "L0",
            "loss_name": "MSELoss",
            "registry_loss_name": "MSE",
            "reduction": LOSS_REDUCTION,
            "huber_delta_model_space": None,
        }
    if loss_name == "HUBER":
        delta = training_config.get("huber_delta")
        if not isinstance(delta, (int, float)) or not math.isfinite(float(delta)):
            raise ValueError("Huber delta must be finite")
        if float(delta) != HUBER_DELTA_MODEL_SPACE:
            raise ValueError("S15 Huber delta must equal 1.0 in model-space")
        return {
            "loss_id": "L1",
            "loss_name": "HuberLoss",
            "registry_loss_name": "HUBER",
            "reduction": LOSS_REDUCTION,
            "huber_delta_model_space": HUBER_DELTA_MODEL_SPACE,
        }
    raise ValueError(f"Unsupported training loss: {loss_name or '<missing>'}")


def build_training_criterion(training_config: dict[str, Any]) -> nn.Module:
    config = criterion_config(training_config)
    if config["registry_loss_name"] == "MSE":
        return nn.MSELoss(reduction=LOSS_REDUCTION)
    return nn.HuberLoss(delta=HUBER_DELTA_MODEL_SPACE, reduction=LOSS_REDUCTION)


def validate_criterion_inputs(prediction: torch.Tensor, target_model: torch.Tensor) -> None:
    if prediction.shape != target_model.shape:
        raise RuntimeError(
            f"Prediction and model-space target shapes must match exactly: "
            f"{tuple(prediction.shape)} != {tuple(target_model.shape)}"
        )
    if prediction.ndim != 2 or prediction.shape[1] != 1:
        raise RuntimeError(f"Loss inputs must have shape [B,1], received {tuple(prediction.shape)}")
    if not bool(torch.isfinite(prediction).all().item()):
        raise RuntimeError("Prediction contains non-finite values")
    if not bool(torch.isfinite(target_model).all().item()):
        raise RuntimeError("Model-space target contains non-finite values")


def huber_regime_diagnostics(
    prediction_model: np.ndarray | torch.Tensor,
    target_model: np.ndarray | torch.Tensor,
    *,
    split_id: str = "VALIDATION",
    delta: float = HUBER_DELTA_MODEL_SPACE,
) -> dict[str, Any]:
    if str(split_id).upper() == "TEST":
        raise PermissionError("Phase 37 Huber diagnostics forbid Test access")
    if str(split_id).upper() != "VALIDATION":
        raise ValueError("Official Phase 37 Huber regime diagnostics require VALIDATION")
    if float(delta) != HUBER_DELTA_MODEL_SPACE:
        raise ValueError("Phase 37 diagnostic delta must equal 1.0 in model-space")
    prediction = (
        prediction_model.detach().cpu().numpy()
        if isinstance(prediction_model, torch.Tensor)
        else np.asarray(prediction_model)
    )
    target = (
        target_model.detach().cpu().numpy()
        if isinstance(target_model, torch.Tensor)
        else np.asarray(target_model)
    )
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match for Huber diagnostics")
    if prediction.ndim != 2 or prediction.shape[1] != 1:
        raise ValueError("Huber diagnostic inputs must have shape [N,1]")
    residual = np.abs(prediction.astype(np.float64) - target.astype(np.float64)).reshape(-1)
    if residual.size == 0 or not np.isfinite(residual).all():
        raise ValueError("Huber diagnostic residuals must be non-empty and finite")
    quadratic = residual <= HUBER_DELTA_MODEL_SPACE
    return {
        "split_id": "VALIDATION",
        "delta_model_space": HUBER_DELTA_MODEL_SPACE,
        "sample_count": int(residual.size),
        "fraction_abs_residual_le_delta": float(np.mean(quadratic)),
        "fraction_abs_residual_gt_delta": float(np.mean(~quadratic)),
        "median_abs_residual": float(np.median(residual)),
        "p75_abs_residual": float(np.percentile(residual, 75)),
        "p90_abs_residual": float(np.percentile(residual, 90)),
        "p95_abs_residual": float(np.percentile(residual, 95)),
        "max_abs_residual": float(np.max(residual)),
        "winner_eligible": False,
        "test_access": "FORBIDDEN",
        "status": "PASS",
    }
