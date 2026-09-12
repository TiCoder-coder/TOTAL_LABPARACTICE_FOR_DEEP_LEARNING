"""Exact E13 level-plus-delta objective in fold-local Y model space."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
import torch
import torch.nn.functional as F
LOSS_POLICY = "HYBRID_LEVEL_PLUS_DELTA"
BETA_MODEL_SPACE = 1.0
ALLOWED_LAMBDAS = (0.10, 0.25, 0.50)
@dataclass(frozen=True)
class HybridLossComponents:
    level_mse: torch.Tensor
    delta_smooth_l1: torch.Tensor
    weighted_delta_loss: torch.Tensor
    total_loss: torch.Tensor
def is_hybrid_policy(training_config: Mapping[str, Any]) -> bool:
    return training_config.get("loss_policy") == LOSS_POLICY
def validate_hybrid_config(training_config: Mapping[str, Any]) -> float:
    if not is_hybrid_policy(training_config): raise ValueError("E13 hybrid loss policy is not enabled")
    if training_config.get("loss_name") != "MSE": raise ValueError("E13 level term must remain MSE")
    value = training_config.get("lambda_delta")
    if value not in ALLOWED_LAMBDAS: raise ValueError("E13 lambda_delta is not predeclared")
    if training_config.get("delta_beta_model_space") != BETA_MODEL_SPACE: raise ValueError("E13 SmoothL1 beta must equal 1.0 in model space")
    return float(value)
def y_scaler_parameters(bundle: Mapping[str, Any]) -> tuple[float, float]:
    scaler = bundle.get("scaler")
    if scaler is None or not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"): raise ValueError("E13 requires a fitted fold-local Y scaler")
    mean, scale = float(scaler.mean_[0]), float(scaler.scale_[0])
    if not scale > 0: raise ValueError("E13 fold-local y_scaler_scale must be positive")
    return mean, scale
def compute_hybrid_loss(prediction_model: torch.Tensor, target_model: torch.Tensor, y_context_raw_wh: torch.Tensor, target_scaler_bundle: Mapping[str, Any], training_config: Mapping[str, Any]) -> HybridLossComponents:
    from course_work.training.losses import validate_criterion_inputs
    validate_criterion_inputs(prediction_model, target_model)
    context_raw = y_context_raw_wh.to(device=prediction_model.device, dtype=prediction_model.dtype)
    if context_raw.shape != target_model.shape or not torch.isfinite(context_raw).all(): raise RuntimeError("E13 y_context_raw_wh must be finite [B,1]")
    y_mean, y_scale = y_scaler_parameters(target_scaler_bundle)
    scale, mean = prediction_model.new_tensor(y_scale), prediction_model.new_tensor(y_mean)
    context_model = (context_raw - mean) / scale
    delta_true_model = target_model - context_model
    delta_pred_model = prediction_model - context_model
    level = F.mse_loss(prediction_model, target_model, reduction="mean")
    delta = F.smooth_l1_loss(delta_pred_model, delta_true_model, beta=BETA_MODEL_SPACE, reduction="mean")
    weighted = validate_hybrid_config(training_config) * delta
    return HybridLossComponents(level, delta, weighted, level + weighted)
def component_values(components: HybridLossComponents) -> dict[str, float]:
    return {"level_mse": float(components.level_mse.detach().item()), "delta_smooth_l1": float(components.delta_smooth_l1.detach().item()), "weighted_delta_loss": float(components.weighted_delta_loss.detach().item()), "total_loss": float(components.total_loss.detach().item())}
