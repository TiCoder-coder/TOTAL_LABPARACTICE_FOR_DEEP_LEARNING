"""Pure conversion contract for V2 residual-to-persistence prediction."""
from __future__ import annotations

from typing import Any
import numpy as np

DIRECT = "DIRECT"
RESIDUAL_TO_PERSISTENCE = "RESIDUAL_TO_PERSISTENCE"


def _target_std(scaler: Any) -> float:
    value = scaler.get("y_std") if isinstance(scaler, dict) else getattr(scaler, "y_std", None)
    if value is None or not np.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError("Residual YS1 conversion requires a positive fold-local y_std")
    return float(value)


def residual_raw_to_model(delta_raw_wh, scaler, target_option: str = "YS1") -> np.ndarray:
    delta = np.asarray(delta_raw_wh, dtype=np.float64)
    if target_option == "YS0":
        return delta
    if target_option != "YS1":
        raise ValueError(f"Unsupported target scaling option: {target_option}")
    return delta / _target_std(scaler)


def residual_model_to_raw(delta_model, scaler, target_option: str = "YS1") -> np.ndarray:
    delta = np.asarray(delta_model, dtype=np.float64)
    if target_option == "YS0":
        return delta
    if target_option != "YS1":
        raise ValueError(f"Unsupported target scaling option: {target_option}")
    return delta * _target_std(scaler)


def compose_residual_prediction_raw(
    y_context_raw_wh,
    predicted_residual_model,
    scaler,
    target_option: str = "YS1",
) -> np.ndarray:
    """Compose residual prediction: y_hat = y_context + delta.

    Both y_context_raw_wh and predicted_residual_model may be:
      - scalar
      - 1-D array
      - 2-D array [B, 1] (PyTorch model output convention)

    Returns 1-D array of shape (n_elements,) for downstream pooling.
    """
    context = np.asarray(y_context_raw_wh, dtype=np.float64).reshape(-1)
    residual = residual_model_to_raw(predicted_residual_model, scaler, target_option).reshape(-1)
    return context + residual
