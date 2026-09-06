from __future__ import annotations

from typing import Any

from .sources import load_seed_bundle_rows
from .contract import (
    assert_residual_sign_semantics,
)


def _sign_label(residual: float) -> str:
    if residual > 0.0:
        return "UNDERPREDICTION"
    if residual < 0.0:
        return "OVERPREDICTION"
    return "EXACT"


def reconstruct_residual_row(bundle_row: dict[str, str]) -> dict[str, Any]:
    y_true = float(bundle_row["y_true_wh"])
    y_pred = float(bundle_row["y_pred_wh"])
    residual = y_true - y_pred
    abs_err = abs(residual)
    sq_err = residual * residual
    sign = _sign_label(residual)
    assert_residual_sign_semantics(sign)
    return {
        "target_id": bundle_row["target_id"],
        "target_timestamp": bundle_row["target_timestamp"],
        "seed": bundle_row["seed"],
        "y_true_wh": y_true,
        "y_pred_wh": y_pred,
        "residual_wh": residual,
        "absolute_error_wh": abs_err,
        "squared_error_wh2": sq_err,
        "residual_sign": sign,
    }


def seed_label(bundle_row: dict[str, str]) -> str:
    return str(bundle_row["seed"])


def reconstruct_residual_long_table(
    seed: int,
    source_sha256: str,
    project_root=None,
) -> list[dict[str, Any]]:
    bundle_rows = load_seed_bundle_rows(seed, project_root=project_root)
    return [
        {**reconstruct_residual_row(r), "source_prediction_sha256": source_sha256}
        for r in bundle_rows
    ]
