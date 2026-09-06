"""Phase 52 — integrity / probability / mutation / reproducibility audits."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from .sources import (
    ATTN_MAX_TOLERANCE,
    ATTN_NONNEGATIVE_EPS,
    NUM_HEADS,
    NUM_LAYERS,
    PRED_EQUIV_ATOL,
    PRED_EQUIV_RTOL,
    ROW_SUM_ATOL,
    ROW_SUM_RTOL,
)


def _attention_tensor_checks(
    name: str,
    arr: np.ndarray,
    expected_ndim: int = 4,
    expected_heads: int | None = None,
) -> dict[str, Any]:
    """Per-batch/per-layer integrity check."""
    finite = bool(np.isfinite(arr).all())
    ndim_ok = arr.ndim == expected_ndim
    shape_valid = ndim_ok
    if expected_heads is not None and ndim_ok:
        shape_valid = shape_valid and (arr.shape[1] == expected_heads)
    min_w = float(arr.min()) if arr.size > 0 else float("nan")
    max_w = float(arr.max()) if arr.size > 0 else float("nan")
    return {
        "tensor": name,
        "ndim": int(arr.ndim),
        "expected_ndim": expected_ndim,
        "shape": list(arr.shape),
        "expected_heads": expected_heads,
        "shape_valid": bool(shape_valid),
        "finite": finite,
        "min_weight": min_w,
        "max_weight": max_w,
        "max_bound_pass": bool(max_w <= ATTN_MAX_TOLERANCE) if finite else False,
    }


def probability_audit(
    seed: int,
    attention_layers: list[np.ndarray],
) -> list[dict[str, Any]]:
    """For each (layer, head), compute row-sum statistics over the batch.

    attention_layers[layer] has shape [B, H, L, L].
    """
    rows: list[dict[str, Any]] = []
    for layer_idx0, layer_attn in enumerate(attention_layers):
        # layer_attn shape: [B, H, L, L]
        B, H, L, S = layer_attn.shape
        for head_idx0 in range(H):
            block = layer_attn[:, head_idx0, :, :]  # [B, L, S]
            row_sums = block.sum(axis=-1)  # [B, L]
            row_count = int(row_sums.size)
            min_row_sum = float(row_sums.min())
            max_row_sum = float(row_sums.max())
            mean_row_sum = float(row_sums.mean())
            max_abs_diff = float(np.abs(row_sums - 1.0).max())

            block_min = float(block.min())
            block_max = float(block.max())
            nonnegative = bool(block_min >= ATTN_NONNEGATIVE_EPS)
            row_sum_pass = bool(
                (np.abs(row_sums - 1.0) <= ROW_SUM_ATOL + ROW_SUM_RTOL * 1.0).all()
            )
            rows.append({
                "seed": seed,
                "layer_idx0": layer_idx0,
                "head_idx0": head_idx0,
                "row_count_checked": row_count,
                "min_row_sum": min_row_sum,
                "max_row_sum": max_row_sum,
                "mean_row_sum": mean_row_sum,
                "max_abs_row_sum_minus_1": max_abs_diff,
                "min_weight": block_min,
                "max_weight": block_max,
                "nonnegative_pass": nonnegative,
                "row_sum_pass": row_sum_pass,
                "status": "PASS" if (nonnegative and row_sum_pass) else "FAIL",
            })
    return rows


def tensor_integrity_summary(
    seed: int,
    attention_layers: list[np.ndarray],
) -> list[dict[str, Any]]:
    """Per-layer shape / finite / bounds / min-weight summary."""
    rows: list[dict[str, Any]] = []
    for layer_idx0, layer_attn in enumerate(attention_layers):
        check = _attention_tensor_checks(
            name=f"seed{seed}_layer{layer_idx0}",
            arr=layer_attn,
            expected_ndim=4,
            expected_heads=NUM_HEADS,
        )
        check["seed"] = seed
        check["layer_idx0"] = layer_idx0
        check["expected_layers"] = NUM_LAYERS
        check["observed_Q"] = layer_attn.shape[2] if layer_attn.ndim == 4 else None
        check["observed_S"] = layer_attn.shape[3] if layer_attn.ndim == 4 else None
        check["status"] = (
            "PASS"
            if (
                check["shape_valid"]
                and check["finite"]
                and check["max_bound_pass"]
            )
            else "FAIL"
        )
        rows.append(check)
    return rows


def prediction_equivalence_check(
    seed: int,
    y_pred_inspection: np.ndarray,
    y_pred_frozen_wh: np.ndarray,
) -> dict[str, Any]:
    """Compare inspection-path prediction vs frozen Phase 47 prediction in Wh.

    Both should already be in the same coordinate. Tolerance: rtol=atol=1e-5.
    """
    if y_pred_inspection.shape != y_pred_frozen_wh.shape:
        return {
            "seed": seed,
            "N_checked": int(min(len(y_pred_inspection), len(y_pred_frozen_wh))),
            "comparison_space": "wh",
            "rtol": PRED_EQUIV_RTOL,
            "atol": PRED_EQUIV_ATOL,
            "shape_match": False,
            "max_abs_difference": float("nan"),
            "max_relative_difference": float("nan"),
            "allclose_fraction": 0.0,
            "all_pass": False,
            "status": "FAIL",
        }

    rtol = PRED_EQUIV_RTOL
    atol = PRED_EQUIV_ATOL
    diff = np.abs(y_pred_inspection - y_pred_frozen_wh)
    max_abs = float(diff.max())
    tol = atol + rtol * np.abs(y_pred_frozen_wh)
    allclose_mask = diff <= tol
    frac = float(allclose_mask.mean())
    allclose = bool(allclose_mask.all())
    return {
        "seed": seed,
        "N_checked": int(len(y_pred_frozen_wh)),
        "comparison_space": "wh",
        "rtol": rtol,
        "atol": atol,
        "shape_match": True,
        "max_abs_difference": max_abs,
        "max_relative_difference": float((diff / np.clip(np.abs(y_pred_frozen_wh), 1e-12, None)).max()),
        "allclose_fraction": frac,
        "all_pass": allclose,
        "status": "PASS" if allclose else "FAIL",
    }


def model_mutation_check(
    seed: int,
    fp_before: str,
    fp_after: str,
) -> dict[str, Any]:
    """Compare pre/post extraction state_dict SHA256."""
    same = fp_before == fp_after
    return {
        "seed": seed,
        "fingerprint_before": fp_before,
        "fingerprint_after": fp_after,
        "same": same,
        "parameter_mutation_detected": not same,
        "status": "PASS" if same else "FAIL",
    }


def reproducibility_check(
    seed: int,
    pred_run1: np.ndarray,
    pred_run2: np.ndarray,
    attn_run1: np.ndarray,
    attn_run2: np.ndarray,
    rtol: float = PRED_EQUIV_RTOL,
    atol: float = PRED_EQUIV_ATOL,
) -> dict[str, Any]:
    """Run inspection twice on a small deterministic set and compare."""
    pred_close = bool(np.allclose(pred_run1, pred_run2, rtol=rtol, atol=atol))
    attn_close = bool(np.allclose(attn_run1, attn_run2, rtol=rtol, atol=atol))
    overall = pred_close and attn_close
    return {
        "seed": seed,
        "N_checked": int(pred_run1.shape[0]),
        "predictions_allclose": pred_close,
        "predictions_max_abs_difference": float(np.abs(pred_run1 - pred_run2).max()) if pred_run1.size > 0 else 0.0,
        "attention_allclose": attn_close,
        "attention_max_abs_difference": float(np.abs(attn_run1 - attn_run2).max()) if attn_run1.size > 0 else 0.0,
        "rtol": rtol,
        "atol": atol,
        "status": "PASS" if overall else "FAIL",
    }
