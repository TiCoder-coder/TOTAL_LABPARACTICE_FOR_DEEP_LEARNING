"""Phase 47 — Test Evaluation.

Evaluation-only Test inference for Phase47.
This module is NEVER imported during Phase45/46 and is only activated after Phase46 release.

Hard rules enforced here:
    - model.eval() + torch.inference_mode() — no dropout
    - NO optimizer.step()
    - NO backward()
    - NO model.train()
    - NO scaler fit/partial_fit
    - All predictions must be finite
    - Metrics computed in raw Wh space
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    def root_mean_squared_error(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

from course_work.data.datasets import (
    PHASE_47_AUTHORIZATION,
    build_test_evaluation_dataset,
)
from course_work.data.scaling import TARGET_COLUMN
from course_work.evaluation.metrics import METRIC_VERSION
from course_work.models.transformer_regressor import TransformerRegressor, TransformerModelConfig
from course_work.phase47 import (
    LOCKED_BOUNDARY_PROTOCOL,
    LOCKED_CANDIDATE,
    LOCKED_CONFIG_FP,
    LOCKED_FEATURES,
    LOCKED_LOOKBACK,
    LOCKED_TARGET_SCALING,
    LSTM_TUNED_DEV,
    OFFICIAL_RUNS,
)
from course_work.phase47.checkpoint_loader import (
    VerifiedCheckpoint,
    build_transformer_model_from_checkpoint,
    load_verified_transformer_checkpoint,
)
from course_work.phase47.scaler_loader import (
    FINAL_SCALING_CHECKSUMS,
    ScalerFitAttemptError,
    load_final_scaling_v1_x_scaler,
    load_final_scaling_v1_y_scaler,
)
from course_work.phase47.test_population import materialize_final_test_pop_v1


# ============================================================================
# Predictions
# ============================================================================

@dataclass
class PredictionBundle:
    """A prediction bundle for Phase47."""
    model_id: str
    run_id: str | None
    seed: int | None
    split_id: str
    sample_idx: np.ndarray
    target_ids: list[str]
    target_timestamps: list[str]
    y_true_wh: np.ndarray
    y_pred_wh: np.ndarray
    residual_wh: np.ndarray
    absolute_error_wh: np.ndarray
    squared_error_wh: np.ndarray
    population_sha256: str
    lookback_steps: int
    horizon_steps: int
    is_finite: bool
    n_samples: int


@dataclass
class MetricBundle:
    """Per-model metric result."""
    model_id: str
    run_id: str | None
    seed: int | None
    split_id: str
    n_samples: int
    mae_wh: float
    rmse_wh: float
    r2: float
    r2_status: str
    population_sha256: str
    lookback_steps: int
    horizon_steps: int
    finite_status: str
    status: str


@dataclass
class SeedAggregateMetrics:
    """Aggregate metrics across 3 Transformer seeds."""
    metric: str
    seed42: float
    seed123: float
    seed2026: float
    mean: float
    sample_sd: float
    min: float
    max: float
    range: float
    seed_count: int = 3


# ============================================================================
# Test Evaluation
# ============================================================================

class Phase47EvaluationError(Exception):
    """Raised for Phase47 evaluation errors."""
    pass


class Phase47TrainingAttemptError(Exception):
    """Raised if training-related calls are detected during Phase47."""
    pass


class Phase47ScalerFitError(Exception):
    """Raised if scaler fit is attempted."""
    pass


def _check_finite(values: np.ndarray, name: str) -> bool:
    """Check if all values are finite."""
    return bool(np.isfinite(values).all())


def _compute_metrics(
    y_true_wh: np.ndarray,
    y_pred_wh: np.ndarray,
) -> tuple[float, float, float, str]:
    """Compute MAE, RMSE, R² in raw Wh space."""
    if not np.isfinite(y_true_wh).all():
        raise Phase47EvaluationError("y_true_wh contains non-finite values")
    if not np.isfinite(y_pred_wh).all():
        raise Phase47EvaluationError("y_pred_wh contains non-finite values")

    mae = float(mean_absolute_error(y_true_wh, y_pred_wh))
    rmse = float(root_mean_squared_error(y_true_wh, y_pred_wh))

    if len(y_true_wh) < 2:
        r2 = float("nan")
        r2_status = "UNDEFINED_TOO_FEW_SAMPLES"
    elif bool(np.all(y_true_wh == y_true_wh[0])):
        r2 = float("nan")
        r2_status = "UNDEFINED_CONSTANT_TARGET"
    else:
        r2 = float(r2_score(y_true_wh, y_pred_wh, force_finite=False))
        if not np.isfinite(r2):
            raise Phase47EvaluationError(f"R² is non-finite: {r2}")
        r2_status = "DEFINED"

    return mae, rmse, r2, r2_status


def evaluate_transformer_seed_on_test(
    seed: int,
    project_root: Path | None = None,
    authorization: str | None = None,
    batch_size: int = 32,
) -> tuple[PredictionBundle, MetricBundle]:
    """Evaluate a single FINAL_REFIT Transformer seed on the Test set.

    This function:
    1. Loads and verifies the checkpoint
    2. Loads the Test windows (raw features)
    3. Applies FINAL_SCALING-v1 X-scaling (transform-only)
    4. Runs inference with eval()+inference_mode() (no dropout)
    5. Inverse-transforms predictions to raw Wh using FINAL_SCALING-v1 Y-scaler
    6. Returns prediction bundle and metrics

    Args:
        seed: Seed (42, 123, or 2026).
        project_root: COURSE_WORK root.
        authorization: Must be PHASE_47_AUTHORIZATION for Test access.
        batch_size: Inference batch size.

    Returns:
        (PredictionBundle, MetricBundle)
    """
    if authorization != PHASE_47_AUTHORIZATION:
        raise PermissionError(
            f"Test evaluation requires authorization {PHASE_47_AUTHORIZATION!r}"
        )

    root = project_root or Path.cwd()

    # Step 1: Load and verify checkpoint
    ckpt = load_verified_transformer_checkpoint(seed, root, strict=True)

    # Step 2: Build model and load weights
    model = build_transformer_model_from_checkpoint(ckpt)
    model.eval()

    # Step 3: Materialize Test population
    test_pop = materialize_final_test_pop_v1(root)
    target_ids = test_pop["target_sample_ids"]
    target_timestamps = test_pop["target_timestamps"]
    test_pop_sha = test_pop["test_population_fingerprint"]

    # Step 4: Get RAW Test windows (not pre-scaled with Phase9)
    # The window_index has target_raw_row_index; we need the input windows
    window_index = pd.read_csv(root / "artifacts/windows/window_index.csv")
    test_windows = window_index[
        (window_index["lookback_steps"] == 72)
        & (window_index["target_split_id"] == "TEST")
        & (window_index["WB0_valid"] == True)
        & (window_index["included_common_population"] == True)
    ].sort_values("target_sample_id")

    # Step 5: Load raw feature view
    from course_work.data.features import load_validated_feature_view
    from course_work.data.feature_sets import load_validated_feature_set_registry
    feature_view = load_validated_feature_view(root)
    feature_registry = load_validated_feature_set_registry(root)
    feature_entry = feature_registry["variants"]["FS2_TF1"]
    locked_features = feature_entry["features"]

    raw_values = feature_view[locked_features].to_numpy(dtype=np.float32, copy=True)

    # Step 6: Build Test windows in raw feature space
    n_test = len(test_windows)
    lookback = 72
    n_features = len(locked_features)

    X_raw = np.zeros((n_test, lookback, n_features), dtype=np.float32)
    y_true_wh = np.zeros(n_test, dtype=np.float64)
    # target_sample_id is a string (e.g., 'TGT_00016774') — store as string array
    target_idx_array = np.array(target_ids, dtype=object)

    target_values = feature_view["Appliances"].to_numpy(dtype=np.float64, copy=True)

    for i, (_, row) in enumerate(test_windows.iterrows()):
        start = int(row["input_start_raw_row_index"])
        end = int(row["input_end_raw_row_index"])
        target_row = int(row["target_raw_row_index"])
        # X_raw[i] = rows[start:end+1]
        X_raw[i] = raw_values[start:end + 1]
        y_true_wh[i] = target_values[target_row]
        # target_sample_id is a string (e.g., 'TGT_00016774') — kept as string
        target_idx_array[i] = row["target_sample_id"]

    # Step 7: Apply FINAL_SCALING-v1 X-scaling (transform only, no fit)
    x_scaler_wrapper = load_final_scaling_v1_x_scaler("FS2_TF1", root)
    # x_scaler_wrapper._scaler is the underlying StandardScaler
    x_scaler_obj = x_scaler_wrapper._scaler

    # Reshape to [N*lookback, n_features], scale, reshape back
    flat = X_raw.reshape(-1, n_features)
    try:
        scaled_flat = x_scaler_wrapper.transform_only(flat)
    except ScalerFitAttemptError as e:
        raise Phase47ScalerFitError(f"Scaler fit attempted during Phase47: {e}")

    X_scaled = scaled_flat.reshape(n_test, lookback, n_features).astype(np.float32)

    # Step 8: Inference with eval() + inference_mode()
    y_pred_scaled = []
    with torch.inference_mode():
        for i in range(n_test):
            x_tensor = torch.tensor(X_scaled[i:i+1], dtype=torch.float32)
            y_model = model(x_tensor)  # [1, 1]
            y_pred_scaled.append(float(y_model.squeeze().cpu().numpy()))

    y_pred_scaled = np.array(y_pred_scaled, dtype=np.float64).reshape(-1, 1)

    # Step 9: Apply FINAL_SCALING-v1 Y inverse transform
    y_scaler_wrapper = load_final_scaling_v1_y_scaler("YS1", root)
    y_scaler_obj = y_scaler_wrapper._scaler
    y_pred_wh = y_scaler_obj.inverse_transform(y_pred_scaled).reshape(-1)

    if not _check_finite(y_true_wh, "y_true_wh"):
        raise Phase47EvaluationError(f"Non-finite y_true for seed {seed}")
    if not _check_finite(y_pred_wh, "y_pred_wh"):
        raise Phase47EvaluationError(f"Non-finite y_pred for seed {seed}")

    # Step 10: Compute residuals and errors
    residual_wh = y_true_wh - y_pred_wh
    abs_error = np.abs(residual_wh)
    sq_error = np.square(residual_wh)

    # Step 11: Create prediction bundle
    bundle = PredictionBundle(
        model_id=f"TRANSFORMER_SEED{seed}",
        run_id=ckpt.run_id,
        seed=seed,
        split_id="TEST",
        sample_idx=target_idx_array,  # string array (e.g., 'TGT_00016774')
        target_ids=target_ids,
        target_timestamps=target_timestamps,
        y_true_wh=y_true_wh,
        y_pred_wh=y_pred_wh,
        residual_wh=residual_wh,
        absolute_error_wh=abs_error,
        squared_error_wh=sq_error,
        population_sha256=test_pop_sha,
        lookback_steps=72,
        horizon_steps=1,
        is_finite=True,
        n_samples=len(y_true_wh),
    )

    # Step 12: Compute metrics
    mae, rmse, r2, r2_status = _compute_metrics(y_true_wh, y_pred_wh)

    metrics = MetricBundle(
        model_id=f"TRANSFORMER_SEED{seed}",
        run_id=ckpt.run_id,
        seed=seed,
        split_id="TEST",
        n_samples=len(y_true_wh),
        mae_wh=mae,
        rmse_wh=rmse,
        r2=r2,
        r2_status=r2_status,
        population_sha256=test_pop_sha,
        lookback_steps=72,
        horizon_steps=1,
        finite_status="ALL_FINITE",
        status="PASS",
    )

    return bundle, metrics


def evaluate_persistence_on_test(
    target_ids: list[str],
    target_timestamps: list[str],
    project_root: Path | None = None,
) -> tuple[PredictionBundle, MetricBundle]:
    """Evaluate Persistence baseline on the exact same Test target IDs.

    Persistence: predict y_hat = y_{j-1} (previous observed Appliances value).

    Args:
        target_ids: Ordered list of Test target IDs.
        target_timestamps: Ordered list of Test target timestamps.
        project_root: COURSE_WORK root.

    Returns:
        (PredictionBundle, MetricBundle)
    """
    root = project_root or Path.cwd()

    # Load raw Appliances values and timestamps
    from course_work.data.features import load_validated_feature_view
    feature_view = load_validated_feature_view(root)

    # Build timestamp -> value lookup using the 'timestamp' column (not the RangeIndex)
    ts_to_value = {str(t): float(v) for t, v in zip(feature_view["timestamp"], feature_view["Appliances"])}

    # Load window index for timestamps
    window_index = pd.read_csv(root / "artifacts/windows/window_index.csv")

    # For each Test target, find the previous observed value
    y_pred_list = []
    y_true_list = []

    for i, (tid, ts) in enumerate(zip(target_ids, target_timestamps)):
        # y_true = actual Appliances at this timestamp
        y_true = ts_to_value.get(str(ts))
        if y_true is None:
            raise Phase47EvaluationError(f"Cannot find y_true for timestamp {ts}")

        y_true_list.append(y_true)

        # For WB0: prior observed value = value at previous timestamp
        # The window_index gives us input timestamps; the target is at the end
        # We need the "previous" timestamp (10 min before)
        ts_dt = pd.Timestamp(ts)
        prev_ts = ts_dt - pd.Timedelta(minutes=10)
        prev_ts_str = str(prev_ts)

        y_prior = ts_to_value.get(prev_ts_str)
        if y_prior is None:
            # If no prior timestamp, use the first available prior
            # (shouldn't happen for WB0 with Test)
            raise Phase47EvaluationError(
                f"Cannot find prior value for timestamp {ts}"
            )
        y_pred_list.append(float(y_prior))

    y_pred_wh = np.array(y_pred_list, dtype=np.float64)
    y_true_wh = np.array(y_true_list, dtype=np.float64)

    if not _check_finite(y_true_wh, "y_true_wh"):
        raise Phase47EvaluationError("Non-finite y_true for Persistence")
    if not _check_finite(y_pred_wh, "y_pred_wh"):
        raise Phase47EvaluationError("Non-finite y_pred for Persistence")

    residual_wh = y_true_wh - y_pred_wh
    mae, rmse, r2, r2_status = _compute_metrics(y_true_wh, y_pred_wh)

    # Compute population fingerprint
    import hashlib, json
    blob = json.dumps(sorted(target_ids), separators=(",", ":")).encode("utf-8")
    pop_sha = hashlib.sha256(blob).hexdigest()

    bundle = PredictionBundle(
        model_id="PERSISTENCE",
        run_id=None,
        seed=None,
        split_id="TEST",
        sample_idx=np.arange(len(target_ids), dtype=np.int64),
        target_ids=target_ids,
        target_timestamps=target_timestamps,
        y_true_wh=y_true_wh,
        y_pred_wh=y_pred_wh,
        residual_wh=residual_wh,
        absolute_error_wh=np.abs(residual_wh),
        squared_error_wh=np.square(residual_wh),
        population_sha256=pop_sha,
        lookback_steps=72,
        horizon_steps=1,
        is_finite=True,
        n_samples=len(y_true_wh),
    )

    metrics = MetricBundle(
        model_id="PERSISTENCE",
        run_id=None,
        seed=None,
        split_id="TEST",
        n_samples=len(y_true_wh),
        mae_wh=mae,
        rmse_wh=rmse,
        r2=r2,
        r2_status=r2_status,
        population_sha256=pop_sha,
        lookback_steps=72,
        horizon_steps=1,
        finite_status="ALL_FINITE",
        status="PASS",
    )

    return bundle, metrics


def aggregate_seed_metrics(
    seed_metrics: list[MetricBundle],
) -> list[SeedAggregateMetrics]:
    """Aggregate per-seed metrics to mean ± sample SD.

    Per Phase47 plan §47: use ddof=1 for sample SD.
    Per Phase47 plan §54: report mean of per-seed R², not R² from averaged predictions.
    """
    if len(seed_metrics) != 3:
        raise ValueError(f"Expected 3 seeds, got {len(seed_metrics)}")

    metrics_by_name = {}
    for m in seed_metrics:
        if m.model_id.startswith("TRANSFORMER_SEED"):
            seed = m.seed
            for name in ["mae_wh", "rmse_wh", "r2"]:
                if name not in metrics_by_name:
                    metrics_by_name[name] = {}
                metrics_by_name[name][seed] = getattr(m, name)

    aggregates = []
    for name, seed_values in metrics_by_name.items():
        vals = [seed_values[42], seed_values[123], seed_values[2026]]
        mean_val = float(np.mean(vals))
        sd_val = float(np.std(vals, ddof=1))
        aggregates.append(SeedAggregateMetrics(
            metric=name,
            seed42=vals[0],
            seed123=vals[1],
            seed2026=vals[2],
            mean=mean_val,
            sample_sd=sd_val,
            min=float(np.min(vals)),
            max=float(np.max(vals)),
            range=float(np.max(vals) - np.min(vals)),
        ))

    return aggregates


def verify_cross_seed_ytrue_equality(
    bundles: list[PredictionBundle],
) -> bool:
    """Verify all Transformer seeds produce the same y_true (hard check per §89)."""
    transformer_bundles = [b for b in bundles if b.model_id.startswith("TRANSFORMER_SEED")]
    if len(transformer_bundles) < 2:
        return True

    first = transformer_bundles[0].y_true_wh
    for bundle in transformer_bundles[1:]:
        if not np.allclose(first, bundle.y_true_wh, rtol=0, atol=0):
            raise Phase47EvaluationError(
                f"Cross-seed y_true mismatch: "
                f"{first[:5]} vs {bundle.y_true_wh[:5]}"
            )
    return True


def verify_rmse_r2_consistency(
    seed_metrics: list[MetricBundle],
) -> bool:
    """Verify RMSE and R² ordering consistency per §55.

    For same y_true: lower RMSE must → higher R².
    """
    for m in seed_metrics:
        if m.r2_status == "DEFINED":
            if m.rmse_wh < 0:
                raise Phase47EvaluationError(f"Negative RMSE: {m.rmse_wh}")
            # Check R² bounds
            if m.r2 > 1.0 + 1e-6:
                raise Phase47EvaluationError(f"R² > 1: {m.r2}")
            if m.r2 < -1.0 - 1e-6:
                # Negative R² is allowed but should be reported
                pass

    # Check ordering consistency
    m_list = sorted(seed_metrics, key=lambda m: m.rmse_wh)
    for i in range(len(m_list) - 1):
        lower_rmse = m_list[i]
        higher_rmse = m_list[i + 1]
        if lower_rmse.r2_status == "DEFINED" and higher_rmse.r2_status == "DEFINED":
            if lower_rmse.r2 <= higher_rmse.r2:
                raise Phase47EvaluationError(
                    f"RMSE/R² ordering inconsistency: "
                    f"RMSE={lower_rmse.rmse_wh:.4f} has R²={lower_rmse.r2:.4f} vs "
                    f"RMSE={higher_rmse.rmse_wh:.4f} has R²={higher_rmse.r2:.4f}"
                )

    return True
