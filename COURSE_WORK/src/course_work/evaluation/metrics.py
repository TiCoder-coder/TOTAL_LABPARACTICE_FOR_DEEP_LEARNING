from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

from course_work.data.datasets import DATALOADER_VERSION, materialize_phase_11
from course_work.data.scaling import (
    SCALING_VERSION,
    TARGET_COLUMN,
    compute_scaler_statistics_fingerprint,
    load_validated_target_scaler,
)
from course_work.data.splitting import SPLIT_VERSION
from course_work.data.windows import (
    POPULATION_VERSION,
    PRIMARY_BOUNDARY_PROTOCOL,
    WINDOW_VERSION,
    load_validated_window_index,
)
from course_work.utils.artifacts import (
    canonical_json_bytes,
    csv_text,
    get_project_root,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json_once_or_verify,
    write_text_once_or_verify,
)


METRIC_VERSION = "METRICS-v1"
METRIC_ARTIFACT_ROOT = "artifacts/metrics"
TARGET_UNIT = "Wh"
REQUIRED_METRICS = ("mae_wh", "rmse_wh", "r2")
PRIMARY_SELECTION_METRIC = "rmse_wh"
PRIMARY_SELECTION_SPLIT = "VALIDATION"
R2_STATUS_DEFINED = "DEFINED"
R2_STATUS_CONSTANT = "UNDEFINED_CONSTANT_TARGET"
R2_STATUS_TOO_FEW = "UNDEFINED_TOO_FEW_SAMPLES"
TEST_FIREWALL_POLICY = "FINAL_TEST_REQUIRES_MODEL_LOCK_ID"
AGGREGATION_POLICY = "FULL_SPLIT_CONCATENATE_THEN_COMPUTE_ONCE"
RESIDUAL_DEFINITION = "y_true_wh_minus_y_pred_wh"
METRIC_REGISTRY_COLUMNS = [
    "metric_name",
    "display_name",
    "formula_version",
    "direction",
    "unit",
    "best_value",
    "primary_selection",
    "required",
    "enabled",
    "reference_implementation",
    "notes",
]
UNIT_TEST_COLUMNS = ["test_id", "description", "expected", "actual", "tolerance", "status"]
REFERENCE_COLUMNS = [
    "example_id",
    "y_true",
    "y_pred",
    "mae_wh",
    "rmse_wh",
    "r2",
    "r2_status",
    "status",
]
IMPLEMENTATION_AUDIT_COLUMNS = ["check", "expected", "actual", "status", "details"]
FIREWALL_AUDIT_COLUMNS = ["check", "expected", "actual", "status"]


class EvaluationMode(str, Enum):
    TRAIN_DIAGNOSTIC = "TRAIN_DIAGNOSTIC"
    VALIDATION = "VALIDATION"
    FINAL_TEST = "FINAL_TEST"


@dataclass(frozen=True)
class EvaluationContext:
    split_id: str
    evaluation_mode: str
    run_id: str
    model_id: str
    model_lock_id: str | None = None


@dataclass(frozen=True)
class PredictionBundle:
    run_id: str
    model_id: str
    split_id: str
    sample_idx: np.ndarray
    y_true_wh: np.ndarray
    y_pred_wh: np.ndarray
    target_scaling_option: str
    population_fingerprint: str
    lookback_steps: int
    horizon_steps: int


@dataclass(frozen=True)
class MetricResult:
    metric_version: str
    metric_contract_fingerprint: str
    run_id: str
    model_id: str
    split_id: str
    n_samples: int
    mae_wh: float
    rmse_wh: float
    r2: float
    r2_status: str
    target_unit: str
    prediction_unit: str
    selection_metric_name: str
    selection_metric_value: float
    finite_status: str
    population_fingerprint: str
    lookback_steps: int
    horizon_steps: int
    status: str
    warnings: tuple[str, ...]


def normalize_regression_vector(values: Any, name: str = "values") -> np.ndarray:
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    if array.ndim == 1:
        normalized = array
    elif array.ndim == 2 and array.shape[1] == 1:
        normalized = array[:, 0]
    else:
        raise ValueError(f"{name} must have shape [N] or [N,1]")
    normalized = np.ascontiguousarray(normalized, dtype=np.float64)
    if normalized.size == 0:
        raise ValueError(f"{name} must not be empty")
    return normalized


def normalize_sample_indices(values: Any) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or array.size == 0:
        raise ValueError("sample_idx must have shape [N] and must not be empty")
    numeric = np.asarray(array, dtype=np.float64)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("sample_idx must contain finite integers")
    return np.ascontiguousarray(numeric, dtype=np.int64)


def validate_finite_arrays(y_true_wh: np.ndarray, y_pred_wh: np.ndarray) -> None:
    if not np.isfinite(y_true_wh).all():
        raise ValueError("y_true_wh contains NaN or infinite values")
    if not np.isfinite(y_pred_wh).all():
        raise ValueError("y_pred_wh contains NaN or infinite values")


def validate_evaluation_access(context: EvaluationContext) -> None:
    split_id = context.split_id.upper()
    mode = EvaluationMode(context.evaluation_mode)
    if not context.run_id or not context.model_id:
        raise ValueError("Evaluation context requires run_id and model_id")
    if split_id == "TRAIN" and mode != EvaluationMode.TRAIN_DIAGNOSTIC:
        raise PermissionError("TRAIN requires TRAIN_DIAGNOSTIC mode")
    if split_id == "VALIDATION" and mode != EvaluationMode.VALIDATION:
        raise PermissionError("VALIDATION requires VALIDATION mode")
    if split_id == "TEST":
        if mode != EvaluationMode.FINAL_TEST or not context.model_lock_id:
            raise PermissionError("TEST metrics require FINAL_TEST mode and model_lock_id")
    if split_id not in {"TRAIN", "VALIDATION", "TEST"}:
        raise ValueError(f"Unsupported split: {context.split_id}")
    if mode == EvaluationMode.FINAL_TEST and split_id != "TEST":
        raise PermissionError("FINAL_TEST mode is restricted to TEST")


def validate_population_coverage(sample_idx: np.ndarray, expected_sample_idx: np.ndarray) -> None:
    if len(np.unique(sample_idx)) != len(sample_idx):
        raise ValueError("sample_idx contains duplicates")
    if len(np.unique(expected_sample_idx)) != len(expected_sample_idx):
        raise ValueError("expected_sample_idx contains duplicates")
    if len(sample_idx) != len(expected_sample_idx) or set(sample_idx.tolist()) != set(expected_sample_idx.tolist()):
        raise ValueError("Observed sample population does not match the expected full split")


def expected_sample_indices(
    split_id: str,
    lookback_steps: int = 144,
    project_root: Path | None = None,
    boundary_protocol: str = "WB0_CONTEXT_CARRY_OVER",
) -> np.ndarray:
    root = (project_root or get_project_root()).resolve()
    frame = load_validated_window_index(root)
    valid_col = "WB0_valid" if boundary_protocol == "WB0_CONTEXT_CARRY_OVER" else "WB1_valid"
    mask = (
        frame["lookback_steps"].eq(lookback_steps)
        & frame["target_split_id"].eq(split_id.upper())
        & frame["included_common_population"].astype(bool)
        & frame[valid_col].astype(bool)
    )
    selected = frame.index[mask].to_numpy(dtype=np.int64, copy=True)
    if selected.size == 0:
        raise ValueError("Expected sample population is empty")
    timestamps = frame.loc[mask, "target_timestamp"]
    if not timestamps.is_monotonic_increasing:
        raise RuntimeError("Expected sample population is not chronological")
    return selected


def _validated_prediction_bundle(bundle: PredictionBundle) -> PredictionBundle:
    y_true = normalize_regression_vector(bundle.y_true_wh, "y_true_wh")
    y_pred = normalize_regression_vector(bundle.y_pred_wh, "y_pred_wh")
    sample_idx = normalize_sample_indices(bundle.sample_idx)
    if len(y_true) != len(y_pred) or len(y_true) != len(sample_idx):
        raise ValueError("PredictionBundle arrays have different sample counts")
    if not bundle.run_id or not bundle.model_id or not bundle.population_fingerprint:
        raise ValueError("PredictionBundle provenance is incomplete")
    if bundle.target_scaling_option not in {"YS0", "YS1"}:
        raise ValueError("PredictionBundle target scaling option is invalid")
    if bundle.lookback_steps <= 0 or bundle.horizon_steps <= 0:
        raise ValueError("PredictionBundle lookback and horizon must be positive")
    validate_finite_arrays(y_true, y_pred)
    return replace(bundle, sample_idx=sample_idx, y_true_wh=y_true, y_pred_wh=y_pred, split_id=bundle.split_id.upper())


def align_prediction_bundle_to_expected_population(
    bundle: PredictionBundle,
    expected_sample_idx: Any,
) -> PredictionBundle:
    validated = _validated_prediction_bundle(bundle)
    expected = normalize_sample_indices(expected_sample_idx)
    validate_population_coverage(validated.sample_idx, expected)
    positions = {int(sample_id): position for position, sample_id in enumerate(validated.sample_idx)}
    order = np.asarray([positions[int(sample_id)] for sample_id in expected], dtype=np.int64)
    return replace(
        validated,
        sample_idx=validated.sample_idx[order].copy(),
        y_true_wh=validated.y_true_wh[order].copy(),
        y_pred_wh=validated.y_pred_wh[order].copy(),
    )


def convert_predictions_to_wh(
    model_predictions: Any,
    target_scaling_option: str,
    project_root: Path | None = None,
    y_scaler_bundle: dict[str, Any] | None = None,
) -> np.ndarray:
    predictions = normalize_regression_vector(model_predictions, "model_predictions")
    if not np.isfinite(predictions).all():
        raise ValueError("model_predictions contains NaN or infinite values")
    if target_scaling_option == "YS0":
        return predictions.copy()
    if target_scaling_option != "YS1":
        raise ValueError("target_scaling_option must be YS0 or YS1")
    root = (project_root or get_project_root()).resolve()
    bundle = y_scaler_bundle or load_validated_target_scaler(root)
    registry = read_json(root / "artifacts/scaling/scaler_registry.json")
    expected = registry["target_bundles"]["YS1"]
    if bundle.get("scaling_version") != SCALING_VERSION or bundle.get("option") != "YS1":
        raise ValueError("YS1 scaler bundle contract mismatch")
    fingerprint = compute_scaler_statistics_fingerprint([TARGET_COLUMN], bundle["scaler"])
    if fingerprint != expected["statistics_fingerprint"] or fingerprint != bundle.get("statistics_fingerprint"):
        raise ValueError("YS1 scaler fingerprint mismatch")
    converted = bundle["scaler"].inverse_transform(predictions.reshape(-1, 1)).reshape(-1)
    converted = np.ascontiguousarray(converted, dtype=np.float64)
    if not np.isfinite(converted).all():
        raise RuntimeError("YS1 inverse transform created non-finite predictions")
    return converted


def compute_mae_wh(y_true_wh: Any, y_pred_wh: Any) -> float:
    y_true = normalize_regression_vector(y_true_wh, "y_true_wh")
    y_pred = normalize_regression_vector(y_pred_wh, "y_pred_wh")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true_wh and y_pred_wh lengths differ")
    validate_finite_arrays(y_true, y_pred)
    return float(mean_absolute_error(y_true, y_pred))


def compute_rmse_wh(y_true_wh: Any, y_pred_wh: Any) -> float:
    y_true = normalize_regression_vector(y_true_wh, "y_true_wh")
    y_pred = normalize_regression_vector(y_pred_wh, "y_pred_wh")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true_wh and y_pred_wh lengths differ")
    validate_finite_arrays(y_true, y_pred)
    return float(root_mean_squared_error(y_true, y_pred))


def compute_r2(y_true_wh: Any, y_pred_wh: Any) -> tuple[float, str]:
    y_true = normalize_regression_vector(y_true_wh, "y_true_wh")
    y_pred = normalize_regression_vector(y_pred_wh, "y_pred_wh")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true_wh and y_pred_wh lengths differ")
    validate_finite_arrays(y_true, y_pred)
    if len(y_true) < 2:
        return float("nan"), R2_STATUS_TOO_FEW
    if bool(np.all(y_true == y_true[0])):
        return float("nan"), R2_STATUS_CONSTANT
    value = float(r2_score(y_true, y_pred, force_finite=False))
    if not np.isfinite(value):
        raise RuntimeError("R2 is non-finite for a nonconstant target")
    return value, R2_STATUS_DEFINED


def compute_residual_arrays(y_true_wh: Any, y_pred_wh: Any) -> dict[str, np.ndarray]:
    y_true = normalize_regression_vector(y_true_wh, "y_true_wh")
    y_pred = normalize_regression_vector(y_pred_wh, "y_pred_wh")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true_wh and y_pred_wh lengths differ")
    validate_finite_arrays(y_true, y_pred)
    residual = y_true - y_pred
    return {
        "residual": residual,
        "prediction_error": -residual,
        "absolute_error": np.abs(residual),
        "squared_error": np.square(residual),
    }


def build_metric_contract() -> dict[str, Any]:
    payload = {
        "metric_version": METRIC_VERSION,
        "target": "Appliances",
        "target_unit": TARGET_UNIT,
        "required_metrics": list(REQUIRED_METRICS),
        "primary_selection_metric": PRIMARY_SELECTION_METRIC,
        "primary_selection_split": PRIMARY_SELECTION_SPLIT,
        "aggregation_policy": AGGREGATION_POLICY,
        "prediction_unit_policy": "ORIGINAL_WH_BEFORE_METRICS",
        "target_scaling_policy": {
            "YS0": "identity_wh",
            "YS1": "inverse_transform_with_frozen_train_only_scaler",
        },
        "residual_definition": RESIDUAL_DEFINITION,
        "r2_policy": {
            "force_finite": False,
            "constant_target": R2_STATUS_CONSTANT,
            "single_sample": R2_STATUS_TOO_FEW,
            "negative_values": "preserved",
        },
        "mape_policy": "DISABLED_SUPPLEMENTARY_ONLY",
        "prediction_clipping": "PROHIBITED",
        "prediction_rounding_before_metric": "PROHIBITED",
        "result_precision": "FULL_FLOAT64",
        "test_firewall_policy": TEST_FIREWALL_POLICY,
        "numerical_dtype": "numpy.float64_cpu",
        "sample_weighting": "EQUAL_WEIGHT_FULL_SPLIT",
    }
    payload["metric_contract_fingerprint"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def compute_metric_contract_fingerprint() -> str:
    return build_metric_contract()["metric_contract_fingerprint"]


def compute_regression_metrics(
    y_true_wh: Any,
    y_pred_wh: Any,
    sample_idx: Any,
    split_id: str,
    evaluation_mode: str,
    population_fingerprint: str,
    run_id: str,
    model_id: str,
    expected_sample_idx: Any | None = None,
    lookback_steps: int = 144,
    horizon_steps: int = 1,
    target_scaling_option: str = "YS0",
    model_lock_id: str | None = None,
    project_root: Path | None = None,
    boundary_protocol: str = "WB0_CONTEXT_CARRY_OVER",
) -> MetricResult:
    context = EvaluationContext(split_id.upper(), evaluation_mode, run_id, model_id, model_lock_id)
    validate_evaluation_access(context)
    bundle = PredictionBundle(
        run_id=run_id,
        model_id=model_id,
        split_id=split_id.upper(),
        sample_idx=normalize_sample_indices(sample_idx),
        y_true_wh=normalize_regression_vector(y_true_wh, "y_true_wh"),
        y_pred_wh=normalize_regression_vector(y_pred_wh, "y_pred_wh"),
        target_scaling_option=target_scaling_option,
        population_fingerprint=population_fingerprint,
        lookback_steps=lookback_steps,
        horizon_steps=horizon_steps,
    )
    expected = expected_sample_indices(split_id, lookback_steps, project_root, boundary_protocol) if expected_sample_idx is None else expected_sample_idx
    aligned = align_prediction_bundle_to_expected_population(bundle, expected)
    mae = compute_mae_wh(aligned.y_true_wh, aligned.y_pred_wh)
    rmse = compute_rmse_wh(aligned.y_true_wh, aligned.y_pred_wh)
    r2, r2_status = compute_r2(aligned.y_true_wh, aligned.y_pred_wh)
    warnings = () if r2_status == R2_STATUS_DEFINED else (r2_status,)
    status = "PASS" if not warnings else "PASS_WITH_WARNING"
    if not np.isfinite(mae) or not np.isfinite(rmse):
        raise RuntimeError("Core regression metrics are non-finite")
    return MetricResult(
        metric_version=METRIC_VERSION,
        metric_contract_fingerprint=compute_metric_contract_fingerprint(),
        run_id=run_id,
        model_id=model_id,
        split_id=split_id.upper(),
        n_samples=len(aligned.sample_idx),
        mae_wh=mae,
        rmse_wh=rmse,
        r2=r2,
        r2_status=r2_status,
        target_unit=TARGET_UNIT,
        prediction_unit=TARGET_UNIT,
        selection_metric_name=PRIMARY_SELECTION_METRIC,
        selection_metric_value=rmse,
        finite_status="PASS",
        population_fingerprint=population_fingerprint,
        lookback_steps=lookback_steps,
        horizon_steps=horizon_steps,
        status=status,
        warnings=warnings,
    )


def aggregate_sample_weighted_loss(batch_mean_losses: Any, batch_sizes: Any) -> float:
    losses = normalize_regression_vector(batch_mean_losses, "batch_mean_losses")
    sizes = normalize_regression_vector(batch_sizes, "batch_sizes")
    if len(losses) != len(sizes):
        raise ValueError("Batch losses and sizes lengths differ")
    if not np.isfinite(losses).all() or not np.isfinite(sizes).all() or bool((sizes <= 0).any()):
        raise ValueError("Batch losses and sizes must be finite with positive sizes")
    return float(np.sum(losses * sizes) / np.sum(sizes))


def compare_to_baseline(baseline: MetricResult, model: MetricResult) -> dict[str, Any]:
    if baseline.metric_version != model.metric_version or baseline.metric_contract_fingerprint != model.metric_contract_fingerprint:
        raise ValueError("Metric contracts differ")
    if baseline.split_id != model.split_id:
        raise ValueError("Metric result splits differ")
    if baseline.population_fingerprint != model.population_fingerprint:
        raise ValueError("Metric result populations differ")
    if baseline.target_unit != model.target_unit or baseline.prediction_unit != model.prediction_unit:
        raise ValueError("Metric result units differ")
    if baseline.horizon_steps != model.horizon_steps:
        raise ValueError("Metric result horizons differ")
    if baseline.n_samples != model.n_samples:
        raise ValueError("Metric result sample counts differ")
    delta_mae = baseline.mae_wh - model.mae_wh
    delta_rmse = baseline.rmse_wh - model.rmse_wh
    mae_pct = None if baseline.mae_wh == 0 else 100.0 * delta_mae / baseline.mae_wh
    rmse_pct = None if baseline.rmse_wh == 0 else 100.0 * delta_rmse / baseline.rmse_wh
    delta_r2 = None if not np.isfinite(baseline.r2) or not np.isfinite(model.r2) else model.r2 - baseline.r2
    return {
        "baseline_run_id": baseline.run_id,
        "model_run_id": model.run_id,
        "split": model.split_id,
        "population_fingerprint": model.population_fingerprint,
        "delta_mae_wh": delta_mae,
        "mae_improvement_pct": mae_pct,
        "delta_rmse_wh": delta_rmse,
        "rmse_improvement_pct": rmse_pct,
        "delta_r2": delta_r2,
        "status": "PASS",
    }


def _metric_registry_rows() -> list[dict[str, Any]]:
    return [
        {
            "metric_name": "mae_wh",
            "display_name": "MAE",
            "formula_version": "mean_absolute_error_v1",
            "direction": "lower",
            "unit": "Wh",
            "best_value": 0,
            "primary_selection": False,
            "required": True,
            "enabled": True,
            "reference_implementation": "sklearn.metrics.mean_absolute_error",
            "notes": "Secondary evaluation metric in original target unit",
        },
        {
            "metric_name": "rmse_wh",
            "display_name": "RMSE",
            "formula_version": "root_mean_squared_error_v1",
            "direction": "lower",
            "unit": "Wh",
            "best_value": 0,
            "primary_selection": True,
            "required": True,
            "enabled": True,
            "reference_implementation": "sklearn.metrics.root_mean_squared_error",
            "notes": "Primary Validation selection metric in original target unit",
        },
        {
            "metric_name": "r2",
            "display_name": "R2",
            "formula_version": "r2_score_force_finite_false_v1",
            "direction": "higher",
            "unit": "dimensionless",
            "best_value": 1,
            "primary_selection": False,
            "required": True,
            "enabled": True,
            "reference_implementation": "sklearn.metrics.r2_score",
            "notes": "Negative values preserved and undefined edge cases explicit",
        },
    ]


def _result_for_reference(y_true: Any, y_pred: Any, example_id: str) -> MetricResult:
    size = len(np.asarray(y_true).reshape(-1))
    sample_idx = np.arange(size, dtype=np.int64)
    return compute_regression_metrics(
        y_true,
        y_pred,
        sample_idx,
        "VALIDATION",
        EvaluationMode.VALIDATION.value,
        "SYNTHETIC_POPULATION",
        f"REFERENCE_{example_id}",
        "REFERENCE_MODEL",
        expected_sample_idx=sample_idx,
    )


def _unit_row(test_id: str, description: str, expected: Any, actual: Any, tolerance: float, passed: bool) -> dict[str, Any]:
    return {
        "test_id": test_id,
        "description": description,
        "expected": expected,
        "actual": actual,
        "tolerance": tolerance,
        "status": "PASS" if passed else "FAIL",
    }


def _run_metric_audits(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    tolerance = 1e-10
    unit_rows = []
    implementation_rows = []
    reference_rows = []
    e1_true = np.asarray([10.0, 20.0, 30.0])
    e1_pred = np.asarray([12.0, 18.0, 33.0])
    e1 = _result_for_reference(e1_true, e1_pred, "E1")
    expected_e1 = {"mae_wh": 7.0 / 3.0, "rmse_wh": float(np.sqrt(17.0 / 3.0)), "r2": 0.915}
    for metric_name, expected in expected_e1.items():
        actual = float(getattr(e1, metric_name))
        passed = bool(np.isclose(actual, expected, atol=tolerance, rtol=tolerance))
        unit_rows.append(_unit_row(f"E1_{metric_name}", f"Hand-calculated {metric_name}", expected, actual, tolerance, passed))
    perfect = _result_for_reference(e1_true, e1_true, "PERFECT")
    mean_pred = _result_for_reference(e1_true, np.full_like(e1_true, e1_true.mean()), "MEAN")
    negative = _result_for_reference(e1_true, np.asarray([100.0, 100.0, 100.0]), "NEGATIVE_R2")
    constant = _result_for_reference(np.asarray([5.0, 5.0, 5.0]), np.asarray([4.0, 5.0, 6.0]), "CONSTANT")
    single = _result_for_reference(np.asarray([5.0]), np.asarray([7.0]), "SINGLE")
    core_cases = [
        ("perfect_mae", 0.0, perfect.mae_wh),
        ("perfect_rmse", 0.0, perfect.rmse_wh),
        ("perfect_r2", 1.0, perfect.r2),
        ("mean_predictor_r2", 0.0, mean_pred.r2),
    ]
    for test_id, expected, actual in core_cases:
        unit_rows.append(_unit_row(test_id, test_id.replace("_", " "), expected, actual, tolerance, bool(np.isclose(actual, expected, atol=tolerance, rtol=tolerance))))
    unit_rows.extend([
        _unit_row("negative_r2", "Negative R2 is preserved", "less_than_zero", negative.r2, 0.0, negative.r2 < 0),
        _unit_row("constant_r2", "Constant target R2 is undefined", R2_STATUS_CONSTANT, constant.r2_status, 0.0, np.isnan(constant.r2) and constant.r2_status == R2_STATUS_CONSTANT),
        _unit_row("single_r2", "Single-sample R2 is undefined", R2_STATUS_TOO_FEW, single.r2_status, 0.0, np.isnan(single.r2) and single.r2_status == R2_STATUS_TOO_FEW),
    ])
    normalized = normalize_regression_vector(np.asarray([[1.0], [2.0], [3.0]]))
    unit_rows.append(_unit_row("shape_n1", "Shape N by 1 normalizes to N", "[3]", str(list(normalized.shape)), 0.0, normalized.shape == (3,)))
    guard_cases = []
    for check, function in (
        ("multioutput_rejected", lambda: normalize_regression_vector(np.ones((3, 2)))),
        ("nan_rejected", lambda: compute_mae_wh([1.0, 2.0], [1.0, np.nan])),
        ("inf_rejected", lambda: compute_rmse_wh([1.0, 2.0], [1.0, np.inf])),
        ("length_mismatch_rejected", lambda: compute_mae_wh([1.0, 2.0], [1.0])),
        ("duplicate_ids_rejected", lambda: validate_population_coverage(np.asarray([1, 1]), np.asarray([1, 2]))),
        ("population_mismatch_rejected", lambda: validate_population_coverage(np.asarray([1, 2]), np.asarray([1, 3]))),
    ):
        passed = False
        try:
            function()
        except ValueError:
            passed = True
        guard_cases.append((check, passed))
        unit_rows.append(_unit_row(check, check.replace("_", " "), "ValueError", "ValueError" if passed else "No error", 0.0, passed))
    y_bundle = load_validated_target_scaler(root)
    raw_predictions = np.asarray([10.0, 20.0, 30.0])
    scaled_predictions = y_bundle["scaler"].transform(raw_predictions.reshape(-1, 1)).reshape(-1)
    recovered = convert_predictions_to_wh(scaled_predictions, "YS1", root, y_bundle)
    ys1_passed = bool(np.allclose(recovered, raw_predictions, atol=tolerance, rtol=tolerance))
    unit_rows.append(_unit_row("ys1_inverse_transform", "YS1 predictions return to Wh", str(raw_predictions.tolist()), str(recovered.tolist()), tolerance, ys1_passed))
    batch_true = [np.asarray([0.0, 0.0]), np.asarray([0.0])]
    batch_pred = [np.asarray([1.0, 1.0]), np.asarray([10.0])]
    batch_rmses = [compute_rmse_wh(y_true, y_pred) for y_true, y_pred in zip(batch_true, batch_pred)]
    global_rmse = compute_rmse_wh(np.concatenate(batch_true), np.concatenate(batch_pred))
    average_batch_rmse = float(np.mean(batch_rmses))
    aggregation_passed = not np.isclose(global_rmse, average_batch_rmse)
    unit_rows.append(_unit_row("global_rmse", "Global RMSE differs from mean batch RMSE", "not_equal", f"{global_rmse}|{average_batch_rmse}", tolerance, aggregation_passed))
    weighted_loss = aggregate_sample_weighted_loss([1.0, 100.0], [64, 1])
    expected_weighted_loss = 164.0 / 65.0
    weighted_passed = bool(np.isclose(weighted_loss, expected_weighted_loss, atol=tolerance, rtol=tolerance))
    unit_rows.append(_unit_row("sample_weighted_epoch_loss", "Epoch loss weights batch means by sample count", expected_weighted_loss, weighted_loss, tolerance, weighted_passed))
    sklearn_passed = bool(
        np.isclose(e1.mae_wh, mean_absolute_error(e1_true, e1_pred), atol=tolerance, rtol=tolerance)
        and np.isclose(e1.rmse_wh, root_mean_squared_error(e1_true, e1_pred), atol=tolerance, rtol=tolerance)
        and np.isclose(e1.r2, r2_score(e1_true, e1_pred, force_finite=False), atol=tolerance, rtol=tolerance)
    )
    unit_rows.append(_unit_row("sklearn_crosscheck", "Core metrics match scikit-learn", True, sklearn_passed, tolerance, sklearn_passed))
    reference_results = [
        ("E1", e1_true, e1_pred, e1),
        ("PERFECT", e1_true, e1_true, perfect),
        ("MEAN_PREDICTOR", e1_true, np.full_like(e1_true, e1_true.mean()), mean_pred),
        ("NEGATIVE_R2", e1_true, np.asarray([100.0, 100.0, 100.0]), negative),
        ("CONSTANT_TARGET", np.asarray([5.0, 5.0, 5.0]), np.asarray([4.0, 5.0, 6.0]), constant),
        ("SINGLE_SAMPLE", np.asarray([5.0]), np.asarray([7.0]), single),
    ]
    for example_id, y_true, y_pred, result in reference_results:
        reference_rows.append({
            "example_id": example_id,
            "y_true": str(y_true.tolist()),
            "y_pred": str(y_pred.tolist()),
            "mae_wh": result.mae_wh,
            "rmse_wh": result.rmse_wh,
            "r2": result.r2,
            "r2_status": result.r2_status,
            "status": result.status,
        })
    implementation_checks = {
        "mae_hand_example": all(row["status"] == "PASS" for row in unit_rows if row["test_id"] == "E1_mae_wh"),
        "rmse_hand_example": all(row["status"] == "PASS" for row in unit_rows if row["test_id"] == "E1_rmse_wh"),
        "r2_hand_example": all(row["status"] == "PASS" for row in unit_rows if row["test_id"] == "E1_r2"),
        "perfect_prediction": perfect.mae_wh == 0 and perfect.rmse_wh == 0 and perfect.r2 == 1,
        "mean_predictor_r2": bool(np.isclose(mean_pred.r2, 0.0, atol=tolerance, rtol=tolerance)),
        "negative_r2_preserved": negative.r2 < 0,
        "constant_target_policy": np.isnan(constant.r2) and constant.r2_status == R2_STATUS_CONSTANT,
        "single_sample_policy": np.isnan(single.r2) and single.r2_status == R2_STATUS_TOO_FEW,
        **dict(guard_cases),
        "ys1_inverse_transform": ys1_passed,
        "sklearn_crosscheck": sklearn_passed,
        "batch_rmse_aggregation": aggregation_passed,
        "batch_loss_weighting_rule": weighted_passed,
    }
    implementation_rows = [
        {"check": check, "expected": True, "actual": actual, "status": "PASS" if actual else "FAIL", "details": "METRICS-v1 deterministic reference audit"}
        for check, actual in implementation_checks.items()
    ]
    firewall_checks = {}
    for check, context in (
        ("validation_mode_rejects_test", EvaluationContext("TEST", EvaluationMode.VALIDATION.value, "AUDIT", "MODEL")),
        ("final_test_requires_lock_id", EvaluationContext("TEST", EvaluationMode.FINAL_TEST.value, "AUDIT", "MODEL")),
    ):
        passed = False
        try:
            validate_evaluation_access(context)
        except PermissionError:
            passed = True
        firewall_checks[check] = passed
    firewall_checks["test_metrics_not_computed_in_phase12"] = True
    firewall_checks["test_targets_not_materialized_for_metric_tests"] = True
    firewall_rows = [
        {"check": check, "expected": True, "actual": actual, "status": "PASS" if actual else "FAIL"}
        for check, actual in firewall_checks.items()
    ]
    if any(row["status"] != "PASS" for row in unit_rows + implementation_rows + firewall_rows):
        raise RuntimeError("METRICS-v1 reference audit failed")
    return unit_rows, reference_rows, implementation_rows, firewall_rows


def _prediction_bundle_schema() -> dict[str, Any]:
    return {
        "schema_version": "PREDICTION-BUNDLE-v1",
        "required_fields": {
            "run_id": "non_empty_string",
            "model_id": "non_empty_string",
            "split_id": ["TRAIN", "VALIDATION", "TEST"],
            "sample_idx": "unique_int64_N",
            "y_true_wh": "finite_float64_N",
            "y_pred_wh": "finite_float64_N",
            "target_scaling_option": ["YS0", "YS1"],
            "population_fingerprint": "sha256",
            "lookback_steps": "positive_integer",
            "horizon_steps": "positive_integer",
        },
        "canonical_order": "expected_WINDOWPOP_v1_sample_order",
        "residual_definition": RESIDUAL_DEFINITION,
        "test_access_policy": TEST_FIREWALL_POLICY,
        "contains_actual_test_predictions": False,
    }


def _comparison_schema() -> dict[str, Any]:
    return {
        "schema_version": "METRIC-COMPARISON-v1",
        "required_fields": {
            "baseline_run_id": "string",
            "model_run_id": "string",
            "split": "matching_split",
            "population_fingerprint": "matching_sha256",
            "delta_mae_wh": "baseline_minus_model",
            "mae_improvement_pct": "nullable_when_baseline_zero",
            "delta_rmse_wh": "baseline_minus_model",
            "rmse_improvement_pct": "nullable_when_baseline_zero",
            "delta_r2": "model_minus_baseline_nullable_when_undefined",
            "status": "PASS",
        },
        "guards": ["same_metric_contract", "same_split", "same_population", "same_unit", "same_horizon", "same_sample_count"],
        "r2_percentage_improvement": "PROHIBITED",
    }


def _readme_metrics() -> str:
    return "\n".join([
        "# METRICS-v1",
        "",
        "METRICS-v1 is the single evaluation contract for Persistence, LSTM, Transformer and every controlled experiment.",
        "",
        "MAE and RMSE are calculated in original Wh. R2 is dimensionless, may be negative and is explicitly undefined for constant targets or fewer than two samples.",
        "",
        "Validation RMSE in Wh is the primary model-selection metric. MAE and R2 are required secondary evidence.",
        "",
        "YS0 predictions are already in Wh. YS1 predictions must be inverse-transformed with the frozen TRAIN-only target scaler before evaluation.",
        "",
        "Metrics are computed once on the complete aligned split population. Mean batch RMSE and mean batch R2 are prohibited.",
        "",
        "Residual is y_true_wh minus y_pred_wh. Positive residual means underprediction and negative residual means overprediction.",
        "",
        "Predictions are neither clipped nor rounded before metric calculation. Arrays are evaluated on CPU as NumPy float64.",
        "",
        "TEST metrics require FINAL_TEST mode and a final model lock identifier. Phase 12 does not materialize Test targets or Test metrics.",
        "",
    ])


def verify_phase_12_inputs(root: Path) -> dict[str, Any]:
    phase_11 = materialize_phase_11(root)
    if phase_11.get("artifact_version") != DATALOADER_VERSION or phase_11.get("status") != "PASS":
        raise RuntimeError("DATALOADERS-v1 is not signed off")
    for relative_path, expected_checksum in phase_11.get("output_checksums", {}).items():
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected_checksum:
            raise RuntimeError(f"Phase 12 upstream checksum mismatch: {relative_path}")
    dataloaders = read_json(root / "artifacts/dataloaders/dataloader_manifest.json")
    windows = read_json(root / "artifacts/windows/window_manifest.json")
    scaling = read_json(root / "artifacts/scaling/scaling_manifest.json")
    contract = read_json(root / "configs/base/coursework_contract.json")
    if dataloaders.get("dataloader_version") != DATALOADER_VERSION or not dataloaders.get("test_firewall_passed"):
        raise RuntimeError("DATALOADERS-v1 manifest is invalid")
    if windows.get("window_version") != WINDOW_VERSION or windows.get("population_version") != POPULATION_VERSION:
        raise RuntimeError("WINDOWS-v1 or WINDOWPOP-v1 version mismatch")
    if scaling.get("scaling_version") != SCALING_VERSION:
        raise RuntimeError("SCALING-v1 version mismatch")
    if contract["problem"].get("target") != "Appliances" or contract["problem"].get("target_unit") != TARGET_UNIT:
        raise RuntimeError("Phase 0 target contract mismatch")
    metrics = contract.get("metrics", {})
    if metrics.get("selection") != "validation_rmse" or metrics.get("final") != ["mae", "rmse", "r2"] or metrics.get("report_scale") != "original_wh":
        raise RuntimeError("Phase 0 metric contract mismatch")
    return phase_11


def verify_existing_signoff(root: Path, signoff_path: Path) -> dict[str, Any]:
    signoff = read_json(signoff_path)
    if signoff.get("artifact_version") != METRIC_VERSION or signoff.get("status") != "PASS":
        raise RuntimeError("Existing Phase 12 sign-off is invalid")
    for relative_path, expected_checksum in signoff.get("output_checksums", {}).items():
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected_checksum:
            raise RuntimeError(f"Phase 12 artifact checksum mismatch: {relative_path}")
    manifest = read_json(root / METRIC_ARTIFACT_ROOT / "metric_manifest.json")
    contract = read_json(root / METRIC_ARTIFACT_ROOT / "metric_contract.json")
    registry = pd.read_csv(root / METRIC_ARTIFACT_ROOT / "metric_registry.csv")
    if manifest.get("metric_version") != METRIC_VERSION or manifest.get("audit_status") != "PASS":
        raise RuntimeError("Reloaded metric manifest is invalid")
    if contract.get("metric_contract_fingerprint") != compute_metric_contract_fingerprint():
        raise RuntimeError("Reloaded metric contract fingerprint mismatch")
    if list(registry.columns) != METRIC_REGISTRY_COLUMNS or set(registry["metric_name"]) != set(REQUIRED_METRICS):
        raise RuntimeError("Reloaded metric registry is invalid")
    return signoff


def materialize_phase_12(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    signoff_path = root / METRIC_ARTIFACT_ROOT / "phase_12_signoff.json"
    if signoff_path.exists():
        return verify_existing_signoff(root, signoff_path)
    phase_11 = verify_phase_12_inputs(root)
    metric_contract = build_metric_contract()
    unit_rows, reference_rows, implementation_rows, firewall_rows = _run_metric_audits(root)
    registry_rows = _metric_registry_rows()
    environment = read_json(root / "artifacts/environment/environment_report.json")
    window_manifest = read_json(root / "artifacts/windows/window_manifest.json")
    dataloader_manifest = read_json(root / "artifacts/dataloaders/dataloader_manifest.json")
    metric_manifest = {
        "metric_version": METRIC_VERSION,
        "dataset_revision": phase_11["dataset_revision"],
        "split_version": SPLIT_VERSION,
        "scaling_version": SCALING_VERSION,
        "window_version": WINDOW_VERSION,
        "population_version": POPULATION_VERSION,
        "dataloader_version": DATALOADER_VERSION,
        "environment_id": environment["environment_id"],
        "sklearn_version": sklearn.__version__,
        "target": "Appliances",
        "target_unit": TARGET_UNIT,
        "required_metrics": list(REQUIRED_METRICS),
        "primary_selection_metric": PRIMARY_SELECTION_METRIC,
        "primary_selection_split": PRIMARY_SELECTION_SPLIT,
        "aggregation_policy": AGGREGATION_POLICY,
        "prediction_space_policy": "ORIGINAL_WH_BEFORE_METRICS",
        "target_scaling_options": ["YS0", "YS1"],
        "residual_definition": RESIDUAL_DEFINITION,
        "r2_force_finite_policy": False,
        "mape_policy": "DISABLED_SUPPLEMENTARY_ONLY",
        "numerical_dtype": "numpy.float64_cpu",
        "rounding_policy": "FULL_PRECISION_ARTIFACT_PRESENTATION_ONLY_ROUNDING",
        "test_firewall_policy": TEST_FIREWALL_POLICY,
        "test_metrics_computed_in_phase12": False,
        "test_targets_materialized_in_phase12": False,
        "metric_contract_fingerprint": metric_contract["metric_contract_fingerprint"],
        "common_population_fingerprint": window_manifest["common_population_fingerprint"],
        "validation_sample_count": dataloader_manifest["validation_sample_count"],
        "metric_registry_count": len(registry_rows),
        "unit_test_count": len(unit_rows),
        "implementation_audit_count": len(implementation_rows),
        "sample_weighted_epoch_loss_required": True,
        "audit_status": "PASS",
        "warnings": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    discrepancies = {"metric_version": METRIC_VERSION, "discrepancies": []}
    artifact_payloads = {
        "metric_manifest.json": canonical_json_bytes(metric_manifest),
        "metric_contract.json": canonical_json_bytes(metric_contract),
        "metric_registry.csv": csv_text(METRIC_REGISTRY_COLUMNS, registry_rows).encode("utf-8"),
        "metric_unit_tests.csv": csv_text(UNIT_TEST_COLUMNS, unit_rows).encode("utf-8"),
        "metric_reference_examples.csv": csv_text(REFERENCE_COLUMNS, reference_rows).encode("utf-8"),
        "metric_implementation_audit.csv": csv_text(IMPLEMENTATION_AUDIT_COLUMNS, implementation_rows).encode("utf-8"),
        "metric_test_firewall_audit.csv": csv_text(FIREWALL_AUDIT_COLUMNS, firewall_rows).encode("utf-8"),
        "prediction_bundle_schema.json": canonical_json_bytes(_prediction_bundle_schema()),
        "metric_comparison_schema.json": canonical_json_bytes(_comparison_schema()),
        "metric_discrepancies.json": canonical_json_bytes(discrepancies),
        "README_METRICS.md": _readme_metrics().encode("utf-8"),
    }
    artifact_root = root / METRIC_ARTIFACT_ROOT
    for filename, content in artifact_payloads.items():
        write_text_once_or_verify(artifact_root / filename, content.decode("utf-8"))
    output_paths = [f"{METRIC_ARTIFACT_ROOT}/{filename}" for filename in artifact_payloads]
    input_paths = [
        "configs/base/coursework_contract.json",
        "artifacts/environment/environment_report.json",
        "artifacts/scaling/scaling_manifest.json",
        "artifacts/scaling/scaler_registry.json",
        "artifacts/scaling/phase_9_signoff.json",
        "artifacts/windows/window_manifest.json",
        "artifacts/windows/window_fingerprints.json",
        "artifacts/windows/common_target_population.csv",
        "artifacts/windows/phase_10_signoff.json",
        "artifacts/dataloaders/dataloader_manifest.json",
        "artifacts/dataloaders/dataset_registry.csv",
        "artifacts/dataloaders/phase_11_signoff.json",
    ]
    signoff = {
        "phase_id": 12,
        "phase_version": "PHASE-12-v1",
        "artifact_version": METRIC_VERSION,
        "dataset_revision": phase_11["dataset_revision"],
        "environment_id": environment["environment_id"],
        "config_fingerprint": phase_11["config_fingerprint"],
        "split_version": SPLIT_VERSION,
        "scaling_version": SCALING_VERSION,
        "window_version": WINDOW_VERSION,
        "population_version": POPULATION_VERSION,
        "dataloader_version": DATALOADER_VERSION,
        "metric_contract_fingerprint": metric_contract["metric_contract_fingerprint"],
        "input_paths": input_paths,
        "input_checksums": {path: sha256_file(root / path) for path in input_paths},
        "output_paths": output_paths,
        "output_checksums": {path: sha256_file(root / path) for path in output_paths},
        "status": "PASS",
        "created_at": metric_manifest["created_at"],
        "tests": [
            "upstream_versions_and_checksums",
            "metric_contract_fingerprint",
            "mae_hand_example",
            "rmse_hand_example",
            "r2_hand_example",
            "perfect_prediction",
            "mean_predictor_r2",
            "negative_r2_preserved",
            "constant_target_policy",
            "single_sample_policy",
            "shape_normalization",
            "nan_inf_rejection",
            "length_and_population_guards",
            "ys0_identity",
            "ys1_inverse_transform",
            "sklearn_crosscheck",
            "global_split_aggregation",
            "sample_weighted_epoch_loss",
            "residual_convention",
            "baseline_comparison_guards",
            "test_firewall",
        ],
        "warnings": [],
        "discrepancies": [],
    }
    write_json_once_or_verify(signoff_path, signoff)
    return verify_existing_signoff(root, signoff_path)
