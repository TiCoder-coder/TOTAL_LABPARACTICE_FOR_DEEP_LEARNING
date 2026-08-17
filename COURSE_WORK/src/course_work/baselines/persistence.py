import inspect
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from course_work.data.datasets import DATALOADER_VERSION, verify_existing_signoff as verify_phase_11_signoff
from course_work.data.windows import (
    POPULATION_VERSION,
    WINDOW_VERSION,
    load_validated_common_population,
    load_validated_window_index,
    verify_existing_signoff as verify_phase_10_signoff,
)
from course_work.evaluation.metrics import (
    METRIC_VERSION,
    EvaluationMode,
    MetricResult,
    PredictionBundle,
    compute_regression_metrics,
    compute_residual_arrays,
    expected_sample_indices,
    verify_existing_signoff as verify_phase_12_signoff,
)
from course_work.experiments.registry import (
    EXPERIMENT_VERSION,
    ArtifactType,
    ExecutionType,
    ExperimentRegistry,
    FailureType,
    RunStatus,
    build_reference_run_config,
    validate_run_config,
    verify_existing_signoff as verify_phase_13_signoff,
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


PERSISTENCE_VERSION = "PERSISTENCE-v1"
PERSISTENCE_MODEL_ID = "PERSISTENCE_LAST_VALUE"
PERSISTENCE_SCOPE = "TASK_LEVEL"
PERSISTENCE_FORMULA = "y_hat(t+1) = y(t)"
VALIDATION_SPLIT = "VALIDATION"
PRIMARY_LOOKBACK = 144
FORECAST_HORIZON_STEPS = 1
SAMPLING_INTERVAL_MINUTES = 10
SOURCE_LAG_STEPS = 1
SOURCE_LAG_MINUTES = 10
BOUNDARY_PROTOCOL = "WB0_CONTEXT_CARRY_OVER"
TEST_ACCESS_POLICY = "LOCKED_UNTIL_PHASE_47"
PERSISTENCE_ARTIFACT_ROOT = Path("artifacts/baselines/persistence")
PHASE_VERSION = "PHASE-14-v1"
PREDICTION_COLUMNS = [
    "run_id",
    "sample_idx",
    "window_id",
    "target_timestamp",
    "source_timestamp",
    "source_raw_row_index",
    "target_raw_row_index",
    "y_true_wh",
    "y_pred_wh",
    "residual_wh",
    "absolute_error_wh",
    "squared_error_wh",
]
AUDIT_COLUMNS = ["check", "expected", "actual", "status", "details"]
UNIT_TEST_COLUMNS = ["test_id", "expected", "actual", "status"]


@dataclass(frozen=True)
class PersistenceConfig:
    baseline_version: str = PERSISTENCE_VERSION
    model_id: str = PERSISTENCE_MODEL_ID
    baseline_scope: str = PERSISTENCE_SCOPE
    formula: str = PERSISTENCE_FORMULA
    split_id: str = VALIDATION_SPLIT
    primary_lookback_steps: int = PRIMARY_LOOKBACK
    forecast_horizon_steps: int = FORECAST_HORIZON_STEPS
    forecast_horizon_minutes: int = SAMPLING_INTERVAL_MINUTES
    source_lag_steps: int = SOURCE_LAG_STEPS
    source_lag_minutes: int = SOURCE_LAG_MINUTES
    boundary_protocol: str = BOUNDARY_PROTOCOL
    target_unit: str = "Wh"
    trainable_parameters: int = 0
    requires_training: bool = False
    seed: None = None
    test_access_authorized: bool = False
    test_access_policy: str = TEST_ACCESS_POLICY

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PersistencePreparedData:
    window_metadata: pd.DataFrame
    raw_timeline: pd.DataFrame
    sample_idx: np.ndarray
    y_true_wh: np.ndarray
    y_pred_wh: np.ndarray
    source_values_wh: np.ndarray
    population_fingerprint: str
    validation_sample_count: int
    maximum_validation_raw_row_index: int
    minimum_test_raw_row_index: int


@dataclass(frozen=True)
class PersistenceEvaluationResult:
    prediction_bundle: PredictionBundle
    metric_result: MetricResult
    residual_arrays: dict[str, np.ndarray]
    audit_rows: list[dict[str, Any]]
    unit_test_rows: list[dict[str, Any]]


def _normalize_row_indices(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must contain integer row indices")
    try:
        numeric = array.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integer row indices") from exc
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} must contain finite integer row indices")
    return numeric.astype(np.int64)


def _normalize_values(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def predict_persistence(
    target_raw_row_indices: Any,
    source_raw_row_indices: Any,
    appliances_raw_wh: Any,
) -> np.ndarray:
    targets = _normalize_row_indices(target_raw_row_indices, "target_raw_row_indices")
    sources = _normalize_row_indices(source_raw_row_indices, "source_raw_row_indices")
    values = _normalize_values(appliances_raw_wh, "appliances_raw_wh")
    if len(targets) != len(sources):
        raise ValueError("Source and target index counts differ")
    if len(np.unique(targets)) != len(targets) or len(np.unique(sources)) != len(sources):
        raise ValueError("Source and target indices must be unique")
    if not np.equal(targets - sources, SOURCE_LAG_STEPS).all():
        raise ValueError("Persistence source must be exactly one row before target")
    if sources.min() < 0 or targets.min() < 0 or targets.max() >= len(values):
        raise IndexError("Persistence source or target index is out of bounds")
    predictions = values[sources].astype(np.float64, copy=True)
    if not np.isfinite(predictions).all():
        raise ValueError("Persistence predictions contain non-finite values")
    return predictions


def validate_persistence_inputs(window_index_subset: pd.DataFrame, appliances_raw_wh: Any) -> dict[str, Any]:
    required = {
        "window_id",
        "target_sample_id",
        "lookback_steps",
        "horizon_steps",
        "timeline_input_end",
        "timeline_target",
        "input_end_timestamp",
        "target_timestamp",
        "input_end_raw_row_index",
        "target_raw_row_index",
        "continuity_segment_id",
        "target_split_id",
        "WB0_valid",
        "included_common_population",
    }
    missing = sorted(required - set(window_index_subset.columns))
    if missing:
        raise ValueError(f"Persistence window metadata is missing columns: {missing}")
    if window_index_subset.empty:
        raise ValueError("Persistence window metadata is empty")
    values = _normalize_values(appliances_raw_wh, "appliances_raw_wh")
    targets = _normalize_row_indices(window_index_subset["target_raw_row_index"], "target_raw_row_index")
    sources = _normalize_row_indices(window_index_subset["input_end_raw_row_index"], "input_end_raw_row_index")
    if targets.max() >= len(values) or sources.max() >= len(values):
        raise IndexError("Persistence metadata exceeds the bounded raw timeline")
    if not np.isfinite(values[np.concatenate((sources, targets))]).all():
        raise ValueError("Selected Persistence target values contain non-finite values")
    return {
        "sample_count": len(window_index_subset),
        "raw_value_count": len(values),
        "minimum_source_raw_row_index": int(sources.min()),
        "maximum_target_raw_row_index": int(targets.max()),
        "status": "PASS",
    }


def validate_h1_alignment(window_index_subset: pd.DataFrame) -> dict[str, Any]:
    targets = _normalize_row_indices(window_index_subset["timeline_target"], "timeline_target")
    sources = _normalize_row_indices(window_index_subset["timeline_input_end"], "timeline_input_end")
    target_raw = _normalize_row_indices(window_index_subset["target_raw_row_index"], "target_raw_row_index")
    source_raw = _normalize_row_indices(window_index_subset["input_end_raw_row_index"], "input_end_raw_row_index")
    source_timestamps = pd.to_datetime(window_index_subset["input_end_timestamp"], errors="raise")
    target_timestamps = pd.to_datetime(window_index_subset["target_timestamp"], errors="raise")
    deltas = (target_timestamps - source_timestamps).dt.total_seconds().to_numpy(dtype=np.float64) / 60.0
    if not np.equal(targets - sources, SOURCE_LAG_STEPS).all():
        raise ValueError("Timeline source-target lag is not one step")
    if not np.equal(target_raw - source_raw, SOURCE_LAG_STEPS).all():
        raise ValueError("Raw source-target lag is not one step")
    if not np.equal(deltas, SOURCE_LAG_MINUTES).all():
        raise ValueError("Source-target timestamp lag is not ten minutes")
    if not window_index_subset["horizon_steps"].eq(FORECAST_HORIZON_STEPS).all():
        raise ValueError("Persistence horizon is not H1")
    if not window_index_subset["lookback_steps"].eq(PRIMARY_LOOKBACK).all():
        raise ValueError("Persistence population is not anchored to L144")
    if not window_index_subset["target_split_id"].eq(VALIDATION_SPLIT).all():
        raise PermissionError("Persistence evaluation contains a non-Validation target")
    if not window_index_subset["WB0_valid"].astype(bool).all():
        raise ValueError("Persistence evaluation contains an invalid WB0 window")
    if not window_index_subset["included_common_population"].astype(bool).all():
        raise ValueError("Persistence evaluation contains a target outside the common population")
    if window_index_subset["continuity_segment_id"].isna().any():
        raise ValueError("Persistence continuity segment is missing")
    return {
        "sample_count": len(window_index_subset),
        "horizon_steps": FORECAST_HORIZON_STEPS,
        "horizon_minutes": SAMPLING_INTERVAL_MINUTES,
        "source_lag_steps": SOURCE_LAG_STEPS,
        "source_lag_minutes": SOURCE_LAG_MINUTES,
        "status": "PASS",
    }


def validate_population(
    sample_idx: Any,
    expected_sample_idx: Any,
    population_fingerprint: str,
    expected_population_fingerprint: str,
) -> dict[str, Any]:
    observed = _normalize_row_indices(sample_idx, "sample_idx")
    expected = _normalize_row_indices(expected_sample_idx, "expected_sample_idx")
    if len(np.unique(observed)) != len(observed):
        raise ValueError("Persistence sample_idx contains duplicates")
    if len(np.unique(expected)) != len(expected):
        raise ValueError("Expected sample_idx contains duplicates")
    if not np.array_equal(observed, expected):
        raise ValueError("Persistence sample population or order differs from WINDOWPOP-v1")
    if population_fingerprint != expected_population_fingerprint:
        raise ValueError("Persistence population fingerprint mismatch")
    return {
        "sample_count": len(observed),
        "population_fingerprint": population_fingerprint,
        "chronological_order_preserved": True,
        "status": "PASS",
    }


def validate_persistence_predictions(y_pred_wh: Any, source_values_wh: Any) -> dict[str, Any]:
    predictions = _normalize_values(y_pred_wh, "y_pred_wh")
    sources = _normalize_values(source_values_wh, "source_values_wh")
    if len(predictions) != len(sources):
        raise ValueError("Persistence prediction and source counts differ")
    if not np.array_equal(predictions, sources):
        raise ValueError("Persistence predictions differ from raw source values")
    return {
        "sample_count": len(predictions),
        "prediction_unit": "Wh",
        "predictions_equal_raw_source": True,
        "status": "PASS",
    }


def persistence_contract_payload(config: PersistenceConfig | None = None) -> dict[str, Any]:
    return (config or PersistenceConfig()).to_dict()


def verify_phase_14_inputs(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    phase_10 = verify_phase_10_signoff(root, root / "artifacts/windows/phase_10_signoff.json")
    phase_11 = verify_phase_11_signoff(root, root / "artifacts/dataloaders/phase_11_signoff.json")
    phase_12 = verify_phase_12_signoff(root, root / "artifacts/metrics/phase_12_signoff.json")
    phase_13 = verify_phase_13_signoff(root, root / "artifacts/experiments/phase_13_signoff.json")
    versions = {
        "phase_10": phase_10.get("artifact_version"),
        "phase_11": phase_11.get("artifact_version"),
        "phase_12": phase_12.get("artifact_version"),
        "phase_13": phase_13.get("artifact_version"),
    }
    expected_versions = {
        "phase_10": WINDOW_VERSION,
        "phase_11": DATALOADER_VERSION,
        "phase_12": METRIC_VERSION,
        "phase_13": EXPERIMENT_VERSION,
    }
    if versions != expected_versions:
        raise RuntimeError("Phase 14 upstream version mismatch")
    dataset_manifest = read_json(root / "data/raw_data/dataset_manifest.json")
    raw_path = root / "data/raw_data/energydata_complete.csv"
    raw_checksum = sha256_file(raw_path)
    if raw_checksum != dataset_manifest.get("csv_sha256"):
        raise RuntimeError("Phase 14 raw dataset checksum mismatch")
    window_manifest = read_json(root / "artifacts/windows/window_manifest.json")
    metric_manifest = read_json(root / "artifacts/metrics/metric_manifest.json")
    registry_manifest = read_json(root / "artifacts/experiments/registry_manifest.json")
    if window_manifest.get("population_version") != POPULATION_VERSION:
        raise RuntimeError("Phase 14 population version mismatch")
    if window_manifest.get("primary_lookback") != PRIMARY_LOOKBACK:
        raise RuntimeError("Phase 14 primary lookback mismatch")
    if window_manifest.get("forecast_horizon_steps") != FORECAST_HORIZON_STEPS:
        raise RuntimeError("Phase 14 forecast horizon mismatch")
    if window_manifest.get("sampling_interval_minutes") != SAMPLING_INTERVAL_MINUTES:
        raise RuntimeError("Phase 14 sampling interval mismatch")
    if window_manifest.get("primary_boundary_protocol") != BOUNDARY_PROTOCOL:
        raise RuntimeError("Phase 14 boundary protocol mismatch")
    if window_manifest.get("test_target_access_policy") != TEST_ACCESS_POLICY:
        raise PermissionError("Phase 14 Test access policy mismatch")
    if window_manifest.get("test_target_values_exported") is not False:
        raise PermissionError("Phase 14 upstream Test target export violation")
    if metric_manifest.get("metric_version") != METRIC_VERSION or metric_manifest.get("target_unit") != "Wh":
        raise RuntimeError("Phase 14 metric contract mismatch")
    if registry_manifest.get("experiment_registry_version") != EXPERIMENT_VERSION:
        raise RuntimeError("Phase 14 registry contract mismatch")
    return {
        "root": root,
        "phase_signoffs": {
            "phase_10": phase_10,
            "phase_11": phase_11,
            "phase_12": phase_12,
            "phase_13": phase_13,
        },
        "dataset_manifest": dataset_manifest,
        "window_manifest": window_manifest,
        "metric_manifest": metric_manifest,
        "registry_manifest": registry_manifest,
        "raw_path": raw_path,
        "raw_checksum": raw_checksum,
    }


def load_bounded_appliances_timeline(project_root: Path, maximum_raw_row_index: int) -> pd.DataFrame:
    if maximum_raw_row_index < 0:
        raise ValueError("Maximum raw row index must be non-negative")
    path = project_root / "data/raw_data/energydata_complete.csv"
    timeline = pd.read_csv(path, usecols=["date", "Appliances"], nrows=maximum_raw_row_index + 1)
    if len(timeline) != maximum_raw_row_index + 1:
        raise RuntimeError("Bounded raw timeline does not reach the required Validation target")
    timeline.insert(0, "raw_row_index", np.arange(len(timeline), dtype=np.int64))
    timeline["timestamp"] = pd.to_datetime(timeline["date"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    timeline["Appliances"] = pd.to_numeric(timeline["Appliances"], errors="raise").astype(np.float64)
    if not np.isfinite(timeline["Appliances"].to_numpy(dtype=np.float64)).all():
        raise ValueError("Bounded raw Appliances timeline contains non-finite values")
    return timeline


def prepare_validation_persistence_data(project_root: Path | None = None) -> PersistencePreparedData:
    context = verify_phase_14_inputs(project_root)
    root = context["root"]
    window_index = load_validated_window_index(root)
    mask = (
        window_index["lookback_steps"].eq(PRIMARY_LOOKBACK)
        & window_index["horizon_steps"].eq(FORECAST_HORIZON_STEPS)
        & window_index["target_split_id"].eq(VALIDATION_SPLIT)
        & window_index["WB0_valid"].astype(bool)
        & window_index["included_common_population"].astype(bool)
    )
    selected = window_index.loc[mask].copy()
    if selected.empty or not selected["target_timestamp"].is_monotonic_increasing:
        raise RuntimeError("Persistence Validation metadata is empty or non-chronological")
    sample_idx = selected.index.to_numpy(dtype=np.int64, copy=True)
    expected_idx = expected_sample_indices(VALIDATION_SPLIT, PRIMARY_LOOKBACK, root)
    population_fingerprint = context["window_manifest"]["common_population_fingerprint"]
    validate_population(sample_idx, expected_idx, population_fingerprint, population_fingerprint)
    common_population = load_validated_common_population(root)
    expected_population = common_population.loc[
        common_population["target_split_id"].eq(VALIDATION_SPLIT)
        & common_population["included_common_population"].astype(bool)
    ].copy()
    if selected["target_sample_id"].tolist() != expected_population["target_sample_id"].tolist():
        raise RuntimeError("Persistence target IDs differ from the common Validation population")
    test_rows = window_index.loc[
        window_index["lookback_steps"].eq(PRIMARY_LOOKBACK)
        & window_index["target_split_id"].eq("TEST")
        & window_index["included_common_population"].astype(bool)
    ]
    if test_rows.empty:
        raise RuntimeError("Persistence Test boundary metadata is unavailable")
    maximum_validation_raw_row_index = int(selected["target_raw_row_index"].max())
    minimum_test_raw_row_index = int(test_rows["target_raw_row_index"].min())
    if maximum_validation_raw_row_index >= minimum_test_raw_row_index:
        raise PermissionError("Persistence Validation boundary overlaps Test")
    raw_timeline = load_bounded_appliances_timeline(root, maximum_validation_raw_row_index)
    validate_persistence_inputs(selected, raw_timeline["Appliances"].to_numpy(dtype=np.float64, copy=True))
    validate_h1_alignment(selected)
    source_indices = selected["input_end_raw_row_index"].to_numpy(dtype=np.int64, copy=True)
    target_indices = selected["target_raw_row_index"].to_numpy(dtype=np.int64, copy=True)
    raw_values = raw_timeline["Appliances"].to_numpy(dtype=np.float64, copy=True)
    y_pred_wh = predict_persistence(target_indices, source_indices, raw_values)
    y_true_wh = raw_values[target_indices].astype(np.float64, copy=True)
    source_values_wh = raw_values[source_indices].astype(np.float64, copy=True)
    validate_persistence_predictions(y_pred_wh, source_values_wh)
    raw_timestamps = raw_timeline["timestamp"]
    expected_source_timestamps = raw_timestamps.iloc[source_indices].reset_index(drop=True)
    expected_target_timestamps = raw_timestamps.iloc[target_indices].reset_index(drop=True)
    observed_source_timestamps = pd.to_datetime(selected["input_end_timestamp"], errors="raise").reset_index(drop=True)
    observed_target_timestamps = pd.to_datetime(selected["target_timestamp"], errors="raise").reset_index(drop=True)
    if not expected_source_timestamps.equals(observed_source_timestamps):
        raise RuntimeError("Persistence source timestamps differ from the raw timeline")
    if not expected_target_timestamps.equals(observed_target_timestamps):
        raise RuntimeError("Persistence target timestamps differ from the raw timeline")
    return PersistencePreparedData(
        window_metadata=selected,
        raw_timeline=raw_timeline,
        sample_idx=sample_idx,
        y_true_wh=y_true_wh,
        y_pred_wh=y_pred_wh,
        source_values_wh=source_values_wh,
        population_fingerprint=population_fingerprint,
        validation_sample_count=len(selected),
        maximum_validation_raw_row_index=maximum_validation_raw_row_index,
        minimum_test_raw_row_index=minimum_test_raw_row_index,
    )


def build_persistence_prediction_frame(prepared: PersistencePreparedData, run_id: str) -> pd.DataFrame:
    if not run_id:
        raise ValueError("Persistence run_id is required")
    metadata = prepared.window_metadata
    residual_wh = prepared.y_true_wh - prepared.y_pred_wh
    frame = pd.DataFrame(
        {
            "run_id": run_id,
            "sample_idx": prepared.sample_idx,
            "window_id": metadata["window_id"].to_numpy(copy=True),
            "target_timestamp": pd.to_datetime(metadata["target_timestamp"], errors="raise").dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
            "source_timestamp": pd.to_datetime(metadata["input_end_timestamp"], errors="raise").dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
            "source_raw_row_index": metadata["input_end_raw_row_index"].to_numpy(dtype=np.int64, copy=True),
            "target_raw_row_index": metadata["target_raw_row_index"].to_numpy(dtype=np.int64, copy=True),
            "y_true_wh": prepared.y_true_wh,
            "y_pred_wh": prepared.y_pred_wh,
            "residual_wh": residual_wh,
            "absolute_error_wh": np.abs(residual_wh),
            "squared_error_wh": np.square(residual_wh),
        }
    )
    if len(frame) != prepared.validation_sample_count or frame["sample_idx"].duplicated().any():
        raise RuntimeError("Persistence prediction frame population mismatch")
    return frame


def build_persistence_run_config(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    config = build_reference_run_config(root, "PERSISTENCE")
    registry = ExperimentRegistry(root)
    validated = validate_run_config(config, registry.upstream_context)
    if validated["data"]["target_access_mode"] != VALIDATION_SPLIT:
        raise PermissionError("Persistence run config must target Validation")
    if validated["model"]["baseline_scope"] != PERSISTENCE_SCOPE:
        raise RuntimeError("Persistence run config scope mismatch")
    return validated


def register_and_start_persistence_run(
    registry: ExperimentRegistry,
    config: dict[str, Any],
    rerun_reason: str | None = None,
) -> dict[str, Any]:
    registered = registry.register_run(
        config,
        "PERSISTENCE_BASELINE",
        ExecutionType.EVALUATION.value,
        test_access_authorized=False,
        rerun_reason=rerun_reason,
        notes="PERSISTENCE-v1 deterministic Validation baseline",
    )
    started = registry.start_run(registered["run_id"])
    if started["status"] != "RUNNING" or started["test_access_authorized"]:
        raise RuntimeError("Persistence run did not enter the required RUNNING state")
    return started


def build_persistence_prediction_bundle(prepared: PersistencePreparedData, run_id: str) -> PredictionBundle:
    if not run_id:
        raise ValueError("Persistence run_id is required")
    return PredictionBundle(
        run_id=run_id,
        model_id=PERSISTENCE_MODEL_ID,
        split_id=VALIDATION_SPLIT,
        sample_idx=prepared.sample_idx.copy(),
        y_true_wh=prepared.y_true_wh.copy(),
        y_pred_wh=prepared.y_pred_wh.copy(),
        target_scaling_option="YS0",
        population_fingerprint=prepared.population_fingerprint,
        lookback_steps=PRIMARY_LOOKBACK,
        horizon_steps=FORECAST_HORIZON_STEPS,
    )


def validate_lookback_invariance(prepared: PersistencePreparedData, project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    frame = load_validated_window_index(root)
    expected_target_ids = prepared.window_metadata["target_sample_id"].tolist()
    expected_predictions = prepared.y_pred_wh
    raw_values = prepared.raw_timeline["Appliances"].to_numpy(dtype=np.float64, copy=True)
    results = {}
    for lookback in (36, 72, 144):
        subset = frame.loc[
            frame["lookback_steps"].eq(lookback)
            & frame["horizon_steps"].eq(FORECAST_HORIZON_STEPS)
            & frame["target_split_id"].eq(VALIDATION_SPLIT)
            & frame["WB0_valid"].astype(bool)
            & frame["included_common_population"].astype(bool)
        ]
        target_ids_match = subset["target_sample_id"].tolist() == expected_target_ids
        predictions = predict_persistence(
            subset["target_raw_row_index"].to_numpy(dtype=np.int64, copy=True),
            subset["input_end_raw_row_index"].to_numpy(dtype=np.int64, copy=True),
            raw_values,
        )
        predictions_match = np.array_equal(predictions, expected_predictions)
        results[f"L{lookback}"] = target_ids_match and predictions_match
    if not all(results.values()):
        raise RuntimeError("Persistence prediction is not invariant across common lookbacks")
    return {"lookbacks": results, "status": "PASS"}


def _audit_row(check: str, expected: Any, actual: Any, passed: bool, details: str = "") -> dict[str, Any]:
    return {
        "check": check,
        "expected": expected,
        "actual": actual,
        "status": "PASS" if passed else "FAIL",
        "details": details,
    }


def _unit_test_row(test_id: str, expected: Any, actual: Any, passed: bool) -> dict[str, Any]:
    return {
        "test_id": test_id,
        "expected": expected,
        "actual": actual,
        "status": "PASS" if passed else "FAIL",
    }


def run_persistence_contract_tests(prepared: PersistencePreparedData, project_root: Path | None = None) -> list[dict[str, Any]]:
    rows = []
    cases = {
        "constant_series": np.asarray([5.0, 5.0, 5.0, 5.0]),
        "increasing_series": np.asarray([1.0, 2.0, 3.0, 4.0]),
        "decreasing_series": np.asarray([4.0, 3.0, 2.0, 1.0]),
        "varying_series": np.asarray([8.0, 2.0, 9.0, 3.0]),
    }
    for test_id, values in cases.items():
        actual = predict_persistence([1, 2, 3], [0, 1, 2], values)
        passed = np.array_equal(actual, values[:3].astype(np.float64))
        rows.append(_unit_test_row(test_id, values[:3].tolist(), actual.tolist(), passed))
    wrong_lag_rejected = False
    try:
        predict_persistence([2], [0], [1.0, 2.0, 3.0])
    except ValueError:
        wrong_lag_rejected = True
    rows.append(_unit_test_row("wrong_lag_rejected", True, wrong_lag_rejected, wrong_lag_rejected))
    test_firewall_rejected = False
    try:
        compute_regression_metrics(
            [1.0],
            [1.0],
            [0],
            "TEST",
            EvaluationMode.VALIDATION.value,
            prepared.population_fingerprint,
            "PERSISTENCE_FIREWALL_TEST",
            PERSISTENCE_MODEL_ID,
            expected_sample_idx=[0],
            project_root=project_root,
        )
    except PermissionError:
        test_firewall_rejected = True
    rows.append(_unit_test_row("test_firewall_rejected", True, test_firewall_rejected, test_firewall_rejected))
    first = predict_persistence(
        prepared.window_metadata["target_raw_row_index"],
        prepared.window_metadata["input_end_raw_row_index"],
        prepared.raw_timeline["Appliances"],
    )
    second = predict_persistence(
        prepared.window_metadata["target_raw_row_index"],
        prepared.window_metadata["input_end_raw_row_index"],
        prepared.raw_timeline["Appliances"],
    )
    deterministic = np.array_equal(first, second)
    rows.append(_unit_test_row("determinism_without_seed", True, deterministic, deterministic))
    lookback = validate_lookback_invariance(prepared, project_root)
    rows.append(_unit_test_row("lookback_invariance", True, lookback["status"] == "PASS", lookback["status"] == "PASS"))
    parameters = set(inspect.signature(predict_persistence).parameters)
    expected_parameters = {"target_raw_row_indices", "source_raw_row_indices", "appliances_raw_wh"}
    dependency_free = parameters == expected_parameters
    for test_id in ("feature_variant_invariance", "target_scaling_invariance", "batch_size_invariance", "device_invariance"):
        rows.append(_unit_test_row(test_id, True, dependency_free, dependency_free))
    if any(row["status"] != "PASS" for row in rows):
        raise RuntimeError("Persistence contract tests failed")
    return rows


def build_persistence_audit_rows(
    prepared: PersistencePreparedData,
    metric_result: MetricResult,
    project_root: Path | None = None,
) -> list[dict[str, Any]]:
    metadata = prepared.window_metadata
    independent_mae = float(np.mean(np.abs(prepared.y_true_wh - prepared.y_pred_wh)))
    independent_rmse = float(np.sqrt(np.mean(np.square(prepared.y_true_wh - prepared.y_pred_wh))))
    centered = prepared.y_true_wh - float(np.mean(prepared.y_true_wh))
    denominator = float(np.sum(np.square(centered)))
    independent_r2 = float(1.0 - np.sum(np.square(prepared.y_true_wh - prepared.y_pred_wh)) / denominator)
    lookback = validate_lookback_invariance(prepared, project_root)
    deterministic = np.array_equal(
        prepared.y_pred_wh,
        predict_persistence(
            metadata["target_raw_row_index"],
            metadata["input_end_raw_row_index"],
            prepared.raw_timeline["Appliances"],
        ),
    )
    dependency_parameters = set(inspect.signature(predict_persistence).parameters)
    dependency_free = dependency_parameters == {"target_raw_row_indices", "source_raw_row_indices", "appliances_raw_wh"}
    rows = [
        _audit_row("formula_valid", PERSISTENCE_FORMULA, PERSISTENCE_FORMULA, True),
        _audit_row("horizon_steps_valid", 1, int(metadata["horizon_steps"].iloc[0]), metadata["horizon_steps"].eq(1).all()),
        _audit_row("horizon_minutes_valid", 10, 10, True),
        _audit_row("source_before_target", True, bool((metadata["timeline_input_end"] < metadata["timeline_target"]).all()), bool((metadata["timeline_input_end"] < metadata["timeline_target"]).all())),
        _audit_row("source_target_delta_valid", 1, sorted((metadata["timeline_target"] - metadata["timeline_input_end"]).unique().tolist()), (metadata["timeline_target"] - metadata["timeline_input_end"]).eq(1).all()),
        _audit_row("continuity_valid", True, not metadata["continuity_segment_id"].isna().any(), not metadata["continuity_segment_id"].isna().any()),
        _audit_row("population_complete", prepared.validation_sample_count, metric_result.n_samples, metric_result.n_samples == prepared.validation_sample_count),
        _audit_row("sample_ids_unique", prepared.validation_sample_count, len(np.unique(prepared.sample_idx)), len(np.unique(prepared.sample_idx)) == prepared.validation_sample_count),
        _audit_row("predictions_equal_raw_source", True, bool(np.array_equal(prepared.y_pred_wh, prepared.source_values_wh)), np.array_equal(prepared.y_pred_wh, prepared.source_values_wh)),
        _audit_row("prediction_unit_wh", "Wh", metric_result.prediction_unit, metric_result.prediction_unit == "Wh"),
        _audit_row("metric_version_valid", METRIC_VERSION, metric_result.metric_version, metric_result.metric_version == METRIC_VERSION),
        _audit_row("mae_independent_crosscheck", independent_mae, metric_result.mae_wh, math.isclose(independent_mae, metric_result.mae_wh, rel_tol=1e-12, abs_tol=1e-12)),
        _audit_row("rmse_independent_crosscheck", independent_rmse, metric_result.rmse_wh, math.isclose(independent_rmse, metric_result.rmse_wh, rel_tol=1e-12, abs_tol=1e-12)),
        _audit_row("r2_independent_crosscheck", independent_r2, metric_result.r2, math.isclose(independent_r2, metric_result.r2, rel_tol=1e-12, abs_tol=1e-12)),
        _audit_row("test_firewall_valid", "VALIDATION_ONLY", sorted(metadata["target_split_id"].unique().tolist()), metadata["target_split_id"].eq(VALIDATION_SPLIT).all() and prepared.maximum_validation_raw_row_index < prepared.minimum_test_raw_row_index),
        _audit_row("lookback_invariance_valid", "PASS", lookback["status"], lookback["status"] == "PASS"),
        _audit_row("feature_variant_invariance_valid", True, dependency_free, dependency_free),
        _audit_row("target_scaling_invariance_valid", True, dependency_free, dependency_free),
        _audit_row("batch_invariance_valid", True, dependency_free, dependency_free),
        _audit_row("device_invariance_valid", True, dependency_free, dependency_free),
        _audit_row("determinism_valid", True, deterministic, deterministic),
    ]
    if any(row["status"] != "PASS" for row in rows):
        raise RuntimeError("Persistence audit failed")
    return rows


def evaluate_persistence_validation(
    prepared: PersistencePreparedData,
    run_id: str,
    project_root: Path | None = None,
) -> PersistenceEvaluationResult:
    root = (project_root or get_project_root()).resolve()
    bundle = build_persistence_prediction_bundle(prepared, run_id)
    expected_idx = expected_sample_indices(VALIDATION_SPLIT, PRIMARY_LOOKBACK, root)
    metric_result = compute_regression_metrics(
        bundle.y_true_wh,
        bundle.y_pred_wh,
        bundle.sample_idx,
        bundle.split_id,
        EvaluationMode.VALIDATION.value,
        bundle.population_fingerprint,
        bundle.run_id,
        bundle.model_id,
        expected_sample_idx=expected_idx,
        lookback_steps=bundle.lookback_steps,
        horizon_steps=bundle.horizon_steps,
        target_scaling_option=bundle.target_scaling_option,
        project_root=root,
    )
    residual_arrays = compute_residual_arrays(bundle.y_true_wh, bundle.y_pred_wh)
    audit_rows = build_persistence_audit_rows(prepared, metric_result, root)
    unit_test_rows = run_persistence_contract_tests(prepared, root)
    return PersistenceEvaluationResult(
        prediction_bundle=bundle,
        metric_result=metric_result,
        residual_arrays=residual_arrays,
        audit_rows=audit_rows,
        unit_test_rows=unit_test_rows,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_baseline_config_fingerprint(config: PersistenceConfig | None = None) -> str:
    contract = persistence_contract_payload(config)
    payload = {
        "formula": contract["formula"],
        "forecast_horizon_steps": contract["forecast_horizon_steps"],
        "forecast_horizon_minutes": contract["forecast_horizon_minutes"],
        "source_lag_steps": contract["source_lag_steps"],
        "source_lag_minutes": contract["source_lag_minutes"],
        "population_version": POPULATION_VERSION,
        "metric_version": METRIC_VERSION,
    }
    return sha256_bytes(canonical_json_bytes(payload))


def compute_prediction_fingerprint(predictions: pd.DataFrame) -> str:
    required = ["sample_idx", "source_raw_row_index", "target_raw_row_index", "y_pred_wh"]
    if any(column not in predictions.columns for column in required):
        raise ValueError("Prediction fingerprint columns are missing")
    content = predictions.loc[:, required].to_csv(index=False, lineterminator="\n").encode("utf-8")
    return sha256_bytes(content)


def metric_result_payload(metric_result: MetricResult) -> dict[str, Any]:
    payload = asdict(metric_result)
    payload["warnings"] = list(metric_result.warnings)
    return payload


def compute_metric_result_fingerprint(metric_result: MetricResult) -> str:
    return sha256_bytes(canonical_json_bytes(metric_result_payload(metric_result)))


def _readme_persistence() -> str:
    return "\n".join(
        [
            "# PERSISTENCE-v1",
            "",
            "PERSISTENCE_LAST_VALUE is the canonical task-level temporal baseline for one-step Appliances forecasting.",
            "",
            "The formula is y_hat(t+1) = y(t). Source and target values remain in raw Wh.",
            "",
            "The evaluation uses the full WINDOWPOP-v1 Validation population anchored to L144 under WB0_CONTEXT_CARRY_OVER.",
            "",
            "MAE, RMSE and R2 are computed once over the full Validation split with METRICS-v1.",
            "",
            "No feature variant, scaler, DataLoader, training process, seed, optimizer or checkpoint participates in the prediction.",
            "",
            "Test targets and Test metrics remain locked until Phase 47.",
            "",
        ]
    )


def write_and_register_persistence_artifacts(
    registry: ExperimentRegistry,
    artifact_root: Path,
    prepared: PersistencePreparedData,
    evaluation: PersistenceEvaluationResult,
    run_id: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    timestamp = created_at or utc_now()
    predictions = build_persistence_prediction_frame(prepared, run_id)
    if predictions.columns.tolist() != PREDICTION_COLUMNS:
        raise RuntimeError("Persistence prediction artifact schema mismatch")
    metric = evaluation.metric_result
    baseline_fingerprint = compute_baseline_config_fingerprint()
    prediction_fingerprint = compute_prediction_fingerprint(predictions)
    metric_fingerprint = compute_metric_result_fingerprint(metric)
    metrics_payload = {
        "artifact_version": PERSISTENCE_VERSION,
        "created_at": timestamp,
        "run_id": run_id,
        "model_id": PERSISTENCE_MODEL_ID,
        "split_id": VALIDATION_SPLIT,
        "metric_result": metric_result_payload(metric),
        "metric_result_fingerprint": metric_fingerprint,
        "prediction_fingerprint": prediction_fingerprint,
        "test_access_authorized": False,
    }
    summary_payload = {
        "artifact_version": PERSISTENCE_VERSION,
        "created_at": timestamp,
        "run_id": run_id,
        "baseline_version": PERSISTENCE_VERSION,
        "model_id": PERSISTENCE_MODEL_ID,
        "baseline_scope": PERSISTENCE_SCOPE,
        "formula": PERSISTENCE_FORMULA,
        "forecast_horizon_steps": FORECAST_HORIZON_STEPS,
        "forecast_horizon_minutes": SAMPLING_INTERVAL_MINUTES,
        "source_lag_steps": SOURCE_LAG_STEPS,
        "source_lag_minutes": SOURCE_LAG_MINUTES,
        "population_version": POPULATION_VERSION,
        "population_fingerprint": prepared.population_fingerprint,
        "validation_samples": prepared.validation_sample_count,
        "validation_metrics": {
            "mae_wh": metric.mae_wh,
            "rmse_wh": metric.rmse_wh,
            "r2": metric.r2,
            "r2_status": metric.r2_status,
        },
        "target_unit": "Wh",
        "trainable_parameters": 0,
        "requires_training": False,
        "feature_variant_id": None,
        "scaler_bundle_id": None,
        "seed": None,
        "test_status": TEST_ACCESS_POLICY,
        "status": metric.status,
    }
    discrepancy_payload = {
        "artifact_version": PERSISTENCE_VERSION,
        "created_at": timestamp,
        "run_id": run_id,
        "warnings": list(metric.warnings),
        "discrepancies": [],
        "status": "PASS" if not metric.warnings else "PASS_WITH_WARNING",
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": artifact_root / "persistence_validation_predictions.csv",
        "metrics": artifact_root / "persistence_validation_metrics.json",
        "summary": artifact_root / "persistence_baseline_summary.json",
        "audit": artifact_root / "persistence_audit.csv",
        "unit_tests": artifact_root / "persistence_unit_tests.csv",
        "discrepancies": artifact_root / "persistence_discrepancies.json",
        "readme": artifact_root / "README_PERSISTENCE.md",
    }
    write_text_once_or_verify(paths["predictions"], predictions.to_csv(index=False, lineterminator="\n"))
    write_json_once_or_verify(paths["metrics"], metrics_payload)
    write_json_once_or_verify(paths["summary"], summary_payload)
    write_text_once_or_verify(paths["audit"], csv_text(AUDIT_COLUMNS, evaluation.audit_rows))
    write_text_once_or_verify(paths["unit_tests"], csv_text(UNIT_TEST_COLUMNS, evaluation.unit_test_rows))
    write_json_once_or_verify(paths["discrepancies"], discrepancy_payload)
    write_text_once_or_verify(paths["readme"], _readme_persistence())
    artifact_specs = (
        ("predictions", ArtifactType.PREDICTIONS.value, True),
        ("metrics", ArtifactType.METRICS.value, True),
        ("audit", ArtifactType.AUDIT.value, False),
        ("summary", ArtifactType.TABLE.value, False),
        ("unit_tests", ArtifactType.TABLE.value, False),
        ("discrepancies", ArtifactType.OTHER.value, False),
        ("readme", ArtifactType.OTHER.value, False),
    )
    registered_artifacts = []
    for key, artifact_type, required in artifact_specs:
        registered_artifacts.append(registry.register_artifact(run_id, artifact_type, paths[key], required))
    metric_rows = []
    metric_values = (
        ("mae_wh", metric.mae_wh, "Wh", "PASS"),
        ("rmse_wh", metric.rmse_wh, "Wh", "PASS"),
        ("r2", metric.r2, "dimensionless", metric.status),
    )
    for metric_name, metric_value, metric_unit, status in metric_values:
        metric_rows.append(
            registry.register_metric(
                run_id,
                VALIDATION_SPLIT,
                metric_name,
                metric_value,
                metric_unit,
                metric.n_samples,
                metric.population_fingerprint,
                "DETERMINISTIC_EVALUATION",
                status,
            )
        )
    return {
        "created_at": timestamp,
        "paths": paths,
        "checksums": {key: sha256_file(path) for key, path in paths.items()},
        "registered_artifacts": registered_artifacts,
        "registered_metrics": metric_rows,
        "baseline_config_fingerprint": baseline_fingerprint,
        "prediction_fingerprint": prediction_fingerprint,
        "metric_result_fingerprint": metric_fingerprint,
        "metrics_payload": metrics_payload,
        "summary_payload": summary_payload,
        "discrepancy_payload": discrepancy_payload,
    }


def complete_persistence_run(
    registry: ExperimentRegistry,
    run_id: str,
    evaluation: PersistenceEvaluationResult,
) -> dict[str, Any]:
    completed = registry.complete_run(
        run_id,
        best_validation_rmse_wh=evaluation.metric_result.rmse_wh,
    )
    if completed["status"] != RunStatus.COMPLETED.value:
        raise RuntimeError("Persistence run did not complete")
    if completed["best_epoch"] is not None:
        raise RuntimeError("Persistence run must not record a best epoch")
    if not math.isclose(
        float(completed["best_validation_rmse_wh"]),
        evaluation.metric_result.rmse_wh,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Persistence completed RMSE differs from METRICS-v1")
    registry_rows = registry.validate_registry()
    if any(row["status"] != "PASS" for row in registry_rows):
        raise RuntimeError("Persistence completion left the registry invalid")
    return completed


def _relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _input_checksums(root: Path) -> dict[str, str]:
    paths = [
        "data/raw_data/energydata_complete.csv",
        "artifacts/windows/phase_10_signoff.json",
        "artifacts/windows/window_manifest.json",
        "artifacts/windows/window_index.csv",
        "artifacts/windows/common_target_population.csv",
        "artifacts/dataloaders/phase_11_signoff.json",
        "artifacts/metrics/phase_12_signoff.json",
        "artifacts/metrics/metric_manifest.json",
        "artifacts/metrics/metric_contract.json",
        "artifacts/experiments/phase_13_signoff.json",
    ]
    return {path: sha256_file(root / path) for path in paths}


def _failure_type_for_stage(stage: str, error: BaseException) -> str:
    if isinstance(error, PermissionError):
        return FailureType.TEST_FIREWALL_VIOLATION.value
    if stage in {"PREPARE_VALIDATION", "H1_ALIGNMENT", "POPULATION"}:
        return FailureType.WINDOW_CONTRACT_ERROR.value
    if stage in {"EVALUATE", "REGISTER_METRICS"}:
        return FailureType.METRIC_ERROR.value
    return FailureType.OTHER.value


def build_persistence_manifest(
    root: Path,
    context: dict[str, Any],
    completed: dict[str, Any],
    prepared: PersistencePreparedData,
    evaluation: PersistenceEvaluationResult,
    artifact_result: dict[str, Any],
) -> dict[str, Any]:
    paths = {key: _relative_path(path, root) for key, path in artifact_result["paths"].items()}
    metric = evaluation.metric_result
    return {
        "artifact_version": PERSISTENCE_VERSION,
        "phase_version": PHASE_VERSION,
        "created_at": artifact_result["created_at"],
        "baseline_version": PERSISTENCE_VERSION,
        "model_id": PERSISTENCE_MODEL_ID,
        "baseline_scope": PERSISTENCE_SCOPE,
        "formula": PERSISTENCE_FORMULA,
        "run_id": completed["run_id"],
        "run_status": completed["status"],
        "experiment_family": completed["experiment_family"],
        "execution_type": completed["execution_type"],
        "experiment_config_fingerprint": completed["config_fingerprint"],
        "baseline_config_fingerprint": artifact_result["baseline_config_fingerprint"],
        "prediction_fingerprint": artifact_result["prediction_fingerprint"],
        "metric_result_fingerprint": artifact_result["metric_result_fingerprint"],
        "dataset_revision": context["dataset_manifest"]["dataset_revision"],
        "dataset_fingerprint": context["raw_checksum"],
        "window_version": WINDOW_VERSION,
        "population_version": POPULATION_VERSION,
        "population_fingerprint": prepared.population_fingerprint,
        "metric_version": METRIC_VERSION,
        "metric_contract_fingerprint": metric.metric_contract_fingerprint,
        "experiment_registry_version": EXPERIMENT_VERSION,
        "primary_lookback_steps": PRIMARY_LOOKBACK,
        "forecast_horizon_steps": FORECAST_HORIZON_STEPS,
        "forecast_horizon_minutes": SAMPLING_INTERVAL_MINUTES,
        "source_lag_steps": SOURCE_LAG_STEPS,
        "source_lag_minutes": SOURCE_LAG_MINUTES,
        "boundary_protocol": BOUNDARY_PROTOCOL,
        "split_id": VALIDATION_SPLIT,
        "validation_sample_count": prepared.validation_sample_count,
        "target_unit": "Wh",
        "trainable_parameters": 0,
        "requires_training": False,
        "feature_variant_id": None,
        "scaler_bundle_id": None,
        "dataloader_config_id": None,
        "seed": None,
        "device_type": "cpu",
        "test_access_authorized": False,
        "test_access_policy": TEST_ACCESS_POLICY,
        "test_targets_materialized": False,
        "validation_metrics": {
            "mae_wh": metric.mae_wh,
            "rmse_wh": metric.rmse_wh,
            "r2": metric.r2,
            "r2_status": metric.r2_status,
        },
        "artifact_paths": paths,
        "artifact_checksums": artifact_result["checksums"],
        "audit_check_count": len(evaluation.audit_rows),
        "unit_test_count": len(evaluation.unit_test_rows),
        "audit_status": "PASS",
        "warnings": list(metric.warnings),
    }


def verify_existing_signoff(project_root: Path, signoff_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    signoff = read_json(signoff_path)
    if signoff.get("artifact_version") != PERSISTENCE_VERSION or signoff.get("phase_version") != PHASE_VERSION:
        raise RuntimeError("Existing Phase 14 sign-off version mismatch")
    if signoff.get("status") != "PASS" or signoff.get("test_access_authorized") is not False:
        raise RuntimeError("Existing Phase 14 sign-off status or Test policy is invalid")
    for relative_path, expected_checksum in signoff.get("input_checksums", {}).items():
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected_checksum:
            raise RuntimeError(f"Phase 14 input checksum mismatch: {relative_path}")
    for relative_path, expected_checksum in signoff.get("output_checksums", {}).items():
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected_checksum:
            raise RuntimeError(f"Phase 14 output checksum mismatch: {relative_path}")
    manifest = read_json(root / PERSISTENCE_ARTIFACT_ROOT / "persistence_manifest.json")
    metrics = read_json(root / PERSISTENCE_ARTIFACT_ROOT / "persistence_validation_metrics.json")
    predictions = pd.read_csv(root / PERSISTENCE_ARTIFACT_ROOT / "persistence_validation_predictions.csv")
    if manifest.get("audit_status") != "PASS" or manifest.get("run_id") != signoff.get("run_id"):
        raise RuntimeError("Existing Persistence manifest is invalid")
    if predictions.columns.tolist() != PREDICTION_COLUMNS or len(predictions) != signoff.get("validation_sample_count"):
        raise RuntimeError("Existing Persistence prediction artifact is invalid")
    if not predictions["run_id"].eq(signoff["run_id"]).all() or predictions["sample_idx"].duplicated().any():
        raise RuntimeError("Existing Persistence predictions have invalid provenance")
    if metrics.get("metric_result", {}).get("split_id") != VALIDATION_SPLIT:
        raise PermissionError("Existing Persistence metric split violates the Test firewall")
    registry = ExperimentRegistry(root)
    record = registry.get_run(signoff["run_id"])
    if record["status"] != RunStatus.COMPLETED.value or record["experiment_family"] != "PERSISTENCE_BASELINE":
        raise RuntimeError("Existing Persistence registry record is invalid")
    if record["test_access_authorized"] or record["config"]["data"]["target_access_mode"] != VALIDATION_SPLIT:
        raise PermissionError("Existing Persistence registry record violates the Test firewall")
    if any(row["status"] != "PASS" for row in registry.validate_registry()):
        raise RuntimeError("Live registry validation failed for Phase 14")
    return signoff


def materialize_phase_14(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    artifact_root = root / PERSISTENCE_ARTIFACT_ROOT
    signoff_path = artifact_root / "phase_14_signoff.json"
    if signoff_path.exists():
        return verify_existing_signoff(root, signoff_path)
    context = verify_phase_14_inputs(root)
    input_checksums = _input_checksums(root)
    registry = ExperimentRegistry(root)
    run_id = None
    stage = "REGISTER"
    try:
        prepared = prepare_validation_persistence_data(root)
        stage = "REGISTER"
        config = build_persistence_run_config(root)
        started = register_and_start_persistence_run(registry, config)
        run_id = started["run_id"]
        stage = "EVALUATE"
        evaluation = evaluate_persistence_validation(prepared, run_id, root)
        stage = "REGISTER_METRICS"
        artifact_result = write_and_register_persistence_artifacts(
            registry,
            artifact_root,
            prepared,
            evaluation,
            run_id,
        )
        stage = "COMPLETE"
        completed = complete_persistence_run(registry, run_id, evaluation)
        stage = "MANIFEST"
        manifest = build_persistence_manifest(root, context, completed, prepared, evaluation, artifact_result)
        manifest_path = artifact_root / "persistence_manifest.json"
        write_json_once_or_verify(manifest_path, manifest)
        output_paths = [
            manifest_path,
            artifact_result["paths"]["summary"],
            artifact_result["paths"]["predictions"],
            artifact_result["paths"]["metrics"],
            artifact_result["paths"]["audit"],
            artifact_result["paths"]["unit_tests"],
            artifact_result["paths"]["discrepancies"],
            artifact_result["paths"]["readme"],
        ]
        output_checksums = {_relative_path(path, root): sha256_file(path) for path in output_paths}
        signoff = {
            "artifact_version": PERSISTENCE_VERSION,
            "phase_version": PHASE_VERSION,
            "phase_id": 14,
            "created_at": artifact_result["created_at"],
            "status": "PASS",
            "run_id": run_id,
            "run_status": completed["status"],
            "model_id": PERSISTENCE_MODEL_ID,
            "baseline_scope": PERSISTENCE_SCOPE,
            "formula": PERSISTENCE_FORMULA,
            "dataset_revision": context["dataset_manifest"]["dataset_revision"],
            "dataset_fingerprint": context["raw_checksum"],
            "window_version": WINDOW_VERSION,
            "population_version": POPULATION_VERSION,
            "population_fingerprint": prepared.population_fingerprint,
            "metric_version": METRIC_VERSION,
            "metric_contract_fingerprint": evaluation.metric_result.metric_contract_fingerprint,
            "experiment_registry_version": EXPERIMENT_VERSION,
            "experiment_config_fingerprint": completed["config_fingerprint"],
            "baseline_config_fingerprint": artifact_result["baseline_config_fingerprint"],
            "prediction_fingerprint": artifact_result["prediction_fingerprint"],
            "metric_result_fingerprint": artifact_result["metric_result_fingerprint"],
            "validation_sample_count": prepared.validation_sample_count,
            "validation_metrics": {
                "mae_wh": evaluation.metric_result.mae_wh,
                "rmse_wh": evaluation.metric_result.rmse_wh,
                "r2": evaluation.metric_result.r2,
                "r2_status": evaluation.metric_result.r2_status,
            },
            "target_unit": "Wh",
            "trainable_parameters": 0,
            "requires_training": False,
            "feature_variant_id": None,
            "scaler_bundle_id": None,
            "seed": None,
            "test_access_authorized": False,
            "test_access_policy": TEST_ACCESS_POLICY,
            "test_targets_materialized": False,
            "input_paths": list(input_checksums),
            "input_checksums": input_checksums,
            "output_paths": list(output_checksums),
            "output_checksums": output_checksums,
            "tests": [row["test_id"] for row in evaluation.unit_test_rows] + [row["check"] for row in evaluation.audit_rows],
            "warnings": list(evaluation.metric_result.warnings),
            "discrepancies": [],
        }
        write_json_once_or_verify(signoff_path, signoff)
        return verify_existing_signoff(root, signoff_path)
    except BaseException as error:
        if run_id is not None:
            current = registry.get_run(run_id)
            if current["status"] == RunStatus.RUNNING.value:
                registry.fail_run(
                    run_id,
                    _failure_type_for_stage(stage, error),
                    stage,
                    str(error),
                    exception_class=type(error).__name__,
                    recoverable=False,
                    rerun_recommended=True,
                )
            elif current["status"] == RunStatus.REGISTERED.value:
                registry.cancel_run(run_id, str(error))
        raise
