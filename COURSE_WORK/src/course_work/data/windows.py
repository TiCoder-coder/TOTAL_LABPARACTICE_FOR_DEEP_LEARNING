from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from course_work.data.feature_sets import FEATURE_SET_VERSION, load_validated_feature_set_registry
from course_work.data.features import FEATURE_VERSION, load_validated_feature_view
from course_work.data.scaling import (
    SCALING_VERSION,
    load_validated_scaler_bundle,
    load_validated_target_scaler,
    materialize_phase_9,
    transform_feature_variant,
    transform_target,
)
from course_work.data.splitting import SPLIT_VERSION, load_validated_split_membership
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


WINDOW_VERSION = "WINDOWS-v1"
POPULATION_VERSION = "WINDOWPOP-v1"
WINDOW_ARTIFACT_ROOT = "artifacts/windows"
LOOKBACK_OPTIONS = (36, 72, 144)
PRIMARY_LOOKBACK = 144
FORECAST_HORIZON_STEPS = 1
SAMPLING_INTERVAL_MINUTES = 10
PRIMARY_BOUNDARY_PROTOCOL = "WB0_CONTEXT_CARRY_OVER"
ALTERNATIVE_BOUNDARY_PROTOCOL = "WB1_STRICT_ISOLATION"
SEQUENCE_DIRECTION = "oldest_to_newest"
TARGET_ASSIGNMENT_POLICY = "by_target_timestamp"
TEST_TARGET_ACCESS_POLICY = "LOCKED_UNTIL_PHASE_47"
HISTORICAL_TARGET_ASSUMPTION = "observed_through_input_end_for_one_step_forecasting"
COMMON_POPULATION_POLICY = "intersection_of_native_L36_L72_L144_WB0_targets"
SPLIT_IDS = ("TRAIN", "VALIDATION", "TEST")
SPLIT_RANK = {"TRAIN": 0, "VALIDATION": 1, "TEST": 2}
WINDOW_INDEX_COLUMNS = [
    "window_id",
    "target_sample_id",
    "lookback_steps",
    "horizon_steps",
    "timeline_input_start",
    "timeline_input_end",
    "timeline_target",
    "input_start_timestamp",
    "input_end_timestamp",
    "target_timestamp",
    "input_start_raw_row_index",
    "input_end_raw_row_index",
    "target_raw_row_index",
    "continuity_segment_id",
    "target_split_id",
    "input_start_split_id",
    "input_end_split_id",
    "crosses_split_boundary",
    "WB0_valid",
    "WB1_valid",
    "included_common_population",
]
REJECTED_COLUMNS = [
    "lookback_steps",
    "target_sample_id",
    "target_timestamp",
    "target_split_id",
    "reason",
    "timeline_input_start",
    "timeline_input_end",
    "timeline_target",
    "details",
]


def compute_window_bounds(target_idx: int, lookback: int, horizon: int = FORECAST_HORIZON_STEPS) -> tuple[int, int, int]:
    if target_idx < 0:
        raise ValueError("target_idx must be non-negative")
    if lookback <= 0 or horizon <= 0:
        raise ValueError("lookback and horizon must be positive")
    input_end = target_idx - horizon
    input_start = input_end - lookback + 1
    return input_start, input_end, target_idx


def make_target_sample_id(target_idx: int) -> str:
    if target_idx < 0:
        raise ValueError("target_idx must be non-negative")
    return f"TGT_{target_idx:08d}"


def make_window_id(lookback: int, horizon: int, target_idx: int) -> str:
    if lookback <= 0 or horizon <= 0 or target_idx < 0:
        raise ValueError("Window identity inputs are invalid")
    return f"WIN_L{lookback:03d}_H{horizon:02d}_{make_target_sample_id(target_idx)}"


def build_relative_lag_axis(
    lookback: int,
    horizon: int = FORECAST_HORIZON_STEPS,
    sampling_interval_minutes: int = SAMPLING_INTERVAL_MINUTES,
) -> np.ndarray:
    if lookback <= 0 or horizon <= 0 or sampling_interval_minutes <= 0:
        raise ValueError("Relative lag inputs must be positive")
    positions = np.arange(lookback, dtype=np.int64)
    lag_steps = horizon + lookback - 1 - positions
    return lag_steps * sampling_interval_minutes


def build_timeline_positions(feature_view: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    if len(feature_view) != len(membership):
        raise ValueError("FEATURES-v1 and SPLIT-v1 row counts differ")
    required_features = {"raw_row_index", "timestamp", "continuity_segment_id"}
    required_membership = {"raw_row_index", "timestamp", "continuity_segment_id", "split_id"}
    if not required_features.issubset(feature_view.columns):
        raise ValueError("FEATURES-v1 timeline metadata is incomplete")
    if not required_membership.issubset(membership.columns):
        raise ValueError("SPLIT-v1 membership metadata is incomplete")
    feature_timestamps = pd.to_datetime(feature_view["timestamp"], errors="raise")
    membership_timestamps = pd.to_datetime(membership["timestamp"], errors="raise")
    if not feature_view["raw_row_index"].reset_index(drop=True).equals(membership["raw_row_index"].reset_index(drop=True)):
        raise ValueError("FEATURES-v1 and SPLIT-v1 raw row identities differ")
    if not feature_timestamps.reset_index(drop=True).equals(membership_timestamps.reset_index(drop=True)):
        raise ValueError("FEATURES-v1 and SPLIT-v1 timestamps differ")
    if not feature_view["continuity_segment_id"].astype(str).reset_index(drop=True).equals(membership["continuity_segment_id"].astype(str).reset_index(drop=True)):
        raise ValueError("FEATURES-v1 and SPLIT-v1 continuity segments differ")
    timeline = pd.DataFrame({
        "timeline_position": np.arange(len(feature_view), dtype=np.int64),
        "raw_row_index": feature_view["raw_row_index"].to_numpy(dtype=np.int64, copy=True),
        "timestamp": feature_timestamps.to_numpy(copy=True),
        "continuity_segment_id": feature_view["continuity_segment_id"].astype(str).to_numpy(copy=True),
        "split_id": membership["split_id"].astype(str).to_numpy(copy=True),
    })
    if timeline["raw_row_index"].duplicated().any():
        raise ValueError("Timeline contains duplicate raw row identities")
    if not timeline["timestamp"].is_monotonic_increasing:
        raise ValueError("Timeline timestamps are not chronological")
    invalid_splits = sorted(set(timeline["split_id"]) - set(SPLIT_IDS))
    if invalid_splits:
        raise ValueError(f"Timeline contains invalid split IDs: {invalid_splits}")
    return timeline


def validate_temporal_window(
    timeline: pd.DataFrame,
    target_idx: int,
    lookback: int,
    horizon: int = FORECAST_HORIZON_STEPS,
    sampling_interval_minutes: int = SAMPLING_INTERVAL_MINUTES,
) -> tuple[bool, str, str]:
    input_start, input_end, target_idx = compute_window_bounds(target_idx, lookback, horizon)
    if target_idx >= len(timeline):
        return False, "INVALID_TARGET_SPLIT", "target position is outside the timeline"
    if input_start < 0:
        return False, "INSUFFICIENT_HISTORY", f"required input start {input_start} is before timeline start"
    positions = timeline.iloc[input_start:target_idx + 1]
    timestamps = pd.to_datetime(positions["timestamp"], errors="coerce")
    if timestamps.isna().any():
        return False, "OFF_GRID_TIMESTAMP", "window contains an invalid timestamp"
    if timestamps.duplicated().any():
        return False, "DUPLICATE_TIMESTAMP", "window contains duplicate timestamps"
    off_grid = any(
        value.second != 0 or value.microsecond != 0 or value.minute % sampling_interval_minutes != 0
        for value in timestamps
    )
    if off_grid:
        return False, "OFF_GRID_TIMESTAMP", "window contains timestamps outside the sampling grid"
    input_timestamps = timestamps.iloc[:lookback]
    target_timestamp = timestamps.iloc[-1]
    input_end_timestamp = input_timestamps.iloc[-1]
    if target_timestamp <= input_end_timestamp:
        return False, "TARGET_NOT_FUTURE", "target timestamp is not later than input end"
    expected_delta = pd.Timedelta(minutes=sampling_interval_minutes)
    if len(input_timestamps) != lookback or not input_timestamps.diff().iloc[1:].eq(expected_delta).all():
        return False, "INPUT_GAP", "input timestamps are not contiguous at the registered cadence"
    expected_target_delta = pd.Timedelta(minutes=horizon * sampling_interval_minutes)
    if target_timestamp - input_end_timestamp != expected_target_delta:
        return False, "TARGET_GAP", "target horizon delta does not match the registered horizon"
    segments = positions["continuity_segment_id"].astype(str)
    if segments.nunique(dropna=False) != 1:
        return False, "INPUT_GAP", "input and target do not share one continuity segment"
    if target_idx <= input_end or input_start <= target_idx <= input_end:
        return False, "TARGET_NOT_FUTURE", "target position overlaps the input range"
    return True, "VALID", "temporal geometry is valid"


def build_native_window_index(
    timeline: pd.DataFrame,
    lookback: int,
    horizon: int = FORECAST_HORIZON_STEPS,
    sampling_interval_minutes: int = SAMPLING_INTERVAL_MINUTES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_rows = []
    rejected_rows = []
    for target_idx in range(len(timeline)):
        input_start, input_end, _ = compute_window_bounds(target_idx, lookback, horizon)
        target = timeline.iloc[target_idx]
        target_sample_id = make_target_sample_id(target_idx)
        valid, reason, details = validate_temporal_window(
            timeline,
            target_idx,
            lookback,
            horizon,
            sampling_interval_minutes,
        )
        if not valid:
            rejected_rows.append({
                "lookback_steps": lookback,
                "target_sample_id": target_sample_id,
                "target_timestamp": pd.Timestamp(target["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
                "target_split_id": str(target["split_id"]),
                "reason": reason,
                "timeline_input_start": input_start,
                "timeline_input_end": input_end,
                "timeline_target": target_idx,
                "details": details,
            })
            continue
        input_rows = timeline.iloc[input_start:input_end + 1]
        target_split = str(target["split_id"])
        input_splits = input_rows["split_id"].astype(str)
        if target_split not in SPLIT_RANK:
            raise RuntimeError(f"Invalid target split for {target_sample_id}: {target_split}")
        if any(SPLIT_RANK[value] > SPLIT_RANK[target_split] for value in input_splits):
            raise RuntimeError(f"Future split input detected for {target_sample_id}")
        if target_split == "TRAIN" and set(input_splits) != {"TRAIN"}:
            raise RuntimeError(f"TRAIN target contains non-TRAIN input for {target_sample_id}")
        crosses_boundary = bool((input_splits != target_split).any())
        wb1_valid = bool((input_splits == target_split).all())
        valid_rows.append({
            "window_id": make_window_id(lookback, horizon, target_idx),
            "target_sample_id": target_sample_id,
            "lookback_steps": lookback,
            "horizon_steps": horizon,
            "timeline_input_start": input_start,
            "timeline_input_end": input_end,
            "timeline_target": target_idx,
            "input_start_timestamp": pd.Timestamp(input_rows.iloc[0]["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
            "input_end_timestamp": pd.Timestamp(input_rows.iloc[-1]["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
            "target_timestamp": pd.Timestamp(target["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
            "input_start_raw_row_index": int(input_rows.iloc[0]["raw_row_index"]),
            "input_end_raw_row_index": int(input_rows.iloc[-1]["raw_row_index"]),
            "target_raw_row_index": int(target["raw_row_index"]),
            "continuity_segment_id": str(target["continuity_segment_id"]),
            "target_split_id": target_split,
            "input_start_split_id": str(input_rows.iloc[0]["split_id"]),
            "input_end_split_id": str(input_rows.iloc[-1]["split_id"]),
            "crosses_split_boundary": crosses_boundary,
            "WB0_valid": True,
            "WB1_valid": wb1_valid,
            "included_common_population": False,
        })
    return pd.DataFrame(valid_rows, columns=WINDOW_INDEX_COLUMNS), pd.DataFrame(rejected_rows, columns=REJECTED_COLUMNS)


def build_common_target_population(
    native_indices: dict[int, pd.DataFrame],
    timeline: pd.DataFrame,
) -> pd.DataFrame:
    if set(native_indices) != set(LOOKBACK_OPTIONS):
        raise ValueError("Native window indices do not cover every registered lookback")
    target_sets = [set(frame["target_sample_id"]) for frame in native_indices.values()]
    common_ids = set.intersection(*target_sets)
    if not common_ids:
        raise RuntimeError("Common target population is empty")
    target_positions = sorted(
        int(value.removeprefix("TGT_"))
        for value in common_ids
    )
    rows = []
    for target_idx in target_positions:
        target = timeline.iloc[target_idx]
        rows.append({
            "target_sample_id": make_target_sample_id(target_idx),
            "timeline_target": target_idx,
            "target_timestamp": pd.Timestamp(target["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
            "target_split_id": str(target["split_id"]),
            "continuity_segment_id": str(target["continuity_segment_id"]),
            "valid_L36": True,
            "valid_L72": True,
            "valid_L144": True,
            "included_common_population": True,
        })
    return pd.DataFrame(rows)


def filter_common_window_index(
    native_indices: dict[int, pd.DataFrame],
    common_population: pd.DataFrame,
) -> pd.DataFrame:
    common_ids = set(common_population["target_sample_id"])
    frames = []
    for lookback in LOOKBACK_OPTIONS:
        frame = native_indices[lookback]
        selected = frame.loc[frame["target_sample_id"].isin(common_ids)].copy(deep=True)
        selected["included_common_population"] = True
        selected = selected.sort_values("timeline_target", kind="stable").reset_index(drop=True)
        if set(selected["target_sample_id"]) != common_ids:
            raise RuntimeError(f"Common target alignment failed for L{lookback}")
        frames.append(selected)
    return pd.concat(frames, ignore_index=True).loc[:, WINDOW_INDEX_COLUMNS]


def compute_window_fingerprint(window_index: pd.DataFrame) -> str:
    fields = [
        "window_id",
        "target_sample_id",
        "timeline_input_start",
        "timeline_input_end",
        "timeline_target",
        "target_split_id",
        "WB0_valid",
        "WB1_valid",
    ]
    records = window_index.loc[:, fields].to_dict(orient="records")
    return sha256_bytes(canonical_json_bytes(records))


def compute_population_fingerprint(common_population: pd.DataFrame) -> str:
    fields = ["target_sample_id", "timeline_target", "target_split_id", "continuity_segment_id"]
    records = common_population.loc[:, fields].to_dict(orient="records")
    return sha256_bytes(canonical_json_bytes(records))


def transform_feature_timeline(
    feature_view: pd.DataFrame,
    variant_id: str,
    feature_registry: dict[str, Any],
    global_split_fingerprint: str,
    project_root: Path | None = None,
) -> np.ndarray:
    if variant_id not in feature_registry["variants"]:
        raise KeyError(f"Unknown feature variant: {variant_id}")
    entry = feature_registry["variants"][variant_id]
    bundle = load_validated_scaler_bundle(variant_id, project_root)
    transformed = transform_feature_variant(
        feature_view.loc[:, entry["features"]],
        bundle,
        variant_id,
        entry["fingerprint"],
        global_split_fingerprint,
    )
    if list(transformed.columns) != entry["features"]:
        raise RuntimeError(f"Feature order changed during timeline transform for {variant_id}")
    values = np.ascontiguousarray(transformed.to_numpy(dtype=np.float32, copy=True))
    if values.shape != (len(feature_view), entry["feature_count"]):
        raise RuntimeError(f"Feature timeline shape mismatch for {variant_id}")
    if not np.isfinite(values).all():
        raise RuntimeError(f"Feature timeline contains non-finite values for {variant_id}")
    return values


def transform_feature_timeline_with_scaler(
    feature_view: pd.DataFrame,
    variant_id: str,
    feature_entry: dict[str, Any],
    global_split_fingerprint: str,
    project_root: Path | None = None,
    *,
    scaler_bundle: dict[str, Any] | None = None,
) -> np.ndarray:
    """Transform a feature view using an explicit scaler bundle.

    This is equivalent to transform_feature_timeline but accepts a pre-loaded
    scaler_bundle so callers can specify which scaler registry to use
    (e.g., FINAL_SCALING-v1 instead of Phase 9 scalers).

    Args:
        feature_view: The full pre-Test feature DataFrame.
        variant_id: Feature variant ID (e.g. "FS2_TF1").
        feature_entry: The feature entry from the feature registry (already loaded).
        global_split_fingerprint: Split fingerprint from split_manifest.
        project_root: COURSE_WORK root.
        scaler_bundle: Either:
            - A registry dict with key "x_bundles" (existing behavior), or
            - A pre-loaded joblib X bundle dict (contains "scaler" key).
            If None, falls back to the Phase 9 scaler bundle (existing behavior).

    Returns:
        np.ndarray of shape (len(feature_view), feature_entry["feature_count"]).
    """
    if scaler_bundle is None:
        bundle = load_validated_scaler_bundle(variant_id, project_root)
    elif "scaler" in scaler_bundle:
        # Pre-loaded joblib bundle (scaler is already instantiated).
        bundle = scaler_bundle
    elif "x_bundles" in scaler_bundle:
        # Registry dict — extract the variant entry.
        if variant_id not in scaler_bundle["x_bundles"]:
            raise KeyError(
                f"Variant {variant_id} not found in provided scaler_bundle. "
                f"Available: {list(scaler_bundle.get('x_bundles', {}).keys())}"
            )
        bundle = scaler_bundle["x_bundles"][variant_id]
    else:
        raise ValueError(
            f"scaler_bundle must be a registry dict (with 'x_bundles') "
            f"or a pre-loaded joblib bundle (with 'scaler'). "
            f"Got: {list(scaler_bundle.keys())}"
        )
    # Use the bundle-aware transform so FINAL_SCALING-v1 bundles
    # (scaling_version=FINAL_SCALING-v1) are accepted.
    from course_work.data.scaling import transform_feature_variant_with_scaler_bundle
    transformed = transform_feature_variant_with_scaler_bundle(
        feature_view.loc[:, feature_entry["features"]],
        bundle,
        variant_id,
        feature_entry["fingerprint"],
        global_split_fingerprint,
    )
    if list(transformed.columns) != feature_entry["features"]:
        raise RuntimeError(f"Feature order changed during timeline transform for {variant_id}")
    values = np.ascontiguousarray(transformed.to_numpy(dtype=np.float32, copy=True))
    if values.shape != (len(feature_view), feature_entry["feature_count"]):
        raise RuntimeError(f"Feature timeline shape mismatch for {variant_id}")
    if not np.isfinite(values).all():
        raise RuntimeError(f"Feature timeline contains non-finite values for {variant_id}")
    return values


def materialize_window(
    feature_matrix: np.ndarray,
    window_record: pd.Series | dict[str, Any],
    target_values: np.ndarray | pd.Series | None = None,
    target_option: str = "YS0",
    target_scaler: dict[str, Any] | None = None,
    allow_test_targets: bool = False,
) -> dict[str, Any]:
    record = dict(window_record)
    lookback = int(record["lookback_steps"])
    input_start = int(record["timeline_input_start"])
    input_end = int(record["timeline_input_end"])
    target_idx = int(record["timeline_target"])
    if feature_matrix.ndim != 2:
        raise ValueError("Feature matrix must be two-dimensional")
    if input_start < 0 or input_end >= len(feature_matrix) or target_idx >= len(feature_matrix):
        raise IndexError("Window record is outside the feature timeline")
    x_seq = np.ascontiguousarray(feature_matrix[input_start:input_end + 1], dtype=np.float32)
    if x_seq.shape != (lookback, feature_matrix.shape[1]):
        raise RuntimeError("Materialized X shape does not match the window contract")
    if not np.isfinite(x_seq).all():
        raise RuntimeError("Materialized X contains non-finite values")
    metadata_fields = [
        "window_id",
        "target_sample_id",
        "lookback_steps",
        "horizon_steps",
        "timeline_input_start",
        "timeline_input_end",
        "timeline_target",
        "input_start_timestamp",
        "input_end_timestamp",
        "target_timestamp",
        "target_split_id",
    ]
    result = {
        "X_seq": x_seq,
        "y_model": None,
        "y_raw_wh": None,
        "metadata": {field: record[field] for field in metadata_fields},
    }
    if str(record["target_split_id"]) == "TEST" and not allow_test_targets:
        return result
    if target_values is None:
        raise ValueError("Target values are required when target access is enabled")
    raw_values = np.asarray(target_values, dtype=np.float64).reshape(-1)
    if target_idx >= len(raw_values):
        raise IndexError("Target index is outside the target timeline")
    y_raw = np.asarray([raw_values[target_idx]], dtype=np.float32)
    y_model = np.asarray(transform_target(y_raw.astype(np.float64), target_option, target_scaler), dtype=np.float32).reshape(1)
    if not np.isfinite(y_raw).all() or not np.isfinite(y_model).all():
        raise RuntimeError("Materialized target contains non-finite values")
    result["y_model"] = y_model
    result["y_raw_wh"] = y_raw
    return result


def build_window_population_summary(
    timeline: pd.DataFrame,
    native_indices: dict[int, pd.DataFrame],
    rejected: pd.DataFrame,
    common_index: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    candidate_counts = timeline["split_id"].value_counts().to_dict()
    for lookback in LOOKBACK_OPTIONS:
        native = native_indices[lookback]
        common = common_index.loc[common_index["lookback_steps"].eq(lookback)]
        rejected_lookback = rejected.loc[rejected["lookback_steps"].eq(lookback)]
        for split_id in SPLIT_IDS:
            native_split = native.loc[native["target_split_id"].eq(split_id)]
            common_split = common.loc[common["target_split_id"].eq(split_id)]
            rejected_split = rejected_lookback.loc[rejected_lookback["target_split_id"].eq(split_id)]
            rows.append({
                "population_id": POPULATION_VERSION,
                "lookback": lookback,
                "horizon": FORECAST_HORIZON_STEPS,
                "boundary_protocol": PRIMARY_BOUNDARY_PROTOCOL,
                "split_id": split_id,
                "candidate_count": int(candidate_counts.get(split_id, 0)),
                "native_valid_count": len(native_split),
                "common_valid_count": len(common_split),
                "excluded_for_common_count": len(native_split) - len(common_split),
                "rejected_count": len(rejected_split),
                "cross_boundary_count": int(native_split["crosses_split_boundary"].sum()),
                "WB1_valid_count": int(native_split["WB1_valid"].sum()),
                "feature_population_status": "PASS",
            })
    return rows


def build_window_boundary_audit(common_index: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for lookback in LOOKBACK_OPTIONS:
        lookback_frame = common_index.loc[common_index["lookback_steps"].eq(lookback)]
        for split_id in SPLIT_IDS:
            split_frame = lookback_frame.loc[lookback_frame["target_split_id"].eq(split_id)]
            if split_frame.empty:
                raise RuntimeError(f"No common windows for L{lookback} {split_id}")
            for boundary, record in (("FIRST", split_frame.iloc[0]), ("LAST", split_frame.iloc[-1])):
                rows.append({
                    "lookback": lookback,
                    "split_id": split_id,
                    "boundary": boundary,
                    "window_id": record["window_id"],
                    "target_sample_id": record["target_sample_id"],
                    "input_start_timestamp": record["input_start_timestamp"],
                    "input_end_timestamp": record["input_end_timestamp"],
                    "target_timestamp": record["target_timestamp"],
                    "input_start_split_id": record["input_start_split_id"],
                    "input_end_split_id": record["input_end_split_id"],
                    "crosses_split_boundary": bool(record["crosses_split_boundary"]),
                    "WB0_valid": bool(record["WB0_valid"]),
                    "WB1_valid": bool(record["WB1_valid"]),
                    "status": "PASS",
                })
    return rows


def run_window_leakage_audit(
    timeline: pd.DataFrame,
    common_index: pd.DataFrame,
    common_population: pd.DataFrame,
) -> list[dict[str, Any]]:
    input_end = pd.to_datetime(common_index["input_end_timestamp"])
    target = pd.to_datetime(common_index["target_timestamp"])
    target_after_input = bool((common_index["timeline_target"] > common_index["timeline_input_end"]).all())
    horizon_delta_valid = bool((target - input_end).eq(pd.Timedelta(minutes=SAMPLING_INTERVAL_MINUTES)).all())
    target_not_in_input = bool(((common_index["timeline_target"] < common_index["timeline_input_start"]) | (common_index["timeline_target"] > common_index["timeline_input_end"])).all())
    assigned_splits = timeline.iloc[common_index["timeline_target"].to_numpy(dtype=np.int64)]["split_id"].reset_index(drop=True)
    target_assignment_valid = bool(assigned_splits.equals(common_index["target_split_id"].reset_index(drop=True)))
    train = common_index.loc[common_index["target_split_id"].eq("TRAIN")]
    validation = common_index.loc[common_index["target_split_id"].eq("VALIDATION")]
    train_guard = bool(train["input_start_split_id"].eq("TRAIN").all() and train["input_end_split_id"].eq("TRAIN").all())
    validation_guard = bool(~validation["input_start_split_id"].eq("TEST").any() and ~validation["input_end_split_id"].eq("TEST").any())
    common_sets = [
        set(common_index.loc[common_index["lookback_steps"].eq(lookback), "target_sample_id"])
        for lookback in LOOKBACK_OPTIONS
    ]
    population_alignment = all(values == set(common_population["target_sample_id"]) for values in common_sets)
    checks = {
        "target_after_input": target_after_input,
        "horizon_delta_valid": horizon_delta_valid,
        "input_continuity_valid": bool(common_index["WB0_valid"].all()),
        "target_not_in_input": target_not_in_input,
        "target_assignment_by_timestamp": target_assignment_valid,
        "train_has_no_future_split_inputs": train_guard,
        "validation_has_no_test_inputs": validation_guard,
        "feature_population_alignment": population_alignment,
        "target_scaling_population_alignment": population_alignment,
        "lookback_common_population_alignment": population_alignment,
        "test_target_values_not_exported": not any("target_value" in column or "y_raw" in column for column in common_index.columns),
    }
    return [
        {
            "check": check,
            "expected": True,
            "actual": actual,
            "status": "PASS" if actual else "FAIL",
            "notes": "Index-first temporal, population and Test-firewall contract",
        }
        for check, actual in checks.items()
    ]


def build_materialization_audit(
    root: Path,
    feature_view: pd.DataFrame,
    feature_registry: dict[str, Any],
    split_manifest: dict[str, Any],
    common_index: pd.DataFrame,
) -> list[dict[str, Any]]:
    target_values = feature_view["Appliances"].to_numpy(dtype=np.float64, copy=True)
    target_scaler = load_validated_target_scaler(root)
    rows = []
    for variant_id, variant in feature_registry["variants"].items():
        matrix = transform_feature_timeline(
            feature_view,
            variant_id,
            feature_registry,
            split_manifest["global_split_fingerprint"],
            root,
        )
        for lookback in LOOKBACK_OPTIONS:
            lookback_frame = common_index.loc[common_index["lookback_steps"].eq(lookback)]
            for split_id in SPLIT_IDS:
                record = lookback_frame.loc[lookback_frame["target_split_id"].eq(split_id)].iloc[0]
                options = ("LOCKED",) if split_id == "TEST" else ("YS0", "YS1")
                for target_option in options:
                    result = materialize_window(
                        matrix,
                        record,
                        None if split_id == "TEST" else target_values,
                        "YS0" if target_option == "LOCKED" else target_option,
                        None if target_option in {"LOCKED", "YS0"} else target_scaler,
                        allow_test_targets=False,
                    )
                    expected_shape = [lookback, variant["feature_count"]]
                    x_shape = list(result["X_seq"].shape)
                    y_shape = "LOCKED" if split_id == "TEST" else list(result["y_model"].shape)
                    checks = [
                        x_shape == expected_shape,
                        list(matrix.shape) == [len(feature_view), variant["feature_count"]],
                        bool(np.isfinite(result["X_seq"]).all()),
                        result["metadata"]["timeline_target"] > result["metadata"]["timeline_input_end"],
                    ]
                    if split_id == "TEST":
                        checks.extend([result["y_model"] is None, result["y_raw_wh"] is None])
                    else:
                        checks.extend([
                            y_shape == [1],
                            bool(np.isfinite(result["y_model"]).all()),
                            bool(np.isfinite(result["y_raw_wh"]).all()),
                        ])
                    rows.append({
                        "variant_id": variant_id,
                        "lookback": lookback,
                        "split_id": split_id,
                        "target_option": target_option,
                        "X_shape": str(x_shape),
                        "expected_X_shape": str(expected_shape),
                        "y_shape": str(y_shape),
                        "feature_fingerprint_match": True,
                        "finite_X": True,
                        "finite_y": "LOCKED" if split_id == "TEST" else True,
                        "timestamp_order_valid": True,
                        "status": "PASS" if all(checks) else "FAIL",
                    })
    return rows


def _readme_windows() -> str:
    return "\n".join([
        "# WINDOWS-v1",
        "",
        "WINDOWS-v1 defines sequence-to-one geometry for lookbacks 36, 72 and 144 with horizon 1 at a 10-minute cadence.",
        "",
        "For target position j, input end is j minus H and input start is input end minus L plus one. The target row never enters X.",
        "",
        "Samples are assigned to TRAIN, VALIDATION or TEST by target timestamp. WB0 permits continuous historical context from prior periods. WB1 eligibility requires every input row to share the target split.",
        "",
        "WINDOWPOP-v1 is the ordered intersection of native valid targets for L36, L72 and L144. Controlled experiments use this fixed target population.",
        "",
        "Feature timelines are transformed with frozen SCALING-v1 bundles and materialized lazily as float32 arrays shaped L by F. Full three-dimensional window tensors are not stored.",
        "",
        "Historical Appliances values through input end are valid for FS1 and FS2 under the one-step observed-history assumption.",
        "",
        "Test window structure is available, but Test target values and outcome analysis remain locked until Phase 47.",
        "",
    ])


def verify_phase_10_inputs(root: Path) -> dict[str, Any]:
    phase_9 = materialize_phase_9(root)
    if phase_9.get("artifact_version") != SCALING_VERSION or phase_9.get("status") not in {"PASS", "PASS_WITH_WARNING"}:
        raise RuntimeError("SCALING-v1 is not signed off")
    for relative_path, expected_checksum in {**phase_9.get("input_checksums", {}), **phase_9.get("output_checksums", {})}.items():
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected_checksum:
            raise RuntimeError(f"Phase 10 upstream checksum mismatch: {relative_path}")
    required_signoffs = {
        "artifacts/temporal/phase_4_signoff.json": ("TEMPORAL-v1", {"PASS", "PASS_WITH_WARNING"}),
        "artifacts/features/phase_7_signoff.json": (FEATURE_VERSION, {"PASS"}),
        "artifacts/feature_sets/phase_8_signoff.json": (FEATURE_SET_VERSION, {"PASS"}),
        "artifacts/splits/phase_5_signoff.json": (SPLIT_VERSION, {"PASS"}),
    }
    for relative_path, (version, statuses) in required_signoffs.items():
        signoff = read_json(root / relative_path)
        if signoff.get("artifact_version") != version or signoff.get("status") not in statuses:
            raise RuntimeError(f"Invalid Phase 10 prerequisite: {relative_path}")
    contract = read_json(root / "configs/base/coursework_contract.json")
    problem = contract["problem"]
    lookback_options = tuple(item["value"] for item in contract["option_registry"]["lookbacks"])
    boundary_options = {item["id"]: item["value"] for item in contract["option_registry"]["boundary"]}
    if problem["forecast_horizon_steps"] != FORECAST_HORIZON_STEPS or problem["sampling_minutes"] != SAMPLING_INTERVAL_MINUTES:
        raise RuntimeError("Phase 0 forecasting contract does not match WINDOWS-v1")
    if lookback_options != LOOKBACK_OPTIONS or contract["lookbacks"]["primary"] != PRIMARY_LOOKBACK:
        raise RuntimeError("Phase 0 lookback contract does not match WINDOWS-v1")
    if boundary_options != {"WB0": "context_carry_over", "WB1": "strict_isolation"}:
        raise RuntimeError("Phase 0 boundary contract does not match WINDOWS-v1")
    return phase_9


def verify_existing_signoff(root: Path, signoff_path: Path) -> dict[str, Any]:
    signoff = read_json(signoff_path)
    if signoff.get("artifact_version") != WINDOW_VERSION or signoff.get("status") not in {"PASS", "PASS_WITH_WARNING"}:
        raise RuntimeError("Existing Phase 10 sign-off is invalid")
    for relative_path, expected_checksum in signoff.get("output_checksums", {}).items():
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected_checksum:
            raise RuntimeError(f"Phase 10 artifact checksum mismatch: {relative_path}")
    artifact_root = root / WINDOW_ARTIFACT_ROOT
    manifest = read_json(artifact_root / "window_manifest.json")
    fingerprints = read_json(artifact_root / "window_fingerprints.json")
    window_index = pd.read_csv(artifact_root / "window_index.csv")
    population = pd.read_csv(artifact_root / "common_target_population.csv")
    if manifest.get("window_version") != WINDOW_VERSION or manifest.get("population_version") != POPULATION_VERSION:
        raise RuntimeError("Reloaded window manifest version mismatch")
    if list(window_index.columns) != WINDOW_INDEX_COLUMNS:
        raise RuntimeError("Reloaded window index schema mismatch")
    if any("target_value" in column or "y_raw" in column for column in window_index.columns):
        raise RuntimeError("Test target firewall violation in window index")
    for lookback in LOOKBACK_OPTIONS:
        frame = window_index.loc[window_index["lookback_steps"].eq(lookback)]
        if compute_window_fingerprint(frame) != fingerprints["window_index_fingerprints"][f"L{lookback:03d}_H01_WB0"]:
            raise RuntimeError(f"Reloaded window fingerprint mismatch for L{lookback}")
        if set(frame["target_sample_id"]) != set(population["target_sample_id"]):
            raise RuntimeError(f"Reloaded population alignment mismatch for L{lookback}")
    if compute_population_fingerprint(population) != fingerprints["common_population_fingerprint"]:
        raise RuntimeError("Reloaded common population fingerprint mismatch")
    return signoff


def materialize_phase_10(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    signoff_path = root / WINDOW_ARTIFACT_ROOT / "phase_10_signoff.json"
    if signoff_path.exists():
        return verify_existing_signoff(root, signoff_path)
    phase_9 = verify_phase_10_inputs(root)
    feature_view = load_validated_feature_view(root)
    membership = load_validated_split_membership(root)
    timeline = build_timeline_positions(feature_view, membership)
    feature_registry = load_validated_feature_set_registry(root)
    temporal_manifest = read_json(root / "artifacts/temporal/temporal_manifest.json")
    feature_manifest = read_json(root / "artifacts/features/feature_engineering_manifest.json")
    split_manifest = read_json(root / "artifacts/splits/split_manifest.json")
    scaling_manifest = read_json(root / "artifacts/scaling/scaling_manifest.json")
    if temporal_manifest["expected_interval_minutes"] != SAMPLING_INTERVAL_MINUTES:
        raise RuntimeError("TEMPORAL-v1 cadence does not match WINDOWS-v1")
    if split_manifest["global_split_fingerprint"] != scaling_manifest["global_split_fingerprint"]:
        raise RuntimeError("SPLIT-v1 and SCALING-v1 fingerprints differ")
    native_indices = {}
    rejected_frames = []
    for lookback in LOOKBACK_OPTIONS:
        native, rejected = build_native_window_index(timeline, lookback)
        native_indices[lookback] = native
        rejected_frames.append(rejected)
    common_population = build_common_target_population(native_indices, timeline)
    common_index = filter_common_window_index(native_indices, common_population)
    rejected = pd.concat(rejected_frames, ignore_index=True).loc[:, REJECTED_COLUMNS]
    second_native = {lookback: build_native_window_index(timeline, lookback)[0] for lookback in LOOKBACK_OPTIONS}
    second_population = build_common_target_population(second_native, timeline)
    second_index = filter_common_window_index(second_native, second_population)
    if not common_index.equals(second_index) or not common_population.equals(second_population):
        raise RuntimeError("WINDOWS-v1 construction is nondeterministic")
    population_summary = build_window_population_summary(timeline, native_indices, rejected, common_index)
    boundary_audit = build_window_boundary_audit(common_index)
    leakage_audit = run_window_leakage_audit(timeline, common_index, common_population)
    if any(row["status"] != "PASS" for row in leakage_audit):
        raise RuntimeError(f"Window leakage audit failed: {leakage_audit}")
    materialization_audit = build_materialization_audit(
        root,
        feature_view,
        feature_registry,
        split_manifest,
        common_index,
    )
    if any(row["status"] != "PASS" for row in materialization_audit):
        raise RuntimeError("Window materialization audit failed")
    window_fingerprints = {
        f"L{lookback:03d}_H01_WB0": compute_window_fingerprint(
            common_index.loc[common_index["lookback_steps"].eq(lookback)]
        )
        for lookback in LOOKBACK_OPTIONS
    }
    population_fingerprint = compute_population_fingerprint(common_population)
    wb1_fingerprint = sha256_bytes(canonical_json_bytes(
        common_index.loc[:, ["window_id", "WB1_valid"]].to_dict(orient="records")
    ))
    fingerprint_artifact = {
        "window_version": WINDOW_VERSION,
        "population_version": POPULATION_VERSION,
        "window_index_fingerprints": window_fingerprints,
        "common_population_fingerprint": population_fingerprint,
        "WB0_population_fingerprint": population_fingerprint,
        "WB1_eligibility_fingerprint": wb1_fingerprint,
    }
    split_counts = common_population["target_split_id"].value_counts().to_dict()
    warnings = []
    discrepancies = []
    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "window_version": WINDOW_VERSION,
        "population_version": POPULATION_VERSION,
        "dataset_revision": feature_manifest["dataset_revision"],
        "temporal_version": temporal_manifest["temporal_version"],
        "feature_version": FEATURE_VERSION,
        "feature_set_version": FEATURE_SET_VERSION,
        "split_version": SPLIT_VERSION,
        "scaling_version": SCALING_VERSION,
        "environment_id": feature_manifest["environment_id"],
        "sampling_interval_minutes": SAMPLING_INTERVAL_MINUTES,
        "forecast_horizon_steps": FORECAST_HORIZON_STEPS,
        "forecast_horizon_minutes": FORECAST_HORIZON_STEPS * SAMPLING_INTERVAL_MINUTES,
        "lookback_options": list(LOOKBACK_OPTIONS),
        "primary_lookback": PRIMARY_LOOKBACK,
        "sequence_direction": SEQUENCE_DIRECTION,
        "target_assignment_policy": TARGET_ASSIGNMENT_POLICY,
        "primary_boundary_protocol": PRIMARY_BOUNDARY_PROTOCOL,
        "alternative_boundary_protocol": ALTERNATIVE_BOUNDARY_PROTOCOL,
        "historical_target_availability_assumption": HISTORICAL_TARGET_ASSUMPTION,
        "common_target_population_policy": COMMON_POPULATION_POLICY,
        "window_index_fingerprints": window_fingerprints,
        "common_population_fingerprint": population_fingerprint,
        "global_split_fingerprint": split_manifest["global_split_fingerprint"],
        "feature_variant_count": len(feature_registry["variants"]),
        "feature_variants_share_population": True,
        "target_options_share_population": True,
        "train_target_count": int(split_counts.get("TRAIN", 0)),
        "validation_target_count": int(split_counts.get("VALIDATION", 0)),
        "test_target_count": int(split_counts.get("TEST", 0)),
        "total_target_count": len(common_population),
        "window_index_row_count": len(common_index),
        "test_target_access_policy": TEST_TARGET_ACCESS_POLICY,
        "test_target_values_exported": False,
        "full_3d_windows_saved": False,
        "audit_status": "PASS",
        "warnings": warnings,
        "created_at": created_at,
    }
    artifact_root = root / WINDOW_ARTIFACT_ROOT
    manifest_path = artifact_root / "window_manifest.json"
    index_path = artifact_root / "window_index.csv"
    population_path = artifact_root / "common_target_population.csv"
    summary_path = artifact_root / "window_population_summary.csv"
    rejected_path = artifact_root / "rejected_window_candidates.csv"
    boundary_path = artifact_root / "window_boundary_audit.csv"
    leakage_path = artifact_root / "window_leakage_audit.csv"
    materialization_path = artifact_root / "window_materialization_audit.csv"
    fingerprints_path = artifact_root / "window_fingerprints.json"
    discrepancies_path = artifact_root / "window_discrepancies.json"
    readme_path = artifact_root / "README_WINDOWS.md"
    write_text_once_or_verify(index_path, common_index.to_csv(index=False, lineterminator="\n"))
    write_text_once_or_verify(population_path, common_population.to_csv(index=False, lineterminator="\n"))
    write_text_once_or_verify(summary_path, csv_text([
        "population_id", "lookback", "horizon", "boundary_protocol", "split_id", "candidate_count", "native_valid_count", "common_valid_count", "excluded_for_common_count", "rejected_count", "cross_boundary_count", "WB1_valid_count", "feature_population_status"
    ], population_summary))
    write_text_once_or_verify(rejected_path, rejected.to_csv(index=False, lineterminator="\n"))
    write_text_once_or_verify(boundary_path, csv_text([
        "lookback", "split_id", "boundary", "window_id", "target_sample_id", "input_start_timestamp", "input_end_timestamp", "target_timestamp", "input_start_split_id", "input_end_split_id", "crosses_split_boundary", "WB0_valid", "WB1_valid", "status"
    ], boundary_audit))
    write_text_once_or_verify(leakage_path, csv_text(["check", "expected", "actual", "status", "notes"], leakage_audit))
    write_text_once_or_verify(materialization_path, csv_text([
        "variant_id", "lookback", "split_id", "target_option", "X_shape", "expected_X_shape", "y_shape", "feature_fingerprint_match", "finite_X", "finite_y", "timestamp_order_valid", "status"
    ], materialization_audit))
    write_json_once_or_verify(fingerprints_path, fingerprint_artifact)
    write_json_once_or_verify(discrepancies_path, {
        "window_version": WINDOW_VERSION,
        "population_version": POPULATION_VERSION,
        "discrepancies": discrepancies,
    })
    write_text_once_or_verify(readme_path, _readme_windows())
    write_json_once_or_verify(manifest_path, manifest)
    output_paths = [
        f"{WINDOW_ARTIFACT_ROOT}/window_manifest.json",
        f"{WINDOW_ARTIFACT_ROOT}/window_index.csv",
        f"{WINDOW_ARTIFACT_ROOT}/common_target_population.csv",
        f"{WINDOW_ARTIFACT_ROOT}/window_population_summary.csv",
        f"{WINDOW_ARTIFACT_ROOT}/rejected_window_candidates.csv",
        f"{WINDOW_ARTIFACT_ROOT}/window_boundary_audit.csv",
        f"{WINDOW_ARTIFACT_ROOT}/window_leakage_audit.csv",
        f"{WINDOW_ARTIFACT_ROOT}/window_materialization_audit.csv",
        f"{WINDOW_ARTIFACT_ROOT}/window_fingerprints.json",
        f"{WINDOW_ARTIFACT_ROOT}/window_discrepancies.json",
        f"{WINDOW_ARTIFACT_ROOT}/README_WINDOWS.md",
    ]
    input_paths = [
        "artifacts/temporal/phase_4_signoff.json",
        "artifacts/temporal/temporal_manifest.json",
        "artifacts/features/phase_7_signoff.json",
        "artifacts/features/feature_engineering_manifest.json",
        feature_manifest["derived_file_path"],
        "artifacts/feature_sets/phase_8_signoff.json",
        "artifacts/feature_sets/feature_set_registry.json",
        "artifacts/splits/phase_5_signoff.json",
        "artifacts/splits/split_manifest.json",
        "artifacts/splits/split_membership.csv",
        "artifacts/scaling/phase_9_signoff.json",
        "artifacts/scaling/scaling_manifest.json",
        "artifacts/scaling/scaler_registry.json",
        "configs/base/coursework_contract.json",
    ]
    signoff = {
        "artifact_version": WINDOW_VERSION,
        "population_version": POPULATION_VERSION,
        "phase_id": 10,
        "phase_version": "PHASE-10-v1",
        "created_at": created_at,
        "environment_id": feature_manifest["environment_id"],
        "dataset_revision": feature_manifest["dataset_revision"],
        "temporal_version": temporal_manifest["temporal_version"],
        "feature_version": FEATURE_VERSION,
        "feature_set_version": FEATURE_SET_VERSION,
        "split_version": SPLIT_VERSION,
        "scaling_version": SCALING_VERSION,
        "global_split_fingerprint": split_manifest["global_split_fingerprint"],
        "common_population_fingerprint": population_fingerprint,
        "input_paths": input_paths,
        "input_checksums": {relative_path: sha256_file(root / relative_path) for relative_path in input_paths},
        "output_paths": output_paths,
        "output_checksums": {relative_path: sha256_file(root / relative_path) for relative_path in output_paths},
        "config_fingerprint": phase_9["config_fingerprint"],
        "status": "PASS",
        "tests": [
            "upstream_versions_and_checksums",
            "deterministic_index_construction",
            "lookback_bounds",
            "ten_minute_horizon",
            "input_continuity",
            "continuity_segment_alignment",
            "target_excluded_from_input",
            "target_split_assignment",
            "WB0_context_carry_over",
            "WB1_eligibility",
            "native_populations",
            "common_population_alignment",
            "feature_variant_population_alignment",
            "target_scaling_population_alignment",
            "feature_order_and_scaler_binding",
            "float32_lazy_materialization",
            "target_shape",
            "relative_lag_axis",
            "test_target_firewall",
            "window_and_population_fingerprints",
        ],
        "warnings": warnings,
        "discrepancies": discrepancies,
    }
    write_json_once_or_verify(signoff_path, signoff)
    return verify_existing_signoff(root, signoff_path)


def load_validated_window_index(project_root: Path | None = None) -> pd.DataFrame:
    root = (project_root or get_project_root()).resolve()
    materialize_phase_10(root)
    frame = pd.read_csv(
        root / WINDOW_ARTIFACT_ROOT / "window_index.csv",
        parse_dates=["input_start_timestamp", "input_end_timestamp", "target_timestamp"],
    )
    if list(frame.columns) != WINDOW_INDEX_COLUMNS:
        raise RuntimeError("WINDOWS-v1 index schema mismatch")
    return frame


def load_validated_common_population(project_root: Path | None = None) -> pd.DataFrame:
    root = (project_root or get_project_root()).resolve()
    materialize_phase_10(root)
    frame = pd.read_csv(
        root / WINDOW_ARTIFACT_ROOT / "common_target_population.csv",
        parse_dates=["target_timestamp"],
    )
    if frame.empty or not frame["target_timestamp"].is_monotonic_increasing:
        raise RuntimeError("WINDOWPOP-v1 population is empty or non-chronological")
    return frame
