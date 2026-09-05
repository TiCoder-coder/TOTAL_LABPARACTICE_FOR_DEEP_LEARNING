from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from course_work.data.schema import dataframe_fingerprint, load_raw_csv, materialize_phase_3
from course_work.utils.artifacts import (
    csv_text,
    get_project_root,
    read_json,
    sha256_file,
    write_json_once_or_verify,
    write_text_once_or_verify,
)


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
EXPECTED_INTERVAL_MINUTES = 10
WINDOW_SAFETY_CONTRACT = {
    "gap_policy": "segment_and_reject_cross_gap_windows",
    "forecast_horizon_steps": 1,
    "lookbacks": [36, 72, 144],
    "requirements": [
        "exact_lookback_rows",
        "adjacent_input_delta_10_minutes",
        "target_delta_10_minutes",
        "single_continuity_segment",
        "unique_timestamps",
        "input_before_target",
    ],
}


def parse_timestamps_strict(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, format=TIMESTAMP_FORMAT, errors="coerce")


def interval_status(delta_minutes: float) -> str:
    if np.isnan(delta_minutes):
        return "START"
    if delta_minutes == EXPECTED_INTERVAL_MINUTES:
        return "EXPECTED"
    if delta_minutes == 0:
        return "DUPLICATE"
    if delta_minutes < 0:
        return "REVERSE"
    if 0 < delta_minutes < EXPECTED_INTERVAL_MINUTES:
        return "TOO_SHORT"
    return "GAP"


def build_temporal_view(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "date" not in dataframe.columns:
        raise ValueError("Timestamp column date is missing")
    view = dataframe.copy(deep=True)
    view.insert(0, "raw_row_index", dataframe.index.to_numpy(copy=True))
    view.insert(1, "timestamp_raw", dataframe["date"].astype(str).to_numpy(copy=True))
    view["timestamp_parsed"] = parse_timestamps_strict(view["date"])
    if view["timestamp_parsed"].isna().any():
        raise ValueError("Strict timestamp parsing failed")
    view = view.sort_values("timestamp_parsed", kind="stable").reset_index(drop=True)
    deltas = view["timestamp_parsed"].diff().dt.total_seconds().div(60)
    view["delta_minutes"] = deltas
    view["interval_status"] = deltas.map(interval_status)
    segment_start = deltas.ne(EXPECTED_INTERVAL_MINUTES)
    segment_numbers = segment_start.cumsum().astype(int)
    view["continuity_segment_id"] = segment_numbers.map(lambda value: f"SEG-{value:04d}")
    return view


def duplicate_timestamp_rows(dataframe: pd.DataFrame, timestamps: pd.Series) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    duplicate_mask = timestamps.duplicated(keep=False)
    for timestamp, indices in timestamps[duplicate_mask].groupby(timestamps[duplicate_mask]).groups.items():
        group = dataframe.loc[list(indices)]
        comparison = group.drop(columns=["date"], errors="ignore")
        exact = len(comparison.drop_duplicates()) == 1
        conflicting = [column for column in comparison.columns if comparison[column].nunique(dropna=False) > 1]
        rows.append({
            "timestamp": timestamp.isoformat(sep=" "),
            "duplicate_type": "DT0_EXACT" if exact else "DT1_CONFLICTING",
            "group_size": len(group),
            "raw_row_indices": ";".join(str(index) for index in group.index),
            "conflicting_columns": ";".join(conflicting),
        })
    return rows


def gap_rows(view: pd.DataFrame) -> list[dict[str, Any]]:
    gaps = view[view["delta_minutes"] > EXPECTED_INTERVAL_MINUTES]
    rows: list[dict[str, Any]] = []
    for gap_id, (index, row) in enumerate(gaps.iterrows(), start=1):
        delta = float(row["delta_minutes"])
        multiple = delta % EXPECTED_INTERVAL_MINUTES == 0
        missing_steps = int(delta / EXPECTED_INTERVAL_MINUTES - 1) if multiple else None
        if delta > 1440:
            severity = "G4"
        elif delta > 60:
            severity = "G3"
        elif delta >= 30:
            severity = "G2"
        else:
            severity = "G1"
        rows.append({
            "gap_id": f"GAP-{gap_id:04d}",
            "raw_row_index_prev": int(view.loc[index - 1, "raw_row_index"]),
            "raw_row_index_current": int(row["raw_row_index"]),
            "previous_timestamp": view.loc[index - 1, "timestamp_parsed"].isoformat(sep=" "),
            "next_timestamp": row["timestamp_parsed"].isoformat(sep=" "),
            "delta_minutes": delta,
            "missing_steps": missing_steps,
            "is_multiple_of_10min": multiple,
            "severity": severity,
            "notes": "segment boundary",
        })
    return rows


def missing_timestamp_rows(view: pd.DataFrame, gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observed = pd.DatetimeIndex(view["timestamp_parsed"].drop_duplicates())
    expected = pd.date_range(observed.min(), observed.max(), freq=f"{EXPECTED_INTERVAL_MINUTES}min")
    missing = expected.difference(observed)
    rows: list[dict[str, Any]] = []
    for timestamp in missing:
        position = observed.searchsorted(timestamp)
        previous_timestamp = observed[position - 1] if position > 0 else None
        next_timestamp = observed[position] if position < len(observed) else None
        gap_id = next(
            (
                gap["gap_id"]
                for gap in gaps
                if gap["previous_timestamp"] == previous_timestamp.isoformat(sep=" ")
                and gap["next_timestamp"] == next_timestamp.isoformat(sep=" ")
            ),
            None,
        )
        rows.append({
            "missing_timestamp": timestamp.isoformat(sep=" "),
            "previous_observed_timestamp": previous_timestamp.isoformat(sep=" ") if previous_timestamp is not None else None,
            "next_observed_timestamp": next_timestamp.isoformat(sep=" ") if next_timestamp is not None else None,
            "gap_id": gap_id,
        })
    return rows


def continuity_segment_rows(view: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment_id, group in view.groupby("continuity_segment_id", sort=False):
        start = group["timestamp_parsed"].iloc[0]
        end = group["timestamp_parsed"].iloc[-1]
        rows.append({
            "segment_id": segment_id,
            "start_timestamp": start.isoformat(sep=" "),
            "end_timestamp": end.isoformat(sep=" "),
            "row_count": len(group),
            "duration_minutes": float((end - start).total_seconds() / 60),
            "supports_L36_H1": len(group) >= 37,
            "supports_L72_H1": len(group) >= 73,
            "supports_L144_H1": len(group) >= 145,
        })
    return rows


def is_temporally_valid_window(
    timestamps: Sequence[pd.Timestamp],
    lookback: int,
    horizon: int = 1,
) -> tuple[bool, str]:
    if lookback < 1 or horizon < 1:
        raise ValueError("lookback and horizon must be positive")
    required = lookback + horizon
    if len(timestamps) != required:
        return False, "INSUFFICIENT_HISTORY"
    values = pd.DatetimeIndex(timestamps)
    if values.has_duplicates:
        return False, "DUPLICATE_TIMESTAMP"
    input_values = values[:lookback]
    input_deltas = np.diff(input_values.asi8) / 60_000_000_000
    if np.any(input_deltas <= 0):
        return False, "TARGET_NOT_FUTURE"
    if np.any(input_deltas != EXPECTED_INTERVAL_MINUTES):
        return False, "INPUT_GAP"
    target_delta = (values[-1] - input_values[-1]).total_seconds() / 60
    if target_delta <= 0:
        return False, "TARGET_NOT_FUTURE"
    if target_delta != EXPECTED_INTERVAL_MINUTES * horizon:
        return False, "TARGET_GAP"
    if any(timestamp.second != 0 or timestamp.minute % EXPECTED_INTERVAL_MINUTES for timestamp in values):
        return False, "OFF_GRID_TIMESTAMP"
    return True, "VALID"


def audit_temporal_dataframe(dataframe: pd.DataFrame) -> dict[str, Any]:
    raw_timestamps = parse_timestamps_strict(dataframe["date"])
    parse_failures = int(raw_timestamps.isna().sum() - dataframe["date"].isna().sum())
    original_deltas = raw_timestamps.diff().dt.total_seconds().div(60)
    raw_order_monotonic = raw_timestamps.is_monotonic_increasing
    duplicate_rows = duplicate_timestamp_rows(dataframe, raw_timestamps)
    view = build_temporal_view(dataframe)
    gaps = gap_rows(view)
    missing = missing_timestamp_rows(view, gaps)
    segments = continuity_segment_rows(view)
    sorted_deltas = view["delta_minutes"].dropna()
    off_grid_mask = (
        view["timestamp_parsed"].dt.second.ne(0)
        | view["timestamp_parsed"].dt.microsecond.ne(0)
        | view["timestamp_parsed"].dt.minute.mod(EXPECTED_INTERVAL_MINUTES).ne(0)
    )
    unique_count = int(view["timestamp_parsed"].nunique())
    expected_grid_count = len(pd.date_range(view["timestamp_parsed"].min(), view["timestamp_parsed"].max(), freq="10min"))
    expected_delta_count = int(sorted_deltas.eq(EXPECTED_INTERVAL_MINUTES).sum())
    denominator = max(len(view) - 1, 1)
    manifest = {
        "row_count": len(view),
        "parsed_timestamp_count": int(raw_timestamps.notna().sum()),
        "parse_failure_count": parse_failures,
        "parse_success_rate": float(raw_timestamps.notna().mean()),
        "min_timestamp": view["timestamp_parsed"].min().isoformat(sep=" "),
        "max_timestamp": view["timestamp_parsed"].max().isoformat(sep=" "),
        "elapsed_duration_minutes": float((view["timestamp_parsed"].max() - view["timestamp_parsed"].min()).total_seconds() / 60),
        "unique_timestamp_count": unique_count,
        "duplicate_timestamp_count": int(raw_timestamps.duplicated(keep=False).sum()),
        "duplicate_timestamp_group_count": len(duplicate_rows),
        "conflicting_duplicate_group_count": sum(row["duplicate_type"] == "DT1_CONFLICTING" for row in duplicate_rows),
        "negative_delta_count": int(original_deltas.lt(0).sum()),
        "zero_delta_count": int(original_deltas.eq(0).sum()),
        "expected_delta_count": expected_delta_count,
        "too_short_delta_count": int(sorted_deltas.between(0, EXPECTED_INTERVAL_MINUTES, inclusive="neither").sum()),
        "gap_count": len(gaps),
        "off_grid_timestamp_count": int(off_grid_mask.sum()),
        "off_grid_gap_count": sum(not row["is_multiple_of_10min"] for row in gaps),
        "expected_grid_count": expected_grid_count,
        "missing_timestamp_count": len(missing),
        "continuity_ratio": expected_delta_count / denominator,
        "coverage_completeness_ratio": unique_count / expected_grid_count,
        "continuity_segment_count": len(segments),
        "largest_gap_minutes": max((row["delta_minutes"] for row in gaps), default=None),
        "raw_order_monotonic": raw_order_monotonic,
        "sorted_view_monotonic": view["timestamp_parsed"].is_monotonic_increasing,
    }
    critical = []
    if parse_failures:
        critical.append("parse_failures")
    if manifest["conflicting_duplicate_group_count"]:
        critical.append("conflicting_duplicates")
    if manifest["off_grid_timestamp_count"] or manifest["off_grid_gap_count"]:
        critical.append("off_grid")
    if not manifest["sorted_view_monotonic"]:
        critical.append("sorted_order")
    warnings = []
    if not raw_order_monotonic:
        warnings.append("raw_order_not_monotonic")
    if duplicate_rows:
        warnings.append("duplicate_timestamps")
    if gaps:
        warnings.append("temporal_gaps_segmented")
    if critical:
        status = "FAIL"
    elif warnings:
        status = "PASS_WITH_WARNING"
    else:
        status = "PASS"
    return {
        "view": view,
        "gaps": gaps,
        "missing": missing,
        "duplicates": duplicate_rows,
        "segments": segments,
        "manifest": manifest,
        "warnings": warnings,
        "critical_failures": critical,
        "status": status,
    }


def verify_existing_signoff(root: Path, signoff_path: Path) -> dict[str, Any]:
    signoff = read_json(signoff_path)
    if signoff.get("status") not in {"PASS", "PASS_WITH_WARNING"} or signoff.get("artifact_version") != "TEMPORAL-v1":
        raise RuntimeError("Existing Phase 4 sign-off is invalid")
    for relative_path, expected_hash in signoff.get("output_checksums", {}).items():
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Phase 4 artifact checksum mismatch: {relative_path}")
    return signoff


def materialize_phase_4(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    phase_3 = materialize_phase_3(root)
    if phase_3.get("status") not in {"PASS", "PASS_WITH_WARNING"}:
        raise RuntimeError("Phase 3 sign-off does not permit Phase 4")
    raw_path = root / "data/raw_data/energydata_complete.csv"
    raw_hash_before = sha256_file(raw_path)
    schema_manifest = read_json(root / "artifacts/schema/schema_manifest.json")
    if raw_hash_before != schema_manifest.get("raw_csv_sha256") or schema_manifest.get("timestamp_column") != "date":
        raise RuntimeError("Temporal inputs do not match DATA-v1 and SCHEMA-v1")
    artifact_root = root / "artifacts/temporal"
    signoff_path = artifact_root / "phase_4_signoff.json"
    if signoff_path.exists():
        return verify_existing_signoff(root, signoff_path)
    dataframe = load_raw_csv(raw_path)
    dataframe_before = dataframe_fingerprint(dataframe)
    audit = audit_temporal_dataframe(dataframe)
    if audit["status"] == "FAIL":
        raise RuntimeError(f"Critical temporal failures: {audit['critical_failures']}")
    view = audit["view"]
    manifest_values = audit["manifest"]
    dataframe_after = dataframe_fingerprint(dataframe)
    if dataframe_after != dataframe_before:
        raise RuntimeError("Raw DataFrame mutated during temporal audit")
    if sha256_file(raw_path) != raw_hash_before:
        raise RuntimeError("Raw CSV changed during temporal audit")
    interval_rows = [
        {
            "delta_minutes": delta,
            "interval_status": interval_status(float(delta)),
            "count": int(count),
        }
        for delta, count in view["delta_minutes"].dropna().value_counts().sort_index().items()
    ]
    daily = view.assign(calendar_day=view["timestamp_parsed"].dt.strftime("%Y-%m-%d")).groupby("calendar_day").size()
    first_day = daily.index[0]
    last_day = daily.index[-1]
    daily_rows = [
        {
            "calendar_day": day,
            "observation_count": int(count),
            "expected_full_day_count": 144,
            "boundary_status": "FIRST_PARTIAL" if day == first_day and count != 144 else "LAST_PARTIAL" if day == last_day and count != 144 else "INTERIOR",
            "status": "PASS" if count == 144 or day in {first_day, last_day} else "WARNING",
        }
        for day, count in daily.items()
    ]
    summary_expectations = {
        "parsed_timestamp_count": len(dataframe),
        "unique_timestamp_count": len(dataframe),
        "duplicate_timestamp_count": 0,
        "negative_delta_count": 0,
        "zero_delta_count": 0,
        "expected_delta_count": len(dataframe) - 1,
        "gap_count": 0,
        "missing_timestamp_count": 0,
        "off_grid_timestamp_count": 0,
        "continuity_segment_count": 1,
        "continuity_ratio": 1.0,
        "coverage_completeness_ratio": 1.0,
    }
    summary_rows = [
        {
            "metric": metric,
            "value": manifest_values[metric],
            "expected": expected,
            "status": "PASS" if manifest_values[metric] == expected else "WARNING",
        }
        for metric, expected in summary_expectations.items()
    ]
    discrepancies = []
    for index, gap in enumerate(audit["gaps"], start=1):
        discrepancies.append({
            "id": f"TD-{index:03d}",
            "severity": "WARNING",
            "category": "GAP",
            "timestamp": gap["next_timestamp"],
            "previous_timestamp": gap["previous_timestamp"],
            "expected": EXPECTED_INTERVAL_MINUTES,
            "actual": gap["delta_minutes"],
            "interpretation": "Observed timestamps are separated by more than one expected interval",
            "recommended_action": "Reject windows crossing the assigned continuity segment boundary",
            "resolved": True,
            "notes": gap["gap_id"],
        })
    temporal_manifest = {
        "temporal_version": "TEMPORAL-v1",
        "dataset_revision": "DATA-v1",
        "schema_version": "SCHEMA-v1",
        "environment_id": "ENV-v1",
        "raw_csv_sha256": raw_hash_before,
        "timestamp_column": "date",
        "timestamp_format": TIMESTAMP_FORMAT,
        "timezone_status": "unspecified_in_raw_data",
        "expected_interval_minutes": EXPECTED_INTERVAL_MINUTES,
        **manifest_values,
        "window_safety_contract": WINDOW_SAFETY_CONTRACT,
        "dataframe_fingerprint_before": dataframe_before,
        "dataframe_fingerprint_after": dataframe_after,
        "audit_status": audit["status"],
        "warnings": audit["warnings"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    temporal_manifest_path = artifact_root / "temporal_manifest.json"
    temporal_summary_path = artifact_root / "temporal_summary.csv"
    gap_path = artifact_root / "timestamp_gaps.csv"
    missing_path = artifact_root / "missing_timestamps.csv"
    duplicate_path = artifact_root / "duplicate_timestamps.csv"
    segment_path = artifact_root / "continuity_segments.csv"
    interval_path = artifact_root / "interval_distribution.csv"
    daily_path = artifact_root / "daily_observation_counts.csv"
    discrepancies_path = artifact_root / "temporal_discrepancies.json"
    write_json_once_or_verify(temporal_manifest_path, temporal_manifest)
    write_text_once_or_verify(temporal_summary_path, csv_text(["metric", "value", "expected", "status"], summary_rows))
    gap_fields = ["gap_id", "raw_row_index_prev", "raw_row_index_current", "previous_timestamp", "next_timestamp", "delta_minutes", "missing_steps", "is_multiple_of_10min", "severity", "notes"]
    write_text_once_or_verify(gap_path, csv_text(gap_fields, audit["gaps"]))
    missing_fields = ["missing_timestamp", "previous_observed_timestamp", "next_observed_timestamp", "gap_id"]
    write_text_once_or_verify(missing_path, csv_text(missing_fields, audit["missing"]))
    duplicate_fields = ["timestamp", "duplicate_type", "group_size", "raw_row_indices", "conflicting_columns"]
    write_text_once_or_verify(duplicate_path, csv_text(duplicate_fields, audit["duplicates"]))
    segment_fields = ["segment_id", "start_timestamp", "end_timestamp", "row_count", "duration_minutes", "supports_L36_H1", "supports_L72_H1", "supports_L144_H1"]
    write_text_once_or_verify(segment_path, csv_text(segment_fields, audit["segments"]))
    write_text_once_or_verify(interval_path, csv_text(["delta_minutes", "interval_status", "count"], interval_rows))
    write_text_once_or_verify(daily_path, csv_text(["calendar_day", "observation_count", "expected_full_day_count", "boundary_status", "status"], daily_rows))
    write_json_once_or_verify(discrepancies_path, {"temporal_version": "TEMPORAL-v1", "discrepancies": discrepancies})
    output_paths = [
        "artifacts/temporal/temporal_manifest.json",
        "artifacts/temporal/temporal_summary.csv",
        "artifacts/temporal/timestamp_gaps.csv",
        "artifacts/temporal/missing_timestamps.csv",
        "artifacts/temporal/duplicate_timestamps.csv",
        "artifacts/temporal/continuity_segments.csv",
        "artifacts/temporal/interval_distribution.csv",
        "artifacts/temporal/daily_observation_counts.csv",
        "artifacts/temporal/temporal_discrepancies.json",
    ]
    output_checksums = {relative: sha256_file(root / relative) for relative in output_paths}
    signoff = {
        "artifact_version": "TEMPORAL-v1",
        "phase_id": 4,
        "phase_version": "PHASE-4-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment_id": "ENV-v1",
        "dataset_revision": "DATA-v1",
        "input_paths": ["artifacts/schema/phase_3_signoff.json", "data/raw_data/energydata_complete.csv"],
        "input_checksums": {
            "artifacts/schema/phase_3_signoff.json": sha256_file(root / "artifacts/schema/phase_3_signoff.json"),
            "data/raw_data/energydata_complete.csv": raw_hash_before,
        },
        "output_paths": output_paths,
        "output_checksums": output_checksums,
        "config_fingerprint": phase_3["config_fingerprint"],
        "status": audit["status"],
        "tests": [
            "strict_timestamp_parse",
            "original_order_before_sort",
            "duplicate_classification",
            "interval_taxonomy",
            "grid_and_gap_detection",
            "continuity_segmentation",
            "window_safety_contract",
            "raw_dataframe_immutability",
        ],
        "warnings": audit["warnings"],
        "discrepancies": [item["id"] for item in discrepancies],
    }
    write_json_once_or_verify(signoff_path, signoff)
    return signoff


def load_validated_temporal_view(project_root: Path | None = None) -> pd.DataFrame:
    root = (project_root or get_project_root()).resolve()
    signoff = materialize_phase_4(root)
    if signoff.get("status") not in {"PASS", "PASS_WITH_WARNING"}:
        raise RuntimeError("TEMPORAL-v1 is not signed off")
    return build_temporal_view(load_raw_csv(root / "data/raw_data/energydata_complete.csv"))
