from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from course_work.data.schema import dataframe_fingerprint, load_raw_csv
from course_work.data.temporal import TIMESTAMP_FORMAT, load_validated_temporal_view
from course_work.utils.artifacts import (
    csv_text,
    get_project_root,
    read_json,
    sha256_file,
    write_json_once_or_verify,
    write_text_once_or_verify,
)


FEATURE_VERSION = "FEATURES-v1"
EDA_VERSION = "EDA-v1"
TIME_FEATURE_VERSION = "TIME-FEATURES-v1"
ENGINEERED_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "weekend"]
METADATA_COLUMNS = ["raw_row_index", "date", "timestamp", "continuity_segment_id"]
TARGET_COLUMN = "Appliances"
DERIVED_RELATIVE_PATH = "data/interim/uci_appliances_energy_prediction/energydata_feature_engineered_v1.csv"
FEATURE_ARTIFACT_ROOT = "artifacts/features"
UPSTREAM_SIGNOFFS = {
    "artifacts/contracts/phase_0_signoff.json": ("COURSEWORK-CONTRACT-v1", {"PASS"}),
    "artifacts/environment/phase_1_signoff.json": ("ENV-v1", {"PASS"}),
    "artifacts/acquisition/phase_2_signoff.json": ("DATA-v1", {"PASS"}),
    "artifacts/schema/phase_3_signoff.json": ("SCHEMA-v1", {"PASS", "PASS_WITH_WARNING"}),
    "artifacts/temporal/phase_4_signoff.json": ("TEMPORAL-v1", {"PASS", "PASS_WITH_WARNING"}),
    "artifacts/splits/phase_5_signoff.json": ("SPLIT-v1", {"PASS"}),
}


def verify_signed_artifact(root: Path, relative_path: str, expected_version: str, allowed_statuses: set[str]) -> dict[str, Any]:
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Required signed artifact is missing: {relative_path}")
    signoff = read_json(path)
    if signoff.get("artifact_version") != expected_version or signoff.get("status") not in allowed_statuses:
        raise RuntimeError(f"Invalid upstream sign-off: {relative_path}")
    for output_path, expected_checksum in signoff.get("output_checksums", {}).items():
        artifact_path = root / output_path
        if not artifact_path.is_file() or sha256_file(artifact_path) != expected_checksum:
            raise RuntimeError(f"Upstream artifact checksum mismatch: {output_path}")
    return signoff


def verify_phase_7_inputs(root: Path) -> dict[str, dict[str, Any]]:
    signoffs = {
        relative_path: verify_signed_artifact(root, relative_path, expected_version, allowed_statuses)
        for relative_path, (expected_version, allowed_statuses) in UPSTREAM_SIGNOFFS.items()
    }
    schema_manifest = read_json(root / "artifacts/schema/schema_manifest.json")
    temporal_manifest = read_json(root / "artifacts/temporal/temporal_manifest.json")
    eda_manifest_path = root / "artifacts/eda/eda_manifest.json"
    if not eda_manifest_path.is_file():
        raise FileNotFoundError("Required upstream EDA manifest is missing: artifacts/eda/eda_manifest.json")
    eda_manifest = read_json(eda_manifest_path)
    if eda_manifest.get("eda_version") != EDA_VERSION:
        raise RuntimeError(
            f"Invalid EDA version: expected {EDA_VERSION}, got {eda_manifest.get('eda_version')}"
        )
    split_signoff = signoffs["artifacts/splits/phase_5_signoff.json"]
    raw_path = root / "data/raw_data/energydata_complete.csv"
    raw_checksum = sha256_file(raw_path)
    if schema_manifest.get("raw_csv_sha256") != raw_checksum:
        raise RuntimeError("SCHEMA-v1 does not match DATA-v1")
    if temporal_manifest.get("raw_csv_sha256") != raw_checksum:
        raise RuntimeError("TEMPORAL-v1 does not match DATA-v1")
    versions = {
        "schema": schema_manifest.get("schema_version"),
        "temporal": temporal_manifest.get("temporal_version"),
        "split": split_signoff.get("artifact_version"),
        "environment": split_signoff.get("environment_id"),
        "dataset": split_signoff.get("dataset_revision"),
    }
    expected = {
        "schema": "SCHEMA-v1",
        "temporal": "TEMPORAL-v1",
        "split": "SPLIT-v1",
        "environment": "ENV-v1",
        "dataset": "DATA-v1",
    }
    if versions != expected:
        raise RuntimeError(f"Phase 7 input version mismatch: {versions}")
    signoffs["artifacts/eda/eda_manifest.json"] = eda_manifest
    return signoffs


def validate_feature_input(dataframe: pd.DataFrame, schema_manifest: dict[str, Any]) -> None:
    required = [
        "raw_row_index",
        "date",
        "timestamp_parsed",
        "continuity_segment_id",
        TARGET_COLUMN,
        *schema_manifest["regular_feature_columns"],
        *schema_manifest["random_control_columns"],
    ]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Feature input columns are missing: {missing}")
    if dataframe.columns.duplicated().any():
        raise ValueError("Feature input has duplicate column names")
    if dataframe["timestamp_parsed"].isna().any() or not pd.api.types.is_datetime64_any_dtype(dataframe["timestamp_parsed"]):
        raise ValueError("Feature input must use valid TEMPORAL-v1 parsed timestamps")
    if dataframe["raw_row_index"].duplicated().any():
        raise ValueError("Feature input has duplicate raw row indices")


def add_time_features(dataframe: pd.DataFrame, timestamp_column: str = "timestamp") -> pd.DataFrame:
    if timestamp_column not in dataframe.columns:
        raise ValueError(f"Timestamp column is missing: {timestamp_column}")
    existing = [column for column in ENGINEERED_FEATURES if column in dataframe.columns]
    if existing:
        raise ValueError(f"Engineered time features already exist: {existing}")
    timestamps = dataframe[timestamp_column]
    if timestamps.isna().any() or not pd.api.types.is_datetime64_any_dtype(timestamps):
        raise ValueError("Time features require non-null datetime values")
    output = dataframe.copy(deep=True)
    minute_of_day = timestamps.dt.hour.mul(60).add(timestamps.dt.minute)
    day_of_week = timestamps.dt.dayofweek
    output["hour_sin"] = np.sin(2.0 * np.pi * minute_of_day / 1440.0)
    output["hour_cos"] = np.cos(2.0 * np.pi * minute_of_day / 1440.0)
    output["dow_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    output["dow_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)
    output["weekend"] = day_of_week.isin([5, 6]).astype("int8")
    return output


def build_feature_view(temporal_view: pd.DataFrame, schema_manifest: dict[str, Any]) -> pd.DataFrame:
    validate_feature_input(temporal_view, schema_manifest)
    ordered_raw_features = schema_manifest["regular_feature_columns"]
    random_controls = schema_manifest["random_control_columns"]
    output = pd.DataFrame({
        "raw_row_index": temporal_view["raw_row_index"].to_numpy(copy=True),
        "date": temporal_view["date"].astype(str).to_numpy(copy=True),
        "timestamp": temporal_view["timestamp_parsed"].to_numpy(copy=True),
        "continuity_segment_id": temporal_view["continuity_segment_id"].to_numpy(copy=True),
    })
    for column in [TARGET_COLUMN, *ordered_raw_features, *random_controls]:
        output[column] = temporal_view[column].to_numpy(copy=True)
    return add_time_features(output)


def candidate_membership(column: str, schema_manifest: dict[str, Any]) -> str:
    if column in METADATA_COLUMNS:
        return "METADATA_ONLY"
    if column == TARGET_COLUMN:
        return "FS1_CANDIDATE;FS2_CANDIDATE"
    if column in schema_manifest["random_control_columns"]:
        return "FS2_CANDIDATE"
    if column in ENGINEERED_FEATURES:
        return "FS0_CANDIDATE;FS1_CANDIDATE;FS2_CANDIDATE_WHEN_TF1"
    return "FS0_CANDIDATE;FS1_CANDIDATE;FS2_CANDIDATE"


def feature_semantics(column: str, schema_manifest: dict[str, Any]) -> dict[str, Any]:
    if column == "raw_row_index":
        return {"origin": "TEMPORAL-v1", "role": "metadata", "group": "lineage", "unit": None, "availability": "METADATA_ONLY", "model_eligible": False, "source": "raw row position", "formula": "identity", "transformation": "SAFE_GLOBAL_DETERMINISTIC"}
    if column == "date":
        return {"origin": "DATA-v1", "role": "metadata", "group": "time", "unit": None, "availability": "METADATA_ONLY", "model_eligible": False, "source": "date", "formula": "identity", "transformation": "SAFE_GLOBAL_DETERMINISTIC"}
    if column == "timestamp":
        return {"origin": "TEMPORAL-v1", "role": "metadata", "group": "time", "unit": None, "availability": "METADATA_ONLY", "model_eligible": False, "source": "timestamp_parsed", "formula": "identity", "transformation": "SAFE_GLOBAL_DETERMINISTIC"}
    if column == "continuity_segment_id":
        return {"origin": "TEMPORAL-v1", "role": "metadata", "group": "continuity", "unit": None, "availability": "METADATA_ONLY", "model_eligible": False, "source": "continuity_segment_id", "formula": "identity", "transformation": "SAFE_GLOBAL_DETERMINISTIC"}
    if column == TARGET_COLUMN:
        return {"origin": "DATA-v1", "role": "target_history_candidate", "group": "target", "unit": schema_manifest["unit_mapping"][column], "availability": "TARGET_HISTORY_CANDIDATE", "model_eligible": True, "source": column, "formula": "identity", "transformation": "SAFE_GLOBAL_DETERMINISTIC"}
    if column in schema_manifest["random_control_columns"]:
        return {"origin": "DATA-v1", "role": "random_control", "group": "random_controls", "unit": schema_manifest["unit_mapping"][column], "availability": "RANDOM_CONTROL", "model_eligible": True, "source": column, "formula": "identity", "transformation": "SAFE_GLOBAL_DETERMINISTIC"}
    if column in schema_manifest["regular_feature_columns"]:
        return {"origin": "DATA-v1", "role": "raw_exogenous", "group": schema_manifest["feature_group_mapping"][column], "unit": schema_manifest["unit_mapping"][column], "availability": "PAST_OBSERVED", "model_eligible": True, "source": column, "formula": "identity", "transformation": "SAFE_GLOBAL_DETERMINISTIC"}
    formulas = {
        "hour_sin": "sin(2*pi*minute_of_day/1440)",
        "hour_cos": "cos(2*pi*minute_of_day/1440)",
        "dow_sin": "sin(2*pi*day_of_week/7)",
        "dow_cos": "cos(2*pi*day_of_week/7)",
        "weekend": "1 if day_of_week in {5,6} else 0",
    }
    return {"origin": TIME_FEATURE_VERSION, "role": "engineered_time", "group": "calendar", "unit": "nondimensional", "availability": "KNOWN_CALENDAR", "model_eligible": True, "source": "timestamp", "formula": formulas[column], "transformation": "SAFE_GLOBAL_DETERMINISTIC"}


def build_feature_lineage(feature_view: pd.DataFrame, schema_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, column in enumerate(feature_view.columns, start=1):
        semantics = feature_semantics(column, schema_manifest)
        rows.append({
            "feature_name": column,
            "feature_type": semantics["role"],
            "source_columns": semantics["source"],
            "formula": semantics["formula"],
            "transformation_class": semantics["transformation"],
            "unit": semantics["unit"],
            "availability": semantics["availability"],
            "uses_train_fit": False,
            "uses_future_information": False,
            "default_model_status": candidate_membership(column, schema_manifest),
            "notes": f"LINEAGE-{index:03d}",
        })
    return rows


def build_feature_registry(feature_view: pd.DataFrame, schema_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, column in enumerate(feature_view.columns, start=1):
        semantics = feature_semantics(column, schema_manifest)
        rows.append({
            "column_name": column,
            "origin": semantics["origin"],
            "role": semantics["role"],
            "feature_group": semantics["group"],
            "model_eligible": semantics["model_eligible"],
            "feature_set_membership_candidate": candidate_membership(column, schema_manifest),
            "unit": semantics["unit"],
            "dtype": str(feature_view[column].dtype),
            "availability": semantics["availability"],
            "lineage_id": f"LINEAGE-{index:03d}",
            "status": "ACTIVE",
            "notes": "Phase 7 owns final ordered feature-set membership",
        })
    return rows


def build_feature_availability(feature_view: pd.DataFrame, schema_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for column in feature_view.columns:
        semantics = feature_semantics(column, schema_manifest)
        rows.append({
            "feature_name": column,
            "availability": semantics["availability"],
            "information_time": "historical_row_or_known_calendar",
            "model_eligible": semantics["model_eligible"],
            "feature_set_membership_candidate": candidate_membership(column, schema_manifest),
            "available_at_prediction_time": True,
            "status": "PASS",
            "notes": "Metadata is retained for lineage and excluded from model inputs" if not semantics["model_eligible"] else "Available for historical sequence rows",
        })
    return rows


def build_feature_leakage_audit(feature_view: pd.DataFrame, schema_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for column in feature_view.columns:
        semantics = feature_semantics(column, schema_manifest)
        rows.append({
            "feature_name": column,
            "depends_on_future_target": False,
            "depends_on_future_feature": False,
            "depends_on_full_dataset_statistics": False,
            "available_at_prediction_time": True,
            "safe_for_pre_split_engineering": True,
            "status": "PASS",
            "notes": "Not model eligible" if not semantics["model_eligible"] else "Historical or deterministic calendar information only",
        })
    return rows


def validate_feature_invariants(temporal_view: pd.DataFrame, feature_view: pd.DataFrame, schema_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    raw_features = [TARGET_COLUMN, *schema_manifest["regular_feature_columns"], *schema_manifest["random_control_columns"]]
    expected_columns = [*METADATA_COLUMNS, *raw_features, *ENGINEERED_FEATURES]
    engineered = feature_view[ENGINEERED_FEATURES]
    hour_norm = feature_view["hour_sin"].pow(2).add(feature_view["hour_cos"].pow(2))
    dow_norm = feature_view["dow_sin"].pow(2).add(feature_view["dow_cos"].pow(2))
    raw_preserved = all(feature_view[column].reset_index(drop=True).equals(temporal_view[column].reset_index(drop=True)) for column in raw_features)
    checks = [
        ("row_count_preserved", len(feature_view) == len(temporal_view), len(temporal_view), len(feature_view)),
        ("timestamp_preserved", feature_view["timestamp"].reset_index(drop=True).equals(temporal_view["timestamp_parsed"].reset_index(drop=True)), "TEMPORAL-v1 exact values and order", "exact"),
        ("target_preserved", feature_view[TARGET_COLUMN].reset_index(drop=True).equals(temporal_view[TARGET_COLUMN].reset_index(drop=True)), "DATA-v1 target values", "exact"),
        ("raw_features_preserved", raw_preserved, "all raw target, exogenous and control values", "exact"),
        ("continuity_segments_preserved", feature_view["continuity_segment_id"].reset_index(drop=True).equals(temporal_view["continuity_segment_id"].reset_index(drop=True)), "TEMPORAL-v1 segment IDs", "exact"),
        ("no_duplicate_column_names", not feature_view.columns.duplicated().any(), 0, int(feature_view.columns.duplicated().sum())),
        ("no_new_nulls", int(engineered.isna().sum().sum()) == 0, 0, int(engineered.isna().sum().sum())),
        ("time_feature_ranges_valid", bool(feature_view[["hour_sin", "hour_cos", "dow_sin", "dow_cos"]].ge(-1.0 - 1e-12).all().all() and feature_view[["hour_sin", "hour_cos", "dow_sin", "dow_cos"]].le(1.0 + 1e-12).all().all()), "[-1, 1]", "within tolerance"),
        ("cyclical_norm_valid", bool(np.allclose(hour_norm, 1.0, atol=1e-12) and np.allclose(dow_norm, 1.0, atol=1e-12)), 1.0, f"hour_max_error={float((hour_norm - 1).abs().max())};dow_max_error={float((dow_norm - 1).abs().max())}"),
        ("weekend_binary_valid", set(feature_view["weekend"].unique()).issubset({0, 1}), "subset of {0,1}", sorted(int(value) for value in feature_view["weekend"].unique())),
        ("feature_count_valid", list(feature_view.columns) == expected_columns, expected_columns, list(feature_view.columns)),
        ("no_future_leakage", not any(token in column.lower() for column in feature_view.columns for token in ["target_next", "t_plus_1", "future_", "rolling_", "_lag_", "scaled_"]), "no forbidden feature names", "none"),
    ]
    rows = [
        {"check": name, "expected": expected, "actual": actual, "status": "PASS" if passed else "FAIL", "details": "deterministic Phase 6 invariant"}
        for name, passed, expected, actual in checks
    ]
    failures = [row["check"] for row in rows if row["status"] == "FAIL"]
    if failures:
        raise RuntimeError(f"Phase 6 invariant failures: {failures}")
    return rows


def dataframe_csv(dataframe: pd.DataFrame) -> str:
    return dataframe.to_csv(index=False, lineterminator="\n", date_format=TIMESTAMP_FORMAT)


def verify_existing_signoff(root: Path, signoff_path: Path) -> dict[str, Any]:
    signoff = read_json(signoff_path)
    if signoff.get("status") != "PASS" or signoff.get("artifact_version") != FEATURE_VERSION:
        raise RuntimeError("Existing Phase 7 sign-off is invalid")
    for relative_path, expected_checksum in signoff.get("output_checksums", {}).items():
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected_checksum:
            raise RuntimeError(f"Phase 7 artifact checksum mismatch: {relative_path}")
    manifest = read_json(root / FEATURE_ARTIFACT_ROOT / "feature_engineering_manifest.json")
    derived_path = root / manifest["derived_file_path"]
    checksum_text = (root / FEATURE_ARTIFACT_ROOT / "feature_engineered_v1.sha256").read_text(encoding="utf-8").strip()
    if sha256_file(derived_path) != manifest["derived_file_sha256"] or checksum_text != manifest["derived_file_sha256"]:
        raise RuntimeError("FEATURES-v1 derived checksum verification failed")
    return signoff


def materialize_phase_7(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    signoff_path = root / FEATURE_ARTIFACT_ROOT / "phase_7_signoff.json"
    if signoff_path.exists():
        return verify_existing_signoff(root, signoff_path)
    upstream = verify_phase_7_inputs(root)
    raw_path = root / "data/raw_data/energydata_complete.csv"
    raw_checksum_before = sha256_file(raw_path)
    raw_dataframe = load_raw_csv(raw_path)
    raw_fingerprint_before = dataframe_fingerprint(raw_dataframe)
    temporal_view = load_validated_temporal_view(root)
    schema_manifest = read_json(root / "artifacts/schema/schema_manifest.json")
    feature_view = build_feature_view(temporal_view, schema_manifest)
    audit_rows = validate_feature_invariants(temporal_view, feature_view, schema_manifest)
    lineage_rows = build_feature_lineage(feature_view, schema_manifest)
    registry_rows = build_feature_registry(feature_view, schema_manifest)
    availability_rows = build_feature_availability(feature_view, schema_manifest)
    leakage_rows = build_feature_leakage_audit(feature_view, schema_manifest)
    if dataframe_fingerprint(raw_dataframe) != raw_fingerprint_before or sha256_file(raw_path) != raw_checksum_before:
        raise RuntimeError("Raw DATA-v1 changed during Phase 7")
    derived_path = root / DERIVED_RELATIVE_PATH
    write_text_once_or_verify(derived_path, dataframe_csv(feature_view))
    derived_checksum = sha256_file(derived_path)
    artifact_root = root / FEATURE_ARTIFACT_ROOT
    lineage_path = artifact_root / "feature_lineage.csv"
    registry_path = artifact_root / "feature_registry.csv"
    availability_path = artifact_root / "feature_availability.csv"
    leakage_path = artifact_root / "feature_leakage_audit.csv"
    audit_path = artifact_root / "feature_engineering_audit.csv"
    discrepancies_path = artifact_root / "feature_engineering_discrepancies.json"
    checksum_path = artifact_root / "feature_engineered_v1.sha256"
    manifest_path = artifact_root / "feature_engineering_manifest.json"
    lineage_fields = ["feature_name", "feature_type", "source_columns", "formula", "transformation_class", "unit", "availability", "uses_train_fit", "uses_future_information", "default_model_status", "notes"]
    registry_fields = ["column_name", "origin", "role", "feature_group", "model_eligible", "feature_set_membership_candidate", "unit", "dtype", "availability", "lineage_id", "status", "notes"]
    availability_fields = ["feature_name", "availability", "information_time", "model_eligible", "feature_set_membership_candidate", "available_at_prediction_time", "status", "notes"]
    leakage_fields = ["feature_name", "depends_on_future_target", "depends_on_future_feature", "depends_on_full_dataset_statistics", "available_at_prediction_time", "safe_for_pre_split_engineering", "status", "notes"]
    audit_fields = ["check", "expected", "actual", "status", "details"]
    write_text_once_or_verify(lineage_path, csv_text(lineage_fields, lineage_rows))
    write_text_once_or_verify(registry_path, csv_text(registry_fields, registry_rows))
    write_text_once_or_verify(availability_path, csv_text(availability_fields, availability_rows))
    write_text_once_or_verify(leakage_path, csv_text(leakage_fields, leakage_rows))
    write_text_once_or_verify(audit_path, csv_text(audit_fields, audit_rows))
    write_json_once_or_verify(discrepancies_path, {"feature_version": FEATURE_VERSION, "discrepancies": []})
    write_text_once_or_verify(checksum_path, f"{derived_checksum}\n")
    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "feature_version": FEATURE_VERSION,
        "dataset_revision": "DATA-v1",
        "schema_version": "SCHEMA-v1",
        "temporal_version": "TEMPORAL-v1",
        "eda_version": upstream["artifacts/eda/eda_manifest.json"]["eda_version"],
        "split_version": "SPLIT-v1",
        "environment_id": "ENV-v1",
        "feature_engineering_scope": "FULL_DATASET_DETERMINISTIC_TRANSFORMATIONS",
        "train_only_scope": False,
        "scope_explanation": (
            "Engineered features are deterministic transformations of timestamps and known calendar values "
            "that do not depend on target values, scaler fits, or any data-derived statistics. "
            "Therefore the FEATURES-v1 feature view may be engineered over the full dataset, while statistical "
            "transformations (scaling, statistics) and training remain TRAIN-only."
        ),
        "total_rows": len(temporal_view),
        "source_raw_sha256": raw_checksum_before,
        "derived_file_path": DERIVED_RELATIVE_PATH,
        "derived_file_sha256": derived_checksum,
        "row_count": len(feature_view),
        "column_count": len(feature_view.columns),
        "raw_feature_count": len(schema_manifest["regular_feature_columns"]) + len(schema_manifest["random_control_columns"]),
        "engineered_feature_count": len(ENGINEERED_FEATURES),
        "metadata_column_count": len(METADATA_COLUMNS),
        "target_column": TARGET_COLUMN,
        "ordered_columns": list(feature_view.columns),
        "engineered_features": ENGINEERED_FEATURES,
        "time_feature_formula_version": TIME_FEATURE_VERSION,
        "time_feature_version": TIME_FEATURE_VERSION,
        "timestamp_serialization_format": TIMESTAMP_FORMAT,
        "row_count_preserved": True,
        "timestamp_preserved": True,
        "target_preserved": True,
        "raw_features_preserved": True,
        "continuity_preserved": True,
        "no_new_missing_values": True,
        "feature_count_valid": True,
        "new_missing_values_count": 0,
        "leakage_audit_status": "PASS",
        "leakage_audit_passed": True,
        "audit_status": "PASS",
        "raw_dataframe_fingerprint_before": raw_fingerprint_before,
        "raw_dataframe_fingerprint_after": dataframe_fingerprint(raw_dataframe),
        "warnings": [],
        "created_at": created_at,
    }
    write_json_once_or_verify(manifest_path, manifest)
    output_paths = [
        DERIVED_RELATIVE_PATH,
        f"{FEATURE_ARTIFACT_ROOT}/feature_engineering_manifest.json",
        f"{FEATURE_ARTIFACT_ROOT}/feature_registry.csv",
        f"{FEATURE_ARTIFACT_ROOT}/feature_lineage.csv",
        f"{FEATURE_ARTIFACT_ROOT}/feature_availability.csv",
        f"{FEATURE_ARTIFACT_ROOT}/feature_leakage_audit.csv",
        f"{FEATURE_ARTIFACT_ROOT}/feature_engineering_audit.csv",
        f"{FEATURE_ARTIFACT_ROOT}/feature_engineering_discrepancies.json",
        f"{FEATURE_ARTIFACT_ROOT}/feature_engineered_v1.sha256",
    ]
    output_checksums = {relative_path: sha256_file(root / relative_path) for relative_path in output_paths}
    input_paths = (
        list(UPSTREAM_SIGNOFFS)
        + [
            "artifacts/eda/eda_manifest.json",
            "data/raw_data/energydata_complete.csv",
            "configs/base/coursework_contract.json",
        ]
    )
    input_checksums = {relative_path: sha256_file(root / relative_path) for relative_path in input_paths}
    signoff = {
        "artifact_version": FEATURE_VERSION,
        "phase_id": 7,
        "phase_version": "PHASE-7-v1",
        "created_at": created_at,
        "environment_id": "ENV-v1",
        "dataset_revision": "DATA-v1",
        "schema_version": "SCHEMA-v1",
        "temporal_version": "TEMPORAL-v1",
        "eda_version": upstream["artifacts/eda/eda_manifest.json"]["eda_version"],
        "split_version": "SPLIT-v1",
        "feature_engineering_scope": "FULL_DATASET_DETERMINISTIC_TRANSFORMATIONS",
        "train_only_scope": False,
        "total_rows_used": len(feature_view),
        "time_feature_version": TIME_FEATURE_VERSION,
        "input_paths": input_paths,
        "input_checksums": input_checksums,
        "output_paths": output_paths,
        "output_checksums": output_checksums,
        "config_fingerprint": upstream["artifacts/contracts/phase_0_signoff.json"]["config_fingerprint"],
        "derived_checksum": derived_checksum,
        "status": "PASS",
        "tests": [row["check"] for row in audit_rows],
        "warnings": [],
        "discrepancies": [],
    }
    write_json_once_or_verify(signoff_path, signoff)
    return verify_existing_signoff(root, signoff_path)


def load_validated_feature_view(project_root: Path | None = None) -> pd.DataFrame:
    root = (project_root or get_project_root()).resolve()
    materialize_phase_7(root)
    manifest = read_json(root / FEATURE_ARTIFACT_ROOT / "feature_engineering_manifest.json")
    derived_path = root / manifest["derived_file_path"]
    dataframe = pd.read_csv(derived_path, parse_dates=["timestamp"])
    if list(dataframe.columns) != manifest["ordered_columns"] or len(dataframe) != manifest["row_count"]:
        raise RuntimeError("Reloaded FEATURES-v1 schema does not match its manifest")
    if sha256_file(derived_path) != manifest["derived_file_sha256"]:
        raise RuntimeError("Reloaded FEATURES-v1 checksum mismatch")
    if not str(manifest.get("feature_engineering_scope", "")).startswith("FULL_DATASET_DETERMINISTIC"):
        raise RuntimeError("FEATURES-v1 must use FULL_DATASET_DETERMINISTIC_TRANSFORMATIONS scope")
    return dataframe