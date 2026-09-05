import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from course_work.data.acquisition import VARIABLE_METADATA, materialize_phase_2
from course_work.utils.artifacts import (
    canonical_json_bytes,
    csv_text,
    get_project_root,
    read_json,
    sha256_file,
    write_json_once_or_verify,
    write_text_once_or_verify,
)


EXPECTED_COLUMNS = [item["name"] for item in VARIABLE_METADATA]
EXPECTED_COLUMN_SET = set(EXPECTED_COLUMNS)
RANDOM_CONTROLS = ["rv1", "rv2"]
INDOOR_TEMPERATURE = ["T1", "T2", "T3", "T4", "T5", "T7", "T8", "T9"]
INDOOR_HUMIDITY = ["RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_7", "RH_8", "RH_9"]
LOCAL_OUTDOOR = ["T6", "RH_6"]
WEATHER_STATION = ["T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"]


def expected_schema() -> dict[str, dict[str, Any]]:
    metadata = {item["name"]: item for item in VARIABLE_METADATA}
    schema: dict[str, dict[str, Any]] = {}
    for column in EXPECTED_COLUMNS:
        if column == "date":
            role = "time"
            group = "time"
            semantic_type = "datetime_like_string"
            alias = None
        elif column == "Appliances":
            role = "target"
            group = "target"
            semantic_type = "continuous_regression_target"
            alias = None
        elif column in RANDOM_CONTROLS:
            role = "random_control"
            group = "random_controls"
            semantic_type = "continuous"
            alias = None
        elif column == "lights":
            role = "feature"
            group = "lighting"
            semantic_type = "numeric"
            alias = None
        elif column in INDOOR_TEMPERATURE:
            role = "feature"
            group = "indoor_temperature"
            semantic_type = "continuous"
            alias = None
        elif column in INDOOR_HUMIDITY:
            role = "feature"
            group = "indoor_humidity"
            semantic_type = "continuous"
            alias = None
        elif column in LOCAL_OUTDOOR:
            role = "feature"
            group = "local_outdoor_sensor"
            semantic_type = "continuous"
            alias = None
        else:
            role = "feature"
            group = "weather_station"
            semantic_type = "continuous"
            alias = {"T_out": "To", "Press_mm_hg": "Pressure"}.get(column)
        schema[column] = {
            "role": role,
            "feature_group": group,
            "expected_semantic_type": semantic_type,
            "unit": metadata[column]["units"] or None,
            "description": metadata[column]["description"],
            "documentation_alias": alias,
        }
    return schema


def load_raw_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def serialized_header(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream), None)
    if not header:
        raise ValueError("Raw CSV has no serialized header")
    return header


def dataframe_fingerprint(dataframe: pd.DataFrame) -> str:
    row_hashes = pd.util.hash_pandas_object(dataframe, index=True).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256()
    digest.update(row_hashes.tobytes())
    digest.update("\x1f".join(str(column) for column in dataframe.columns).encode("utf-8"))
    digest.update("\x1f".join(str(dtype) for dtype in dataframe.dtypes).encode("utf-8"))
    return digest.hexdigest()


def audit_raw_dataframe(dataframe: pd.DataFrame, raw_header: list[str]) -> dict[str, Any]:
    schema = expected_schema()
    columns = list(dataframe.columns)
    missing = [column for column in EXPECTED_COLUMNS if column not in columns]
    unexpected = [column for column in columns if column not in EXPECTED_COLUMN_SET]
    duplicate_names = sorted({name for name in raw_header if raw_header.count(name) > 1})
    whitespace_columns = [column for column in raw_header if column != column.strip()]
    null_counts = dataframe.isna().sum().astype(int).to_dict()
    unique_counts = dataframe.nunique(dropna=False).astype(int).to_dict()
    all_null = [column for column in columns if null_counts[column] == len(dataframe)]
    constant = [column for column in columns if unique_counts[column] == 1]
    numeric_columns = [column for column in EXPECTED_COLUMNS if column != "date" and column in columns]
    numeric_coercion_failures: dict[str, int] = {}
    non_finite_columns: dict[str, int] = {}
    for column in numeric_columns:
        probe = pd.to_numeric(dataframe[column], errors="coerce")
        introduced = int(probe.isna().sum() - dataframe[column].isna().sum())
        if introduced > 0:
            numeric_coercion_failures[column] = introduced
        finite_failures = int((~np.isfinite(probe.dropna().to_numpy(dtype=float))).sum())
        if finite_failures > 0:
            non_finite_columns[column] = finite_failures
    if "date" in columns:
        date_probe = pd.to_datetime(dataframe["date"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        timestamp_failures = int(date_probe.isna().sum() - dataframe["date"].isna().sum())
        first_raw_date = None if dataframe.empty else str(dataframe["date"].iloc[0])
        last_raw_date = None if dataframe.empty else str(dataframe["date"].iloc[-1])
    else:
        timestamp_failures = len(dataframe)
        first_raw_date = None
        last_raw_date = None
    role_mapping = {column: schema[column]["role"] for column in EXPECTED_COLUMNS if column in columns}
    group_mapping = {column: schema[column]["feature_group"] for column in EXPECTED_COLUMNS if column in columns}
    unit_mapping = {column: schema[column]["unit"] for column in EXPECTED_COLUMNS if column in columns}
    regular_features = [column for column in columns if role_mapping.get(column) == "feature"]
    random_controls = [column for column in columns if role_mapping.get(column) == "random_control"]
    critical_failures = []
    if len(dataframe) != 19735:
        critical_failures.append("row_count")
    if missing:
        critical_failures.append("missing_columns")
    if unexpected:
        critical_failures.append("unexpected_columns")
    if duplicate_names:
        critical_failures.append("duplicate_column_names")
    if whitespace_columns:
        critical_failures.append("column_whitespace")
    if columns.count("Appliances") != 1 or columns.count("date") != 1:
        critical_failures.append("target_or_timestamp_cardinality")
    if "Appliances" in all_null or "date" in all_null:
        critical_failures.append("target_or_timestamp_all_null")
    if "Appliances" in constant:
        critical_failures.append("target_constant")
    if numeric_coercion_failures:
        critical_failures.append("numeric_coercion")
    if non_finite_columns:
        critical_failures.append("non_finite")
    if timestamp_failures:
        critical_failures.append("timestamp_parse")
    schema_descriptor = [
        {"column_name": column, "raw_dtype": str(dataframe[column].dtype)}
        for column in columns
    ]
    schema_fingerprint = hashlib.sha256(canonical_json_bytes(schema_descriptor)).hexdigest()
    return {
        "row_count": len(dataframe),
        "column_count": len(columns),
        "ordered_columns": columns,
        "dtype_mapping": {column: str(dataframe[column].dtype) for column in columns},
        "role_mapping": role_mapping,
        "feature_group_mapping": group_mapping,
        "unit_mapping": unit_mapping,
        "target_column": "Appliances" if columns.count("Appliances") == 1 else None,
        "timestamp_column": "date" if columns.count("date") == 1 else None,
        "regular_feature_columns": regular_features,
        "random_control_columns": random_controls,
        "missing_expected_columns": missing,
        "unexpected_columns": unexpected,
        "duplicate_column_names": duplicate_names,
        "whitespace_columns": whitespace_columns,
        "column_order_matches_expected": columns == EXPECTED_COLUMNS,
        "null_counts": null_counts,
        "all_null_columns": all_null,
        "constant_columns": constant,
        "numeric_coercion_failures": numeric_coercion_failures,
        "non_finite_columns": non_finite_columns,
        "timestamp_parse_failure_count": timestamp_failures,
        "first_raw_date": first_raw_date,
        "last_raw_date": last_raw_date,
        "memory_usage_bytes": int(dataframe.memory_usage(deep=True).sum()),
        "schema_fingerprint": schema_fingerprint,
        "critical_failures": critical_failures,
    }


def variable_dictionary_rows(dataframe: pd.DataFrame, audit: dict[str, Any]) -> list[dict[str, Any]]:
    schema = expected_schema()
    rows: list[dict[str, Any]] = []
    for position, column in enumerate(dataframe.columns):
        reference = schema.get(column, {})
        numeric_ok = column == "date" or column not in audit["numeric_coercion_failures"]
        status = "PASS" if column in schema and numeric_ok else "FAIL"
        rows.append({
            "position": position,
            "column_name": column,
            "raw_dtype": str(dataframe[column].dtype),
            "expected_semantic_type": reference.get("expected_semantic_type"),
            "role": reference.get("role", "unexpected"),
            "feature_group": reference.get("feature_group", "unexpected"),
            "unit": reference.get("unit"),
            "description": reference.get("description"),
            "documentation_alias": reference.get("documentation_alias"),
            "null_count": audit["null_counts"][column],
            "unique_count": audit["null_counts"][column] if column not in audit["null_counts"] else int(dataframe[column].nunique(dropna=False)),
            "constant_flag": column in audit["constant_columns"],
            "numeric_coercion_ok": numeric_ok,
            "status": status,
            "notes": "random control retained for FS2 ablation" if column in RANDOM_CONTROLS else "",
        })
    return rows


def schema_comparison_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    checks = [
        ("row_count", 19735, audit["row_count"], audit["row_count"] == 19735),
        ("raw_column_count", 29, audit["column_count"], audit["column_count"] == 29),
        ("target", "Appliances exactly once", audit["target_column"], audit["target_column"] == "Appliances"),
        ("timestamp", "date exactly once", audit["timestamp_column"], audit["timestamp_column"] == "date"),
        ("missing_expected_columns", 0, len(audit["missing_expected_columns"]), not audit["missing_expected_columns"]),
        ("unexpected_columns", 0, len(audit["unexpected_columns"]), not audit["unexpected_columns"]),
        ("duplicate_column_names", 0, len(audit["duplicate_column_names"]), not audit["duplicate_column_names"]),
        ("all_null_columns", 0, len(audit["all_null_columns"]), not audit["all_null_columns"]),
        ("numeric_coercion_failures", 0, sum(audit["numeric_coercion_failures"].values()), not audit["numeric_coercion_failures"]),
        ("non_finite_values", 0, sum(audit["non_finite_columns"].values()), not audit["non_finite_columns"]),
        ("timestamp_parse_failures", 0, audit["timestamp_parse_failure_count"], audit["timestamp_parse_failure_count"] == 0),
        ("column_order", "canonical", "canonical" if audit["column_order_matches_expected"] else "different", audit["column_order_matches_expected"]),
    ]
    return [
        {"check": name, "expected": expected, "actual": actual, "status": "PASS" if passed else "FAIL"}
        for name, expected, actual, passed in checks
    ]


def verify_existing_signoff(root: Path, signoff_path: Path) -> dict[str, Any]:
    signoff = read_json(signoff_path)
    if signoff.get("status") not in {"PASS", "PASS_WITH_WARNING"} or signoff.get("artifact_version") != "SCHEMA-v1":
        raise RuntimeError("Existing Phase 3 sign-off is invalid")
    for relative_path, expected_hash in signoff.get("output_checksums", {}).items():
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Phase 3 artifact checksum mismatch: {relative_path}")
    return signoff


def materialize_phase_3(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    phase_2 = materialize_phase_2(root)
    if phase_2.get("status") != "PASS":
        raise RuntimeError("Phase 2 sign-off is not PASS")
    raw_path = root / "data/raw_data/energydata_complete.csv"
    manifest = read_json(root / "data/raw_data/dataset_manifest.json")
    raw_hash_before = sha256_file(raw_path)
    if raw_hash_before != manifest.get("csv_sha256"):
        raise RuntimeError("DATA-v1 checksum mismatch")
    artifact_root = root / "artifacts/schema"
    signoff_path = artifact_root / "phase_3_signoff.json"
    if signoff_path.exists():
        return verify_existing_signoff(root, signoff_path)
    dataframe = load_raw_csv(raw_path)
    dataframe_before = dataframe_fingerprint(dataframe)
    audit = audit_raw_dataframe(dataframe, serialized_header(raw_path))
    if audit["critical_failures"]:
        raise RuntimeError(f"Critical schema failures: {audit['critical_failures']}")
    dictionary_rows = variable_dictionary_rows(dataframe, audit)
    comparison_rows = schema_comparison_rows(audit)
    dataframe_after = dataframe_fingerprint(dataframe)
    if dataframe_after != dataframe_before:
        raise RuntimeError("Raw DataFrame mutated during schema audit")
    if sha256_file(raw_path) != raw_hash_before:
        raise RuntimeError("Raw CSV changed during schema audit")
    discrepancies = [
        {
            "id": "SD-001",
            "severity": "WARNING",
            "field": "feature_count_display",
            "expected": "UCI detailed page reports 28 features",
            "actual": "Raw CSV has 29 total columns: 1 time field, 1 target, 25 regular features and 2 random controls",
            "source_reference": "UCI dataset 374 detailed page and raw DATA-v1",
            "interpretation": "UCI feature count includes the date field and all non-target columns; raw total also includes the target",
            "action": "Preserve all raw columns and use explicit semantic roles",
            "resolved": True,
        }
    ]
    warnings = ["SD-001"]
    audit_status = "PASS_WITH_WARNING"
    summary_rows = [
        {"metric": "row_count", "value": audit["row_count"]},
        {"metric": "column_count", "value": audit["column_count"]},
        {"metric": "time_columns", "value": 1},
        {"metric": "target_columns", "value": 1},
        {"metric": "regular_feature_columns", "value": len(audit["regular_feature_columns"])},
        {"metric": "random_control_columns", "value": len(audit["random_control_columns"])},
        {"metric": "memory_usage_bytes", "value": audit["memory_usage_bytes"]},
    ]
    manifest_output = {
        "schema_version": "SCHEMA-v1",
        "dataset_revision": "DATA-v1",
        "environment_id": "ENV-v1",
        "raw_csv_sha256": raw_hash_before,
        "row_count": audit["row_count"],
        "column_count": audit["column_count"],
        "ordered_columns": audit["ordered_columns"],
        "target_column": audit["target_column"],
        "timestamp_column": audit["timestamp_column"],
        "regular_feature_columns": audit["regular_feature_columns"],
        "random_control_columns": audit["random_control_columns"],
        "dtype_mapping": audit["dtype_mapping"],
        "role_mapping": audit["role_mapping"],
        "feature_group_mapping": audit["feature_group_mapping"],
        "unit_mapping": audit["unit_mapping"],
        "missing_expected_columns": audit["missing_expected_columns"],
        "unexpected_columns": audit["unexpected_columns"],
        "duplicate_column_names": audit["duplicate_column_names"],
        "whitespace_columns": audit["whitespace_columns"],
        "column_order_matches_expected": audit["column_order_matches_expected"],
        "all_null_columns": audit["all_null_columns"],
        "constant_columns": audit["constant_columns"],
        "non_finite_columns": audit["non_finite_columns"],
        "numeric_coercion_failures": audit["numeric_coercion_failures"],
        "timestamp_parse_failure_count": audit["timestamp_parse_failure_count"],
        "first_raw_date": audit["first_raw_date"],
        "last_raw_date": audit["last_raw_date"],
        "memory_usage_bytes": audit["memory_usage_bytes"],
        "schema_fingerprint": audit["schema_fingerprint"],
        "dataframe_fingerprint_before": dataframe_before,
        "dataframe_fingerprint_after": dataframe_after,
        "audit_status": audit_status,
        "warnings": warnings,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = artifact_root / "schema_summary.csv"
    dictionary_path = artifact_root / "variable_dictionary.csv"
    comparison_path = artifact_root / "schema_comparison.csv"
    manifest_path = artifact_root / "schema_manifest.json"
    fingerprint_path = artifact_root / "schema_fingerprint.txt"
    discrepancies_path = artifact_root / "schema_discrepancies.json"
    write_text_once_or_verify(summary_path, csv_text(["metric", "value"], summary_rows))
    dictionary_fields = [
        "position", "column_name", "raw_dtype", "expected_semantic_type", "role", "feature_group",
        "unit", "description", "documentation_alias", "null_count", "unique_count", "constant_flag",
        "numeric_coercion_ok", "status", "notes",
    ]
    write_text_once_or_verify(dictionary_path, csv_text(dictionary_fields, dictionary_rows))
    write_text_once_or_verify(comparison_path, csv_text(["check", "expected", "actual", "status"], comparison_rows))
    write_json_once_or_verify(manifest_path, manifest_output)
    write_text_once_or_verify(fingerprint_path, f"{audit['schema_fingerprint']}\n")
    write_json_once_or_verify(discrepancies_path, {"schema_version": "SCHEMA-v1", "discrepancies": discrepancies})
    output_paths = [
        "artifacts/schema/schema_summary.csv",
        "artifacts/schema/variable_dictionary.csv",
        "artifacts/schema/schema_comparison.csv",
        "artifacts/schema/schema_manifest.json",
        "artifacts/schema/schema_fingerprint.txt",
        "artifacts/schema/schema_discrepancies.json",
    ]
    output_checksums = {relative: sha256_file(root / relative) for relative in output_paths}
    signoff = {
        "artifact_version": "SCHEMA-v1",
        "phase_id": 3,
        "phase_version": "PHASE-3-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment_id": "ENV-v1",
        "dataset_revision": "DATA-v1",
        "input_paths": ["artifacts/acquisition/phase_2_signoff.json", "data/raw_data/energydata_complete.csv"],
        "input_checksums": {
            "artifacts/acquisition/phase_2_signoff.json": sha256_file(root / "artifacts/acquisition/phase_2_signoff.json"),
            "data/raw_data/energydata_complete.csv": raw_hash_before,
        },
        "output_paths": output_paths,
        "output_checksums": output_checksums,
        "config_fingerprint": phase_2["config_fingerprint"],
        "status": audit_status,
        "tests": [
            "data_v1_checksum",
            "raw_shape_and_order",
            "target_and_timestamp_cardinality",
            "raw_dtype_audit",
            "semantic_role_mapping",
            "null_constant_and_non_finite_audit",
            "numeric_coercion",
            "timestamp_parse_probe",
            "raw_dataframe_immutability",
        ],
        "warnings": warnings,
        "discrepancies": ["SD-001"],
    }
    write_json_once_or_verify(signoff_path, signoff)
    return signoff
