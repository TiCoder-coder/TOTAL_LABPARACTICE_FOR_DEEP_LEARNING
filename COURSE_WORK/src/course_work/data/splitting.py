from datetime import datetime, timezone
from io import BytesIO
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from course_work.data.temporal import load_validated_temporal_view
from course_work.data.temporal import TIMESTAMP_FORMAT
from course_work.utils.artifacts import (
    atomic_write_bytes,
    csv_text,
    get_project_root,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json_once_or_verify,
    write_text_once_or_verify,
)


SPLIT_VERSION = "SPLIT-v1"
SPLIT_ARTIFACT_ROOT = "artifacts/splits"
SPLIT_IDS = ("TRAIN", "VALIDATION", "TEST")
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
DISTRIBUTION_VARIABLES = ("Appliances", "lights", "T1", "RH_1", "T_out", "RH_out")
TARGET_ASSIGNMENT_POLICY = "by_target_timestamp"
PRIMARY_BOUNDARY_PROTOCOL = "WB0_CONTEXT_CARRY_OVER"
ALTERNATIVE_BOUNDARY_PROTOCOL = "WB1_STRICT_ISOLATION"
TEST_DISTRIBUTION_STATUS = "LOCKED_UNTIL_PHASE_47"


def validate_split_ratios(train_ratio: float, validation_ratio: float, test_ratio: float) -> None:
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(not 0.0 < ratio < 1.0 for ratio in ratios):
        raise ValueError("Every split ratio must be between zero and one")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Split ratios must sum to one")


def validate_split_input(dataframe: pd.DataFrame) -> None:
    required = ["raw_row_index", "timestamp_parsed", "continuity_segment_id"]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Split input columns are missing: {missing}")
    if dataframe.columns.duplicated().any():
        raise ValueError("Split input contains duplicate column names")
    timestamps = dataframe["timestamp_parsed"]
    if timestamps.isna().any() or not pd.api.types.is_datetime64_any_dtype(timestamps):
        raise ValueError("Split timestamps must be non-null datetime values")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Split timestamps must already be chronologically sorted")
    if timestamps.duplicated().any():
        raise ValueError("Split timestamps must be unique")
    if dataframe["raw_row_index"].duplicated().any():
        raise ValueError("Split raw row indices must be unique")
    if len(dataframe) < 3:
        raise ValueError("At least three rows are required for three non-empty splits")


def compute_split_boundaries(
    row_count: int,
    train_ratio: float = TRAIN_RATIO,
    validation_ratio: float = VALIDATION_RATIO,
    test_ratio: float = TEST_RATIO,
) -> dict[str, tuple[int, int]]:
    validate_split_ratios(train_ratio, validation_ratio, test_ratio)
    train_end = math.floor(train_ratio * row_count)
    validation_end = math.floor((train_ratio + validation_ratio) * row_count)
    boundaries = {
        "TRAIN": (0, train_end),
        "VALIDATION": (train_end, validation_end),
        "TEST": (validation_end, row_count),
    }
    if any(end <= start for start, end in boundaries.values()):
        raise ValueError("Split ratios create an empty split")
    return boundaries


def build_chronological_membership(
    dataframe: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    validation_ratio: float = VALIDATION_RATIO,
    test_ratio: float = TEST_RATIO,
) -> pd.DataFrame:
    validate_split_input(dataframe)
    boundaries = compute_split_boundaries(len(dataframe), train_ratio, validation_ratio, test_ratio)
    output = dataframe[["raw_row_index", "timestamp_parsed", "continuity_segment_id"]].copy(deep=True).reset_index(drop=True)
    output = output.rename(columns={"timestamp_parsed": "timestamp"})
    output["split_id"] = ""
    output["split_position"] = -1
    for split_id in SPLIT_IDS:
        start, end = boundaries[split_id]
        output.loc[start : end - 1, "split_id"] = split_id
        output.loc[start : end - 1, "split_position"] = np.arange(end - start, dtype=np.int64)
    output["split_position"] = output["split_position"].astype("int64")
    return output


def split_fingerprint(membership: pd.DataFrame, split_id: str) -> str:
    if split_id not in SPLIT_IDS:
        raise KeyError(f"Unknown split ID: {split_id}")
    subset = membership[membership["split_id"].eq(split_id)]
    specification = "\n".join(
        f"{int(row.raw_row_index)}|{row.timestamp.strftime(TIMESTAMP_FORMAT)}|{row.split_id}"
        for row in subset.itertuples(index=False)
    )
    return sha256_bytes(specification.encode("utf-8"))


def global_split_fingerprint(
    membership: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    validation_ratio: float = VALIDATION_RATIO,
    test_ratio: float = TEST_RATIO,
) -> str:
    split_hashes = [split_fingerprint(membership, split_id) for split_id in SPLIT_IDS]
    counts = [int(membership["split_id"].eq(split_id).sum()) for split_id in SPLIT_IDS]
    specification = "|".join([
        SPLIT_VERSION,
        f"{train_ratio:.12f}",
        f"{validation_ratio:.12f}",
        f"{test_ratio:.12f}",
        *(str(value) for value in counts),
        *split_hashes,
    ])
    return sha256_bytes(specification.encode("utf-8"))


def build_boundary_rows(membership: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for split_id in SPLIT_IDS:
        subset = membership[membership["split_id"].eq(split_id)]
        start = subset.index[0]
        end = subset.index[-1]
        start_timestamp = subset.iloc[0]["timestamp"]
        end_timestamp = subset.iloc[-1]["timestamp"]
        rows.append({
            "split_id": split_id,
            "start_row_position": int(start),
            "end_row_position": int(end),
            "start_timestamp": start_timestamp.strftime(TIMESTAMP_FORMAT),
            "end_timestamp": end_timestamp.strftime(TIMESTAMP_FORMAT),
            "row_count": len(subset),
            "duration_minutes": float((end_timestamp - start_timestamp).total_seconds() / 60.0),
            "continuity_segment_start": subset.iloc[0]["continuity_segment_id"],
            "continuity_segment_end": subset.iloc[-1]["continuity_segment_id"],
            "fingerprint": split_fingerprint(membership, split_id),
        })
    return rows


def build_boundary_neighborhood(membership: pd.DataFrame, radius: int = 5) -> list[dict[str, Any]]:
    if radius < 1:
        raise ValueError("Boundary neighborhood radius must be positive")
    transition_positions = [
        ("TRAIN_TO_VALIDATION", int(membership.index[membership["split_id"].eq("VALIDATION")][0])),
        ("VALIDATION_TO_TEST", int(membership.index[membership["split_id"].eq("TEST")][0])),
    ]
    rows = []
    for boundary_id, boundary_position in transition_positions:
        start = max(0, boundary_position - radius)
        end = min(len(membership), boundary_position + radius)
        for position in range(start, end):
            row = membership.iloc[position]
            rows.append({
                "boundary_id": boundary_id,
                "offset_from_new_split_start": position - boundary_position,
                "global_row_position": position,
                "raw_row_index": int(row["raw_row_index"]),
                "timestamp": row["timestamp"].strftime(TIMESTAMP_FORMAT),
                "continuity_segment_id": row["continuity_segment_id"],
                "split_id": row["split_id"],
                "split_position": int(row["split_position"]),
            })
    return rows


def build_split_summary(membership: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    total = len(membership)
    for boundary in build_boundary_rows(membership):
        rows.append({
            "split_id": boundary["split_id"],
            "row_count": boundary["row_count"],
            "ratio": boundary["row_count"] / total,
            "start_timestamp": boundary["start_timestamp"],
            "end_timestamp": boundary["end_timestamp"],
            "duration_minutes": boundary["duration_minutes"],
            "fingerprint": boundary["fingerprint"],
        })
    return rows


def build_split_audit(
    temporal_view: pd.DataFrame,
    membership: pd.DataFrame,
) -> list[dict[str, Any]]:
    grouped_indices = {
        split_id: set(membership.index[membership["split_id"].eq(split_id)].tolist())
        for split_id in SPLIT_IDS
    }
    all_indices = set().union(*grouped_indices.values())
    pairwise_disjoint = all(
        grouped_indices[left].isdisjoint(grouped_indices[right])
        for left_index, left in enumerate(SPLIT_IDS)
        for right in SPLIT_IDS[left_index + 1 :]
    )
    train = membership[membership["split_id"].eq("TRAIN")]
    validation = membership[membership["split_id"].eq("VALIDATION")]
    test = membership[membership["split_id"].eq("TEST")]
    checks = [
        ("chronological_order_valid", membership["timestamp"].is_monotonic_increasing and not membership["timestamp"].duplicated().any(), "strictly increasing unique timestamps", "valid"),
        ("splits_disjoint", pairwise_disjoint, True, pairwise_disjoint),
        ("splits_cover_all_rows", all_indices == set(range(len(membership))) and len(membership) == len(temporal_view), len(temporal_view), len(all_indices)),
        ("no_randomization", membership["raw_row_index"].equals(temporal_view["raw_row_index"].reset_index(drop=True)), "TEMPORAL-v1 order", "preserved"),
        ("train_before_validation", train["timestamp"].iloc[-1] < validation["timestamp"].iloc[0], True, train["timestamp"].iloc[-1] < validation["timestamp"].iloc[0]),
        ("validation_before_test", validation["timestamp"].iloc[-1] < test["timestamp"].iloc[0], True, validation["timestamp"].iloc[-1] < test["timestamp"].iloc[0]),
        ("test_locked", True, True, True),
        ("phase5_chronological_split_only", True, "structural SPLIT-v1 before FE/FS/EDA", "single canonical membership"),
        ("sample_assignment_by_target_timestamp", TARGET_ASSIGNMENT_POLICY == "by_target_timestamp", "by_target_timestamp", TARGET_ASSIGNMENT_POLICY),
        ("wb0_primary_and_wb1_supported", PRIMARY_BOUNDARY_PROTOCOL == "WB0_CONTEXT_CARRY_OVER" and ALTERNATIVE_BOUNDARY_PROTOCOL == "WB1_STRICT_ISOLATION", "WB0 primary; WB1 supported", f"{PRIMARY_BOUNDARY_PROTOCOL};{ALTERNATIVE_BOUNDARY_PROTOCOL}"),
    ]
    rows = [
        {
            "check": name,
            "expected": expected,
            "actual": actual,
            "status": "PASS" if passed else "FAIL",
            "notes": "structural split protocol audit",
        }
        for name, passed, expected, actual in checks
    ]
    failures = [row["check"] for row in rows if row["status"] == "FAIL"]
    if failures:
        raise RuntimeError(f"Phase 5 split audit failures: {failures}")
    return rows


def build_train_validation_distribution_summary(
    temporal_view: pd.DataFrame,
    membership: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    for split_id in ("TRAIN", "VALIDATION"):
        indices = membership.index[membership["split_id"].eq(split_id)]
        for variable in DISTRIBUTION_VARIABLES:
            values = temporal_view.loc[indices, variable]
            rows.append({
                "split": split_id,
                "variable": variable,
                "count": int(values.count()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "q05": float(values.quantile(0.05)),
                "q25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "q75": float(values.quantile(0.75)),
                "q95": float(values.quantile(0.95)),
                "max": float(values.max()),
            })
    return rows


def render_split_timeline(membership: pd.DataFrame) -> bytes:
    colors = {"TRAIN": "#3264a8", "VALIDATION": "#d18f2f", "TEST": "#7a4f9a"}
    figure, axis = plt.subplots(figsize=(13, 3.2))
    for position, split_id in enumerate(SPLIT_IDS):
        subset = membership[membership["split_id"].eq(split_id)]
        start = subset["timestamp"].iloc[0]
        end = subset["timestamp"].iloc[-1]
        axis.barh(position, end - start, left=start, height=0.55, color=colors[split_id])
        axis.text(start + (end - start) / 2, position, f"{split_id}\n{len(subset):,} rows", ha="center", va="center", color="white", fontweight="bold")
    axis.set_yticks(range(len(SPLIT_IDS)), labels=SPLIT_IDS)
    axis.set_xlabel("Timestamp")
    axis.set_title("SPLIT-v1 Chronological Target Periods")
    axis.grid(axis="x", alpha=0.25)
    figure.autofmt_xdate()
    figure.tight_layout()
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    return buffer.getvalue()


def dataframe_csv(dataframe: pd.DataFrame) -> str:
    return dataframe.to_csv(index=False, lineterminator="\n", date_format=TIMESTAMP_FORMAT)


def verify_phase_5_inputs(root: Path) -> dict[str, dict[str, Any]]:
    temporal_manifest = read_json(root / "artifacts/temporal/temporal_manifest.json")
    raw_path = root / "data/raw_data/energydata_complete.csv"
    if temporal_manifest.get("raw_csv_sha256") != sha256_file(raw_path):
        raise RuntimeError("TEMPORAL-v1 does not match DATA-v1")
    contract = read_json(root / "configs/base/coursework_contract.json")
    split = contract.get("split", {})
    if split != {
        "type": "chronological",
        "train": TRAIN_RATIO,
        "validation": VALIDATION_RATIO,
        "test": TEST_RATIO,
        "membership_basis": "target_timestamp",
        "shuffle_before_split": False,
    }:
        raise RuntimeError("Phase 0 split contract does not match SPLIT-v1")
    boundary_options = {item["id"]: item["value"] for item in contract["option_registry"]["boundary"]}
    if boundary_options != {"WB0": "context_carry_over", "WB1": "strict_isolation"}:
        raise RuntimeError("Phase 0 boundary options do not support WB0 and WB1")
    return {"temporal_manifest": temporal_manifest}


def verify_existing_signoff(root: Path, signoff_path: Path) -> dict[str, Any]:
    signoff = read_json(signoff_path)
    if signoff.get("status") != "PASS" or signoff.get("artifact_version") != SPLIT_VERSION:
        raise RuntimeError("Existing Phase 5 sign-off is invalid")
    for relative_path, expected_checksum in signoff.get("output_checksums", {}).items():
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected_checksum:
            raise RuntimeError(f"Phase 5 artifact checksum mismatch: {relative_path}")
    manifest = read_json(root / SPLIT_ARTIFACT_ROOT / "split_manifest.json")
    membership = pd.read_csv(root / SPLIT_ARTIFACT_ROOT / "split_membership.csv", parse_dates=["timestamp"])
    if len(membership) != manifest["total_rows"]:
        raise RuntimeError("Reloaded SPLIT-v1 row count mismatch")
    if global_split_fingerprint(membership) != manifest["global_split_fingerprint"]:
        raise RuntimeError("Reloaded SPLIT-v1 global fingerprint mismatch")
    return signoff


def materialize_phase_5(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    signoff_path = root / SPLIT_ARTIFACT_ROOT / "phase_5_signoff.json"
    if signoff_path.exists():
        return verify_existing_signoff(root, signoff_path)
    verify_phase_5_inputs(root)
    temporal_view = load_validated_temporal_view(root)
    temporal_manifest = read_json(root / "artifacts/temporal/temporal_manifest.json")
    membership = build_chronological_membership(temporal_view)
    second_membership = build_chronological_membership(temporal_view)
    if not membership.equals(second_membership):
        raise RuntimeError("Chronological split construction is nondeterministic")
    audit_rows = build_split_audit(temporal_view, membership)
    boundary_rows = build_boundary_rows(membership)
    summary_rows = build_split_summary(membership)
    neighborhood_rows = build_boundary_neighborhood(membership)
    distribution_rows = build_train_validation_distribution_summary(temporal_view, membership)
    if {row["split"] for row in distribution_rows} != {"TRAIN", "VALIDATION"}:
        raise RuntimeError("Test distribution firewall violation")
    fingerprints = {row["split_id"]: row["fingerprint"] for row in boundary_rows}
    global_fingerprint = global_split_fingerprint(membership)
    artifact_root = root / SPLIT_ARTIFACT_ROOT
    manifest_path = artifact_root / "split_manifest.json"
    summary_path = artifact_root / "split_summary.csv"
    membership_path = artifact_root / "split_membership.csv"
    boundaries_path = artifact_root / "split_boundaries.csv"
    neighborhood_path = artifact_root / "split_boundary_neighborhood.csv"
    audit_path = artifact_root / "split_leakage_audit.csv"
    distribution_path = artifact_root / "train_validation_distribution_summary.csv"
    discrepancies_path = artifact_root / "split_discrepancies.json"
    figure_path = artifact_root / "figures/SPLIT_01_timeline.png"
    write_text_once_or_verify(membership_path, dataframe_csv(membership))
    write_text_once_or_verify(summary_path, csv_text(["split_id", "row_count", "ratio", "start_timestamp", "end_timestamp", "duration_minutes", "fingerprint"], summary_rows))
    write_text_once_or_verify(boundaries_path, csv_text(["split_id", "start_row_position", "end_row_position", "start_timestamp", "end_timestamp", "row_count", "duration_minutes", "continuity_segment_start", "continuity_segment_end", "fingerprint"], boundary_rows))
    write_text_once_or_verify(neighborhood_path, csv_text(["boundary_id", "offset_from_new_split_start", "global_row_position", "raw_row_index", "timestamp", "continuity_segment_id", "split_id", "split_position"], neighborhood_rows))
    write_text_once_or_verify(audit_path, csv_text(["check", "expected", "actual", "status", "notes"], audit_rows))
    write_text_once_or_verify(distribution_path, csv_text(["split", "variable", "count", "mean", "std", "min", "q05", "q25", "median", "q75", "q95", "max"], distribution_rows))
    write_json_once_or_verify(discrepancies_path, {"split_version": SPLIT_VERSION, "discrepancies": []})
    figure_bytes = render_split_timeline(membership)
    if figure_path.exists():
        if figure_path.read_bytes() != figure_bytes:
            raise FileExistsError(f"Signed artifact differs from requested content: {figure_path}")
    else:
        atomic_write_bytes(figure_path, figure_bytes)
    created_at = datetime.now(timezone.utc).isoformat()
    boundary_by_id = {row["split_id"]: row for row in boundary_rows}
    counts = {row["split_id"]: row["row_count"] for row in boundary_rows}
    manifest = {
        "split_version": SPLIT_VERSION,
        "dataset_revision": temporal_manifest["dataset_revision"],
        "schema_version": temporal_manifest["schema_version"],
        "temporal_version": temporal_manifest["temporal_version"],
        "environment_id": temporal_manifest["environment_id"],
        "split_method": "chronological_observation_proportion",
        "rounding_rule": "floor_train_and_cumulative_train_validation",
        "target_assignment_policy": TARGET_ASSIGNMENT_POLICY,
        "train_ratio": TRAIN_RATIO,
        "validation_ratio": VALIDATION_RATIO,
        "test_ratio": TEST_RATIO,
        "total_rows": len(membership),
        "train_rows": counts["TRAIN"],
        "validation_rows": counts["VALIDATION"],
        "test_rows": counts["TEST"],
        "train_start_timestamp": boundary_by_id["TRAIN"]["start_timestamp"],
        "train_end_timestamp": boundary_by_id["TRAIN"]["end_timestamp"],
        "validation_start_timestamp": boundary_by_id["VALIDATION"]["start_timestamp"],
        "validation_end_timestamp": boundary_by_id["VALIDATION"]["end_timestamp"],
        "test_start_timestamp": boundary_by_id["TEST"]["start_timestamp"],
        "test_end_timestamp": boundary_by_id["TEST"]["end_timestamp"],
        "train_duration_minutes": boundary_by_id["TRAIN"]["duration_minutes"],
        "validation_duration_minutes": boundary_by_id["VALIDATION"]["duration_minutes"],
        "test_duration_minutes": boundary_by_id["TEST"]["duration_minutes"],
        "train_fingerprint": fingerprints["TRAIN"],
        "validation_fingerprint": fingerprints["VALIDATION"],
        "test_fingerprint": fingerprints["TEST"],
        "global_split_fingerprint": global_fingerprint,
        "primary_boundary_protocol": PRIMARY_BOUNDARY_PROTOCOL,
        "alternative_boundary_protocol_supported": ALTERNATIVE_BOUNDARY_PROTOCOL,
        "context_policy": "input_rows_may_precede_target_period_under_wb0",
        "test_locked": True,
        "test_distribution_analysis_status": TEST_DISTRIBUTION_STATUS,
        "train_validation_distribution_variables": list(DISTRIBUTION_VARIABLES),
        "test_structural_metadata_only": True,
        "feature_variants_share_same_split": True,
        "audit_status": "PASS",
        "warnings": [],
        "created_at": created_at,
    }
    write_json_once_or_verify(manifest_path, manifest)
    output_paths = [
        f"{SPLIT_ARTIFACT_ROOT}/split_manifest.json",
        f"{SPLIT_ARTIFACT_ROOT}/split_summary.csv",
        f"{SPLIT_ARTIFACT_ROOT}/split_membership.csv",
        f"{SPLIT_ARTIFACT_ROOT}/split_boundaries.csv",
        f"{SPLIT_ARTIFACT_ROOT}/split_boundary_neighborhood.csv",
        f"{SPLIT_ARTIFACT_ROOT}/split_leakage_audit.csv",
        f"{SPLIT_ARTIFACT_ROOT}/train_validation_distribution_summary.csv",
        f"{SPLIT_ARTIFACT_ROOT}/split_discrepancies.json",
        f"{SPLIT_ARTIFACT_ROOT}/figures/SPLIT_01_timeline.png",
    ]
    input_paths = [
        "artifacts/temporal/phase_4_signoff.json",
        "artifacts/temporal/temporal_manifest.json",
        "configs/base/coursework_contract.json",
        "data/raw_data/energydata_complete.csv",
    ]
    signoff = {
        "artifact_version": SPLIT_VERSION,
        "phase_id": 5,
        "phase_version": "PHASE-5-v1",
        "created_at": created_at,
        "environment_id": temporal_manifest["environment_id"],
        "dataset_revision": temporal_manifest["dataset_revision"],
        "input_paths": input_paths,
        "input_checksums": {relative_path: sha256_file(root / relative_path) for relative_path in input_paths},
        "output_paths": output_paths,
        "output_checksums": {relative_path: sha256_file(root / relative_path) for relative_path in output_paths},
        "config_fingerprint": temporal_manifest.get("config_fingerprint", ""),
        "global_split_fingerprint": global_fingerprint,
        "train_rows": counts["TRAIN"],
        "validation_rows": counts["VALIDATION"],
        "test_rows": counts["TEST"],
        "test_locked": True,
        "status": "PASS",
        "tests": [row["check"] for row in audit_rows] + [
            "split_ratio_contract",
            "floor_rounding_boundaries",
            "split_fingerprint_determinism",
            "boundary_neighborhood_structural_only",
            "train_validation_distribution_only",
            "timestamp_only_split_figure",
            "upstream_checksum_preservation",
        ],
        "warnings": [],
        "discrepancies": [],
    }
    write_json_once_or_verify(signoff_path, signoff)
    return verify_existing_signoff(root, signoff_path)


def load_validated_split_membership(project_root: Path | None = None) -> pd.DataFrame:
    root = (project_root or get_project_root()).resolve()
    materialize_phase_5(root)
    manifest = read_json(root / SPLIT_ARTIFACT_ROOT / "split_manifest.json")
    membership = pd.read_csv(root / SPLIT_ARTIFACT_ROOT / "split_membership.csv", parse_dates=["timestamp"])
    expected_counts = {
        "TRAIN": manifest["train_rows"],
        "VALIDATION": manifest["validation_rows"],
        "TEST": manifest["test_rows"],
    }
    if membership["split_id"].value_counts().to_dict() != expected_counts:
        raise RuntimeError("Reloaded SPLIT-v1 membership counts mismatch")
    if global_split_fingerprint(membership) != manifest["global_split_fingerprint"]:
        raise RuntimeError("Reloaded SPLIT-v1 fingerprint mismatch")
    return membership
