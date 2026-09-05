"""Preflight audit + shared-data parity checks."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from course_work.lstm_tuning.shared_data_contract import (
    SharedDataContract,
    assert_required_fields,
    assert_test_locked,
)
from course_work.utils.artifacts import canonical_json_bytes, sha256_bytes, write_text_once_or_verify


def _csv_lines(rows: list[list[Any]]) -> str:
    lines = []
    for row in rows:
        cells = ["" if cell is None else str(cell) for cell in row]
        lines.append(",".join(cells))
    return "\n".join(lines) + "\n"


def build_preflight_audit(
    contract: SharedDataContract,
    handoff_lookback: int,
    handoff_feature_variant: str,
    handoff_target_scaling: str,
    handoff_boundary_protocol: str,
    phase42_signoff_status: str,
    test_locked_handoff: bool,
) -> list[list[Any]]:
    assert_required_fields(contract)
    assert_test_locked(contract)

    rows: list[list[Any]] = [
        ["phase42_signoff_status", phase42_signoff_status, "PASS" if phase42_signoff_status in {"PASS", "PASS_WITH_WARNING"} else "FAIL"],
        ["handoff_lookback", handoff_lookback, "PASS" if handoff_lookback == contract.lookback_steps else "FAIL"],
        ["handoff_feature_variant", handoff_feature_variant, "PASS" if handoff_feature_variant == contract.feature_variant_id else "FAIL"],
        ["handoff_target_scaling", handoff_target_scaling, "PASS" if handoff_target_scaling == contract.target_scaling_id else "FAIL"],
        ["handoff_boundary_protocol", handoff_boundary_protocol, "PASS" if handoff_boundary_protocol == contract.boundary_protocol else "FAIL"],
        ["shared_lookback", contract.lookback_steps, "PASS" if contract.lookback_steps == handoff_lookback else "FAIL"],
        ["shared_feature_variant", contract.feature_variant_id, "PASS"],
        ["shared_target_scaling", contract.target_scaling_id, "PASS"],
        ["shared_boundary_protocol", contract.boundary_protocol, "PASS"],
        ["window_population_version", contract.window_population_version, "PASS"],
        ["train_target_ids_fingerprint_present", "yes" if contract.train_target_ids_fingerprint else "no", "PASS" if contract.train_target_ids_fingerprint else "FAIL"],
        ["validation_target_ids_fingerprint_present", "yes" if contract.validation_target_ids_fingerprint else "no", "PASS" if contract.validation_target_ids_fingerprint else "FAIL"],
        ["x_scaler_checksum_present", "yes" if contract.x_scaler_checksum else "no", "PASS" if contract.x_scaler_checksum else "FAIL"],
        ["y_scaler_checksum_present", "yes" if contract.target_scaler_checksum else "no", "PASS" if contract.target_scaler_checksum else "FAIL"],
        ["test_locked_handoff", test_locked_handoff, "PASS" if test_locked_handoff else "FAIL"],
        ["test_locked_contract", contract.test_locked, "PASS" if contract.test_locked else "FAIL"],
    ]
    return rows


def build_common_data_audit(
    contract: SharedDataContract,
    candidate_contract: SharedDataContract | None = None,
) -> list[list[Any]]:
    target = candidate_contract or contract
    rows: list[list[Any]] = [
        ["shared", "train_sample_count", contract.train_sample_count, target.train_sample_count, "PASS" if contract.train_sample_count == target.train_sample_count else "FAIL"],
        ["shared", "validation_sample_count", contract.validation_sample_count, target.validation_sample_count, "PASS" if contract.validation_sample_count == target.validation_sample_count else "FAIL"],
        ["shared", "feature_variant_id", contract.feature_variant_id, target.feature_variant_id, "PASS" if contract.feature_variant_id == target.feature_variant_id else "FAIL"],
        ["shared", "feature_fingerprint", contract.feature_fingerprint, target.feature_fingerprint, "PASS" if contract.feature_fingerprint == target.feature_fingerprint else "FAIL"],
        ["shared", "x_scaler_checksum", contract.x_scaler_checksum, target.x_scaler_checksum, "PASS" if contract.x_scaler_checksum == target.x_scaler_checksum else "FAIL"],
        ["shared", "target_scaler_checksum", contract.target_scaler_checksum, target.target_scaler_checksum, "PASS" if contract.target_scaler_checksum == target.target_scaler_checksum else "FAIL"],
        ["shared", "lookback_steps", contract.lookback_steps, target.lookback_steps, "PASS" if contract.lookback_steps == target.lookback_steps else "FAIL"],
        ["shared", "boundary_protocol", contract.boundary_protocol, target.boundary_protocol, "PASS" if contract.boundary_protocol == target.boundary_protocol else "FAIL"],
        ["shared", "population_fingerprint", contract.population_fingerprint, target.population_fingerprint, "PASS" if contract.population_fingerprint == target.population_fingerprint else "FAIL"],
        ["shared", "test_locked", contract.test_locked, target.test_locked, "PASS" if contract.test_locked == target.test_locked else "FAIL"],
    ]
    return rows


def write_preflight_csv(project_root: Path, rows: list[list[Any]]) -> Path:
    path = project_root / "artifacts" / "lstm_tuning" / "phase43_preflight_audit.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "metric,expected,actual,status\n" + _csv_lines(rows)
    write_text_once_or_verify(path, content)
    return path


def write_common_data_audit_csv(
    project_root: Path,
    rows: list[list[Any]],
    filename: str = "lstm_common_data_audit.csv",
) -> Path:
    path = project_root / "artifacts" / "lstm_tuning" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "scope,field,contract_value,candidate_value,status\n" + _csv_lines(rows)
    write_text_once_or_verify(path, content)
    return path


def contract_checksum(contract: SharedDataContract) -> str:
    payload = {
        "feature_variant_id": contract.feature_variant_id,
        "lookback_steps": contract.lookback_steps,
        "boundary_protocol": contract.boundary_protocol,
        "target_scaling_id": contract.target_scaling_id,
        "window_population_version": contract.window_population_version,
        "x_scaler_checksum": contract.x_scaler_checksum,
        "target_scaler_checksum": contract.target_scaler_checksum,
        "feature_fingerprint": contract.feature_fingerprint,
        "population_fingerprint": contract.population_fingerprint,
        "metric_version": contract.metric_version,
        "test_locked": contract.test_locked,
        "batch_size": contract.batch_size,
    }
    return sha256_bytes(canonical_json_bytes(payload))
