from __future__ import annotations

import json
import math
import platform
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from course_work.utils.artifacts import get_project_root, read_json, sha256_file


class PhaseState(str, Enum):
    VALID_REUSABLE = "VALID_REUSABLE"
    LOG_MISSING = "LOG_MISSING"
    LOG_STALE = "LOG_STALE"
    DERIVED_ARTIFACT_MISSING = "DERIVED_ARTIFACT_MISSING"
    CONDITION_INCOMPLETE = "CONDITION_INCOMPLETE"
    SIGNOFF_INVALID = "SIGNOFF_INVALID"
    UPSTREAM_INVALID = "UPSTREAM_INVALID"
    ENVIRONMENT_INVALID = "ENVIRONMENT_INVALID"
    RUNNING = "RUNNING"
    FAILED = "FAILED"


class PhaseAction(str, Enum):
    RENDER_ONLY = "RENDER_ONLY"
    REBUILD_LOG_ONLY = "REBUILD_LOG_ONLY"
    REBUILD_DERIVED_ONLY = "REBUILD_DERIVED_ONLY"
    EXECUTE_MISSING_ONLY = "EXECUTE_MISSING_ONLY"
    WAIT_FOR_RUNNING_PROCESS = "WAIT_FOR_RUNNING_PROCESS"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class SweepPhaseSpec:
    phase_id: int
    phase_name: str
    family_id: str
    sweep_code: str
    artifact_directory: str
    winner_filename: str
    reference_filename: str
    log_filename: str
    prerequisite_paths: tuple[str, ...]
    condition_path: tuple[str, ...]
    condition_values: tuple[tuple[str, Any], ...]
    reference_condition: str
    manifest_filename: str = "sweep_manifest.json"
    results_filename: str = "results.csv"

    @property
    def artifact_root(self) -> Path:
        return Path("artifacts/sweeps") / self.artifact_directory

    @property
    def manifest_path(self) -> Path:
        return self.artifact_root / self.manifest_filename

    @property
    def results_path(self) -> Path:
        return self.artifact_root / self.results_filename

    @property
    def signoff_path(self) -> Path:
        return self.artifact_root / f"phase_{self.phase_id}_signoff.json"

    @property
    def winner_path(self) -> Path:
        return self.artifact_root / self.winner_filename

    @property
    def reference_path(self) -> Path:
        return self.artifact_root / self.reference_filename

    @property
    def processing_log_path(self) -> Path:
        return Path("docs/save_log_in_processing") / self.log_filename

    @property
    def expected_conditions(self) -> tuple[str, ...]:
        return tuple(condition_id for condition_id, _ in self.condition_values)


SWEEP_PHASE_SPECS: dict[int, SweepPhaseSpec] = {
    23: SweepPhaseSpec(23, "S1 Feature-Set Sweep", "S1_FEATURE_SET", "S1", "s1_feature_set", "s1_feature_set_winner.json", "s1_reference_update.json", "phase_23_s1_feature_set_log.json", ("artifacts/learning_diagnostics/phase_22_signoff.json",), ("data", "feature_variant_id"), (("FS0_TF1", "FS0_TF1"), ("FS1_TF1", "FS1_TF1"), ("FS2_TF1", "FS2_TF1")), "FS1_TF1"),
    24: SweepPhaseSpec(24, "S2 Time-Feature Sweep", "S2_TIME_FEATURES", "S2", "s2_time_feature", "s2_time_feature_winner.json", "s2_reference_update.json", "phase_24_s2_time_feature_log.json", ("artifacts/sweeps/s1_feature_set/phase_23_signoff.json", "artifacts/sweeps/s1_feature_set/s1_feature_set_winner.json", "artifacts/sweeps/s1_feature_set/s1_reference_update.json"), ("data", "feature_variant_id"), (("TF0", "FS1_TF0"), ("TF1", "FS1_TF1")), "TF1"),
    25: SweepPhaseSpec(25, "S3 Target-Scaling Sweep", "S3_TARGET_SCALING", "S3", "s3_target_scaling", "s3_target_scaling_winner.json", "s3_reference_update.json", "phase_25_s3_target_scaling_log.json", ("artifacts/sweeps/s2_time_feature/phase_24_signoff.json", "artifacts/sweeps/s2_time_feature/s2_time_feature_winner.json", "artifacts/sweeps/s2_time_feature/s2_reference_update.json"), ("data", "target_scaling_option"), (("YS0", "YS0"), ("YS1", "YS1")), "YS1"),
    26: SweepPhaseSpec(26, "S4 Lookback Sweep", "S4_LOOKBACK", "S4", "s4_lookback", "s4_lookback_winner.json", "s4_reference_update.json", "phase_26_s4_lookback_log.json", ("artifacts/sweeps/s3_target_scaling/phase_25_signoff.json", "artifacts/sweeps/s3_target_scaling/s3_target_scaling_winner.json", "artifacts/sweeps/s3_target_scaling/s3_reference_update.json"), ("data", "lookback_steps"), (("L36", 36), ("L72", 72), ("L144", 144)), "L144"),
    27: SweepPhaseSpec(27, "S5 Pooling Sweep", "S5_POOLING", "S5", "s5_pooling", "s5_pooling_winner.json", "s5_reference_update.json", "phase_27_s5_pooling_log.json", ("artifacts/sweeps/s4_lookback/phase_26_signoff.json", "artifacts/sweeps/s4_lookback/s4_lookback_winner.json", "artifacts/sweeps/s4_lookback/s4_reference_update.json"), ("model", "pooling"), (("LAST_STEP", "LAST_STEP"), ("MEAN", "MEAN")), "LAST_STEP"),
    28: SweepPhaseSpec(28, "S6 Activation Sweep", "S6_ACTIVATION", "S6", "s6_activation", "s6_activation_winner.json", "s6_reference_update.json", "phase_28_s6_activation_log.json", ("artifacts/sweeps/s5_pooling/phase_27_signoff.json", "artifacts/sweeps/s5_pooling/s5_pooling_winner.json", "artifacts/sweeps/s5_pooling/s5_reference_update.json"), ("model", "activation"), (("RELU", "RELU"), ("GELU", "GELU")), "GELU"),
    29: SweepPhaseSpec(29, "S7 Batch-Size Sweep", "S7_BATCH_SIZE", "S7", "s7_batch_size", "s7_batch_winner.json", "s7_reference_update.json", "phase_29_s7_batch_size_log.json", ("artifacts/sweeps/s6_activation/phase_28_signoff.json", "artifacts/sweeps/s6_activation/s6_activation_winner.json", "artifacts/sweeps/s6_activation/s6_reference_update.json"), ("training", "batch_size"), (("B32", 32), ("B64", 64)), "B64"),
    30: SweepPhaseSpec(30, "S8 Learning-Rate Sweep", "S8_LEARNING_RATE", "S8", "s8_learning_rate", "s8_learning_rate_winner.json", "s8_reference_update.json", "phase_30_s8_learning_rate_log.json", ("artifacts/sweeps/s7_batch_size/phase_29_signoff.json", "artifacts/sweeps/s7_batch_size/s7_batch_winner.json", "artifacts/sweeps/s7_batch_size/s7_reference_update.json"), ("training", "learning_rate"), (("LR1", 0.0001), ("LR2", 0.0003), ("LR3", 0.001)), "LR2"),
    31: SweepPhaseSpec(31, "S9 Weight-Decay Sweep", "S9_WEIGHT_DECAY", "S9", "S9_weight_decay", "s9_weight_decay_winner.json", "s9_reference_update.json", "phase_31_s9_weight_decay_log.json", ("artifacts/sweeps/s8_learning_rate/phase_30_signoff.json", "artifacts/sweeps/s8_learning_rate/s8_learning_rate_winner.json", "artifacts/sweeps/s8_learning_rate/s8_reference_update.json"), ("training", "weight_decay"), (("WD0", 0.0), ("WD1", 0.0001), ("WD2", 0.001)), "WD1", "s9_weight_decay_sweep_manifest.json", "s9_weight_decay_metrics.csv"),
    32: SweepPhaseSpec(32, "S10 Dropout Sweep", "S10_DROPOUT", "S10", "S10_dropout", "s10_dropout_winner.json", "s10_reference_update.json", "phase_32_s10_dropout_log.json", ("artifacts/sweeps/S9_weight_decay/phase_31_signoff.json", "artifacts/sweeps/S9_weight_decay/s9_weight_decay_winner.json", "artifacts/sweeps/S9_weight_decay/s9_reference_update.json"), ("model", "dropout"), (("DR01", 0.1), ("DR02", 0.2), ("DR03", 0.3)), "DR01", "s10_dropout_sweep_manifest.json", "s10_dropout_metrics.csv"),
    33: SweepPhaseSpec(33, "S11 d_model Sweep", "S11_D_MODEL", "S11", "S11_d_model", "s11_d_model_winner.json", "s11_reference_update.json", "phase_33_s11_d_model_log.json", ("artifacts/sweeps/S10_dropout/phase_32_signoff.json", "artifacts/sweeps/S10_dropout/s10_dropout_winner.json", "artifacts/sweeps/S10_dropout/s10_reference_update.json"), ("model", "d_model"), (("D32", 32), ("D64", 64)), "D64", "s11_d_model_sweep_manifest.json", "s11_d_model_metrics.csv"),
    34: SweepPhaseSpec(34, "S12 Head Sweep", "S12_HEADS", "S12", "S12_heads", "s12_head_winner.json", "s12_reference_update.json", "phase_34_s12_head_log.json", ("artifacts/sweeps/S11_d_model/phase_33_signoff.json", "artifacts/sweeps/S11_d_model/s11_d_model_winner.json", "artifacts/sweeps/S11_d_model/s11_reference_update.json"), ("model", "num_heads"), (("H2", 2), ("H4", 4)), "H4", "s12_head_sweep_manifest.json", "s12_head_metrics.csv"),
}


STATE_ACTIONS = {
    PhaseState.VALID_REUSABLE: PhaseAction.RENDER_ONLY,
    PhaseState.LOG_MISSING: PhaseAction.REBUILD_LOG_ONLY,
    PhaseState.LOG_STALE: PhaseAction.REBUILD_LOG_ONLY,
    PhaseState.DERIVED_ARTIFACT_MISSING: PhaseAction.REBUILD_DERIVED_ONLY,
    PhaseState.CONDITION_INCOMPLETE: PhaseAction.EXECUTE_MISSING_ONLY,
    PhaseState.SIGNOFF_INVALID: PhaseAction.BLOCK,
    PhaseState.UPSTREAM_INVALID: PhaseAction.BLOCK,
    PhaseState.ENVIRONMENT_INVALID: PhaseAction.BLOCK,
    PhaseState.RUNNING: PhaseAction.WAIT_FOR_RUNNING_PROCESS,
    PhaseState.FAILED: PhaseAction.EXECUTE_MISSING_ONLY,
}


def resolve_phase_action(state: PhaseState | str) -> PhaseAction:
    return STATE_ACTIONS[PhaseState(state)]


def get_sweep_phase_spec(phase_id: int) -> SweepPhaseSpec:
    if phase_id not in SWEEP_PHASE_SPECS:
        raise ValueError(f"Selective sweep execution supports Phase 23-34, received Phase {phase_id}")
    return SWEEP_PHASE_SPECS[phase_id]


def _project_path(root: Path, relative_path: str | Path) -> Path:
    candidate = (root / relative_path).resolve()
    candidate.relative_to(root)
    return candidate


def _json_status(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "MISSING"
    try:
        value = read_json(path)
    except (OSError, ValueError, TypeError):
        return None, "INVALID_JSON"
    if not isinstance(value, dict):
        return None, "INVALID_OBJECT"
    return value, None


def _validate_declared_artifacts(root: Path, record: dict[str, Any], path_field: str, checksum_field: str) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    paths = record.get(path_field, [])
    checksums = record.get(checksum_field, {})
    if not isinstance(paths, list) or not isinstance(checksums, dict):
        return [{"path": path_field, "reason": "INVALID_DECLARATION"}]
    for relative_path in paths:
        if not isinstance(relative_path, str):
            issues.append({"path": str(relative_path), "reason": "INVALID_PATH"})
            continue
        try:
            path = _project_path(root, relative_path)
        except (ValueError, OSError):
            issues.append({"path": relative_path, "reason": "PATH_OUTSIDE_PROJECT"})
            continue
        if not path.is_file():
            issues.append({"path": relative_path, "reason": "MISSING"})
            continue
        expected = checksums.get(relative_path)
        if not isinstance(expected, str):
            issues.append({"path": relative_path, "reason": "CHECKSUM_MISSING"})
            continue
        if sha256_file(path) != expected:
            issues.append({"path": relative_path, "reason": "CHECKSUM_MISMATCH"})
    return issues


def _inspect_prerequisites(root: Path, spec: SweepPhaseSpec) -> dict[str, Any]:
    records = []
    valid = True
    for relative_path in spec.prerequisite_paths:
        path = _project_path(root, relative_path)
        value, error = _json_status(path)
        item: dict[str, Any] = {"path": relative_path, "status": "PASS" if error is None else error}
        if error is None and relative_path.endswith("signoff.json"):
            status = value.get("status")
            issues = _validate_declared_artifacts(root, value, "output_paths", "output_checksums")
            if status not in {"PASS", "PASS_WITH_WARNING"}:
                item["status"] = "SIGNOFF_NOT_PASS"
            if issues:
                item["status"] = "SIGNOFF_ARTIFACT_INVALID"
                item["issues"] = issues
        if item["status"] != "PASS":
            valid = False
        records.append(item)
    return {"valid": valid, "records": records}


def _inspect_signoff(root: Path, spec: SweepPhaseSpec) -> dict[str, Any]:
    relative_path = str(spec.signoff_path)
    value, error = _json_status(_project_path(root, relative_path))
    if error is not None:
        return {"valid": False, "path": relative_path, "status": error, "issues": []}
    issues = _validate_declared_artifacts(root, value, "output_paths", "output_checksums")
    status_valid = value.get("status") in {"PASS", "PASS_WITH_WARNING"}
    identity_valid = value.get("phase_id") == spec.phase_id and value.get("phase_version") == f"PHASE-{spec.phase_id}-v1"
    if not status_valid:
        issues.append({"path": relative_path, "reason": "SIGNOFF_NOT_PASS"})
    if not identity_valid:
        issues.append({"path": relative_path, "reason": "SIGNOFF_IDENTITY_MISMATCH"})
    return {"valid": not issues, "path": relative_path, "status": value.get("status"), "issues": issues, "record": value}


def _inspect_required_phase_artifacts(root: Path, spec: SweepPhaseSpec) -> dict[str, Any]:
    relative_paths = (spec.manifest_path, spec.results_path, spec.winner_path, spec.reference_path, spec.signoff_path)
    records = []
    for relative_path in relative_paths:
        path = _project_path(root, relative_path)
        records.append({"path": str(relative_path), "status": "PASS" if path.is_file() else "MISSING"})
    return {"valid": all(item["status"] == "PASS" for item in records), "records": records}


def _inspect_processing_log(root: Path, spec: SweepPhaseSpec) -> dict[str, Any]:
    relative_path = str(spec.processing_log_path)
    value, error = _json_status(_project_path(root, relative_path))
    if error is not None:
        return {"valid": False, "path": relative_path, "status": error, "issues": []}
    issues = []
    for source in value.get("source_artifacts", []):
        source_path = source.get("path")
        expected = source.get("sha256")
        if not isinstance(source_path, str) or not isinstance(expected, str):
            issues.append({"path": str(source_path), "reason": "INVALID_SOURCE_DECLARATION"})
            continue
        try:
            path = _project_path(root, source_path)
        except (ValueError, OSError):
            issues.append({"path": source_path, "reason": "PATH_OUTSIDE_PROJECT"})
            continue
        if not path.is_file():
            issues.append({"path": source_path, "reason": "MISSING"})
        elif sha256_file(path) != expected:
            issues.append({"path": source_path, "reason": "CHECKSUM_MISMATCH"})
    identity_valid = value.get("phase_id") == spec.phase_id
    if not identity_valid:
        issues.append({"path": relative_path, "reason": "LOG_IDENTITY_MISMATCH"})
    return {"valid": not issues, "path": relative_path, "status": value.get("status"), "issues": issues, "record": value}


def _nested_value(value: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _condition_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
    return actual == expected


def _load_registry_records(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    path = root / "artifacts/experiments/experiment_registry.jsonl"
    if not path.is_file():
        return [], [{"path": str(path.relative_to(root)), "reason": "MISSING"}]
    records = []
    issues = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            issues.append({"path": str(path.relative_to(root)), "reason": f"INVALID_JSON_LINE_{line_number}"})
            continue
        if isinstance(value, dict):
            records.append(value)
        else:
            issues.append({"path": str(path.relative_to(root)), "reason": f"INVALID_RECORD_LINE_{line_number}"})
    return records, issues


def _validate_run_artifacts(root: Path, record: dict[str, Any]) -> list[dict[str, str]]:
    issues = []
    required_artifacts = [item for item in record.get("artifacts", []) if item.get("required") is True]
    required_types = {item.get("artifact_type") for item in required_artifacts}
    for required_type in ("CONFIG", "STATUS", "METRICS"):
        if required_type not in required_types:
            issues.append({"path": str(record.get("run_id")), "reason": f"REQUIRED_{required_type}_MISSING"})
    for artifact in required_artifacts:
        relative_path = artifact.get("artifact_path")
        expected_checksum = artifact.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected_checksum, str):
            issues.append({"path": str(relative_path), "reason": "INVALID_ARTIFACT_DECLARATION"})
            continue
        try:
            path = _project_path(root, relative_path)
        except (ValueError, OSError):
            issues.append({"path": relative_path, "reason": "PATH_OUTSIDE_PROJECT"})
            continue
        if not path.is_file():
            issues.append({"path": relative_path, "reason": "MISSING"})
        elif sha256_file(path) != expected_checksum:
            issues.append({"path": relative_path, "reason": "CHECKSUM_MISMATCH"})
    return issues


def _validation_metrics(record: dict[str, Any]) -> dict[str, float] | None:
    values = {}
    for metric in record.get("metrics", []):
        if metric.get("split_id") != "VALIDATION" or metric.get("status") != "PASS":
            continue
        name = metric.get("metric_name")
        value = metric.get("metric_value")
        if name in {"mae_wh", "rmse_wh", "r2"} and isinstance(value, (int, float)) and math.isfinite(float(value)):
            values[name] = float(value)
    if set(values) != {"mae_wh", "rmse_wh", "r2"}:
        return None
    return values


def _load_run_config(root: Path, run_id: str) -> tuple[dict[str, Any] | None, str | None]:
    payload, error = _json_status(root / "artifacts/runs" / run_id / "config.json")
    if error is not None:
        return None, error
    if payload.get("run_id") != run_id or not isinstance(payload.get("config"), dict):
        return None, "CONFIG_IDENTITY_MISMATCH"
    return payload, None


def _reference_run_id(root: Path, spec: SweepPhaseSpec) -> str | None:
    if spec.phase_id == 23:
        return None
    for relative_path in reversed(spec.prerequisite_paths):
        if not relative_path.endswith(".json") or relative_path.endswith("signoff.json"):
            continue
        value, error = _json_status(_project_path(root, relative_path))
        if error is not None:
            continue
        for key in ("winner_run_id", "reference_run_id", "current_reference_run_id", "source_winner_run_id"):
            run_id = value.get(key)
            if isinstance(run_id, str) and run_id:
                return run_id
    return None


def resolve_condition_values(phase_id: int, project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    spec = get_sweep_phase_spec(phase_id)
    values = dict(spec.condition_values)
    if phase_id != 24:
        return values
    reference_run_id = _reference_run_id(root, spec)
    if reference_run_id is None:
        return values
    payload, error = _load_run_config(root, reference_run_id)
    if error is not None:
        return values
    reference_variant = _nested_value(payload["config"], ("data", "feature_variant_id"))
    if not isinstance(reference_variant, str) or "_TF" not in reference_variant:
        return values
    feature_set = reference_variant.rsplit("_TF", 1)[0]
    return {"TF0": f"{feature_set}_TF0", "TF1": f"{feature_set}_TF1"}


def resolve_phase_conditions(phase_id: int, project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    spec = get_sweep_phase_spec(phase_id)
    records, registry_issues = _load_registry_records(root)
    reference_run_id = _reference_run_id(root, spec)
    verified = []
    invalid = []
    running = []
    failed = []
    condition_values = resolve_condition_values(phase_id, root)
    for condition_id in spec.expected_conditions:
        expected_value = condition_values[condition_id]
        candidates = []
        condition_invalid = []
        condition_running = []
        condition_failed = []
        for record in records:
            run_id = record.get("run_id")
            if not isinstance(run_id, str):
                continue
            config_payload, config_error = _load_run_config(root, run_id)
            if config_error is not None:
                continue
            actual_value = _nested_value(config_payload["config"], spec.condition_path)
            if not _condition_matches(actual_value, expected_value):
                continue
            is_reference = condition_id == spec.reference_condition
            family_matches = record.get("experiment_family") == spec.family_id
            reference_matches = is_reference and reference_run_id is not None and run_id == reference_run_id
            phase_23_reference = phase_id == 23 and is_reference and record.get("experiment_family") == "TRANSFORMER_BASELINE"
            if not family_matches and not reference_matches and not phase_23_reference:
                continue
            candidates.append((record, config_payload))
        valid_candidates = []
        for record, config_payload in candidates:
            run_id = record["run_id"]
            status = record.get("status")
            evidence_issues = _validate_run_artifacts(root, record)
            metrics = _validation_metrics(record)
            config_fingerprint_matches = config_payload.get("config_fingerprint") == record.get("config_fingerprint")
            if status == "RUNNING":
                condition_running.append({"condition_id": condition_id, "run_id": run_id})
                continue
            if status == "FAILED":
                condition_failed.append({"condition_id": condition_id, "run_id": run_id})
                continue
            if status != "COMPLETED":
                continue
            if evidence_issues:
                condition_invalid.append({"condition_id": condition_id, "run_id": run_id, "reasons": evidence_issues})
                continue
            if metrics is None:
                condition_invalid.append({"condition_id": condition_id, "run_id": run_id, "reasons": [{"path": run_id, "reason": "VALIDATION_METRICS_INCOMPLETE"}]})
                continue
            if not config_fingerprint_matches:
                condition_invalid.append({"condition_id": condition_id, "run_id": run_id, "reasons": [{"path": run_id, "reason": "CONFIG_FINGERPRINT_MISMATCH"}]})
                continue
            valid_candidates.append((record, metrics, expected_value))
        if len(valid_candidates) == 1:
            record, metrics, factor_value = valid_candidates[0]
            verified.append({"condition_id": condition_id, "factor_value": factor_value, "run_id": record["run_id"], "rmse_wh": metrics["rmse_wh"], "mae_wh": metrics["mae_wh"], "r2": metrics["r2"], "reused_reference": condition_id == spec.reference_condition})
        elif len(valid_candidates) > 1:
            invalid.append({"condition_id": condition_id, "run_id": None, "reasons": [{"path": condition_id, "reason": "MULTIPLE_VALID_RUNS"}]})
        else:
            invalid.extend(condition_invalid)
            running.extend(condition_running)
            failed.extend(condition_failed)
    verified_ids = {item["condition_id"] for item in verified}
    missing = [condition_id for condition_id in spec.expected_conditions if condition_id not in verified_ids]
    return {
        "expected_conditions": list(spec.expected_conditions),
        "verified_conditions": verified,
        "missing_conditions": missing,
        "invalid_conditions": invalid,
        "running_conditions": running,
        "failed_conditions": failed,
        "reference_condition": spec.reference_condition,
        "reference_run_id": reference_run_id,
        "registry_issues": registry_issues,
        "complete": not missing and not invalid and not registry_issues,
    }


def inspect_execution_readiness(phase_id: int, project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    spec = get_sweep_phase_spec(phase_id)
    prerequisites = _inspect_prerequisites(root, spec)
    signoff_path = root / "artifacts/environment/phase_1_signoff.json"
    report_path = root / "artifacts/environment/environment_report.json"
    signoff, signoff_error = _json_status(signoff_path)
    report, report_error = _json_status(report_path)
    environment_issues = []
    if signoff_error is not None:
        environment_issues.append({"path": str(signoff_path.relative_to(root)), "reason": signoff_error})
    elif signoff.get("status") != "PASS":
        environment_issues.append({"path": str(signoff_path.relative_to(root)), "reason": "SIGNOFF_NOT_PASS"})
    else:
        environment_issues.extend(_validate_declared_artifacts(root, signoff, "output_paths", "output_checksums"))
    if report_error is not None:
        environment_issues.append({"path": str(report_path.relative_to(root)), "reason": report_error})
    else:
        if report.get("python_version") != platform.python_version():
            environment_issues.append({"path": str(report_path.relative_to(root)), "reason": "PYTHON_VERSION_MISMATCH"})
        signed_executable = report.get("python_executable")
        if not isinstance(signed_executable, str) or Path(signed_executable).resolve() != Path(sys.executable).resolve():
            environment_issues.append({"path": str(report_path.relative_to(root)), "reason": "PYTHON_EXECUTABLE_MISMATCH"})
    if not prerequisites["valid"]:
        state = PhaseState.UPSTREAM_INVALID
    elif environment_issues:
        state = PhaseState.ENVIRONMENT_INVALID
    else:
        state = PhaseState.VALID_REUSABLE
    return {
        "ready": state is PhaseState.VALID_REUSABLE,
        "state": state.value,
        "prerequisites": prerequisites,
        "environment": {
            "valid": not environment_issues,
            "issues": environment_issues,
            "current_python_version": platform.python_version(),
            "current_python_executable": str(Path(sys.executable).resolve()),
        },
    }


def validate_condition_request(phase_id: int, condition_id: str, project_root: Path | None = None) -> dict[str, Any]:
    spec = get_sweep_phase_spec(phase_id)
    if condition_id not in spec.expected_conditions:
        raise ValueError(f"Condition {condition_id} is not registered for Phase {phase_id}")
    readiness = inspect_execution_readiness(phase_id, project_root)
    conditions = resolve_phase_conditions(phase_id, project_root)
    already_verified = condition_id in {item["condition_id"] for item in conditions["verified_conditions"]}
    already_running = condition_id in {item["condition_id"] for item in conditions["running_conditions"]}
    return {
        "phase_id": phase_id,
        "condition_id": condition_id,
        "allowed": readiness["ready"] and not already_verified and not already_running,
        "readiness": readiness,
        "already_verified": already_verified,
        "already_running": already_running,
    }


def _block_reasons(inspection: dict[str, Any], readiness: dict[str, Any]) -> list[str]:
    reasons = []
    for record in inspection["prerequisites"]["records"]:
        if record["status"] == "PASS":
            continue
        issues = record.get("issues", [])
        if issues:
            reasons.extend(f"{issue['path']}: {issue['reason']}" for issue in issues)
        else:
            reasons.append(f"{record['path']}: {record['status']}")
    for issue in inspection["signoff"].get("issues", []):
        reasons.append(f"{issue['path']}: {issue['reason']}")
    for issue in inspection["conditions"].get("registry_issues", []):
        reasons.append(f"{issue['path']}: {issue['reason']}")
    for issue in readiness["environment"].get("issues", []):
        reasons.append(f"{issue['path']}: {issue['reason']}")
    return list(dict.fromkeys(reasons))


def plan_phase_resume(phase_id: int, project_root: Path | None = None, allow_execution: bool = False) -> dict[str, Any]:
    inspection = inspect_phase_state(phase_id, project_root)
    resolved_action = PhaseAction(inspection["action"])
    readiness = inspect_execution_readiness(phase_id, project_root)
    effective_action = resolved_action
    reasons = _block_reasons(inspection, readiness) if resolved_action is PhaseAction.BLOCK else []
    if resolved_action is PhaseAction.EXECUTE_MISSING_ONLY:
        if not readiness["ready"]:
            effective_action = PhaseAction.BLOCK
            reasons.append(readiness["state"])
        elif not allow_execution:
            effective_action = PhaseAction.BLOCK
            reasons.append("SCIENTIFIC_EXECUTION_NOT_AUTHORIZED")
    return {
        "phase_id": phase_id,
        "state": inspection["state"],
        "resolved_action": resolved_action.value,
        "effective_action": effective_action.value,
        "execution_authorized": allow_execution,
        "reasons": reasons,
        "inspection": inspection,
        "readiness": readiness,
    }


def inspect_phase_state(phase_id: int, project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    spec = get_sweep_phase_spec(phase_id)
    prerequisites = _inspect_prerequisites(root, spec)
    signoff = _inspect_signoff(root, spec)
    artifacts = _inspect_required_phase_artifacts(root, spec)
    processing_log = _inspect_processing_log(root, spec)
    conditions = resolve_phase_conditions(phase_id, root)
    if not prerequisites["valid"]:
        state = PhaseState.UPSTREAM_INVALID
    elif conditions["running_conditions"]:
        state = PhaseState.RUNNING
    elif conditions["missing_conditions"] and conditions["failed_conditions"]:
        state = PhaseState.FAILED
    elif conditions["missing_conditions"]:
        state = PhaseState.CONDITION_INCOMPLETE
    elif signoff["status"] != "MISSING" and not signoff["valid"]:
        state = PhaseState.SIGNOFF_INVALID
    elif not signoff["valid"]:
        state = PhaseState.DERIVED_ARTIFACT_MISSING
    elif not artifacts["valid"]:
        state = PhaseState.DERIVED_ARTIFACT_MISSING
    elif processing_log["status"] == "MISSING":
        state = PhaseState.LOG_MISSING
    elif not processing_log["valid"]:
        state = PhaseState.LOG_STALE
    else:
        state = PhaseState.VALID_REUSABLE
    action = resolve_phase_action(state)
    return {
        "phase_id": phase_id,
        "phase_name": spec.phase_name,
        "state": state.value,
        "action": action.value,
        "artifact_root": str(spec.artifact_root),
        "prerequisites": prerequisites,
        "signoff": signoff,
        "artifacts": artifacts,
        "processing_log": processing_log,
        "conditions": conditions,
    }
