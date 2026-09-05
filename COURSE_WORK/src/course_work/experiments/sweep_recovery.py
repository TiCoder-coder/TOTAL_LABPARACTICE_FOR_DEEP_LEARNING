from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

from course_work.experiments.phase_execution import inspect_phase_state
from course_work.utils.artifacts import get_project_root, read_json, sha256_file
from course_work.utils.environment import environment_inventory


SCIENTIFICALLY_REUSABLE_STATES = {"VALID_REUSABLE", "LOG_MISSING", "LOG_STALE"}


def _load_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "MISSING"
    try:
        value = read_json(path)
    except (OSError, ValueError, TypeError):
        return None, "INVALID_JSON"
    if not isinstance(value, dict):
        return None, "INVALID_OBJECT"
    return value, None


def _validate_outputs(root: Path, record: dict[str, Any]) -> list[dict[str, str]]:
    paths = record.get("output_paths")
    checksums = record.get("output_checksums")
    if not isinstance(paths, list) or not isinstance(checksums, dict):
        return [{"path": "output_paths", "reason": "INVALID_DECLARATION"}]
    issues = []
    for relative_path in paths:
        if not isinstance(relative_path, str):
            issues.append({"path": str(relative_path), "reason": "INVALID_PATH"})
            continue
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            issues.append({"path": relative_path, "reason": "PATH_OUTSIDE_PROJECT"})
            continue
        if not candidate.is_file():
            issues.append({"path": relative_path, "reason": "MISSING"})
            continue
        expected = checksums.get(relative_path)
        if isinstance(expected, str) and sha256_file(candidate) != expected:
            issues.append({"path": relative_path, "reason": "CHECKSUM_MISMATCH"})
    return issues


def inspect_environment_recovery(project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    report_path = root / "artifacts/environment/environment_report.json"
    signoff_path = root / "artifacts/environment/phase_1_signoff.json"
    report, report_error = _load_object(report_path)
    signoff, signoff_error = _load_object(signoff_path)
    current = environment_inventory(root)
    comparisons = []
    fields = (
        ("python_version", platform.python_version()),
        ("python_executable", str(Path(sys.executable).resolve())),
        ("project_root", str(root)),
    )
    for field, current_value in fields:
        signed_value = report.get(field) if report is not None else None
        comparisons.append(
            {
                "field": field,
                "signed": signed_value,
                "current": current_value,
                "matches": signed_value == current_value,
            }
        )
    signed_executable = report.get("python_executable") if report is not None else None
    signed_executable_exists = isinstance(signed_executable, str) and Path(signed_executable).is_file()
    signoff_issues = []
    if signoff_error is not None:
        signoff_issues.append({"path": "artifacts/environment/phase_1_signoff.json", "reason": signoff_error})
    elif signoff.get("status") != "PASS":
        signoff_issues.append({"path": "artifacts/environment/phase_1_signoff.json", "reason": "SIGNOFF_NOT_PASS"})
    else:
        signoff_issues.extend(_validate_outputs(root, signoff))
    if report_error is not None:
        signoff_issues.append({"path": "artifacts/environment/environment_report.json", "reason": report_error})
    accelerator_available = bool(current.get("cuda_available") or current.get("mps_available"))
    kernel_matches = bool(current.get("kernel", {}).get("matches_interpreter"))
    identity_matches = all(item["matches"] for item in comparisons)
    ready = not signoff_issues and signed_executable_exists and identity_matches and accelerator_available and kernel_matches
    reasons = []
    reasons.extend(f"{item['path']}: {item['reason']}" for item in signoff_issues)
    if not signed_executable_exists:
        reasons.append("SIGNED_PYTHON_EXECUTABLE_MISSING")
    reasons.extend(f"{item['field'].upper()}_MISMATCH" for item in comparisons if not item["matches"])
    if not kernel_matches:
        reasons.append("KERNEL_INTERPRETER_MISMATCH")
    if not accelerator_available:
        reasons.append("CUDA_OR_MPS_REQUIRED")
    return {
        "ready_for_training": ready,
        "signed_evidence_preserved": True,
        "signed_artifacts_valid": not signoff_issues,
        "signed_executable_exists": signed_executable_exists,
        "identity_comparisons": comparisons,
        "current_runtime": {
            "python_version": current.get("python_version"),
            "python_executable": current.get("python_executable"),
            "project_root": current.get("project_root"),
            "kernel_name": current.get("kernel", {}).get("kernel_name"),
            "kernel_executable": current.get("kernel", {}).get("kernel_executable"),
            "kernel_matches_interpreter": kernel_matches,
            "selected_device": current.get("selected_device"),
            "cuda_available": current.get("cuda_available"),
            "mps_available": current.get("mps_available"),
        },
        "reasons": list(dict.fromkeys(reasons)),
    }


def inspect_phase_22_recovery(project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    signoff_path = root / "artifacts/learning_diagnostics/phase_22_signoff.json"
    summary_path = root / "artifacts/learning_diagnostics/learning_diagnostics_summary.csv"
    signoff, error = _load_object(signoff_path)
    baseline_sources = []
    issues = []
    for phase_id, relative_path in (
        (20, "artifacts/lstm_baseline/phase_20_signoff.json"),
        (21, "artifacts/transformer_b0/phase_21_signoff.json"),
    ):
        source, source_error = _load_object(root / relative_path)
        source_issues = []
        if source_error is not None:
            source_issues.append({"path": relative_path, "reason": source_error})
        else:
            if source.get("status") not in {"PASS", "PASS_WITH_WARNING"}:
                source_issues.append({"path": relative_path, "reason": "SIGNOFF_NOT_PASS"})
            source_issues.extend(_validate_outputs(root, source))
        baseline_sources.append(
            {
                "phase_id": phase_id,
                "state": "VALID_REUSABLE" if not source_issues else "CANONICAL_EVIDENCE_INVALID",
                "scientifically_reusable": not source_issues,
                "issues": source_issues,
                "required_action": "REUSE" if not source_issues else "REGENERATE_BASELINE_EVIDENCE",
            }
        )
        issues.extend(source_issues)
    if error is not None:
        issues.append({"path": str(signoff_path.relative_to(root)), "reason": error})
    else:
        if signoff.get("status") not in {"PASS", "PASS_WITH_WARNING"}:
            issues.append({"path": str(signoff_path.relative_to(root)), "reason": "SIGNOFF_NOT_PASS"})
        issues.extend(_validate_outputs(root, signoff))
    if not summary_path.is_file():
        issues.append({"path": str(summary_path.relative_to(root)), "reason": "MISSING"})
    return {
        "phase_id": 22,
        "state": "VALID_REUSABLE" if not issues else "CANONICAL_EVIDENCE_INVALID",
        "scientifically_reusable": not issues,
        "issues": issues,
        "required_action": "REUSE" if not issues else "REPAIR_PHASE_22" if all(item["scientifically_reusable"] for item in baseline_sources) else "REGENERATE_BASELINE_EVIDENCE",
        "upstream_baselines": baseline_sources,
    }


def inspect_recovery_chain(target_phase: int = 30, project_root: Path | None = None) -> dict[str, Any]:
    if target_phase < 23 or target_phase > 34:
        raise ValueError("Recovery target must be between Phase 23 and Phase 34")
    root = Path(project_root or get_project_root()).resolve()
    environment = inspect_environment_recovery(root)
    phase_22 = inspect_phase_22_recovery(root)
    phase_records = [*phase_22["upstream_baselines"], {key: value for key, value in phase_22.items() if key != "upstream_baselines"}]
    for phase_id in range(23, target_phase + 1):
        inspection = inspect_phase_state(phase_id, root)
        condition_state = inspection["conditions"]
        invalid_ids = list(
            dict.fromkeys(
                item["condition_id"]
                for item in condition_state["invalid_conditions"] + condition_state["failed_conditions"]
                if isinstance(item.get("condition_id"), str)
            )
        )
        reusable = inspection["state"] in SCIENTIFICALLY_REUSABLE_STATES
        if reusable and inspection["state"] == "VALID_REUSABLE":
            required_action = "REUSE"
        elif reusable:
            required_action = "REFRESH_PRESENTATION"
        elif inspection["state"] in {"CONDITION_INCOMPLETE", "FAILED"}:
            required_action = "EXECUTE_UNRESOLVED_CONDITIONS"
        elif inspection["state"] in {"DERIVED_ARTIFACT_MISSING", "SIGNOFF_INVALID"} and condition_state["complete"]:
            required_action = "FINALIZE_CANONICAL_ARTIFACTS"
        else:
            required_action = "WAIT_FOR_UPSTREAM_RECOVERY"
        phase_records.append(
            {
                "phase_id": phase_id,
                "state": inspection["state"],
                "scientifically_reusable": reusable,
                "verified_conditions": [item["condition_id"] for item in condition_state["verified_conditions"]],
                "missing_conditions": condition_state["missing_conditions"],
                "invalid_conditions": invalid_ids,
                "required_action": required_action,
            }
        )
    earliest_invalid = next((item["phase_id"] for item in phase_records if not item["scientifically_reusable"]), None)
    required_phases = list(range(earliest_invalid, target_phase + 1)) if earliest_invalid is not None else []
    direct_target_allowed = earliest_invalid in {None, target_phase}
    return {
        "target_phase": target_phase,
        "audit_only": True,
        "scientific_artifacts_written": False,
        "environment": environment,
        "earliest_invalid_phase": earliest_invalid,
        "required_phases": required_phases,
        "direct_target_allowed": direct_target_allowed,
        "execution_ready": environment["ready_for_training"] and earliest_invalid is None,
        "phases": phase_records,
    }
