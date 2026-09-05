"""Phase 22: Learning-Curve Diagnostics.

Materializes learning-curve diagnostics for LSTM B0 and Transformer B0
using the ``LearningCurveDiagnostics`` analyzer from
``course_work.diagnostics.learning_curves``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from course_work.diagnostics.learning_curves import LearningCurveDiagnostics
from course_work.utils.artifacts import (
    get_project_root,
    read_json,
    sha256_file,
    write_json_once_or_verify,
)


PHASE_22_VERSION = "PHASE-22-v1"
PHASE_22_ARTIFACT_ROOT = Path("artifacts/learning_diagnostics")
DIAGNOSTIC_SUMMARY_COLUMNS = [
    "finding_id",
    "model",
    "diagnostic_code",
    "severity",
    "confidence",
    "action_type",
    "title",
    "epoch_range",
    "interpretation",
    "mapped_future_phase",
]


def validate_phase_22_signoff(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    signoff_path = root / PHASE_22_ARTIFACT_ROOT / "phase_22_signoff.json"
    if not signoff_path.is_file():
        return {"valid": False, "issues": [{"path": str(signoff_path.relative_to(root)), "reason": "MISSING"}], "record": None}
    signoff = read_json(signoff_path)
    issues = []
    if signoff.get("status") != "PASS":
        issues.append({"path": str(signoff_path.relative_to(root)), "reason": "SIGNOFF_NOT_PASS"})
    output_paths = signoff.get("output_paths")
    output_checksums = signoff.get("output_checksums")
    if not isinstance(output_paths, list) or not isinstance(output_checksums, dict):
        issues.append({"path": str(signoff_path.relative_to(root)), "reason": "INVALID_OUTPUT_DECLARATION"})
    else:
        for relative_path in output_paths:
            path = root / relative_path
            if not path.is_file():
                issues.append({"path": relative_path, "reason": "MISSING"})
            elif output_checksums.get(relative_path) != sha256_file(path):
                issues.append({"path": relative_path, "reason": "CHECKSUM_MISMATCH"})
    summary_relative = str(PHASE_22_ARTIFACT_ROOT / "learning_diagnostics_summary.csv")
    if not (root / summary_relative).is_file() and {"path": summary_relative, "reason": "MISSING"} not in issues:
        issues.append({"path": summary_relative, "reason": "MISSING"})
    return {"valid": not issues, "issues": issues, "record": signoff}


def materialize_phase_22(project_root: Path | None = None) -> dict[str, Any]:
    """Materialize Phase 22 learning-curve diagnostics."""

    root = (project_root or get_project_root()).resolve()
    artifact_root = root / PHASE_22_ARTIFACT_ROOT
    artifact_root.mkdir(parents=True, exist_ok=True)

    # Run artifacts live under <project_root>/artifacts/runs, not <project_root>/runs,
    # so the diagnostics analyzer must be anchored at <project_root>/artifacts.
    runs_root = root / "artifacts"
    manifest_path = artifact_root / "learning_diagnostics_manifest.json"
    summary_csv_path = artifact_root / "learning_diagnostics_summary.csv"
    signoff_path = artifact_root / "phase_22_signoff.json"

    # Idempotency: if a previous sign-off exists, verify it against the on-disk
    # artifacts and return it unchanged. This matches the pattern used by every
    # other phase and lets the notebook be re-run safely.
    if signoff_path.exists():
        validation = validate_phase_22_signoff(root)
        if validation["valid"]:
            return validation["record"]
        raise RuntimeError(
            f"Existing Phase 22 sign-off is not reusable: {validation['issues']}"
        )

    analyzer = LearningCurveDiagnostics(artifacts_dir=runs_root)
    load_status = analyzer.load_source_data()
    if load_status.get("status") != "READY":
        raise RuntimeError(f"Phase 22 cannot load histories: {load_status}")

    findings = []
    for history, model_id in [
        (analyzer.lstm_history, "lstm_b0"),
        (analyzer.transformer_history, "transformer_b0"),
    ]:
        if history is None:
            continue
        best_epoch, best_rmse = analyzer.recompute_best_epoch(history)
        initial_diag = analyzer.compute_initial_diagnostics(history, model_id)
        tail_diag = analyzer.compute_tail_diagnostics(history, model_id, best_epoch)
        grad_diag = analyzer.compute_gradient_diagnostics(history, model_id)
        findings.extend(
            analyzer.classify_diagnostic_findings(
                history, model_id, best_epoch, initial_diag, tail_diag, grad_diag
            )
        )

    manifest = {
        "artifact_version": "LEARN-DIAG-v1",
        "phase_id": 22,
        "phase_version": PHASE_22_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "load_status": load_status,
        "finding_count": len(findings),
        "models": sorted({f.model for f in findings}),
    }
    write_json_once_or_verify(manifest_path, manifest)

    import csv

    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIAGNOSTIC_SUMMARY_COLUMNS)
        writer.writeheader()
        for finding in findings:
            writer.writerow({
                "finding_id": finding.finding_id,
                "model": finding.model,
                "diagnostic_code": finding.diagnostic_code.value,
                "severity": finding.severity.value,
                "confidence": finding.confidence.value,
                "action_type": finding.action_type.value,
                "title": finding.title,
                "epoch_range": finding.epoch_range,
                "interpretation": finding.interpretation,
                "mapped_future_phase": finding.mapped_future_phase,
            })

    output_checksums = {
        str(PHASE_22_ARTIFACT_ROOT / "learning_diagnostics_manifest.json"): sha256_file(manifest_path),
        str(PHASE_22_ARTIFACT_ROOT / "learning_diagnostics_summary.csv"): sha256_file(summary_csv_path),
    }

    signoff = {
        "artifact_version": "LEARN-DIAG-v1",
        "phase_id": 22,
        "phase_version": PHASE_22_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment_id": "ENV-v1",
        "dataset_revision": None,
        "input_paths": ["artifacts/runs"],
        "input_checksums": {},
        "output_paths": list(output_checksums),
        "output_checksums": output_checksums,
        "config_fingerprint": PHASE_22_VERSION,
        "status": "PASS",
        "tests": ["history_load", "diagnostic_findings_written"],
        "warnings": [],
        "discrepancies": [],
        "summary": {
            "model_count": len(manifest["models"]),
            "finding_count": len(findings),
        },
    }
    if signoff_path.exists():
        existing = read_json(signoff_path)
        if existing.get("status") == "PASS" and existing.get("output_checksums") == output_checksums:
            return existing
        raise RuntimeError("Existing Phase 22 sign-off does not match current artifacts")
    write_json_once_or_verify(signoff_path, signoff)
    return signoff


def recover_phase_22(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    validation = validate_phase_22_signoff(root)
    if validation["valid"]:
        return validation["record"]
    for relative_path in (
        "artifacts/lstm_baseline/phase_20_signoff.json",
        "artifacts/transformer_b0/phase_21_signoff.json",
    ):
        source_path = root / relative_path
        if not source_path.is_file():
            raise RuntimeError(f"Phase 22 source sign-off is missing: {relative_path}")
        source = read_json(source_path)
        source_issues = []
        if source.get("status") not in {"PASS", "PASS_WITH_WARNING"}:
            source_issues.append({"path": relative_path, "reason": "SIGNOFF_NOT_PASS"})
        for output_path, expected in source.get("output_checksums", {}).items():
            candidate = root / output_path
            if not candidate.is_file():
                source_issues.append({"path": output_path, "reason": "MISSING"})
            elif sha256_file(candidate) != expected:
                source_issues.append({"path": output_path, "reason": "CHECKSUM_MISMATCH"})
        if source_issues:
            raise RuntimeError(f"Phase 22 source evidence is invalid: {source_issues}")
    analyzer = LearningCurveDiagnostics(artifacts_dir=root / "artifacts")
    load_status = analyzer.load_source_data()
    if load_status.get("status") != "READY":
        raise RuntimeError(f"Phase 22 source histories are incomplete: {load_status}")
    artifact_root = root / PHASE_22_ARTIFACT_ROOT
    targets = (
        artifact_root / "learning_diagnostics_manifest.json",
        artifact_root / "learning_diagnostics_summary.csv",
        artifact_root / "phase_22_signoff.json",
    )
    existing = [path for path in targets if path.exists()]
    if existing:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        history_root = artifact_root / "_history" / stamp
        history_root.mkdir(parents=True, exist_ok=False)
        for source in existing:
            source.replace(history_root / source.name)
    result = materialize_phase_22(root)
    verified = validate_phase_22_signoff(root)
    if not verified["valid"]:
        raise RuntimeError(f"Recovered Phase 22 failed verification: {verified['issues']}")
    return result
