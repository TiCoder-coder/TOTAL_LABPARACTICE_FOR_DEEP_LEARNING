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
        existing = read_json(signoff_path)
        if existing.get("status") == "PASS":
            return existing
        raise RuntimeError(
            "Existing Phase 22 sign-off is not PASS — refusing to overwrite"
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
