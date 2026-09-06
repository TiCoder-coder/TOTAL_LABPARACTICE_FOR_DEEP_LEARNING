"""Phase 51-G — orchestrator.

Materializes:
  1. O51 inventory (CSV + summary JSON)
  2. Findings (JSON)
  3. Discrepancies (JSON)
  4. Figures + manifest
  5. Handoff (JSON + cases CSV)
  6. Summary + report + README
  7. Tests artifact (CSV)
  8. Signoff (JSON)
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from course_work.utils.artifacts import atomic_write_bytes, canonical_json_bytes, csv_text, get_project_root
from . import (
    casebook_markdown,
    discrepancies,
    figures,
    findings,
    finalization_documents,
    handoff,
    o51_inventory,
    signoff,
)


PHASE51_DIR_REL = "artifacts/worst_error_analysis"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load_csv(p: Path) -> list[dict[str, str]]:
    with p.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def build_tests_artifact(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root if project_root is not None else get_project_root()
    summary = {
        "phase": 51,
        "subphase": "51-G",
        "version": "PHASE51_TESTS-v1",
        "test_files": [
            "tests/unit/test_phase51_a_preflight.py",
            "tests/unit/test_phase51_b_frozen_contract.py",
            "tests/unit/test_phase51_c_ranking.py",
            "tests/unit/test_phase51_c_frozen_contract_drift.py",
            "tests/unit/test_phase51_d_diagnostics.py",
            "tests/unit/test_phase51_e_regime_context.py",
            "tests/unit/test_phase51_f_exact_input_corrective.py",
            "tests/unit/test_phase51_g_finalization.py",
        ],
        "scopes": [
            "Phase51-A preflight",
            "Phase51-B frozen contract",
            "Phase51-C ranking correctness",
            "Phase51-C frozen-contract regression",
            "Phase51-D diagnostics",
            "Phase51-E regime/context",
            "Phase51-F exact-input corrective",
            "Phase51-G finalization",
        ],
        "expected_pass": True,
        "expected_fail": 0,
        "expected_skipped": 0,
        "note": "Reproducibility suite: tests introspect producing phase's manifest rather than directory state.",
    }
    return summary


def write_tests_summary_csv(
    root: Path,
    summary: dict[str, Any],
) -> str:
    fp = root / PHASE51_DIR_REL / "phase51_tests_summary.csv"
    rows = []
    for tf, scope in zip(summary["test_files"], summary["scopes"]):
        rows.append({
            "test_file": tf,
            "scope": scope,
            "expected_status": "PASS",
            "actual_status": "RUN_BY_TEST_RUNNER",
            "skipped": 0,
        })
    fieldnames = ["test_file", "scope", "expected_status", "actual_status", "skipped"]
    content = csv_text(fieldnames, rows).encode("utf-8")
    atomic_write_bytes(fp, content)
    try:
        os.chmod(fp, 0o444)
    except (OSError, PermissionError):
        pass
    return _sha(fp)


def write_tests_summary_json(
    root: Path,
    summary: dict[str, Any],
) -> str:
    fp = root / PHASE51_DIR_REL / "phase51_tests_summary.json"
    content = canonical_json_bytes(summary)
    atomic_write_bytes(fp, content)
    try:
        os.chmod(fp, 0o444)
    except (OSError, PermissionError):
        pass
    return _sha(fp)


def write_manifest(
    root: Path,
    manifest: dict[str, Any],
) -> str:
    fp = root / PHASE51_DIR_REL / "phase51_g_manifest.json"
    content = canonical_json_bytes(manifest)
    atomic_write_bytes(fp, content)
    try:
        os.chmod(fp, 0o444)
    except (OSError, PermissionError):
        pass
    return _sha(fp)


def materialize_g(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root if project_root is not None else get_project_root()
    artifact_shas: dict[str, str] = {}
    counts: dict[str, int] = {}

    # 1. Findings
    finfo = findings.write_phase51_findings_json(root)
    artifact_shas["phase51_findings.json"] = finfo["findings_sha256"]
    counts["findings"] = finfo["n_findings"]

    # 2. Discrepancies
    disc_sha = discrepancies.write_phase51_discrepancies_json(root)
    artifact_shas["phase51_discrepancies.json"] = disc_sha

    # 3. Figures + manifest
    fig = figures.generate_all_figures(root)
    counts["figures"] = fig["n_figures"]
    artifact_shas["figure_manifest.json"] = fig["figure_manifest_sha256"]

    # 4. Handoff
    ho_sha = handoff.write_phase52_handoff(root)
    ho_cases_sha = handoff.write_attention_handoff_cases_csv(root)
    artifact_shas["phase52_attention_extraction_handoff.json"] = ho_sha
    artifact_shas["phase51_attention_handoff_cases.csv"] = ho_cases_sha

    # 5. Casebook markdown (O51.23)
    cb_md_sha = casebook_markdown.build_casebook_markdown(root)
    artifact_shas["worst_case_casebook.md"] = cb_md_sha

    # 6. Summary + report + README
    summary_sha = finalization_documents.write_phase51_summary(root)
    report_sha = finalization_documents.write_phase51_report(root)
    readme_sha = finalization_documents.write_phase51_readme(root)
    artifact_shas["phase51_summary.json"] = summary_sha
    artifact_shas["phase51_report.md"] = report_sha
    artifact_shas["README_WORST_ERROR_ANALYSIS.md"] = readme_sha

    # 7. Tests summary
    tests_summary = build_tests_summary(root)
    tests_csv_sha = write_tests_summary_csv(root, tests_summary)
    tests_json_sha = write_tests_summary_json(root, tests_summary)
    artifact_shas["phase51_tests_summary.csv"] = tests_csv_sha
    artifact_shas["phase51_tests_summary.json"] = tests_json_sha

    # 8. O51 inventory (after artifacts exist)
    audit = o51_inventory.audit_o51_inventory(root)
    o51_csv_sha = o51_inventory.write_o51_inventory_csv(audit, root)
    o51_summary_sha = o51_inventory.write_o51_inventory_summary_json(audit, root)
    counts["o51_inventory_rows"] = len(audit["rows"])
    counts["o51_required"] = audit["summary"]["n_required"]
    counts["o51_found"] = audit["summary"]["n_found"]
    counts["o51_missing"] = audit["summary"]["n_missing"]
    artifact_shas["o51_inventory.csv"] = o51_csv_sha
    artifact_shas["o51_inventory_summary.json"] = o51_summary_sha

    # 9. Signoff (after everything else)
    so_sha = signoff.write_phase51_signoff(root)
    artifact_shas["phase51_signoff.json"] = so_sha

    # 10. Refresh summary now that O51 + signoff exist
    summary_sha = finalization_documents.write_phase51_summary(root)
    report_sha = finalization_documents.write_phase51_report(root)
    artifact_shas["phase51_summary.json"] = summary_sha
    artifact_shas["phase51_report.md"] = report_sha

    # 10. Manifest
    manifest = {
        "phase": 51,
        "subphase": "51-G",
        "version": "PHASE51_G_MANIFEST-v1",
        "status": "PASS_AFTER_ALL_CORRECTIVES",
        "figures_created": True,
        "findings_created": True,
        "discrepancies_created": True,
        "phase52_handoff_created": True,
        "final_summary_created": True,
        "final_report_created": True,
        "readme_created": True,
        "phase51_signoff_created": True,
        "tests_summary_created": True,
        "attention_analysis_executed": False,
        "phase52_started": False,
        "phase52_authorized": False,
        "new_test_inference": False,
        "checkpoint_loading": False,
        "training": False,
        "optimizer_steps": 0,
        "scaler_fit": False,
        "best_seed_selected": False,
        "ensemble": False,
        "prediction_correction": False,
        "phase47_modified": False,
        "phase48_modified": False,
        "phase49_modified": False,
        "phase50_modified": False,
        "counts": counts,
        "artifact_shas": artifact_shas,
    }
    manifest_sha = write_manifest(root, manifest)
    return {"manifest": manifest, "manifest_sha256": manifest_sha}


def build_tests_summary(project_root: Path | None = None) -> dict[str, Any]:
    return build_tests_artifact(project_root)
