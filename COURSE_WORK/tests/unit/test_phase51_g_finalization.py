"""Phase 51-G finalization focused tests.

Validates that the G-phase orchestrator produces all required artifacts
with the expected content invariants.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2] / "COURSE_WORK"
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path("/Users/vientu/Deep Learning/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK")


PHASE51_DIR = PROJECT_ROOT / "artifacts" / "worst_error_analysis"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load_csv(p: Path):
    with p.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_g01_summary_json_exists():
    fp = PHASE51_DIR / "phase51_summary.json"
    assert fp.exists()
    s = json.loads(fp.read_text())
    assert s["phase"] == 51
    assert s["subphase"] == "51-G"


def test_g02_report_md_exists():
    fp = PHASE51_DIR / "phase51_report.md"
    assert fp.exists()
    content = fp.read_text()
    assert "Phase 51 — Worst-Error Analysis Report" in content


def test_g03_readme_exists():
    fp = PHASE51_DIR / "README_WORST_ERROR_ANALYSIS.md"
    assert fp.exists()
    content = fp.read_text()
    assert "Phase 51 — Worst-Error Analysis (README)" in content
    assert "phase52_authorized = false" in content


def test_g04_findings_exist():
    fp = PHASE51_DIR / "phase51_findings.json"
    assert fp.exists()
    f = json.loads(fp.read_text())
    assert f["n_findings"] >= 5
    assert f["kind"] == "DESCRIPTIVE_POST_HOC_NON_CAUSAL"


def test_g05_discrepancies_exist():
    fp = PHASE51_DIR / "phase51_discrepancies.json"
    assert fp.exists()
    d = json.loads(fp.read_text())
    assert d["n_total"] >= 5
    assert d["n_critical_or_high_open"] == 0
    assert d["signoff_gate_pass"] is True


def test_g06_o51_inventory_complete():
    fp = PHASE51_DIR / "o51_inventory_summary.json"
    assert fp.exists()
    s = json.loads(fp.read_text())["summary"]
    assert s["n_missing"] == 0
    assert s["overall_status"] == "PASS"
    assert s["n_found"] == s["n_required"]


def test_g07_figure_manifest_exists():
    fp = PHASE51_DIR / "figure_manifest.json"
    assert fp.exists()
    fm = json.loads(fp.read_text())
    assert fm["n_figures"] >= 12
    assert fm["descriptive_only"] is True
    assert fm["no_causal_claims"] is True


def test_g08_figures_files_exist():
    fig_dir = PHASE51_DIR / "figures"
    assert fig_dir.exists()
    pngs = list(fig_dir.glob("*.png"))
    assert len(pngs) >= 12


def test_g09_handoff_exists():
    fp = PHASE51_DIR / "phase52_attention_extraction_handoff.json"
    assert fp.exists()
    h = json.loads(fp.read_text())
    assert h["phase52_authorized"] is False
    assert h["attention_analysis_executed"] is False
    flcl = h["final_candidate_lineage"]
    assert flcl["lookback"] == 72
    assert flcl["feature_count"] == 33
    assert flcl["feature_set"] == "FS2_TF1"
    assert flcl["boundary_protocol"] == "WB0_CONTEXT_CARRY_OVER"
    assert flcl["selection_contract_sha256"] == (
        "ec798326cb03586e85ce7ba09d53be03016a234fe15e1ba5fb4b3fbf0eb967d4"
    )


def test_g10_attention_handoff_cases_csv_exists():
    fp = PHASE51_DIR / "phase51_attention_handoff_cases.csv"
    assert fp.exists()
    rows = _load_csv(fp)
    assert len(rows) == 160
    # all phase52_authorized should be 0
    for r in rows:
        assert r["phase52_authorized"] == "0"


def test_g11_signoff_pass():
    fp = PHASE51_DIR / "phase51_signoff.json"
    assert fp.exists()
    s = json.loads(fp.read_text())
    assert s["phase51_status"] == "PASS"
    assert s["phase52_authorized"] is False
    assert s["ready_for_phase51_h"] is True
    assert s["ready_for_phase52"] is True


def test_g12_no_residual_renormalization():
    # Ensure we have NOT applied any prediction correction.
    s = json.loads((PHASE51_DIR / "phase51_summary.json").read_text())
    safety = s["safety_invariants"]
    assert safety["prediction_correction"] is False
    assert safety["training"] is False
    assert safety["scaler_fit"] is False
    assert safety["checkpoint_loading"] is False


def test_g13_manifest_exists():
    fp = PHASE51_DIR / "phase51_g_manifest.json"
    assert fp.exists()
    m = json.loads(fp.read_text())
    assert m["phase52_authorized"] is False
    assert m["attention_analysis_executed"] is False
    assert m["figures_created"] is True
    assert m["findings_created"] is True
    assert m["phase52_handoff_created"] is True
    assert m["final_summary_created"] is True
    assert m["final_report_created"] is True
    assert m["readme_created"] is True
    assert m["phase51_signoff_created"] is True


def test_g14_artifact_visualization_paths_descriptive_only():
    """Each figure must have descriptive_only=True."""
    fp = PHASE51_DIR / "figure_manifest.json"
    fm = json.loads(fp.read_text())
    for fig in fm["figures"]:
        assert fig["descriptive_only"] is True


def test_g15_tests_summary_exists():
    fp = PHASE51_DIR / "phase51_tests_summary.json"
    assert fp.exists()
    fp2 = PHASE51_DIR / "phase51_tests_summary.csv"
    assert fp2.exists()


def test_g16_casebook_counts():
    summary = json.loads((PHASE51_DIR / "phase51_summary.json").read_text())
    assert summary["n_casebook_memberships"] == 160
    assert summary["n_casebook_unique_targets"] == 44


def test_g17_exact_input_counts():
    summary = json.loads((PHASE51_DIR / "phase51_summary.json").read_text())
    assert summary["n_exact_input_reconstructions"] == 44
    assert summary["exact_input_verified_count"] == 44


def test_g18_lstm_status_correct():
    summary = json.loads((PHASE51_DIR / "phase51_summary.json").read_text())
    assert "NOT_ELIGIBLE" in summary["lstm_status"]


def test_g19_phase50_assignment_sha_correct():
    summary = json.loads((PHASE51_DIR / "phase51_summary.json").read_text())
    assert summary["phase50_regime_assignment_sha256"] == (
        "e90553cfc747a3f15e0e9ec9e6868ae497e7ade797dc14a81999e416b74219ac"
    )


def test_g20_population_fingerprint_correct():
    summary = json.loads((PHASE51_DIR / "phase51_summary.json").read_text())
    assert summary["candidate_lineage"]["test_population_fingerprint_sha256"] == (
        "d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87"
    )


def test_g21_signoff_all_gates_pass():
    fp = PHASE51_DIR / "phase51_signoff.json"
    s = json.loads(fp.read_text())
    for g in s["gates"]:
        assert g["status"] == "PASS", f"Gate {g['id']} failed: {g['name']}: {g['details']}"


def test_g22_required_figure_files_present():
    """Check that figures exist for each manifest."""
    fp = PHASE51_DIR / "figure_manifest.json"
    fm = json.loads(fp.read_text())
    for fig in fm["figures"]:
        path = PHASE51_DIR / "figures" / fig["filename"]
        assert path.exists(), f"missing figure file {fig['filename']}"


def test_g23_processing_log_will_be_updated_in_phase_g_separately():
    # Sanity: just ensure no validation errors here
    pass
