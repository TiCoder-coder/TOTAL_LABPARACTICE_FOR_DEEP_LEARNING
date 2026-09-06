"""Phase 50-G focused tests — finalization: figures, findings, discrepancies, report, handoff, signoff."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

import pytest

from course_work.analysis.error_regime_analysis import contract, sources
from course_work.analysis.error_regime_analysis.materialize_g import main as g_main


def test_g_runs_clean():
    r = g_main()
    assert r["status"] == "PASS"


def test_figures_present():
    fig_dir = Path("artifacts/error_by_regime/figures")
    files = sorted(p.name for p in fig_dir.iterdir() if p.suffix == ".png")
    expected = {
        "fig_phase50_test_regime_prevalence.png",
        "fig_phase50_target_level_mae.png",
        "fig_phase50_target_level_rmse.png",
        "fig_phase50_extreme_high_rmse.png",
        "fig_phase50_change_magnitude_rmse.png",
        "fig_phase50_change_direction_rmse.png",
        "fig_phase50_time_of_day_mae.png",
        "fig_phase50_day_type_mae.png",
        "fig_phase50_sse_contribution_r1.png",
        "fig_phase50_rmse_lift_r1.png",
        "fig_phase50_cross_seed_rmse_stability.png",
        "fig_phase50_persistence_vs_transformer_mae.png",
        "fig_phase50_seed_spread_r1.png",
        "fig_phase50_sign_consensus_r1.png",
    }
    missing = expected - set(files)
    assert not missing, f"missing figures: {missing}"


def test_signoff_pass():
    so = json.loads(Path("artifacts/error_by_regime/phase_50_signoff.json").read_text())
    assert so["status"] == "PASS"
    assert so["ready_for_phase51"] is True
    assert so["ready_for_phase52_plus"] is False


def test_signoff_safety_invariants():
    so = json.loads(Path("artifacts/error_by_regime/phase_50_signoff.json").read_text())
    assert so["training"] is False
    assert so["new_test_inference"] is False
    assert so["checkpoint_loading"] is False
    assert so["optimizer_steps"] == 0
    assert so["scaler_fit"] is False
    assert so["best_seed_selected"] is False
    assert so["ensemble"] is False
    assert so["three_n_iid_interpretation"] is False
    assert so["test_derived_threshold_used"] is False
    assert so["cartesian_regime_mining"] is False
    assert so["prediction_correction"] is False
    assert so["worst_error_ranking_executed"] is False
    assert so["attention_analysis_executed"] is False
    assert so["phase47_modified"] is False
    assert so["phase48_modified"] is False
    assert so["phase49_modified"] is False


def test_signoff_six_regime_families():
    so = json.loads(Path("artifacts/error_by_regime/phase_50_signoff.json").read_text())
    assert len(so["regime_families_implemented"]) == 6


def test_signoff_no_best_seed_text():
    so_text = Path("artifacts/error_by_regime/phase_50_signoff.json").read_text()
    # Must NOT contain phrases asserting best-seed as positive
    assert "best_seed=true" not in so_text
    assert "winner_seed=true" not in so_text


def test_phase51_handoff_present_and_pass():
    h = json.loads(Path("artifacts/error_by_regime/phase51_handoff.json").read_text())
    assert h["status"] == "PASS"
    assert h["phase50_worst_error_ranking_executed"] is False
    assert h["phase51_first_authorized_phase_for_worst_error_ranking"] is True
    assert h["phase50_did_not_modify_phase47_48_49"] is True
    assert h["frozen_test_assignment_sha256"]


def test_phase51_handoff_inherits_frozen_test_assignment():
    fp = json.loads(Path("artifacts/error_by_regime/test_regime_assignment_fingerprint.json").read_text())
    h = json.loads(Path("artifacts/error_by_regime/phase51_handoff.json").read_text())
    assert h["frozen_test_assignment_sha256"] == fp["assignment_sha256"]


def test_findings_present():
    p = Path("artifacts/error_by_regime/phase50_findings.json")
    assert p.exists()
    findings = json.loads(p.read_text())
    assert "by_regime_mean_rmse" in findings
    assert "transformer_global_mae_per_seed" in findings


def test_findings_r1_only_has_three_labels():
    p = Path("artifacts/error_by_regime/phase50_findings.json")
    findings = json.loads(p.read_text())
    r1 = [k for k in findings["by_regime_mean_rmse"] if k.startswith("R1_TARGET_LEVEL/")]
    assert set(r1) == {"R1_TARGET_LEVEL/TL_LOW", "R1_TARGET_LEVEL/TL_MID", "R1_TARGET_LEVEL/TL_HIGH"}


def test_findings_descriptive_not_causal():
    """No causal language in findings JSON."""
    text = Path("artifacts/error_by_regime/phase50_findings.json").read_text()
    for forbidden in ["because the Transformer fails", "causes", "optimize"]:
        assert forbidden not in text, f"causal phrasing detected: {forbidden}"


def test_discrepancies_present():
    p = Path("artifacts/error_by_regime/phase50_discrepancies.json")
    assert p.exists()
    d = json.loads(p.read_text())
    assert "discrepancies" in d
    assert len(d["discrepancies"]) >= 1


def test_discrepancies_have_required_fields():
    p = Path("artifacts/error_by_regime/phase50_discrepancies.json")
    d = json.loads(p.read_text())
    for entry in d["discrepancies"]:
        for f in ["id", "severity", "description", "evidence", "impact", "resolution", "remaining_status"]:
            assert f in entry, f"missing {f} in {entry.get('id')}"


def test_report_present():
    p = Path("artifacts/error_by_regime/phase50_report.md")
    assert p.exists()
    text = p.read_text()
    assert "Q25_y = 50.0" in text
    assert "Q75_y = 100.0" in text
    assert "Q90_y = 210.0" in text


def test_readme_present():
    p = Path("artifacts/error_by_regime/README.md")
    assert p.exists()
    text = p.read_text()
    assert "Phase 50" in text


def test_no_unresolved_critical_discrepancies():
    p = Path("artifacts/error_by_regime/phase50_discrepancies.json")
    d = json.loads(p.read_text())
    for entry in d["discrepancies"]:
        if entry["severity"] == "CRITICAL":
            assert entry["remaining_status"] != "OPEN", f"unresolved critical: {entry['id']}"


def test_phase47_unchanged_post_50g():
    import hashlib
    expected = {
        "artifacts/final_test/phase_47_signoff.json": "80614523b6091e21",
        "artifacts/final_test/predictions/final_test_predictions_seed42.csv": "246ee0d725af972b",
        "artifacts/final_test/predictions/final_test_predictions_persistence.csv": "7115af1c479b8957",
    }
    for p, exp in expected.items():
        sha = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        assert sha.startswith(exp)


def test_phase48_unchanged_post_50g():
    import hashlib
    sha = hashlib.sha256(Path("artifacts/prediction_analysis/phase_48_signoff.json").read_bytes()).hexdigest()
    assert sha.startswith("e8c102d582a35dd2")


def test_phase49_unchanged_post_50g():
    import hashlib
    expected = {
        "artifacts/residual_analysis/phase_49_signoff.json": "9d5fc659717dbc15",
        "artifacts/residual_analysis/phase50_handoff.json": "ebfe2cdfbdb8523f",
        "artifacts/residual_analysis/residual_long_table.csv": "8418a99110bfda70",
        "artifacts/residual_analysis/residual_wide_table.csv": "931ff9109aef236d",
        "artifacts/residual_analysis/phase49_cross_seed_sign_consensus.csv": "41e862ff18693f8c",
    }
    for p, exp in expected.items():
        sha = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        assert sha.startswith(exp)
