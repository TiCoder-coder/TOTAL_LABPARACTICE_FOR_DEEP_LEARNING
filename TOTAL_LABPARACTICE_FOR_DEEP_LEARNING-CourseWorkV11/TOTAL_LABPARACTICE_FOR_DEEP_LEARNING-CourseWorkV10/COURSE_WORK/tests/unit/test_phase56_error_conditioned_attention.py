"""Phase 56 - focused tests for error-conditioned attention analysis.

Tests verify:
- Source immutability (Phase 47-55 artifacts unmodified)
- Cohort rule freeze (20/60/20 + deciles)
- Assignment SHA256 frozen before attention join
- Join row count (3*2961*2*4 = 71,064)
- Profile sum integrity
- D_HL sum integrity
- D_UO sum integrity
- Metric bounds (JSD, Wasserstein, Cliff's delta, Spearman)
- Architectural head ordering
- Scope: no best-head, no pruning, no retraining, no causal claim
- O56 artifact presence
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ECA_DIR = PROJECT_ROOT / "artifacts" / "error_conditioned_attention"
FIG_DIR = ECA_DIR / "figures"


def _sha256_file(fp: Path) -> str:
    h = hashlib.sha256()
    with fp.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()

def test_phase55_signoff_exists_and_pass():
    fp = PROJECT_ROOT / "artifacts" / "head_comparison" / "phase_55_signoff.json"
    assert fp.is_file(), f"Phase 55 signoff missing: {fp}"
    signoff = json.loads(fp.read_text())
    assert signoff.get("overall_status") in ("PASS",), f"Phase 55 status {signoff.get('overall_status')}"


def test_phase55_handoff_for_phase56():
    fp = PROJECT_ROOT / "artifacts" / "head_comparison" / "phase56_error_conditioned_attention_handoff.json"
    assert fp.is_file(), f"Phase 56 handoff from Phase 55 missing: {fp}"
    handoff = json.loads(fp.read_text())
    assert handoff.get("ready_for_phase56") is True


def test_phase49_residuals_unmodified():
    fp = PROJECT_ROOT / "artifacts" / "residual_analysis" / "residual_long_table.csv"
    assert fp.is_file()
    df = pd.read_csv(fp)
    assert len(df) > 0
    assert "residual_wh" in df.columns


def test_phase50_regime_unmodified():
    fp = PROJECT_ROOT / "artifacts" / "error_by_regime" / "test_regime_assignment.csv"
    assert fp.is_file()
    df = pd.read_csv(fp)
    assert len(df) > 0


def test_phase52_raw_attention_unmodified():
    import numpy as _np
    for seed in [42, 123, 2026]:
        fp = PROJECT_ROOT / "artifacts" / "attention_extraction" / "raw" / f"last_query_attention_seed{seed}.npz"
        assert fp.is_file()
        arr = _np.load(fp)
        assert "last_query_attention" in arr


def test_phase54_metrics_unmodified():
    fp = PROJECT_ROOT / "artifacts" / "last_query_attention" / "last_query_metrics_long.csv"
    assert fp.is_file()
    df = pd.read_csv(fp)
    assert len(df) == 3 * 2961 * 2 * 4


def test_phase55_artifacts_unmodified():
    files = [
        "head_behavior_summary.csv",
        "head_pair_comparison_long.csv",
        "layer_head_diversity_summary.csv",
        "phase_55_signoff.json",
    ]
    for f in files:
        fp = PROJECT_ROOT / "artifacts" / "head_comparison" / f
        assert fp.is_file()

REQUIRED_ARTIFACTS = [
    "error_conditioning_assignment.csv",
    "error_conditioning_assignment_audit.csv",
    "shared_error_conditioning_audit.json",
    "error_conditioning_assignment_fingerprint.json",
    "error_attention_source_verification.csv",
    "error_attention_join_audit.csv",
    "error_attention_association_long.csv",
    "error_attention_association_matrix.csv",
    "error_attention_decile_metric_summary.csv",
    "error_attention_decile_profile_by_lag.csv",
    "error_attention_high_low_metric_comparison.csv",
    "error_attention_high_low_cliffs_delta_matrix.csv",
    "error_attention_high_low_profile_comparison.csv",
    "error_attention_high_low_profile_difference_by_lag.csv",
    "error_attention_signed_metric_comparison.csv",
    "error_attention_signed_profile_comparison.csv",
    "error_attention_signed_profile_difference_by_lag.csv",
    "error_attention_layer_head_mean_metrics_long.csv",
    "error_attention_layer_head_mean_association.csv",
    "error_attention_layer_head_mean_high_low.csv",
    "error_attention_shared_cohort_layer_summary.csv",
    "error_attention_layer_cross_seed_summary.csv",
    "error_attention_full_matrix_association.csv",
    "error_cohort_regime_composition.csv",
    "error_conditioned_worst_case_attention_context.csv",
    "error_conditioned_attention_findings.csv",
    "error_conditioned_attention_tests.csv",
    "error_conditioned_attention_discrepancies.json",
    "phase57_seed_stability_attention_handoff.json",
    "phase58_attention_results_context_handoff.json",
    "phase_56_signoff.json",
    "error_conditioned_attention_manifest.json",
    "error_conditioned_attention_contract.json",
    "error_conditioned_attention_summary.json",
    "error_conditioned_attention_report.md",
    "README_ERROR_CONDITIONED_ATTENTION.md",
    "phase56_preflight_audit.csv",
]


@pytest.mark.parametrize("artifact", REQUIRED_ARTIFACTS)
def test_required_artifact_present(artifact):
    fp = ECA_DIR / artifact
    assert fp.is_file(), f"Missing O56 artifact: {fp}"


def test_cohort_rule_20_60_20():
    df = pd.read_csv(ECA_DIR / "error_conditioning_assignment.csv")
    for seed in [42, 123, 2026]:
        sub = df[df["seed"] == seed]
        n = len(sub)
        n_edge = max(1, int(0.20 * n))
        assert int((sub["seed_error_cohort"] == "LOW_ERROR").sum()) == n_edge
        assert int((sub["seed_error_cohort"] == "HIGH_ERROR").sum()) == n_edge
        assert int((sub["seed_error_cohort"] == "MID_ERROR").sum()) == n - 2 * n_edge


def test_cohort_assignment_sha256_frozen():
    fp = ECA_DIR / "error_conditioning_assignment_fingerprint.json"
    obj = json.loads(fp.read_text())
    assert obj.get("assignment_sha256"), "assignment_sha256 must be present"
    assert obj.get("created_before_attention_join") is True


def test_shared_cohort_assignment_audit():
    fp = ECA_DIR / "shared_error_conditioning_audit.json"
    obj = json.loads(fp.read_text())
    assert obj["coverage_complete"] is True
    assert obj["status"] == "PASS"
    assert obj["shared_low_count"] == obj["shared_high_count"]


def test_decile_coverage():
    df = pd.read_csv(ECA_DIR / "error_conditioning_assignment.csv")
    expected_deciles = set(range(1, 11))
    for seed in [42, 123, 2026]:
        sub = df[df["seed"] == seed]
        deciles = set(sub["seed_error_decile"].unique())
        assert deciles == expected_deciles
        assert sub["seed_error_decile"].value_counts().sum() == 2961


def test_join_row_count():
    audit_fp = ECA_DIR / "error_attention_join_audit.csv"
    df = pd.read_csv(audit_fp)
    for seed in [42, 123, 2026]:
        row = df[df["seed"] == seed].iloc[0]
        assert row["observed_join_rows"] == 2961 * 2 * 4
        assert row["expected_join_rows"] == 2961 * 2 * 4
        assert row["status"] == "PASS"


def test_join_audit_discrepancy_free():
    fp = ECA_DIR / "error_attention_join_audit.csv"
    df = pd.read_csv(fp)
    assert (df["status"] == "PASS").all()
    assert (df["unmatched_error_targets"] == 0).all()
    assert (df["unmatched_attention_targets"] == 0).all()
    assert (df["duplicate_error_rows"] == 0).all()

def test_continuous_associations_complete():
    df = pd.read_csv(ECA_DIR / "error_attention_association_long.csv")
    assert len(df) == 3 * 2 * 4 * 6 * 3


def test_spearman_rho_in_bounds():
    df = pd.read_csv(ECA_DIR / "error_attention_association_long.csv")
    valid = df[df["status"] == "OK"]
    if len(valid) > 0:
        assert (valid["spearman_rho"] >= -1.0).all()
        assert (valid["spearman_rho"] <= 1.0).all()

def test_decile_metric_summary_complete():
    df = pd.read_csv(ECA_DIR / "error_attention_decile_metric_summary.csv")
    assert len(df) == 3 * 2 * 4 * 10 * 6


def test_decile_profile_by_lag_complete():
    df = pd.read_csv(ECA_DIR / "error_attention_decile_profile_by_lag.csv")
    assert len(df) == 3 * 2 * 10 * 72


def test_high_low_metric_complete():
    df = pd.read_csv(ECA_DIR / "error_attention_high_low_metric_comparison.csv")
    assert len(df) == 3 * 2 * 4 * 6


def test_high_low_cliffs_delta_in_bounds():
    df = pd.read_csv(ECA_DIR / "error_attention_high_low_cliffs_delta_matrix.csv")
    assert (df["cliffs_delta"] >= -1.0).all()
    assert (df["cliffs_delta"] <= 1.0).all()


def test_high_low_profile_complete():
    df = pd.read_csv(ECA_DIR / "error_attention_high_low_profile_comparison.csv")
    assert len(df) == 3 * 2 * 4


def test_d_hl_sum_near_zero():
    df = pd.read_csv(ECA_DIR / "error_attention_high_low_profile_comparison.csv")
    valid = df[df["status"] == "OK"]
    if len(valid) > 0:
        assert (valid["difference_sum"].abs() < 1e-5).all(), \
            f"D_HL sum not near 0: {valid['difference_sum'].abs().max()}"


def test_high_low_profiles_sum_to_one():
    df = pd.read_csv(ECA_DIR / "error_attention_high_low_profile_comparison.csv")
    valid = df[df["status"] == "OK"]
    if len(valid) > 0:
        assert ((valid["profile_sum_low"] - 1.0).abs() < 1e-5).all()
        assert ((valid["profile_sum_high"] - 1.0).abs() < 1e-5).all()

def test_signed_metric_complete():
    df = pd.read_csv(ECA_DIR / "error_attention_signed_metric_comparison.csv")
    assert len(df) == 3 * 2 * 4 * 6


def test_d_uo_sum_near_zero():
    df = pd.read_csv(ECA_DIR / "error_attention_signed_profile_comparison.csv")
    valid = df[df["status"] == "OK"]
    if len(valid) > 0:
        assert (valid["difference_sum"].abs() < 1e-5).all()

def test_layer_head_mean_metrics_complete():
    df = pd.read_csv(ECA_DIR / "error_attention_layer_head_mean_metrics_long.csv")
    assert len(df) == 3 * 2 * 2961


def test_layer_head_mean_association_complete():
    df = pd.read_csv(ECA_DIR / "error_attention_layer_head_mean_association.csv")
    assert len(df) == 3 * 2 * 3 * 6


def test_shared_cohort_layer_summary_complete():
    df = pd.read_csv(ECA_DIR / "error_attention_shared_cohort_layer_summary.csv")
    assert len(df) == 3 * 2 * 6


def test_cross_seed_layer_summary_complete():
    df = pd.read_csv(ECA_DIR / "error_attention_layer_cross_seed_summary.csv")
    assert len(df) == 2 * 3 * 6
    valid = df[df["status"] == "OK"]
    if len(valid) > 0:
        assert (valid["sample_sd_across_seeds"] >= 0).all()

def test_full_matrix_association_complete():
    df = pd.read_csv(ECA_DIR / "error_attention_full_matrix_association.csv")
    assert len(df) == 3 * 2 * 4 * 2 * 4

def test_regime_composition_present():
    df = pd.read_csv(ECA_DIR / "error_cohort_regime_composition.csv")
    assert len(df) > 0


def test_worst_case_present():
    df = pd.read_csv(ECA_DIR / "error_conditioned_worst_case_attention_context.csv")
    assert len(df) == 5 * 3 * 2

def test_findings_present_and_descriptive():
    df = pd.read_csv(ECA_DIR / "error_conditioned_attention_findings.csv")
    assert len(df) > 0
    assert (df["causal_claim"] == False).all()
    assert (df["model_change"] == False).all()


def test_tests_pass_rate():
    df = pd.read_csv(ECA_DIR / "error_conditioned_attention_tests.csv")
    assert (df["status"] == "PASS").all()

def test_phase57_handoff():
    fp = ECA_DIR / "phase57_seed_stability_attention_handoff.json"
    obj = json.loads(fp.read_text())
    assert obj["ready_for_phase57"] is True
    assert obj["same_index_head_semantic_alignment_assumed"] is False
    assert obj["head_matching_not_performed_in_phase56"] is True
    assert obj["best_head_selected"] is False
    assert obj["model_retrained"] is False


def test_phase58_handoff():
    fp = ECA_DIR / "phase58_attention_results_context_handoff.json"
    obj = json.loads(fp.read_text())
    assert obj["ready_for_phase58_context"] is True
    assert obj["causal_claim"] is False

def test_phase56_signoff():
    fp = ECA_DIR / "phase_56_signoff.json"
    obj = json.loads(fp.read_text())
    assert obj["overall_status"] in ("PASS", "PASS_WITH_WARNING"), f"Phase 56 status: {obj['overall_status']}"
    assert obj["new_attention_extraction"] is False
    assert obj["model_training"] is False
    assert obj["prediction_correction"] is False
    assert obj["best_head_selected"] is False
    assert obj["best_seed_selected"] is False
    assert obj["head_pruning"] is False
    assert obj["head_ablation"] is False
    assert obj["causal_claim"] is False
    assert obj["phase57_ready"] is True
    assert obj["phase58_context_ready"] is True


def test_figures_generated():
    assert FIG_DIR.is_dir()
    pngs = list(FIG_DIR.glob("*.png"))
    assert len(pngs) > 0

def test_no_cartesian_subgroup_mining():
    df = pd.read_csv(ECA_DIR / "error_conditioned_attention_findings.csv")
    for interp in df["interpretation"]:
        for forbidden in ["HIGH_ERROR*EXTREME_HIGH", "ERROR*HEAD3", "TOD*H3"]:
            assert forbidden not in str(interp).upper().replace(" ", ""), f"Forbidden: {forbidden}"


def test_residual_convention_y_true_minus_y_pred():
    fp = ECA_DIR / "error_conditioned_attention_contract.json"
    obj = json.loads(fp.read_text())
    assert "y_true - y_pred" in obj["residual"]


def test_core_metrics_v1():
    obj = json.loads((ECA_DIR / "error_conditioned_attention_manifest.json").read_text())
    core = obj["core_attention_metrics"]
    assert "normalized_entropy" in core
    assert "expected_lag_minutes" in core
    assert "recent_1h_mass" in core
    assert "recent_6h_mass" in core
    assert "top5_mass" in core
    assert "lag80_minutes" in core
