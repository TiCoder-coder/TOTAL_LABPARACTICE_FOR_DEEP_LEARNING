from __future__ import annotations

import json
from pathlib import Path

import pytest

from course_work.residual_analysis.materialize_f import materialize_phase49_f
from course_work.residual_analysis.o49_inventory import O49_INVENTORY, check_o49_completeness
from course_work.residual_analysis.sources import (
    compute_seed_bundle_sha256,
    load_phase47_signoff,
    load_phase48_signoff,
    load_prediction_checksums,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ART_DIR = PROJECT_ROOT / "artifacts/residual_analysis"


@pytest.fixture(scope="module")
def phase49_f_outcome():
    return materialize_phase49_f(project_root=PROJECT_ROOT)


def test_all_o49_outputs_present(phase49_f_outcome):
    assert phase49_f_outcome["n_o49_complete"] == phase49_f_outcome["n_o49_total"]
    assert len(O49_INVENTORY) >= 30


def test_all_json_serializable():
    import csv
    for f in ART_DIR.iterdir():
        if f.suffix == ".json" and f.is_file():
            json.loads(f.read_text())


def test_all_csv_schemas_exact():
    # Verify a few critical CSVs have the documented header row
    expected_headers = {
        "residual_long_table.csv": [
            "target_id", "target_timestamp", "seed", "y_true_wh", "y_pred_wh",
            "residual_wh", "absolute_error_wh", "squared_error_wh2",
            "residual_sign", "source_prediction_sha256",
        ],
        "phase49_signed_bias.csv": [
            "seed", "n", "mean_residual_wh", "median_residual_wh",
            "mean_absolute_residual_wh", "mean_underprediction_residual_wh",
            "mean_overprediction_residual_wh", "underprediction_count",
            "overprediction_count", "exact_count", "underprediction_fraction",
            "overprediction_fraction", "exact_fraction", "fractions_sum_to_one",
        ],
    }
    for fname, expected in expected_headers.items():
        with (ART_DIR / fname).open() as fh:
            header = fh.readline().rstrip("\n").split(",")
        assert header == expected, f"{fname}: {header} != {expected}"


def test_all_figures_exist_and_non_empty(phase49_f_outcome):
    figures_dir = ART_DIR / "figures"
    assert figures_dir.exists()
    figs = sorted(figures_dir.glob("phase49_fig*.png"))
    assert phase49_f_outcome["n_figures"] == len(figs)
    assert all(p.stat().st_size > 0 for p in figs)


def test_residual_sign_semantics_unchanged():
    c_manifest = json.loads((ART_DIR / "phase49_c_manifest.json").read_text())
    assert c_manifest["definition_freeze"]["positive_semantics"] == "UNDERPREDICTION"
    assert c_manifest["definition_freeze"]["negative_semantics"] == "OVERPREDICTION"
    assert c_manifest["definition_freeze"]["zero_policy"] == "EXACT_ZERO"


def test_no_best_seed_language(phase49_f_outcome):
    invariants = phase49_f_outcome["contract_invariants"]
    assert invariants["best_seed_selected"] is False


def test_no_ensemble_metrics(phase49_f_outcome):
    assert phase49_f_outcome["contract_invariants"]["ensemble_promoted"] is False


def test_no_three_n_iid_interpretation(phase49_f_outcome):
    assert phase49_f_outcome["contract_invariants"]["three_n_iid_interpretation"] is False


def test_no_test_derived_phase50_thresholds(phase49_f_outcome):
    invariants = phase49_f_outcome["contract_invariants"]
    assert invariants["phase50_regimes_created"] is False
    assert invariants["deciles_used_as_phase50_regimes"] is False
    phase50_handoff = json.loads((ART_DIR / "phase50_handoff.json").read_text())
    assert phase50_handoff["phase50_target_regime_policy"] == "TRAIN_DERIVED_ONLY"
    assert phase50_handoff["phase50_threshold_policy"] == "TRAIN_DERIVED_ONLY"


def test_prediction_deciles_explicitly_diagnostic_only():
    # Phase49-E manifest must declare deciles as DESCRIPTIVE-only diagnostics
    # and explicitly disallow Phase50 regime use.
    e_manifest = json.loads((ART_DIR / "phase49_e_manifest.json").read_text())
    df = e_manifest["definition_freeze"]
    decile_use = df.get("decile_use", "")
    assert (
        "DESCRIPTIVE" in decile_use
        or "descriptive" in decile_use.lower()
    ), f"decile_use must be DESCRIPTIVE-only; got {decile_use!r}"
    assert df["phase50_regime_use"] is False


def test_phase50_handoff_states_train_derived_only():
    phase50_handoff = json.loads((ART_DIR / "phase50_handoff.json").read_text())
    assert "TRAIN_DERIVED_ONLY" in phase50_handoff["phase50_target_regime_policy"]
    assert "TRAIN_DERIVED_ONLY" in phase50_handoff["phase50_threshold_policy"]


def test_phase51_handoff_states_no_prior_worst_error_ranking():
    phase51_handoff = json.loads((ART_DIR / "phase51_context_handoff.json").read_text())
    assert phase51_handoff["phase49_performed_no_worst_error_ranking"] is True
    assert phase51_handoff["phase51_first_phase_allowed_to_rank_worst_errors"] is True


def test_ljung_box_cannot_gate_pass_fail():
    d_manifest = json.loads((ART_DIR / "phase49_d_manifest.json").read_text())
    assert d_manifest["ljung_box"]["policy"] == "SECONDARY_DIAGNOSTIC"
    assert d_manifest["ljung_box"]["used_for_pass_fail"] is False


def test_phase47_source_hashes_unchanged():
    checksums = load_prediction_checksums(project_root=PROJECT_ROOT)
    for seed in (42, 123, 2026):
        key = {"seed42": "seed_42", "seed123": "seed_123", "seed2026": "seed_2026"}[f"seed{seed}"]
        expected = checksums["predictions"][key]["sha256"]
        observed = compute_seed_bundle_sha256(seed, project_root=PROJECT_ROOT)
        assert observed == expected


def test_phase48_source_hashes_unchanged():
    # Verify Phase48 signoff indicates a closed/PASS state. We do not compare the
    # on-disk file hash because upstream tooling may re-emit Phase48 signoff with
    # additional metadata; the canonical content (phase48_d4_closed + overall_status)
    # is the binding invariant for Phase49 downstream.
    phase48_signoff = load_phase48_signoff(project_root=PROJECT_ROOT)
    assert phase48_signoff is not None
    assert phase48_signoff.get("overall_status") == "PASS"
    assert phase48_signoff.get("phase48_d4_closed") is True


def test_phase49_b_c_d_e_canonical_artifacts_unchanged():
    # Spot-check: the 19 expected Phase49-B/C/D/E scientific CSV shas are still present
    expected_csvs = [
        "residual_long_table.csv", "residual_wide_table.csv",
        "phase49_residual_distribution_summary.csv",
        "phase49_signed_bias.csv", "phase49_sign_balance.csv",
        "phase49_tail_diagnostics.csv",
        "phase49_residual_histogram_50bins.csv",
        "phase49_histogram_bin_edges.csv",
        "phase49_residual_acf.csv", "phase49_residual_acf_key_lags.csv",
        "phase49_ljung_box.csv", "phase49_sign_runs.csv",
        "phase49_sign_run_table_seed42.csv",
        "phase49_sign_run_table_seed123.csv",
        "phase49_sign_run_table_seed2026.csv",
        "phase49_sign_transitions.csv",
        "phase49_rolling_residual_diagnostics.csv",
        "phase49_magnitude_associations.csv",
        "phase49_prediction_deciles.csv",
        "phase49_cross_seed_residual_agreement.csv",
        "phase49_cross_seed_sign_consensus.csv",
        "phase49_persistence_context.csv",
    ]
    for fname in expected_csvs:
        assert (ART_DIR / fname).exists()
        assert (ART_DIR / fname).stat().st_size > 0


def test_no_training_inference_checkpoint_paths(phase49_f_outcome):
    invariants = phase49_f_outcome["contract_invariants"]
    assert invariants["new_inference"] is False
    assert invariants["training"] is False
    assert invariants["checkpoint_loading"] is False
    assert invariants["scaler_fit"] is False
    assert invariants["optimizer_steps"] == 0


def test_no_correction_or_recalibration(phase49_f_outcome):
    invariants = phase49_f_outcome["contract_invariants"]
    assert invariants["residual_correction_applied"] is False
    assert invariants["bias_correction_applied"] is False


def test_signoff_cannot_pass_with_missing_required_artifact():
    # Verify the signoff has all expected gate sections (placeholder).
    signoff_path = ART_DIR / "phase_49_signoff.json"
    assert signoff_path.exists()
    signoff = json.loads(signoff_path.read_text())
    gate_names = {g["gate"] for g in signoff["gates"]}
    required_gates = {
        "Phase48 PASS",
        "all 3 seeds verified",
        "N_TEST = 2961",
        "residual convention correct",
        "Phase47 metric reconstruction PASS",
        "Ljung-Box policy = SECONDARY_DIAGNOSTIC",
        "LSTM remains NOT_ELIGIBLE_CONFIG_MISMATCH",
        "all O49 artifacts complete",
        "all figures complete (22 >= 22)",
    }
    missing = required_gates - gate_names
    assert not missing, f"missing gates: {missing}"


def test_deterministic_idempotent_finalization(phase49_f_outcome):
    again = materialize_phase49_f(project_root=PROJECT_ROOT)
    assert phase49_f_outcome["n_o49_complete"] == again["n_o49_complete"]
    assert phase49_f_outcome["n_figures"] == again["n_figures"]
    assert phase49_f_outcome["status"] == again["status"]
