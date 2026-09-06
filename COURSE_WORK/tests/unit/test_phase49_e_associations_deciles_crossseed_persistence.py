from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from course_work.residual_analysis.cross_seed import (
    SEED_PAIRS,
    compute_cross_seed_sign_consensus,
    compute_pair_residual_agreement,
)
from course_work.residual_analysis.distributions import (
    load_phase49_b_long_table,
    residuals_for_seed,
)
from course_work.residual_analysis.magnitude_associations import (
    compute_magnitude_associations,
    pearson,
    spearman,
)
from course_work.residual_analysis.materialize_e import SEED_LIST, materialize_phase49_e
from course_work.residual_analysis.persistence_context import (
    compute_persistence_residual_context,
    load_persistence_rows,
    verify_persistence_against_phase47,
)
from course_work.residual_analysis.prediction_deciles import (
    PHASE49_DECILE_COUNT,
    assign_deciles,
    build_prediction_decile_edges,
    compute_decile_diagnostics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def phase49_e_manifest():
    return materialize_phase49_e(project_root=PROJECT_ROOT)


@pytest.fixture(scope="module")
def long_table():
    return load_phase49_b_long_table(project_root=PROJECT_ROOT)


def _seed_data(long_table, seed):
    rows = sorted(
        [r for r in long_table if r["seed"] == seed],
        key=lambda r: r["target_id"],
    )
    return (
        np.asarray([float(r["residual_wh"]) for r in rows], dtype=np.float64),
        np.asarray([float(r["y_true_wh"]) for r in rows], dtype=np.float64),
        np.asarray([float(r["y_pred_wh"]) for r in rows], dtype=np.float64),
        [r["residual_sign"] for r in rows],
    )


def test_residual_convention_unchanged(long_table):
    sample = long_table[0]
    expected = float(sample["y_true_wh"]) - float(sample["y_pred_wh"])
    assert abs(float(sample["residual_wh"]) - expected) == 0.0


def test_magnitude_association_inputs_correct():
    residuals = np.asarray([1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float64)
    y_true = np.asarray([10.0, 12.0, 13.0, 16.0, 25.0], dtype=np.float64)
    y_pred = np.asarray([9.0, 14.0, 10.0, 20.0, 20.0], dtype=np.float64)
    rows = compute_magnitude_associations(residuals, y_true, y_pred)
    assert len(rows) == 8
    assert {r["association_type"] for r in rows} == {"pearson", "spearman"}
    assert {r["x_variable"] for r in rows} == {"residual", "|residual|"}
    assert all(r["y_variable"] in {"y_true_wh", "y_pred_wh"} for r in rows)
    assert all(r["causal_interpretation"] == "NONE" for r in rows)
    assert all(r["status"] == "DESCRIPTIVE" for r in rows)


def test_pearson_deterministic():
    x = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    y = np.asarray([2.0, 4.0, 6.0, 8.0, 10.0], dtype=np.float64)
    a = pearson(x, y)
    b = pearson(x, y)
    assert a == b
    assert abs(a - 1.0) < 1e-12


def test_spearman_deterministic():
    x = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    y = np.asarray([1.0, 4.0, 9.0, 16.0, 25.0], dtype=np.float64)
    a = spearman(x, y)
    b = spearman(x, y)
    assert a == b
    assert abs(a - 1.0) < 1e-12


def test_no_causal_label_language(phase49_e_manifest):
    for row in phase49_e_manifest["magnitude_associations"]["rows_by_seed"]["42"]:
        assert row["causal_interpretation"] == "NONE"
        assert row["status"] == "DESCRIPTIVE"
    assert phase49_e_manifest["contract_invariants"]["causal_interpretation_emitted"] is False


def test_prediction_deciles_use_y_pred_only(phase49_e_manifest):
    assert phase49_e_manifest["prediction_deciles"]["decile_source"] == "y_pred_only"
    assert phase49_e_manifest["definition_freeze"]["decile_source"] == "y_pred_only"
    assert phase49_e_manifest["definition_freeze"]["phase50_regime_use"] is False
    assert phase49_e_manifest["contract_invariants"]["deciles_used_as_phase50_regimes"] is False


def test_intended_decile_count_is_10(phase49_e_manifest):
    assert phase49_e_manifest["prediction_deciles"]["decile_count"] == 10
    assert PHASE49_DECILE_COUNT == 10


def test_duplicate_quantile_edges_handled_explicitly():
    y_pred = np.asarray([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float64)
    edges, meta = build_prediction_decile_edges(y_pred)
    assert edges.size == 11
    assert meta["duplicate_edges_present"] is True
    assert meta["n_unique_edges"] == 1
    assignments = assign_deciles(y_pred, edges)
    assert (assignments >= 1).all() and (assignments <= 10).all()


def test_no_silent_bin_dropping():
    y_pred = np.asarray(list(range(1, 101)), dtype=np.float64)
    edges, meta = build_prediction_decile_edges(y_pred)
    assert edges.size == 11
    assert meta["n_unique_edges"] == 11


def test_decile_counts_sum_to_n_per_seed(phase49_e_manifest):
    for seed in SEED_LIST:
        deciles = phase49_e_manifest["prediction_deciles"]["rows_by_seed"][seed]
        total = sum(d["N"] for d in deciles)
        assert total == 2961


def test_decile_diagnostics_deterministic(phase49_e_manifest):
    a = materialize_phase49_e(project_root=PROJECT_ROOT)
    for seed in SEED_LIST:
        for k, d in enumerate(phase49_e_manifest["prediction_deciles"]["rows_by_seed"][seed]):
            assert d["N"] == a["prediction_deciles"]["rows_by_seed"][seed][k]["N"]
            assert d["mae"] == a["prediction_deciles"]["rows_by_seed"][seed][k]["mae"]


def test_decile_thresholds_not_marked_as_phase50_regimes(phase49_e_manifest):
    for seed in SEED_LIST:
        for decile in phase49_e_manifest["prediction_deciles"]["rows_by_seed"][seed]:
            assert "phase50" not in str(decile).lower()
    assert phase49_e_manifest["contract_invariants"]["phase50_regimes_created"] is False


def test_exactly_3_cross_seed_pairs(phase49_e_manifest):
    assert phase49_e_manifest["cross_seed_residual_agreement"]["n_pairs"] == 3
    assert len(SEED_PAIRS) == 3
    assert SEED_PAIRS == (("42", "123"), ("42", "2026"), ("123", "2026"))


def test_residual_pair_alignment_exact(long_table):
    ids42 = [r["target_id"] for r in long_table if r["seed"] == "42"]
    ids123 = [r["target_id"] for r in long_table if r["seed"] == "123"]
    ids2026 = [r["target_id"] for r in long_table if r["seed"] == "2026"]
    assert ids42 == ids123 == ids2026


def test_residual_pair_correlations_correct(phase49_e_manifest):
    rows = phase49_e_manifest["cross_seed_residual_agreement"]["rows"]
    for r in rows:
        assert r["scope"] == "CROSS_SEED_RESIDUAL_AGREEMENT_DIAGNOSTIC"
        assert r["purpose"] == "NOT model-performance comparison"
        assert r["N"] == 2961
        assert isinstance(r["pearson_residual_correlation"], float)
        assert isinstance(r["spearman_residual_correlation"], float)


def test_no_best_seed_ranking(phase49_e_manifest):
    assert phase49_e_manifest["contract_invariants"]["best_seed_selected"] is False


def test_sign_consensus_uses_exact_zero(phase49_e_manifest):
    assert phase49_e_manifest["cross_seed_sign_consensus"]["zero_policy"] == "EXACT_ZERO"
    assert phase49_e_manifest["definition_freeze"]["zero_policy"] == "EXACT_ZERO"


def test_sign_consensus_fractions_sum_to_one(phase49_e_manifest):
    assert phase49_e_manifest["cross_seed_sign_consensus"]["fractions_sum_to_one"] is True
    fractions = [r["fraction"] for r in phase49_e_manifest["cross_seed_sign_consensus"]["rows"]]
    assert abs(sum(fractions) - 1.0) < 1e-12


def test_persistence_checksum_exact(phase49_e_manifest):
    assert phase49_e_manifest["persistence_context"]["source_sha256_match"] is True
    from course_work.residual_analysis.sources import load_prediction_checksums, sha256_file as src_sha256_file
    checksums = load_prediction_checksums(project_root=PROJECT_ROOT)
    expected = checksums["predictions"]["persistence"]["sha256"]
    observed = src_sha256_file(
        PROJECT_ROOT / "artifacts/final_test/predictions/final_test_predictions_persistence.csv"
    )
    assert observed == expected


def test_persistence_n_is_2961(phase49_e_manifest):
    assert phase49_e_manifest["persistence_context"]["verification"]["N_value"] == 2961
    assert phase49_e_manifest["persistence_context"]["verification"]["N_match"] is True


def test_persistence_target_alignment_exact(phase49_e_manifest):
    v = phase49_e_manifest["persistence_context"]["verification"]
    assert v["target_id_match"] is True
    assert v["y_true_match"] is True
    assert v["timestamp_match"] is True


def test_persistence_residual_convention_correct(phase49_e_manifest):
    ctx = phase49_e_manifest["persistence_context"]["context"]
    assert ctx["residual_convention"] == "y_true - y_pred"


def test_lstm_remains_ineligible(phase49_e_manifest):
    lstm = phase49_e_manifest["lstm_policy"]
    assert lstm["lstm_tuned_dev"] == "NOT_ELIGIBLE_CONFIG_MISMATCH"
    assert lstm["lstm_residual_table_created"] is False
    assert lstm["lstm_placeholder_residuals_created"] is False


def test_no_lstm_residual_artifact():
    target = PROJECT_ROOT / "artifacts/residual_analysis"
    assert not (target / "lstm_residual_table.csv").exists()
    assert not (target / "phase49_lstm_context.csv").exists()


def test_no_ensemble(phase49_e_manifest):
    assert phase49_e_manifest["contract_invariants"]["ensemble_promoted"] is False


def test_no_three_n_iid_pooling(phase49_e_manifest):
    assert phase49_e_manifest["contract_invariants"]["three_n_iid_interpretation"] is False


def test_no_worst_error_ranking(phase49_e_manifest):
    assert phase49_e_manifest["contract_invariants"]["worst_error_ranking_executed"] is False


def test_no_phase50_implementation(phase49_e_manifest):
    assert phase49_e_manifest["phase50_authorized"] is False
    assert phase49_e_manifest["phase51_authorized"] is False
    assert phase49_e_manifest["phase52_plus_authorized"] is False
    assert phase49_e_manifest["contract_invariants"]["phase50_regimes_created"] is False
    assert phase49_e_manifest["contract_invariants"]["target_regime_analysis_deferred_to_phase50"] is True


def test_phase47_unchanged(phase49_e_manifest):
    assert phase49_e_manifest["contract_invariants"]["phase47_modified"] is False
    from course_work.residual_analysis.sources import compute_seed_bundle_sha256, load_prediction_checksums
    checksums = load_prediction_checksums(project_root=PROJECT_ROOT)
    for seed in (42, 123, 2026):
        key = {"seed42": "seed_42", "seed123": "seed_123", "seed2026": "seed_2026"}[f"seed{seed}"]
        expected = checksums["predictions"][key]["sha256"]
        observed = compute_seed_bundle_sha256(seed, project_root=PROJECT_ROOT)
        assert observed == expected


def test_phase48_unchanged(phase49_e_manifest):
    assert phase49_e_manifest["contract_invariants"]["phase48_modified"] is False


def test_phase49_b_c_d_artifacts_unchanged(phase49_e_manifest):
    assert phase49_e_manifest["contract_invariants"]["phase49_b_modified"] is False
    assert phase49_e_manifest["contract_invariants"]["phase49_c_modified"] is False
    assert phase49_e_manifest["contract_invariants"]["phase49_d_modified"] is False


def test_outputs_deterministic(phase49_e_manifest):
    a = materialize_phase49_e(project_root=PROJECT_ROOT)
    for seed in SEED_LIST:
        a_pairs = [r for r in a["cross_seed_residual_agreement"]["rows"]]
        b_pairs = [r for r in phase49_e_manifest["cross_seed_residual_agreement"]["rows"]]
        for k in range(len(a_pairs)):
            for metric in (
                "pearson_residual_correlation",
                "spearman_residual_correlation",
                "mean_absolute_residual_difference",
                "rmse_between_residuals",
                "max_absolute_residual_difference",
            ):
                assert a_pairs[k][metric] == b_pairs[k][metric]


def test_phase49_e_artifacts_exist(phase49_e_manifest):
    for k in (
        "magnitude_associations_csv",
        "prediction_deciles_csv",
        "cross_seed_residual_agreement_csv",
        "cross_seed_sign_consensus_csv",
        "persistence_context_csv",
    ):
        p = Path(phase49_e_manifest["artifacts"][k])
        assert p.exists(), k
        assert p.stat().st_size > 0


def test_phase49_e_persistence_persists_only_descriptive_baseline(phase49_e_manifest):
    ctx = phase49_e_manifest["persistence_context"]["context"]
    assert ctx["scope"] == "BASELINE_RESIDUAL_CONTEXT"
    assert ctx["purpose"] == "NOT a model ranking"


def test_phase47_baseline_interpretation_preserved(phase49_e_manifest):
    interp = phase49_e_manifest["definition_freeze"]["phase47_baseline_interpretation"]
    assert "RMSE" in interp["transformer_better_metrics"]
    assert "R2" in interp["transformer_better_metrics"]
    assert "MAE" in interp["persistence_better_metric"]
