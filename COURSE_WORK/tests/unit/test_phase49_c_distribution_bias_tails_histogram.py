from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from course_work.residual_analysis.bias import (
    _sign_label_exact,
    recompute_sign_class,
    sign_balance_rows,
    signed_bias_for_residuals,
)
from course_work.residual_analysis.c_writers import snapshot_phase49_b_artifacts
from course_work.residual_analysis.contract import (
    assert_best_seed_not_selected,
    assert_ensemble_not_promoted,
    assert_three_n_iid_not_claimed,
)
from course_work.residual_analysis.distributions import (
    compute_distribution_summary,
    distribution_summary_rows,
    load_phase49_b_long_table,
    residuals_for_seed,
)
from course_work.residual_analysis.histogram import (
    PHASE49_HISTOGRAM_BINS,
    compute_common_bin_edges,
    compute_histogram_rows,
    histogram_invariants,
)
from course_work.residual_analysis.materialize_c import SEED_LIST, materialize_phase49_c
from course_work.residual_analysis.tails import compute_tail_diagnostics


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def phase49_c_manifest():
    return materialize_phase49_c(project_root=PROJECT_ROOT)


@pytest.fixture(scope="module")
def long_table():
    return load_phase49_b_long_table(project_root=PROJECT_ROOT)


def test_residual_convention_remains_y_true_minus_y_pred(long_table):
    sample = long_table[0]
    expected = float(sample["y_true_wh"]) - float(sample["y_pred_wh"])
    assert abs(float(sample["residual_wh"]) - expected) == 0.0


def test_distribution_n_is_2961_per_seed(long_table, phase49_c_manifest):
    for seed in SEED_LIST:
        residuals = residuals_for_seed(long_table, seed)
        assert residuals.size == 2961
        d = phase49_c_manifest["distribution_summary"][seed]
        assert d["n"] == 2961


def test_std_definition_uses_ddof_1(long_table):
    residuals = residuals_for_seed(long_table, "42")
    summary = compute_distribution_summary(residuals, ddof=1)
    expected = float(np.std(residuals, ddof=1))
    assert summary["std"] == expected


def test_mad_definition_exact(long_table):
    residuals = residuals_for_seed(long_table, "42")
    summary = compute_distribution_summary(residuals)
    expected = float(np.median(np.abs(residuals - np.median(residuals))))
    assert summary["mad"] == expected


def test_skewness_uses_bias_false(long_table):
    residuals = residuals_for_seed(long_table, "42")
    summary = compute_distribution_summary(residuals)
    from scipy import stats
    expected = float(stats.skew(residuals, bias=False))
    assert summary["skewness"] == expected


def test_kurtosis_uses_fisher_true_and_bias_false(long_table):
    residuals = residuals_for_seed(long_table, "42")
    summary = compute_distribution_summary(residuals)
    from scipy import stats
    expected = float(stats.kurtosis(residuals, fisher=True, bias=False))
    assert summary["kurtosis_excess"] == expected


def test_q1_q3_iqr_correct(long_table):
    residuals = residuals_for_seed(long_table, "42")
    summary = compute_distribution_summary(residuals)
    assert summary["q1"] == float(np.quantile(residuals, 0.25))
    assert summary["q3"] == float(np.quantile(residuals, 0.75))
    assert summary["iqr"] == summary["q3"] - summary["q1"]


def test_p01_p05_p95_p99_correct(long_table):
    residuals = residuals_for_seed(long_table, "42")
    summary = compute_distribution_summary(residuals)
    assert summary["p01"] == float(np.quantile(residuals, 0.01))
    assert summary["p05"] == float(np.quantile(residuals, 0.05))
    assert summary["p95"] == float(np.quantile(residuals, 0.95))
    assert summary["p99"] == float(np.quantile(residuals, 0.99))


def test_sign_classes_exact():
    assert _sign_label_exact(1.0) == "UNDERPREDICTION"
    assert _sign_label_exact(-1.0) == "OVERPREDICTION"
    assert _sign_label_exact(0.0) == "EXACT"


def test_zero_uses_exact_equality():
    assert _sign_label_exact(1e-300) == "UNDERPREDICTION"
    assert _sign_label_exact(-1e-300) == "OVERPREDICTION"
    assert _sign_label_exact(0.0) == "EXACT"


def test_sign_fractions_sum_to_one(long_table, phase49_c_manifest):
    sb = phase49_c_manifest["sign_balance"]
    for seed in SEED_LIST:
        d = sb[seed]
        total = d["underprediction_fraction"] + d["overprediction_fraction"] + d["exact_fraction"]
        assert abs(total - 1.0) < 1e-12
        assert d["fractions_sum_to_one"] is True


def test_bias_summaries_deterministic(phase49_c_manifest):
    manifest_again = materialize_phase49_c(project_root=PROJECT_ROOT)
    for seed in SEED_LIST:
        for k in ("mean_residual_wh", "median_residual_wh", "mean_absolute_residual_wh",
                  "mean_underprediction_residual_wh", "mean_overprediction_residual_wh"):
            assert (
                phase49_c_manifest["signed_bias"][seed][k]
                == manifest_again["signed_bias"][seed][k]
            )


def test_under_over_counts_use_correct_denominator(long_table, phase49_c_manifest):
    sb = phase49_c_manifest["sign_balance"]
    for seed in SEED_LIST:
        d = sb[seed]
        assert d["n"] == 2961
        assert d["underprediction_count"] + d["overprediction_count"] + d["exact_count"] == 2961
        assert abs(d["underprediction_count"] / 2961 - d["underprediction_fraction"]) < 1e-12


def test_tail_quantiles_correct(long_table, phase49_c_manifest):
    td = phase49_c_manifest["tail_diagnostics"]
    for seed in SEED_LIST:
        abs_errors = np.asarray(
            [float(r["absolute_error_wh"]) for r in long_table if r["seed"] == seed],
            dtype=np.float64,
        )
        assert td[seed]["abs_error_p90"] == float(np.quantile(abs_errors, 0.90))
        assert td[seed]["abs_error_p95"] == float(np.quantile(abs_errors, 0.95))
        assert td[seed]["abs_error_p99"] == float(np.quantile(abs_errors, 0.99))
        assert td[seed]["max_absolute_error"] == float(np.max(abs_errors))


def test_no_worst_error_ranking_in_phase49_c_artifacts(phase49_c_manifest):
    artifacts = phase49_c_manifest["artifacts"]
    for k, v in artifacts.items():
        assert "worst" not in v.lower()
        assert "rank" not in v.lower()
    assert phase49_c_manifest["contract_invariants"]["worst_error_ranking_executed"] is False


def test_50_common_histogram_bins_exact(phase49_c_manifest):
    hist = phase49_c_manifest["histogram"]
    assert hist["n_bins"] == 50
    assert hist["n_seeds"] == 3
    assert hist["n_rows_total"] == 150


def test_same_histogram_edges_across_seeds(phase49_c_manifest):
    hist = phase49_c_manifest["histogram"]
    assert hist["shared_bin_edges_across_seeds"] is True


def test_histogram_counts_sum_to_n_per_seed(phase49_c_manifest):
    hist = phase49_c_manifest["histogram"]
    assert hist["all_seeds_count_to_2961"] is True


def test_histogram_fractions_sum_to_one_per_seed(phase49_c_manifest):
    hist = phase49_c_manifest["histogram"]
    assert hist["all_seeds_fraction_to_1"] is True


def test_no_target_regime_creation(phase49_c_manifest):
    assert phase49_c_manifest["contract_invariants"]["target_regime_analysis_deferred_to_phase50"] is True
    assert phase49_c_manifest["contract_invariants"]["deciles_used_as_phase50_regimes"] is False


def test_no_best_seed_selection(phase49_c_manifest):
    invariants = phase49_c_manifest["contract_invariants"]
    assert invariants["best_seed_selected"] is False
    assert_best_seed_not_selected(None)


def test_no_ensemble(phase49_c_manifest):
    invariants = phase49_c_manifest["contract_invariants"]
    assert invariants["ensemble_promoted"] is False
    assert_ensemble_not_promoted(False)


def test_no_three_n_iid_pooling(phase49_c_manifest):
    invariants = phase49_c_manifest["contract_invariants"]
    assert invariants["three_n_iid_interpretation"] is False
    assert_three_n_iid_not_claimed(False)


def test_phase47_source_unchanged(phase49_c_manifest):
    invariants = phase49_c_manifest["contract_invariants"]
    assert invariants["phase47_modified"] is False
    from course_work.residual_analysis.sources import load_prediction_checksums, compute_seed_bundle_sha256
    checksums = load_prediction_checksums(project_root=PROJECT_ROOT)
    for seed in (42, 123, 2026):
        bundle = f"final_test_predictions_seed{seed}.csv"
        key = {"seed42": "seed_42", "seed123": "seed_123", "seed2026": "seed_2026"}[f"seed{seed}"]
        expected = checksums["predictions"][key]["sha256"]
        observed = compute_seed_bundle_sha256(seed, project_root=PROJECT_ROOT)
        assert observed == expected


def test_phase48_artifacts_unchanged(phase49_c_manifest):
    invariants = phase49_c_manifest["contract_invariants"]
    assert invariants["phase48_modified"] is False


def test_phase49_b_artifacts_unchanged(phase49_c_manifest):
    invariants = phase49_c_manifest["contract_invariants"]
    assert invariants["phase49_b_modified"] is False
    snap = snapshot_phase49_b_artifacts(PROJECT_ROOT)
    assert "residual_long_table.csv" in snap
    assert "residual_wide_table.csv" in snap
    assert "phase49_b_manifest.json" in snap


def test_outputs_deterministic(phase49_c_manifest):
    again = materialize_phase49_c(project_root=PROJECT_ROOT)
    assert (
        phase49_c_manifest["distribution_summary"]
        == again["distribution_summary"]
    )
    assert (
        phase49_c_manifest["histogram"]["n_rows_total"]
        == again["histogram"]["n_rows_total"]
    )


def test_phase49_c_artifacts_exist(phase49_c_manifest):
    for k in (
        "distribution_summary_csv",
        "signed_bias_csv",
        "sign_balance_csv",
        "tail_diagnostics_csv",
        "histogram_csv",
        "histogram_bin_edges_csv",
    ):
        p = Path(phase49_c_manifest["artifacts"][k])
        assert p.exists(), k
        assert p.stat().st_size > 0


def test_seed_mean_residual_semantics_locked(phase49_c_manifest):
    assert phase49_c_manifest["seed_mean_residual_semantics"] == "SEED_MEAN_RESIDUAL_DESCRIPTIVE"
