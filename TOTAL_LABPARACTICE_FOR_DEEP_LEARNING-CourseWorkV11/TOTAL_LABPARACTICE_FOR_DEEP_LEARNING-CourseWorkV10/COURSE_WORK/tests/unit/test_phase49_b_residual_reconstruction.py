from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from course_work.analysis.residual_analysis.contract import (
    assert_best_seed_not_selected,
    assert_ensemble_not_promoted,
    assert_residual_sign_semantics,
    assert_three_n_iid_not_claimed,
    contract_n_test,
    contract_seed_list,
    contract_test_population_sha256,
    contract_zero_policy,
)
from course_work.analysis.residual_analysis.materialize_b import materialize_phase49_b
from course_work.analysis.residual_analysis.metrics import (
    mae_from_residuals,
    r2_from_residuals_and_y_true,
    reconstruct_seed_metrics,
    rmse_from_residuals,
)
from course_work.analysis.residual_analysis.residual import reconstruct_residual_row
from course_work.analysis.residual_analysis.sources import (
    compute_seed_bundle_sha256,
    load_seed_bundle_rows,
    verify_source_bundles,
)
from course_work.analysis.residual_analysis.wide_table import (
    build_residual_wide_table_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def phase49_b_manifest():
    return materialize_phase49_b(project_root=PROJECT_ROOT)


def test_residual_definition_is_y_true_minus_y_pred():
    bundle_row = {
        "seed": "42",
        "target_id": "TGT_0",
        "target_timestamp": "2016-05-07 04:40:00",
        "y_true_wh": "60.0",
        "y_pred_wh": "45.3431230037968",
    }
    r = reconstruct_residual_row(bundle_row)
    assert r["residual_wh"] == 60.0 - 45.3431230037968
    assert r["residual_wh"] > 0


def test_positive_residual_means_underprediction():
    bundle_row = {
        "seed": "42",
        "target_id": "TGT_1",
        "target_timestamp": "2016-05-07 04:50:00",
        "y_true_wh": "100.0",
        "y_pred_wh": "60.0",
    }
    r = reconstruct_residual_row(bundle_row)
    assert r["residual_wh"] == 40.0
    assert r["residual_sign"] == "UNDERPREDICTION"


def test_negative_residual_means_overprediction():
    bundle_row = {
        "seed": "123",
        "target_id": "TGT_2",
        "target_timestamp": "2016-05-07 05:00:00",
        "y_true_wh": "20.0",
        "y_pred_wh": "80.0",
    }
    r = reconstruct_residual_row(bundle_row)
    assert r["residual_wh"] == -60.0
    assert r["residual_sign"] == "OVERPREDICTION"


def test_exact_zero_residual_means_exact():
    bundle_row = {
        "seed": "2026",
        "target_id": "TGT_3",
        "target_timestamp": "2016-05-07 05:10:00",
        "y_true_wh": "50.0",
        "y_pred_wh": "50.0",
    }
    r = reconstruct_residual_row(bundle_row)
    assert r["residual_wh"] == 0.0
    assert r["residual_sign"] == "EXACT"


def test_absolute_error_equals_abs_residual():
    bundle_row = {
        "seed": "42",
        "target_id": "TGT_4",
        "target_timestamp": "2016-05-07 05:20:00",
        "y_true_wh": "10.0",
        "y_pred_wh": "13.5",
    }
    r = reconstruct_residual_row(bundle_row)
    assert r["absolute_error_wh"] == abs(r["residual_wh"])


def test_squared_error_wh2_equals_residual_squared():
    bundle_row = {
        "seed": "42",
        "target_id": "TGT_5",
        "target_timestamp": "2016-05-07 05:30:00",
        "y_true_wh": "10.0",
        "y_pred_wh": "13.5",
    }
    r = reconstruct_residual_row(bundle_row)
    assert r["squared_error_wh2"] == r["residual_wh"] ** 2


def test_legacy_squared_error_wh_matches_derived_wh2_per_seed():
    for seed in (42, 123, 2026):
        rows = load_seed_bundle_rows(seed, project_root=PROJECT_ROOT)
        for r in rows[:200]:
            derived = (float(r["y_true_wh"]) - float(r["y_pred_wh"])) ** 2
            assert abs(float(r["squared_error_wh"]) - derived) == 0.0


def test_source_sha_verification_works():
    payload = verify_source_bundles(project_root=PROJECT_ROOT)
    assert payload["__overall_pass__"] is True
    for seed in (42, 123, 2026):
        sha = compute_seed_bundle_sha256(seed, project_root=PROJECT_ROOT)
        assert payload[str(seed)]["observed_sha256"] == sha


def test_n_test_is_2961():
    assert contract_n_test() == 2961
    for seed in (42, 123, 2026):
        rows = load_seed_bundle_rows(seed, project_root=PROJECT_ROOT)
        assert len(rows) == 2961


def test_target_ids_unique_per_seed():
    for seed in (42, 123, 2026):
        rows = load_seed_bundle_rows(seed, project_root=PROJECT_ROOT)
        ids = [r["target_id"] for r in rows]
        assert len(set(ids)) == len(ids) == 2961


def test_all_three_seeds_aligned():
    rows42 = load_seed_bundle_rows(42, project_root=PROJECT_ROOT)
    rows123 = load_seed_bundle_rows(123, project_root=PROJECT_ROOT)
    rows2026 = load_seed_bundle_rows(2026, project_root=PROJECT_ROOT)
    assert [r["target_id"] for r in rows42] == [r["target_id"] for r in rows123]
    assert [r["target_id"] for r in rows42] == [r["target_id"] for r in rows2026]
    assert [r["target_timestamp"] for r in rows42] == [r["target_timestamp"] for r in rows123]
    assert [r["y_true_wh"] for r in rows42] == [r["y_true_wh"] for r in rows123]
    assert [r["y_true_wh"] for r in rows42] == [r["y_true_wh"] for r in rows2026]


def test_residual_long_table_has_8883_rows(phase49_b_manifest):
    assert phase49_b_manifest["long_table_rows"] == 8883
    assert phase49_b_manifest["long_table_rows_expected"] == 2961 * 3


def test_mae_reconstruction_exact():
    residuals = [1.0, -2.0, 3.0, -4.0, 5.0]
    y_true = [10.0, 12.0, 13.0, 16.0, 25.0]
    out = reconstruct_seed_metrics(residuals, y_true)
    assert out["mae_wh"] == mae_from_residuals(residuals)
    assert out["mae_wh"] == 3.0
    assert out["rmse_wh"] == rmse_from_residuals(residuals)
    assert out["r2"] == r2_from_residuals_and_y_true(residuals, y_true)


def test_rmse_reconstruction_exact():
    residuals = [1.0, -2.0, 3.0, -4.0, 5.0]
    expected = ((1 + 4 + 9 + 16 + 25) / 5) ** 0.5
    assert rmse_from_residuals(residuals) == expected


def test_r2_reconstruction_exact():
    y_true = [3.0, -0.5, 2.0, 7.0]
    y_pred = [2.5, 0.0, 2.0, 8.0]
    residuals = [y - p for y, p in zip(y_true, y_pred)]
    assert r2_from_residuals_and_y_true(residuals, y_true) == pytest.approx(
        0.9486081370449679
    )


def test_phase47_metric_agreement_within_tolerance(phase49_b_manifest):
    assert phase49_b_manifest["per_seed_metrics_match_phase47"] is True
    for seed_str, metrics in phase49_b_manifest["per_seed_metrics"].items():
        assert metrics["n"] == 2961
        assert isinstance(metrics["mae_wh"], float)
        assert isinstance(metrics["rmse_wh"], float)
        assert isinstance(metrics["r2"], float)


def test_no_best_seed_selection(phase49_b_manifest):
    invariants = phase49_b_manifest["contract_invariants"]
    assert invariants["best_seed_selected"] is False
    assert_best_seed_not_selected(None)


def test_no_ensemble_evaluation(phase49_b_manifest):
    invariants = phase49_b_manifest["contract_invariants"]
    assert invariants["ensemble_promoted"] is False
    assert_ensemble_not_promoted(False)


def test_no_three_n_iid_interpretation(phase49_b_manifest):
    invariants = phase49_b_manifest["contract_invariants"]
    assert invariants["three_n_iid_interpretation"] is False
    assert_three_n_iid_not_claimed(False)


def test_no_training_or_inference(phase49_b_manifest):
    invariants = phase49_b_manifest["contract_invariants"]
    assert invariants["new_inference"] is False
    assert invariants["training"] is False
    assert invariants["checkpoint_loading"] is False
    assert invariants["scaler_fit"] is False
    assert invariants["optimizer_steps"] == 0


def test_phase47_artifacts_unchanged(phase49_b_manifest):
    assert phase49_b_manifest["phase47_canonical_artifacts_unchanged"] is True
    assert phase49_b_manifest["contract_invariants"]["phase47_modified"] is False


def test_phase48_artifacts_unchanged(phase49_b_manifest):
    assert phase49_b_manifest["phase48_canonical_artifacts_unchanged"] is True
    assert phase49_b_manifest["contract_invariants"]["phase48_modified"] is False


def test_phase49_artifacts_exist(phase49_b_manifest):
    for key in (
        "field_consistency_audit_path",
        "metric_reconstruction_audit_path",
        "long_table_path",
        "wide_table_path",
    ):
        p = Path(phase49_b_manifest[key])
        assert p.exists(), key
        assert p.stat().st_size > 0


def test_outputs_deterministic(phase49_b_manifest):
    seed42_metrics = phase49_b_manifest["per_seed_metrics"]["42"]
    payload2 = materialize_phase49_b(project_root=PROJECT_ROOT)
    seed42_metrics_again = payload2["per_seed_metrics"]["42"]
    assert seed42_metrics == seed42_metrics_again


def test_wide_table_seeded_correctly():
    long42 = [
        {
            "target_id": "TGT_1",
            "target_timestamp": "2016-05-07 04:40:00",
            "seed": "42",
            "y_true_wh": 60.0,
            "y_pred_wh": 45.0,
            "residual_wh": 15.0,
            "absolute_error_wh": 15.0,
            "squared_error_wh2": 225.0,
            "residual_sign": "UNDERPREDICTION",
            "source_prediction_sha256": "abc",
        }
    ]
    long123 = [
        {
            "target_id": "TGT_1",
            "target_timestamp": "2016-05-07 04:40:00",
            "seed": "123",
            "y_true_wh": 60.0,
            "y_pred_wh": 50.0,
            "residual_wh": 10.0,
            "absolute_error_wh": 10.0,
            "squared_error_wh2": 100.0,
            "residual_sign": "UNDERPREDICTION",
            "source_prediction_sha256": "def",
        }
    ]
    long2026 = [
        {
            "target_id": "TGT_1",
            "target_timestamp": "2016-05-07 04:40:00",
            "seed": "2026",
            "y_true_wh": 60.0,
            "y_pred_wh": 55.0,
            "residual_wh": 5.0,
            "absolute_error_wh": 5.0,
            "squared_error_wh2": 25.0,
            "residual_sign": "UNDERPREDICTION",
            "source_prediction_sha256": "ghi",
        }
    ]
    rows = build_residual_wide_table_rows({42: long42, 123: long123, 2026: long2026})
    assert len(rows) == 1
    r = rows[0]
    assert r["seed42_residual_wh"] == 15.0
    assert r["seed123_residual_wh"] == 10.0
    assert r["seed2026_residual_wh"] == 5.0
    assert r["seed_mean_residual_wh"] == 10.0
    assert r["seed_mean_residual_semantics"] == "SEED_MEAN_RESIDUAL_DESCRIPTIVE"


def test_seed_mean_field_carries_descriptive_label(phase49_b_manifest):
    wide_path = Path(phase49_b_manifest["wide_table_path"])
    with wide_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            assert row["seed_mean_residual_semantics"] == "SEED_MEAN_RESIDUAL_DESCRIPTIVE"


def test_sign_semantics_assertion():
    assert_residual_sign_semantics("UNDERPREDICTION")
    assert_residual_sign_semantics("OVERPREDICTION")
    assert_residual_sign_semantics("EXACT")
    with pytest.raises(Exception):
        assert_residual_sign_semantics("UNKNOWN")


def test_contract_seed_list_and_population_fingerprint():
    assert contract_seed_list() == [42, 123, 2026]
    assert contract_test_population_sha256() == (
        "d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87"
    )
    assert contract_zero_policy() == "EXACT_ZERO"


def test_phase50_51_52_plus_remain_unauthorized(phase49_b_manifest):
    assert phase49_b_manifest["phase50_authorized"] is False
    assert phase49_b_manifest["phase51_authorized"] is False
    assert phase49_b_manifest["phase52_plus_authorized"] is False
    assert phase49_b_manifest["contract_invariants"]["worst_error_ranking_executed"] is False
    assert phase49_b_manifest["contract_invariants"]["attention_analysis_executed"] is False
