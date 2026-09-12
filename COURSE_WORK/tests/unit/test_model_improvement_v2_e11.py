from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from course_work.experiments.registry import ExperimentRegistry, V2_NAMESPACE_IDS
from course_work.model_improvement_v2.contracts import assert_one_primary_change
from course_work.model_improvement_v2.e11_pretest import (
    CHALLENGER_VARIANT,
    CONTROL_VARIANT,
    ROLLING_DEFINITIONS,
    ROLLING_FEATURES,
    build_e11_pretest_datasets,
    challenger_feature_order,
    derive_past_only_rolling_features,
)
from course_work.model_improvement_v2.e11_runner import (
    CHALLENGER_ID,
    CONTROL_ID,
    EXPECTED_COMMON_FINGERPRINT,
    build_e11_candidates,
    evaluate_promotion,
    load_e11_config,
    main,
    project_root,
    run_preflight,
)


EXPECTED_FOLD_FINGERPRINTS = {
    "RO1": "2474242da40cfbebb33a86ff7d6601d47e03cdc504d8237b0c3fc3e20211a62f",
    "RO2": "7b81b8d969850e62deec44fee6fabdf45c8e1a9c28d51465ce9036625110e0b1",
    "RO3": EXPECTED_COMMON_FINGERPRINT,
}


def test_exact_rolling_definitions_order_count_and_no_delta_carry_forward():
    expected = (
        "appliances_roll_mean_3",
        "appliances_roll_mean_6",
        "appliances_roll_mean_12",
        "appliances_roll_std_6",
        "appliances_roll_std_12",
        "appliances_roll_max_12",
        "appliances_roll_min_12",
    )
    assert ROLLING_FEATURES == expected
    assert list(ROLLING_DEFINITIONS) == list(expected)
    assert len(challenger_feature_order()) == 40
    assert challenger_feature_order()[-7:] == expected
    assert not {
        "appliances_delta_1",
        "appliances_abs_delta_1",
        "appliances_delta_2",
    }.intersection(challenger_feature_order())


def test_rolling_formulas_are_trailing_ddof_zero_and_future_independent():
    y = np.arange(1.0, 18.0)
    values = derive_past_only_rolling_features(y)
    assert np.isnan(values[:2, 0]).all()
    assert np.isnan(values[:5, 1]).all()
    assert np.isnan(values[:11, [2, 4, 5, 6]]).all()
    s = 11
    np.testing.assert_allclose(
        values[s],
        [
            np.mean(y[s - 2 : s + 1]),
            np.mean(y[s - 5 : s + 1]),
            np.mean(y[s - 11 : s + 1]),
            np.std(y[s - 5 : s + 1], ddof=0),
            np.std(y[s - 11 : s + 1], ddof=0),
            np.max(y[s - 11 : s + 1]),
            np.min(y[s - 11 : s + 1]),
        ],
    )
    modified = y.copy()
    modified[s + 1 :] = -999999.0
    np.testing.assert_allclose(
        derive_past_only_rolling_features(modified)[: s + 1],
        values[: s + 1],
        equal_nan=True,
    )


def test_common_population_folds_and_hard_test_firewall():
    datasets, evidence, audit = build_e11_pretest_datasets(project_root())
    control_ids = datasets[CONTROL_VARIANT].window_records["target_id"].tolist()
    challenger_ids = datasets[CHALLENGER_VARIANT].window_records["target_id"].tolist()
    assert control_ids == challenger_ids
    assert len(control_ids) == 16630
    assert audit.common_population_fingerprint == EXPECTED_COMMON_FINGERPRINT
    assert {str(fold.fold_id): fold.fold_population_fingerprint for fold in evidence.folds} == EXPECTED_FOLD_FINGERPRINTS
    assert all(value == "PASS" for value in evidence.fingerprint_status.values())
    assert audit.control.test_rows_read == audit.challenger.test_rows_read == 0
    assert audit.control.test_target_ids_seen == audit.challenger.test_target_ids_seen == 0
    assert audit.no_future_target_access is True
    assert datasets[CONTROL_VARIANT]._feature_matrix.shape[1] == 33
    assert datasets[CHALLENGER_VARIANT]._feature_matrix.shape[1] == 40


def test_e10_rejection_incumbent_and_one_primary_change_are_locked():
    root = project_root()
    document = load_e11_config(root)
    report = run_preflight(root)
    assert report["predecessor_audit"]["human_decision"] == "E10_REJECTED"
    assert report["accepted_incumbent"] == "E01_DIRECT_FS2_TF1"
    assert document["predecessor_human_decision"]["carry_forward_delta_features"] is False
    assert len(document["predecessor_human_decision"]["reporting_gaps_recorded"]) >= 4
    changed = assert_one_primary_change(
        "E11", document["control"]["config"], document["challenger"]["config"]
    )
    assert changed
    bad = deepcopy(document["challenger"]["config"])
    bad["training"]["learning_rate"] = 1e-4
    with pytest.raises(ValueError):
        assert_one_primary_change("E11", document["control"]["config"], bad)


def test_candidate_pair_retrains_common_population_from_fresh_seed():
    document = load_e11_config(project_root())
    specs = build_e11_candidates(document)
    assert {spec.candidate_id for spec in specs} == {CONTROL_ID, CHALLENGER_ID}
    assert all(item["training_required"] is True for item in (document["control"], document["challenger"]))
    assert document["control"]["reuse_e01_metrics"] is False
    assert all(
        item["initialization_policy"] == "FRESH_FROM_SEED_42_NO_CHECKPOINT_LOAD"
        for item in (document["control"], document["challenger"])
    )
    assert document["control"]["config"]["training"] == document["challenger"]["config"]["training"]


def test_registry_namespace_is_explicit_and_isolated(tmp_path: Path):
    assert "V2_E11" in V2_NAMESPACE_IDS
    registry = ExperimentRegistry(
        project_root=project_root(),
        registry_root=tmp_path / "registry",
        run_root=tmp_path / "runs",
        run_id_namespace="V2_E11",
        upstream_context_override={
            "lineage": {},
            "feature_sets": {},
            "window_fingerprints": {},
        },
    )
    run_id = registry.allocate_run_id(
        "TRANSFORMER_ENCODER", "ROLLING_ORIGIN", "abcdef012345", run_stage="RO1_A"
    )
    assert run_id.startswith("RUN_V2_TR_E11_RO1_A_")
    with pytest.raises(ValueError):
        ExperimentRegistry(
            project_root=project_root(),
            registry_root=tmp_path / "bad",
            run_root=tmp_path / "bad_runs",
            run_id_namespace="V2_ANY",
        )


def test_promotion_requires_all_four_guardrails_and_human_decision():
    control = {"rmse_wh": 60.0, "mae_wh": 25.0}
    challenger = {"rmse_wh": 59.9, "mae_wh": 25.25}
    passed = evaluate_promotion(control, challenger, [59.0, 60.0, 61.0], [59.0, 60.0, 61.5])
    assert passed["eligible_for_human_promotion"] is True
    assert passed["automatic_promotion"] is False
    assert passed["human_decision_required"] is True
    failed = evaluate_promotion(
        control, {"rmse_wh": 59.9, "mae_wh": 25.251}, [59, 60, 61], [59, 60, 61.5]
    )
    assert failed["eligible_for_human_promotion"] is False


def test_preflight_and_official_human_gate():
    report = run_preflight()
    assert report["status"] == "PASS"
    assert report["expected_total_training_runs"] == 12
    assert report["training_candidates"] == [CONTROL_ID, CHALLENGER_ID]
    assert report["population"]["test_rows_read"] == 0
    assert main(["--experiment", "E11", "--mode", "official", "--seed", "42"]) == 3
