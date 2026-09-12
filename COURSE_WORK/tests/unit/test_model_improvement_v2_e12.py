from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from course_work.data.feature_sets import get_feature_list
from course_work.experiments.registry import ExperimentRegistry, V2_NAMESPACE_IDS
from course_work.model_improvement_v2.contracts import assert_one_primary_change
from course_work.model_improvement_v2.e12_pretest import (
    CHALLENGER_VARIANT,
    CONTROL_VARIANT,
    FIRST_ELIGIBLE_TARGET_POSITION,
    LAG_DEFINITION,
    LAG_FEATURE,
    LAG_STEPS,
    build_e12_pretest_datasets,
    challenger_feature_order,
    derive_past_only_lag144_feature,
)
from course_work.model_improvement_v2.e12_runner import (
    CHALLENGER_ID,
    CONTROL_ID,
    EXPECTED_COMMON_FINGERPRINT,
    EXPECTED_FOLD_FINGERPRINTS,
    EXPECTED_LOST_TARGET_FINGERPRINT,
    _expected_configs,
    _audit_resume_contract,
    build_e12_candidates,
    build_e12_run_context,
    evaluate_promotion,
    load_e12_config,
    main,
    project_root,
    run_preflight,
)
from course_work.rolling_origin.real_run import assert_context_invariants


def test_exact_lag144_definition_feature_order_and_rejected_blocks_absent():
    order = challenger_feature_order()
    assert LAG_FEATURE == "Appliances_lag_144"
    assert LAG_STEPS == 144
    assert LAG_DEFINITION == "y[s-144]"
    assert len(order) == 34
    assert order[-1] == LAG_FEATURE
    assert order[:-1] == tuple(get_feature_list(CONTROL_VARIANT))
    rejected = {
        "appliances_delta_1",
        "appliances_abs_delta_1",
        "appliances_delta_2",
        "appliances_roll_mean_3",
        "appliances_roll_mean_6",
        "appliances_roll_mean_12",
        "appliances_roll_std_6",
        "appliances_roll_std_12",
        "appliances_roll_max_12",
        "appliances_roll_min_12",
    }
    assert not rejected.intersection(order)


def test_lag144_is_exact_causal_alignment_without_off_by_one():
    y = np.arange(400.0)
    lag = derive_past_only_lag144_feature(y)
    assert np.isnan(lag[:144]).all()
    assert lag[144, 0] == y[0]
    assert lag[145, 0] == y[1]
    assert lag[215, 0] == y[71]
    assert lag[216, 0] == y[72]
    modified = y.copy()
    modified[216:] = -999999.0
    modified_lag = derive_past_only_lag144_feature(modified)
    np.testing.assert_array_equal(modified_lag[:360], lag[:360])


def test_common_population_lost_targets_folds_and_test_firewall():
    datasets, evidence, audit = build_e12_pretest_datasets(project_root())
    control = datasets[CONTROL_VARIANT]
    challenger = datasets[CHALLENGER_VARIANT]
    control_ids = control.window_records["target_id"].tolist()
    challenger_ids = challenger.window_records["target_id"].tolist()
    assert control_ids == challenger_ids
    assert len(control_ids) == 16558
    assert control_ids[0] == "TGT_00000216"
    assert FIRST_ELIGIBLE_TARGET_POSITION == 216
    assert int(challenger.window_records.iloc[0]["timeline_input_start"]) == 144
    assert int(challenger.window_records.iloc[0]["timeline_input_end"]) == 215
    assert int(challenger.window_records.iloc[0]["timeline_target"]) == 216
    assert challenger._feature_matrix[144, -1] == challenger._target_values[0]
    assert challenger._feature_matrix[215, -1] == challenger._target_values[71]
    assert audit.original_phase44_target_count == 16630
    assert audit.lost_target_count == 72
    assert audit.lost_target_ids[0] == "TGT_00000144"
    assert audit.lost_target_ids[-1] == "TGT_00000215"
    assert audit.lost_target_fingerprint == EXPECTED_LOST_TARGET_FINGERPRINT
    assert audit.common_population_fingerprint == EXPECTED_COMMON_FINGERPRINT
    assert {
        str(fold.fold_id): fold.fold_population_fingerprint for fold in evidence.folds
    } == EXPECTED_FOLD_FINGERPRINTS
    assert audit.common_ordered_target_ids is True
    assert audit.no_future_target_access is True
    assert audit.off_by_one_alignment_verified is True
    assert audit.control.test_rows_read == audit.challenger.test_rows_read == 0
    assert audit.control.test_target_ids_seen == audit.challenger.test_target_ids_seen == 0


def test_predecessor_decisions_incumbent_and_recovery_are_locked():
    report = run_preflight(project_root())
    assert report["accepted_incumbent"] == "E01_DIRECT_FS2_TF1"
    assert report["predecessor_audit"]["E10"]["decision"] == "REJECTED"
    assert report["predecessor_audit"]["E11"]["decision"] == "REJECTED"
    recovery = report["predecessor_audit"]["E11_RECOVERY"]
    assert recovery == {
        "status": "PASS",
        "canonical_completed_runs": 12,
        "stage_a_reused": 6,
        "stage_b_reused": 1,
        "stage_b_trained_during_recovery": 5,
        "failed_interrupted_evidence_preserved": True,
    }


def test_one_primary_change_and_e01_model_training_equivalence():
    root = project_root()
    document = load_e12_config(root)
    control, challenger = _expected_configs(root)
    changed = assert_one_primary_change("E12", control, challenger)
    assert changed == (
        "data.feature_count",
        "data.feature_variant_id",
        "lineage.feature_fingerprint",
        "lineage.scaler_bundle_checksum",
        "lineage.scaler_bundle_id",
        "model.input_size",
    )
    e01 = json.loads(
        (root / "artifacts/model_improvement_v2/experiments/E01/e01_config_snapshot.json").read_text()
    )["v1_config"]
    control_model = deepcopy(control["model"])
    challenger_model = deepcopy(challenger["model"])
    e01_model = deepcopy(e01["model"])
    assert control_model.pop("input_size") == e01_model.pop("input_size") == 33
    assert challenger_model.pop("input_size") == 34
    assert control_model == challenger_model == e01_model
    assert control["training"] == challenger["training"] == e01["training"]
    bad = deepcopy(challenger)
    bad["training"]["learning_rate"] = 1e-4
    with pytest.raises(ValueError):
        assert_one_primary_change("E12", control, bad)
    assert document["control"]["reuse_previous_metrics"] is False


def test_candidates_are_fresh_and_context_satisfies_e12_invariants():
    root = project_root()
    document = load_e12_config(root)
    candidates = build_e12_candidates(document, root)
    assert {candidate.candidate_id for candidate in candidates} == {CONTROL_ID, CHALLENGER_ID}
    assert all(item["training_required"] is True for item in (document["control"], document["challenger"]))
    assert all(
        item["initialization_policy"] == "FRESH_FROM_SEED_42_NO_CHECKPOINT_LOAD"
        for item in (document["control"], document["challenger"])
    )
    context = build_e12_run_context(root, document)
    assert context.reuse_completed_runs is False
    assert context.seed_before_model_construction is True
    assert context.apply_fold_x_scaling is True
    assert_context_invariants(context)


def test_registry_namespace_is_explicit_and_isolated(tmp_path: Path):
    assert "V2_E12" in V2_NAMESPACE_IDS
    registry = ExperimentRegistry(
        project_root=project_root(),
        registry_root=tmp_path / "registry",
        run_root=tmp_path / "runs",
        run_id_namespace="V2_E12",
        upstream_context_override={"lineage": {}, "feature_sets": {}, "window_fingerprints": {}},
    )
    run_id = registry.allocate_run_id(
        "TRANSFORMER_ENCODER", "ROLLING_ORIGIN", "abcdef012345", run_stage="RO1_A"
    )
    assert run_id.startswith("RUN_V2_TR_E12_RO1_A_")
    with pytest.raises(ValueError):
        ExperimentRegistry(
            project_root=project_root(),
            registry_root=tmp_path / "bad",
            run_root=tmp_path / "bad_runs",
            run_id_namespace="V2_E12_ARBITRARY",
        )


def test_registry_accepts_exact_e12_lag144_variant(tmp_path: Path):
    root = project_root()
    document = load_e12_config(root)
    _, challenger = _expected_configs(root)
    context = build_e12_run_context(root, document)
    registry = ExperimentRegistry(
        project_root=root,
        registry_root=tmp_path / "registry",
        run_root=tmp_path / "runs",
        run_id_namespace="V2_E12",
        upstream_context_override=context.registry_upstream_context,
    )
    challenger["lineage"]["rolling_origin_candidate_id"] = CHALLENGER_ID
    challenger["lineage"]["rolling_origin_fold_id"] = "RO1"
    challenger["lineage"]["rolling_origin_stage"] = "A"
    record = registry.register_run(
        challenger,
        experiment_family="ROLLING_ORIGIN",
        execution_type="ROBUSTNESS",
        candidate_id=CHALLENGER_ID,
        sweep_stage="RO1_A",
    )
    assert record["run_id"].startswith("RUN_V2_TR_E12_RO1_A_")


def test_promotion_requires_all_guardrails_and_human_decision():
    passed = evaluate_promotion(
        {"rmse_wh": 60.0, "mae_wh": 25.0},
        {"rmse_wh": 59.9, "mae_wh": 25.25},
        [59.0, 60.0, 61.0],
        [59.0, 60.0, 61.5],
    )
    assert passed["eligible_for_human_promotion"] is True
    assert passed["automatic_promotion"] is False
    assert passed["human_decision_required"] is True
    failed = evaluate_promotion(
        {"rmse_wh": 60.0, "mae_wh": 25.0},
        {"rmse_wh": 59.9, "mae_wh": 25.251},
        [59.0, 60.0, 61.0],
        [59.0, 60.0, 61.5],
    )
    assert failed["eligible_for_human_promotion"] is False


def test_preflight_and_official_human_gate():
    root = project_root()
    registry_path = root / "artifacts/model_improvement_v2/experiments/E12/registry/experiment_registry.jsonl"
    before = registry_path.read_bytes() if registry_path.exists() else None
    report = run_preflight(root)
    assert report["status"] == "PASS"
    assert report["expected_total_training_runs"] == 12
    assert report["training_candidates"] == [CONTROL_ID, CHALLENGER_ID]
    assert report["population"]["ordered_target_ids_identical"] is True
    assert report["population"]["test_rows_read"] == 0
    assert report["population"]["test_target_ids_seen"] == 0
    after = registry_path.read_bytes() if registry_path.exists() else None
    assert after == before
    assert main(["--experiment", "E12", "--mode", "official", "--seed", "42"]) == 3
    final = registry_path.read_bytes() if registry_path.exists() else None
    assert final == before
    assert main(["--experiment", "E12", "--mode", "resume-partial", "--seed", "42"]) == 3
    final = registry_path.read_bytes() if registry_path.exists() else None
    assert final == before


def test_current_e12_partial_state_is_checksum_locked_without_mutation():
    root = project_root()
    registry_path = root / "artifacts/model_improvement_v2/experiments/E12/registry/experiment_registry.jsonl"
    if not registry_path.exists():
        pytest.skip("E12 official execution has not created a partial lineage")
    before = registry_path.read_bytes()
    audit = _audit_resume_contract(root)
    assert audit["status"] == "PASS"
    assert audit["reused_completed_count"] >= 1
    assert audit["reused_completed_count"] + audit["training_run_count_required"] == 12
    assert audit["test_status"] == "NOT_ACCESSED"
    assert registry_path.read_bytes() == before
