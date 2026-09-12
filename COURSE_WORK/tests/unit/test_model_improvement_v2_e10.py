from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from course_work.experiments.registry import ExperimentRegistry, V2_NAMESPACE_IDS
from course_work.model_improvement_v2.contracts import assert_one_primary_change
from course_work.model_improvement_v2.e10_pretest import (
    CHALLENGER_VARIANT,
    CONTROL_VARIANT,
    DELTA_FEATURES,
    build_e10_pretest_datasets,
    challenger_feature_order,
    derive_causal_target_deltas,
)
from course_work.model_improvement_v2.e10_runner import (
    CHALLENGER_ID,
    CONTROL_ID,
    build_e10_candidates,
    evaluate_promotion,
    load_e10_config,
    main,
    project_root,
    run_preflight,
)


def test_exact_delta_definitions_and_no_future_leakage():
    y = np.array([10.0, 13.0, 20.0, 99.0])
    delta = derive_causal_target_deltas(y)
    assert np.isnan(delta[0]).all()
    np.testing.assert_allclose(delta[1], [3.0, 3.0, np.nan], equal_nan=True)
    np.testing.assert_allclose(delta[2], [7.0, 7.0, 10.0])
    modified = y.copy(); modified[3] = -10000.0
    np.testing.assert_allclose(derive_causal_target_deltas(modified)[:3], delta[:3], equal_nan=True)
    assert DELTA_FEATURES == ("appliances_delta_1", "appliances_abs_delta_1", "appliances_delta_2")


def test_feature_order_and_one_primary_change():
    root = project_root(); document = load_e10_config(root)
    assert tuple(document["control"]["feature_order"]) == challenger_feature_order()[:-3]
    assert tuple(document["challenger"]["feature_order"]) == challenger_feature_order()
    assert document["control"]["feature_count"] == 33
    assert document["challenger"]["feature_count"] == 36
    changed = assert_one_primary_change("E10", document["control"]["config"], document["challenger"]["config"])
    assert changed
    bad = deepcopy(document["challenger"]["config"]); bad["training"]["learning_rate"] = 1e-4
    with pytest.raises(ValueError): assert_one_primary_change("E10", document["control"]["config"], bad)


def test_common_population_fold_evidence_and_test_firewall():
    datasets, evidence, audit = build_e10_pretest_datasets(project_root())
    assert set(datasets) == {CONTROL_VARIANT, CHALLENGER_VARIANT}
    assert datasets[CONTROL_VARIANT].window_records["target_id"].tolist() == datasets[CHALLENGER_VARIANT].window_records["target_id"].tolist()
    assert all(value == "PASS" for value in evidence.fingerprint_status.values())
    assert audit.control.test_rows_read == audit.challenger.test_rows_read == 0
    assert audit.control.test_target_ids_seen == audit.challenger.test_target_ids_seen == 0
    assert datasets[CONTROL_VARIANT]._feature_matrix.shape[1] == 33
    assert datasets[CHALLENGER_VARIANT]._feature_matrix.shape[1] == 36


def test_candidate_pair_is_fresh_and_e01_training_is_unchanged():
    document = load_e10_config(project_root()); specs = build_e10_candidates(document)
    assert {s.candidate_id for s in specs} == {CONTROL_ID, CHALLENGER_ID}
    assert document["control"]["training_required"] is True
    assert document["control"]["reuse_e01_metrics"] is False
    assert document["control"]["config"]["training"] == document["challenger"]["config"]["training"]
    assert document["control"]["config"]["model"] | {"input_size": 36} == document["challenger"]["config"]["model"]


def test_registry_namespace_is_explicit_and_isolated(tmp_path: Path):
    assert "V2_E10" in V2_NAMESPACE_IDS
    registry = ExperimentRegistry(project_root=project_root(), registry_root=tmp_path/"registry", run_root=tmp_path/"runs", run_id_namespace="V2_E10", upstream_context_override={"lineage":{},"feature_sets":{},"window_fingerprints":{}})
    run_id = registry.allocate_run_id("TRANSFORMER_ENCODER", "ROLLING_ORIGIN", "abcdef012345", run_stage="RO1_A")
    assert run_id.startswith("RUN_V2_TR_E10_RO1_A_")
    with pytest.raises(ValueError): ExperimentRegistry(project_root=project_root(), registry_root=tmp_path/"bad", run_root=tmp_path/"bad_runs", run_id_namespace="V2_ANY")


def test_promotion_requires_all_four_guards():
    control={"rmse_wh":60.0,"mae_wh":25.0}; challenger={"rmse_wh":59.9,"mae_wh":25.25}
    passed=evaluate_promotion(control,challenger,[59.0,60.0,61.0],[59.0,60.0,61.5])
    assert passed["eligible_for_human_promotion"] is True and passed["automatic_promotion"] is False
    failed=evaluate_promotion(control,{"rmse_wh":59.9,"mae_wh":25.251},[59,60,61],[59,60,61.5])
    assert failed["eligible_for_human_promotion"] is False


def test_preflight_and_official_human_gate():
    report=run_preflight(); assert report["status"] == "PASS"
    assert report["expected_total_training_runs"] == 12
    assert report["training_candidates"] == [CONTROL_ID,CHALLENGER_ID]
    assert main(["--experiment","E10","--mode","official","--seed","42"]) == 3
