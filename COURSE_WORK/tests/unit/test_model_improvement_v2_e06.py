"""Focused static/unit tests for E06 constant AdamW LR audit."""
from __future__ import annotations

from copy import deepcopy

import pytest

from course_work.experiments.registry import V2_NAMESPACE_IDS
from course_work.model_improvement_v2.contracts import assert_one_primary_change
from course_work.model_improvement_v2.e06_runner import (
    ALL_LRS,
    CANDIDATE_IDS,
    CHALLENGER_LRS,
    CONTROL_LR,
    EXPECTED_CONFIG_CHANGES,
    EXPECTED_CONTROL_METRICS,
    PROMOTION_THRESHOLD_RMSE_WH,
    build_e06_candidates,
    build_e06_run_context,
    build_e06_resume_context,
    load_e06_config,
    main,
    run_preflight,
    validate_e06_document,
)
from course_work.model_improvement_v2.pretest_adapter import build_v2_pretest_dataset
from course_work.rolling_origin.real_run import assert_context_invariants
from course_work.rolling_origin.stages import evaluate_stage_c

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


def _e01_config():
    return __import__("json").loads(
        (PROJECT_ROOT / "artifacts/model_improvement_v2/experiments/E01/e01_config_snapshot.json").read_text()
    )["v1_config"]


def test_e06_preflight_has_three_challengers_and_reused_control():
    audit = run_preflight(PROJECT_ROOT)
    assert audit["status"] == "PASS"
    assert audit["control_learning_rate"] == CONTROL_LR == 3e-4
    assert audit["control_reuse"] == "CONTROL_REUSED_FROM_E01"
    assert audit["control_audit"]["metrics"] == EXPECTED_CONTROL_METRICS
    assert audit["control_audit"]["run_count"] == 6
    assert audit["training_candidate_count"] == 3
    assert audit["training_candidates"] == list(CHALLENGER_LRS)
    assert CONTROL_LR not in audit["training_candidates"]
    assert audit["candidate_ids"] == [CANDIDATE_IDS[lr] for lr in CHALLENGER_LRS]
    assert audit["changed_config_fields"] == list(EXPECTED_CONFIG_CHANGES)
    assert audit["scheduler"] == "OFF"
    assert audit["promotion_threshold_rmse_wh"] == PROMOTION_THRESHOLD_RMSE_WH
    assert audit["test_access"] == "NO"
    assert audit["checkpoint_loaded"] is False


def test_each_e06_challenger_changes_learning_rate_only():
    document = load_e06_config(PROJECT_ROOT)
    baseline = _e01_config()
    assert document["primary_change"]["candidate_values"] == list(ALL_LRS)
    for item in document["training_candidates"]:
        config = item["config"]
        assert assert_one_primary_change("E06", baseline, config) == ("training.learning_rate",)
        assert config["training"]["learning_rate"] == item["learning_rate"]
        assert config["training"]["optimizer_name"] == "AdamW"
        assert config["training"]["weight_decay"] == 1e-3
        assert config["training"]["scheduler_name"] is None
        assert config["training"]["scheduler_config"] is None
        assert config["model"].get("prediction_formulation", "DIRECT") == "DIRECT"
        assert config["data"]["feature_variant_id"] == "FS2_TF1"
        assert config["data"]["lookback_steps"] == 72


def test_e06_rejects_any_second_scientific_change():
    document = load_e06_config(PROJECT_ROOT)
    baseline = _e01_config()
    bad = deepcopy(document["training_candidates"][0]["config"])
    bad["training"]["weight_decay"] = 0.0
    with pytest.raises(ValueError, match="outside its primary factor"):
        assert_one_primary_change("E06", baseline, bad)


def test_e06_context_is_three_fresh_candidates_in_isolated_namespace():
    document = load_e06_config(PROJECT_ROOT)
    candidates = build_e06_candidates(document)
    context = build_e06_run_context(PROJECT_ROOT, document)
    assert len(candidates) == len(context.candidate_specs) == 3
    assert {candidate.candidate_id for candidate in candidates} == set(CANDIDATE_IDS.values())
    assert {candidate.config["training"]["learning_rate"] for candidate in candidates} == set(CHALLENGER_LRS)
    assert context.registry_namespace == "V2_E06"
    assert "V2_E06" in V2_NAMESPACE_IDS
    assert all("/experiments/E06" in path.as_posix() for path in (context.artifact_dir, context.registry_root, context.run_root))
    assert context.reuse_completed_runs is False
    assert context.seed_before_model_construction is True
    assert_context_invariants(context)


def test_e06_dataset_factory_reuses_one_pretest_dataset_for_all_lrs():
    document = load_e06_config(PROJECT_ROOT)
    context = build_e06_run_context(PROJECT_ROOT, document)
    first = context.dataset_factory(context.candidate_specs[0], None)
    second = context.dataset_factory(context.candidate_specs[1], None)
    third = context.dataset_factory(context.candidate_specs[2], None)
    assert first is second is third


def test_e06_pretest_adapter_firewall_and_population():
    document = load_e06_config(PROJECT_ROOT)
    dataset, _evidence, audit = build_v2_pretest_dataset(
        PROJECT_ROOT, document["feature_order"], experiment_id="E06", feature_variant_id="FS2_TF1",
    )
    assert len(dataset) > 0
    assert audit.test_rows_read == 0
    assert audit.test_target_ids_seen == 0
    assert audit.population_status == "PASS"
    assert set(audit.fold_fingerprint_status.values()) == {"PASS"}


def test_e06_rejects_test_access():
    document = deepcopy(load_e06_config(PROJECT_ROOT))
    document["training_candidates"][0]["config"]["data"]["target_access_mode"] = "TEST"
    with pytest.raises(PermissionError):
        validate_e06_document(document, PROJECT_ROOT)


def test_e06_rejects_training_control_or_wrong_candidate_count():
    document = deepcopy(load_e06_config(PROJECT_ROOT))
    document["training_candidates"].append(deepcopy(document["training_candidates"][0]))
    with pytest.raises(Exception, match="exactly three"):
        validate_e06_document(document, PROJECT_ROOT)


def test_e06_official_refuses_without_authorization(monkeypatch):
    invoked = False
    def forbidden(*_args, **_kwargs):
        nonlocal invoked
        invoked = True
        raise AssertionError("official execution must not be reached")
    monkeypatch.setattr("course_work.model_improvement_v2.e06_runner.run_official", forbidden)
    assert main(["--experiment", "E06", "--mode", "official", "--seed", "42"]) == 3
    assert invoked is False


class _DirectColumnModel(__import__("torch").nn.Module):
    def forward(self, x):
        values = __import__("torch").arange(1, x.shape[0] + 1, dtype=x.dtype)
        return values.reshape(-1, 1)


class _ColumnPreservingScaler:
    target_scaling_option = "YS1"
    def inverse_transform_y(self, values):
        return values * 10.0 + 5.0


def test_e06_stage_c_flattens_direct_predictions_without_value_change():
    import numpy as np
    import torch
    batch = {
        "x": torch.zeros(2, 72, 33),
        "y_raw_wh": torch.tensor([[15.0], [25.0]]),
        "target_id": ["a", "b"],
        "target_timestamp": ["t1", "t2"],
    }
    result = evaluate_stage_c(
        model=_DirectColumnModel(), outer_eval_loader=[batch],
        scaler_bundle=_ColumnPreservingScaler(), device=torch.device("cpu"),
        model_id="E06_SHAPE_TEST", fold_id="RO1", model_run_id="RUN_TEST",
        refit_epoch=1, prediction_formulation="DIRECT",
        flatten_metric_predictions=True,
    )
    assert result.y_true_wh.shape == result.y_pred_wh.shape == (2,)
    assert np.array_equal(result.y_pred_wh, np.array([15.0, 25.0]))
    assert np.array_equal(result.residuals_wh, np.zeros(2))


def test_stage_c_shape_fix_is_opt_in_so_v1_e01_e02_default_is_unchanged():
    import inspect
    parameter = inspect.signature(evaluate_stage_c).parameters["flatten_metric_predictions"]
    assert parameter.default is False


def test_e06_resume_context_locks_all_completed_runs_without_new_ids():
    document = load_e06_config(PROJECT_ROOT)
    before = (PROJECT_ROOT / "artifacts/model_improvement_v2/experiments/E06/registry/experiment_registry.jsonl").read_bytes()
    context = build_e06_resume_context(PROJECT_ROOT, document)
    after = (PROJECT_ROOT / "artifacts/model_improvement_v2/experiments/E06/registry/experiment_registry.jsonl").read_bytes()
    assert before == after
    assert context.reuse_completed_runs is True
    assert context.reuse_completed_run_ids == document["resume_stage_c"]["locked_run_ids"]
    assert len(context.reuse_completed_run_ids) == 18
    assert_context_invariants(context)


def test_e06_preflight_rejects_checkpoint_hash_tampering():
    document = deepcopy(load_e06_config(PROJECT_ROOT))
    key = next(iter(document["resume_stage_c"]["stage_b_checkpoint_sha256"]))
    document["resume_stage_c"]["stage_b_checkpoint_sha256"][key] = "0" * 64
    with pytest.raises(Exception, match="checkpoint checksum mismatch"):
        validate_e06_document(document, PROJECT_ROOT)


def test_e06_resume_cli_routes_without_training_authorization(monkeypatch):
    invoked = False
    def fake_resume(*_args, **_kwargs):
        nonlocal invoked
        invoked = True
        return 0
    monkeypatch.setattr("course_work.model_improvement_v2.e06_runner.run_resume_stage_c", fake_resume)
    assert main(["--experiment", "E06", "--mode", "resume-stage-c", "--seed", "42"]) == 0
    assert invoked is True
    invoked = False
    assert main(["--experiment", "E06", "--mode", "resume-stage-c", "--seed", "42", "--authorize-training"]) == 2
    assert invoked is False
