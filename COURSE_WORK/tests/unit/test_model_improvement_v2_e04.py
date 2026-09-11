"""Focused static/unit tests for E04 residual LINEAR vs MLP head."""
from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from course_work.experiments.registry import V2_NAMESPACE_IDS
from course_work.model_improvement_v2.contracts import assert_one_primary_change
from course_work.model_improvement_v2.e04_runner import (
    EXPECTED_CONFIG_CHANGES,
    EXPECTED_GLOBAL_METRICS,
    EXPECTED_HEAD_SPEC,
    EXPECTED_LOCAL_METRICS,
    GLOBAL_PROMOTION_THRESHOLD_RMSE_WH,
    build_e04_candidate,
    build_e04_run_context,
    load_e04_config,
    main,
    run_preflight,
    validate_e04_document,
)
from course_work.model_improvement_v2.pretest_adapter import build_v2_pretest_dataset
from course_work.model_improvement_v2.residual import (
    compose_residual_prediction_raw,
    residual_raw_to_model,
)
from course_work.models.transformer_regressor import TransformerRegressor

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


def test_e04_preflight_reuses_locked_controls_and_changes_one_factor():
    audit = run_preflight(PROJECT_ROOT)
    assert audit["status"] == "PASS"
    assert audit["changed_config_fields"] == list(EXPECTED_CONFIG_CHANGES)
    assert audit["one_primary_change"] == "PASS"
    assert audit["local_control_reuse"] == "CONTROL_REUSED_FROM_E03"
    assert audit["local_control_audit"]["metrics"] == EXPECTED_LOCAL_METRICS
    assert audit["global_baseline_audit"]["metrics"] == EXPECTED_GLOBAL_METRICS
    assert audit["global_promotion_threshold_rmse_wh"] == GLOBAL_PROMOTION_THRESHOLD_RMSE_WH
    assert audit["head_spec"] == list(EXPECTED_HEAD_SPEC)
    assert audit["training_candidates"] == ["RESIDUAL_TO_PERSISTENCE_MLP"]
    assert audit["fresh_initialization"] is True
    assert audit["checkpoint_loaded"] is False
    assert audit["test_access"] == "NO"


def test_e04_config_diff_is_prediction_head_only():
    document = load_e04_config(PROJECT_ROOT)
    e03 = __import__("json").loads(
        (PROJECT_ROOT / "artifacts/model_improvement_v2/experiments/E03/e03_config_snapshot.json").read_text()
    )
    changed = assert_one_primary_change("E04", e03["challenger"]["config"], document["challenger"]["config"])
    assert changed == ("model.prediction_head",)
    bad = deepcopy(document["challenger"]["config"])
    bad["training"]["learning_rate"] = 1e-4
    with pytest.raises(ValueError, match="outside its primary factor"):
        assert_one_primary_change("E04", e03["challenger"]["config"], bad)


def test_e04_mlp_head_is_exact_approved_architecture():
    config = load_e04_config(PROJECT_ROOT)["challenger"]["config"]["model"]
    model = TransformerRegressor(config)
    assert model.prediction_head == "MLP"
    assert isinstance(model.head, nn.Sequential)
    assert len(model.head) == 4
    assert isinstance(model.head[0], nn.Linear)
    assert (model.head[0].in_features, model.head[0].out_features) == (64, 64)
    assert isinstance(model.head[1], nn.GELU)
    assert isinstance(model.head[2], nn.Dropout) and model.head[2].p == 0.1
    assert isinstance(model.head[3], nn.Linear)
    assert (model.head[3].in_features, model.head[3].out_features) == (64, 1)
    assert model.checkpoint_metadata()["config"]["prediction_head"] == "MLP"


def test_v1_and_e03_default_linear_construction_is_unchanged():
    model_config = deepcopy(load_e04_config(PROJECT_ROOT)["challenger"]["config"]["model"])
    model_config.pop("prediction_head")
    torch.manual_seed(42)
    implicit = TransformerRegressor(model_config)
    torch.manual_seed(42)
    explicit = TransformerRegressor({**model_config, "prediction_head": "LINEAR"})
    assert implicit.prediction_head == explicit.prediction_head == "LINEAR"
    assert isinstance(implicit.head, nn.Linear)
    assert isinstance(explicit.head, nn.Linear)
    assert implicit.config.to_dict() == explicit.config.to_dict()
    assert "prediction_head" not in implicit.config.to_dict()
    assert "prediction_head" not in implicit.checkpoint_metadata()["config"]
    assert tuple(implicit.state_dict()) == tuple(explicit.state_dict())
    for name, value in implicit.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[name])


def test_e04_residual_math_remains_identical():
    class Scaler:
        y_std = 20.0
    delta_model = residual_raw_to_model(10.0, Scaler(), "YS1")
    assert delta_model == 0.5
    assert compose_residual_prediction_raw(100.0, delta_model, Scaler(), "YS1") == 110.0


def test_e04_namespace_and_single_fresh_candidate():
    document = load_e04_config(PROJECT_ROOT)
    candidate = build_e04_candidate(document)
    context = build_e04_run_context(PROJECT_ROOT, document)
    assert "V2_E04" in V2_NAMESPACE_IDS
    assert context.registry_namespace == "V2_E04"
    assert "/experiments/E04" in context.artifact_dir.as_posix()
    assert "/experiments/E04" in context.registry_root.as_posix()
    assert "/experiments/E04" in context.run_root.as_posix()
    assert context.reuse_completed_runs is False
    assert context.seed_before_model_construction is True
    assert context.candidate_specs == (candidate,)
    assert candidate.config["model"]["prediction_head"] == "MLP"


def test_e04_pretest_adapter_firewall_and_population():
    document = load_e04_config(PROJECT_ROOT)
    dataset, _evidence, audit = build_v2_pretest_dataset(
        PROJECT_ROOT, document["challenger"]["feature_order"],
        experiment_id="E04", feature_variant_id="FS2_TF1",
    )
    assert len(dataset) > 0
    assert audit.test_rows_read == 0
    assert audit.test_target_ids_seen == 0
    assert audit.population_status == "PASS"
    assert set(audit.fold_fingerprint_status.values()) == {"PASS"}


def test_e04_rejects_test_access():
    document = deepcopy(load_e04_config(PROJECT_ROOT))
    document["challenger"]["config"]["data"]["target_access_mode"] = "TEST"
    with pytest.raises(PermissionError):
        validate_e04_document(document, PROJECT_ROOT)


def test_e04_official_refuses_without_authorization(monkeypatch):
    invoked = False
    def forbidden(*_args, **_kwargs):
        nonlocal invoked
        invoked = True
        raise AssertionError("official execution must not be reached")
    monkeypatch.setattr("course_work.model_improvement_v2.e04_runner.run_official", forbidden)
    assert main(["--experiment", "E04", "--mode", "official", "--seed", "42"]) == 3
    assert invoked is False
