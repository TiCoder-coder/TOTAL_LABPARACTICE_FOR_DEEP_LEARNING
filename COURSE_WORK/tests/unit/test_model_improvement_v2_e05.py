"""Focused static/unit tests for E05 residual gate OFF vs ON."""
from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from course_work.experiments.registry import V2_NAMESPACE_IDS
from course_work.model_improvement_v2.contracts import assert_one_primary_change
from course_work.model_improvement_v2.e05_runner import (
    EXPECTED_CONFIG_CHANGES,
    EXPECTED_GATE_SPEC,
    EXPECTED_GLOBAL_METRICS,
    EXPECTED_LOCAL_METRICS,
    GLOBAL_PROMOTION_THRESHOLD_RMSE_WH,
    build_e05_candidate,
    build_e05_run_context,
    load_e05_config,
    main,
    run_preflight,
    validate_e05_document,
)
from course_work.model_improvement_v2.pretest_adapter import build_v2_pretest_dataset
from course_work.model_improvement_v2.residual import compose_residual_prediction_raw
from course_work.models.transformer_regressor import TransformerRegressor

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


def test_e05_preflight_reuses_e03_and_rejects_e04_carry_forward():
    audit = run_preflight(PROJECT_ROOT)
    assert audit["status"] == "PASS"
    assert audit["changed_config_fields"] == list(EXPECTED_CONFIG_CHANGES)
    assert audit["one_primary_change"] == "PASS"
    assert audit["control_gate"] == "OFF" and audit["challenger_gate"] == "ON"
    assert audit["prediction_head"] == "LINEAR"
    assert audit["gate_spec"] == EXPECTED_GATE_SPEC
    assert audit["local_control_reuse"] == "CONTROL_REUSED_FROM_E03"
    assert audit["local_control_audit"]["metrics"] == EXPECTED_LOCAL_METRICS
    assert audit["global_baseline_audit"]["metrics"] == EXPECTED_GLOBAL_METRICS
    assert audit["rejected_e04_audit"] == {
        "status": "PASS", "rmse_wh": 60.66433497520889, "carried_forward": False,
    }
    assert audit["global_promotion_threshold_rmse_wh"] == GLOBAL_PROMOTION_THRESHOLD_RMSE_WH
    assert audit["training_candidates"] == ["RESIDUAL_TO_PERSISTENCE_LINEAR_GATE_ON"]
    assert audit["fresh_initialization"] is True
    assert audit["deterministic_gate_initialization"] is True
    assert audit["gate_initialization"] == {
        "weight": 0.0, "bias": 0.0, "initial_gate": 0.5,
    }
    assert audit["checkpoint_loaded"] is False
    assert audit["test_access"] == "NO"


def test_e05_config_diff_is_residual_gate_only():
    document = load_e05_config(PROJECT_ROOT)
    e03 = __import__("json").loads(
        (PROJECT_ROOT / "artifacts/model_improvement_v2/experiments/E03/e03_config_snapshot.json").read_text()
    )
    control = e03["challenger"]["config"]
    challenger = document["challenger"]["config"]
    assert assert_one_primary_change("E05", control, challenger) == ("model.residual_gate",)
    assert challenger["model"].get("prediction_head", "LINEAR") == "LINEAR"
    bad = deepcopy(challenger)
    bad["model"]["prediction_head"] = "MLP"
    with pytest.raises(ValueError, match="outside its primary factor"):
        assert_one_primary_change("E05", control, bad)


def test_gate_uses_pooled_representation_and_exact_formula():
    config = load_e05_config(PROJECT_ROOT)["challenger"]["config"]["model"]
    model = TransformerRegressor(config)
    assert model.prediction_head == "LINEAR"
    assert model.residual_gate == "ON"
    assert isinstance(model.head, nn.Linear)
    assert isinstance(model.residual_gate_head, nn.Linear)
    assert (model.residual_gate_head.in_features, model.residual_gate_head.out_features) == (64, 1)
    assert torch.count_nonzero(model.residual_gate_head.weight).item() == 0
    assert torch.count_nonzero(model.residual_gate_head.bias).item() == 0
    assert torch.equal(
        torch.sigmoid(model.residual_gate_head.bias.detach()),
        torch.full_like(model.residual_gate_head.bias, 0.5),
    )
    with torch.no_grad():
        model.head.weight.zero_(); model.head.bias.fill_(2.0)
    pooled = torch.zeros(3, 64)
    gated_delta_model = model._predict_from_pooled(pooled)
    assert torch.equal(gated_delta_model, torch.ones(3, 1))  # sigmoid(0) * 2 == 1
    class Scaler:
        y_std = 20.0
    y_hat = compose_residual_prediction_raw(100.0, float(gated_delta_model[0, 0].detach()), Scaler(), "YS1")
    assert y_hat == 120.0


def test_gate_parameters_are_explicitly_zero_initialized_and_deterministic():
    config = load_e05_config(PROJECT_ROOT)["challenger"]["config"]["model"]
    torch.manual_seed(42); first = TransformerRegressor(config)
    torch.manual_seed(42); second = TransformerRegressor(config)
    assert torch.count_nonzero(first.residual_gate_head.weight).item() == 0
    assert torch.count_nonzero(first.residual_gate_head.bias).item() == 0
    assert torch.equal(
        torch.sigmoid(first.residual_gate_head.bias.detach()),
        torch.tensor([0.5]),
    )
    assert tuple(first.state_dict()) == tuple(second.state_dict())
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])
    assert first.checkpoint_metadata()["config"]["residual_gate"] == "ON"


def test_v1_e03_default_gate_off_is_unchanged():
    config = deepcopy(load_e05_config(PROJECT_ROOT)["challenger"]["config"]["model"])
    config.pop("residual_gate")
    torch.manual_seed(42); implicit = TransformerRegressor(config)
    torch.manual_seed(42); explicit = TransformerRegressor({**config, "residual_gate": "OFF"})
    assert implicit.residual_gate == explicit.residual_gate == "OFF"
    assert not hasattr(implicit, "residual_gate_head")
    assert not hasattr(explicit, "residual_gate_head")
    assert tuple(implicit.state_dict()) == tuple(explicit.state_dict())
    for name, value in implicit.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[name])
    assert "residual_gate" not in implicit.checkpoint_metadata()["config"]


def test_e05_namespace_and_single_fresh_candidate():
    document = load_e05_config(PROJECT_ROOT)
    candidate = build_e05_candidate(document)
    context = build_e05_run_context(PROJECT_ROOT, document)
    assert "V2_E05" in V2_NAMESPACE_IDS
    assert context.registry_namespace == "V2_E05"
    assert all("/experiments/E05" in path.as_posix() for path in (
        context.artifact_dir, context.registry_root, context.run_root,
    ))
    assert context.reuse_completed_runs is False
    assert context.seed_before_model_construction is True
    assert context.candidate_specs == (candidate,)
    assert candidate.config["model"]["residual_gate"] == "ON"
    assert candidate.config["model"].get("prediction_head", "LINEAR") == "LINEAR"


def test_e05_pretest_adapter_firewall_and_population():
    document = load_e05_config(PROJECT_ROOT)
    dataset, _evidence, audit = build_v2_pretest_dataset(
        PROJECT_ROOT, document["challenger"]["feature_order"],
        experiment_id="E05", feature_variant_id="FS2_TF1",
    )
    assert len(dataset) > 0
    assert audit.test_rows_read == 0
    assert audit.test_target_ids_seen == 0
    assert audit.population_status == "PASS"
    assert set(audit.fold_fingerprint_status.values()) == {"PASS"}


def test_e05_rejects_test_access():
    document = deepcopy(load_e05_config(PROJECT_ROOT))
    document["challenger"]["config"]["data"]["target_access_mode"] = "TEST"
    with pytest.raises(PermissionError):
        validate_e05_document(document, PROJECT_ROOT)


def test_e05_official_refuses_without_authorization(monkeypatch):
    invoked = False
    def forbidden(*_args, **_kwargs):
        nonlocal invoked
        invoked = True
        raise AssertionError("official execution must not be reached")
    monkeypatch.setattr("course_work.model_improvement_v2.e05_runner.run_official", forbidden)
    assert main(["--experiment", "E05", "--mode", "official", "--seed", "42"]) == 3
    assert invoked is False
