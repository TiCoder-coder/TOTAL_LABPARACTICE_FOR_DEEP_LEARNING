from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from course_work.model_improvement_v2.e09_runner import (
    CANDIDATE_IDS,
    CHALLENGER_GRID,
    EXECUTION_TRACK,
    OPTIMIZER_CONFIG,
    PROMOTION_THRESHOLD_RMSE_WH,
    REGISTRY_NAMESPACE,
    _write_success_contracts,
    build_e09_candidates,
    build_e09_run_context,
    load_e09_config,
    main,
    project_root,
    run_preflight,
    validate_e09_document,
)
from course_work.model_improvement_v2.optimizer import build_optimizer
from course_work.rolling_origin.real_run import (
    V2_DIRECT_METRIC_FLATTEN_TRACKS,
    V2_SHARED_PRETEST_CACHE_KEYS,
    assert_context_invariants,
)
from course_work.rolling_origin.stages import evaluate_stage_c


def test_e09_preflight_locks_exact_six_candidate_grid():
    result = run_preflight()
    assert result["status"] == "PASS"
    assert result["training_candidate_count"] == 6
    assert result["candidate_grid"] == [
        {"learning_rate": lr, "weight_decay": wd} for lr, wd in CHALLENGER_GRID
    ]
    assert set(result["candidate_ids"]) == set(CANDIDATE_IDS.values())
    assert result["optimizer_config"] == {"momentum": 0.9, "nesterov": False}
    assert result["scheduler"] == "OFF"
    assert result["expected_stage_a_runs"] == 18
    assert result["expected_stage_b_runs"] == 18
    assert result["test_access"] == "NO"
    assert result["training_executed"] is False


def test_optimizer_factory_preserves_adamw_e08_sgd_and_supports_e09_sgdm():
    adamw = build_optimizer(torch.nn.Linear(1, 1).parameters(), None, 3e-4, 1e-3)
    assert isinstance(adamw, torch.optim.AdamW)
    e08 = build_optimizer(
        torch.nn.Linear(1, 1).parameters(), "SGD", 3e-3, 1e-3,
        {"momentum": 0.0, "nesterov": False},
    )
    assert isinstance(e08, torch.optim.SGD)
    assert e08.param_groups[0]["momentum"] == 0.0
    e09 = build_optimizer(
        torch.nn.Linear(1, 1).parameters(), "SGD", 1e-2, 1e-4,
        OPTIMIZER_CONFIG,
    )
    group = e09.param_groups[0]
    assert group["lr"] == 1e-2
    assert group["weight_decay"] == 1e-4
    assert group["momentum"] == 0.9
    assert group["nesterov"] is False


@pytest.mark.parametrize(
    "config",
    [
        {"momentum": 0.8, "nesterov": False},
        {"momentum": 0.9, "nesterov": True},
        {"momentum": 0.9},
    ],
)
def test_optimizer_factory_rejects_unapproved_sgd_config(config):
    with pytest.raises(ValueError):
        build_optimizer(torch.nn.Linear(1, 1).parameters(), "SGD", 1e-3, 0.0, config)


def test_e09_candidates_are_exact_and_fresh():
    document = load_e09_config(project_root())
    specs = build_e09_candidates(document)
    assert len(specs) == 6
    assert len({spec.candidate_id for spec in specs}) == 6
    observed = set()
    for spec in specs:
        training = spec.config["training"]
        observed.add((training["learning_rate"], training["weight_decay"]))
        assert training["optimizer_name"] == "SGD"
        assert training["optimizer_config"] == OPTIMIZER_CONFIG
        assert training["scheduler_name"] is None
        assert training["scheduler_config"] is None
        assert spec.config["model"].get("prediction_formulation", "DIRECT") == "DIRECT"
    assert observed == set(CHALLENGER_GRID)


def test_e09_rejects_missing_duplicate_and_out_of_factor_candidates():
    root = project_root()
    document = load_e09_config(root)
    document["training_candidates"] = document["training_candidates"][:-1]
    with pytest.raises(RuntimeError, match="exactly six"):
        validate_e09_document(document, root)

    document = load_e09_config(root)
    document["training_candidates"][1] = document["training_candidates"][0]
    with pytest.raises(RuntimeError):
        validate_e09_document(document, root)

    document = load_e09_config(root)
    document["training_candidates"][0]["config"]["model"]["dropout"] = 0.2
    with pytest.raises(RuntimeError, match="config mismatch"):
        validate_e09_document(document, root)


def test_e09_context_invariant_and_isolation():
    root = project_root()
    context = build_e09_run_context(root, load_e09_config(root))
    assert context.registry_namespace == REGISTRY_NAMESPACE
    assert context.execution_track == EXECUTION_TRACK
    assert len(context.candidate_specs) == 6
    assert all(spec.candidate_id != "TR_C2_ALT_LOOKBACK" for spec in context.candidate_specs)
    assert context.enable_stage_b_lr_replay is False
    assert context.seed_before_model_construction is True
    assert context.reuse_completed_runs is False
    assert_context_invariants(context)


def test_e09_registry_namespace_and_run_ids(tmp_path):
    from course_work.experiments.registry import ExperimentRegistry, V2_NAMESPACE_IDS

    assert "V2_E08" in V2_NAMESPACE_IDS
    assert "V2_E09" in V2_NAMESPACE_IDS
    registry = ExperimentRegistry(
        project_root=tmp_path,
        registry_root=tmp_path / "registry",
        run_root=tmp_path / "runs",
        run_id_namespace="V2_E09",
        upstream_context_override={"lineage": {}, "feature_sets": {}},
    )
    run_id = registry.allocate_run_id(
        "TRANSFORMER_ENCODER", "ROLLING_ORIGIN", "abcd1234" * 8,
        run_stage="RO1_A",
    )
    assert run_id == "RUN_V2_TR_E09_RO1_A_0001_ABCD1234"


def test_e09_projection_and_shared_dataset_have_no_test_rows():
    from course_work.model_improvement_v2.pretest_adapter import build_v2_pretest_dataset

    root = project_root()
    document = load_e09_config(root)
    dataset, evidence, audit = build_v2_pretest_dataset(
        root, document["feature_order"], experiment_id="E09", feature_variant_id="FS2_TF1"
    )
    assert dataset.window_records["target_id"].tolist() == list(evidence.train_ids + evidence.validation_ids)
    assert audit.test_rows_read == audit.test_target_ids_seen == 0
    assert audit.feature_order_status == audit.population_status == "PASS"

    context = build_e09_run_context(root, document)
    datasets = [context.dataset_factory(candidate, None) for candidate in context.candidate_specs]
    assert len({id(value) for value in datasets}) == 1
    assert V2_SHARED_PRETEST_CACHE_KEYS[EXECUTION_TRACK] == "MODEL_IMPROVEMENT_V2_E09_SHARED_PRETEST"


def test_e09_rejects_unapproved_projection():
    from course_work.model_improvement_v2.pretest_adapter import build_v2_pretest_dataset

    root = project_root()
    document = load_e09_config(root)
    with pytest.raises(RuntimeError, match="Unsupported V2 pre-Test projection"):
        build_v2_pretest_dataset(
            root, document["feature_order"], experiment_id="E09", feature_variant_id="FS1_TF1"
        )


def test_e09_stage_a_b_are_fresh_constant_lr_without_replay():
    from course_work.training.engine import TrainingEngine
    from course_work.rolling_origin.refit_engine import RefitEngine
    import inspect

    assert "build_optimizer" in inspect.getsource(TrainingEngine.train)
    assert "build_optimizer" in inspect.getsource(RefitEngine.refit)
    context = build_e09_run_context(project_root(), load_e09_config(project_root()))
    assert context.enable_stage_b_lr_replay is False
    candidate = context.candidate_specs[0]
    training = candidate.config["training"]
    first = build_optimizer(torch.nn.Linear(1, 1).parameters(), "SGD", training["learning_rate"], training["weight_decay"], training["optimizer_config"])
    second = build_optimizer(torch.nn.Linear(1, 1).parameters(), "SGD", training["learning_rate"], training["weight_decay"], training["optimizer_config"])
    assert first is not second
    assert first.param_groups[0]["lr"] == training["learning_rate"]


class _ColumnModel(torch.nn.Module):
    def forward(self, x):
        return torch.arange(1, x.shape[0] + 1, dtype=x.dtype).reshape(-1, 1)


class _Scaler:
    target_scaling_option = "YS1"
    def inverse_transform_y(self, values):
        return values * 10.0 + 5.0


def test_e09_stage_c_direct_prediction_is_one_dimensional():
    assert V2_DIRECT_METRIC_FLATTEN_TRACKS == frozenset({
        "MODEL_IMPROVEMENT_V2_E06", "MODEL_IMPROVEMENT_V2_E07",
        "MODEL_IMPROVEMENT_V2_E08", "MODEL_IMPROVEMENT_V2_E09",
    })
    result = evaluate_stage_c(
        model=_ColumnModel(),
        outer_eval_loader=[{
            "x": torch.zeros(2, 72, 33),
            "y_raw_wh": torch.tensor([[15.0], [25.0]]),
            "target_id": ["a", "b"],
            "target_timestamp": ["t1", "t2"],
        }],
        scaler_bundle=_Scaler(),
        device=torch.device("cpu"),
        model_id="E09_SHAPE_TEST",
        fold_id="RO1",
        model_run_id="RUN_TEST",
        refit_epoch=1,
        prediction_formulation="DIRECT",
        flatten_metric_predictions=True,
    )
    assert result.y_true_wh.shape == result.y_pred_wh.shape == (2,)
    assert np.array_equal(result.y_pred_wh, np.array([15.0, 25.0]))


def test_e09_recovery_context_never_retrains(monkeypatch):
    from course_work.model_improvement_v2 import e09_runner

    fake = SimpleNamespace(reuse_completed_runs=False, reuse_completed_run_ids=None)
    monkeypatch.setattr(e09_runner, "_audit_resume_contract", lambda _root: {"locked_run_ids": {"k": "r"}})
    monkeypatch.setattr(e09_runner, "build_e09_run_context", lambda _root, _doc: fake)
    context, _audit = e09_runner.build_e09_resume_context(project_root(), {})
    assert context.reuse_completed_runs is True
    assert context.reuse_completed_run_ids == {"k": "r"}


def test_e09_finalization_uses_rmse_threshold_and_writes_contracts(tmp_path):
    document = load_e09_config(project_root())
    output = tmp_path / "artifacts/model_improvement_v2/experiments/E09"
    output.mkdir(parents=True)
    metrics = {}
    for index, pair in enumerate(CHALLENGER_GRID):
        metrics[CANDIDATE_IDS[pair]] = {
            "rmse_wh": 59.70 if index == 2 else 61.0 + index,
            "mae_wh": 100.0 if index == 2 else 1.0,
            "r2": 0.0,
        }
    result = SimpleNamespace(
        pooled_metrics_by_cid=metrics,
        stage_a_run_ids={},
        stage_b_run_ids={},
        inner_best_epochs={},
    )
    _write_success_contracts(tmp_path, document, result, {"status": "PASS"})
    manifest = json.loads((output / "e09_execution_manifest.json").read_text())
    comparison = json.loads((output / "e09_sgdm_ablation_comparison.json").read_text())
    assert manifest["control_provenance"]["source_experiment"] == "E01"
    assert manifest["control_run_ids"] == document["control"]["run_ids"]
    assert len(manifest["challenger_configs"]) == 6
    assert {item["candidate_id"] for item in manifest["challenger_configs"]} == set(CANDIDATE_IDS.values())
    assert all(item["config_fingerprint"] for item in manifest["challenger_configs"])
    assert len(comparison["results"]) == 7
    assert all(row.get("config_fingerprint") for row in comparison["results"][1:])
    assert comparison["primary_metric"] == "pooled_rmse_wh"
    assert comparison["promotion_threshold_rmse_wh"] == PROMOTION_THRESHOLD_RMSE_WH
    assert comparison["decision"]["action"] == "PROMOTE_SGDM"
    assert comparison["decision"]["selected_candidate_id"] == CANDIDATE_IDS[CHALLENGER_GRID[2]]
    assert comparison["test_status"] == "NOT_ACCESSED"


def test_e09_cli_authorization_guards(capsys):
    assert main(["--experiment", "E09", "--mode", "preflight", "--seed", "42", "--authorize-training"]) == 2
    assert "valid only in official mode" in capsys.readouterr().err
    assert main(["--experiment", "E09", "--mode", "official", "--seed", "42"]) == 3
    assert "requires --authorize-training" in capsys.readouterr().err
