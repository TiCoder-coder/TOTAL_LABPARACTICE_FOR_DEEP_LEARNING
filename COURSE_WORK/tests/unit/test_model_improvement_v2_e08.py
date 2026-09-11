from __future__ import annotations

import inspect

import pytest
import torch

from course_work.model_improvement_v2.e08_runner import (
    CANDIDATE_IDS,
    CHALLENGER_LRS,
    CONTROL_METRICS,
    EXECUTION_TRACK,
    OPTIMIZER_CONFIG,
    PROMOTION_THRESHOLD_RMSE_WH,
    REGISTRY_NAMESPACE,
    build_e08_candidates,
    build_e08_run_context,
    load_e08_config,
    main,
    project_root,
    run_preflight,
    validate_e08_document,
)
from course_work.model_improvement_v2.optimizer import build_optimizer
from course_work.rolling_origin.real_run import (
    V2_DIRECT_METRIC_FLATTEN_TRACKS,
    V2_SHARED_PRETEST_CACHE_KEYS,
    assert_context_invariants,
)


def test_optimizer_factory_preserves_adamw_default_and_supports_plain_sgd():
    legacy_model = torch.nn.Linear(2, 1)
    legacy = build_optimizer(legacy_model.parameters(), None, 3e-4, 1e-3)
    assert isinstance(legacy, torch.optim.AdamW)
    assert legacy.param_groups[0]["lr"] == 3e-4
    assert legacy.param_groups[0]["weight_decay"] == 1e-3
    legacy_with_ignored_extra = build_optimizer(
        torch.nn.Linear(2, 1).parameters(), None, 3e-4, 1e-3, {"legacy": True}
    )
    assert isinstance(legacy_with_ignored_extra, torch.optim.AdamW)

    model = torch.nn.Linear(2, 1)
    sgd = build_optimizer(model.parameters(), "SGD", 3e-3, 1e-3, OPTIMIZER_CONFIG)
    assert isinstance(sgd, torch.optim.SGD)
    group = sgd.param_groups[0]
    assert group["lr"] == 3e-3
    assert group["momentum"] == 0.0
    assert group["nesterov"] is False
    assert group["weight_decay"] == 1e-3


@pytest.mark.parametrize(
    "name,config",
    [
        ("RMSprop", {}),
        ("SGD", {}),
        ("SGD", {"momentum": 0.8, "nesterov": False}),
        ("SGD", {"momentum": 0.0, "nesterov": True}),
    ],
)
def test_optimizer_factory_rejects_unsupported_or_non_plain_sgd(name, config):
    with pytest.raises(ValueError):
        build_optimizer(torch.nn.Linear(1, 1).parameters(), name, 1e-3, 1e-3, config)


def test_e08_preflight_locks_exact_matrix_and_control_reuse():
    result = run_preflight()
    assert result["status"] == "PASS"
    assert result["training_candidate_count"] == 4
    assert result["learning_rates"] == list(CHALLENGER_LRS)
    assert set(result["candidate_ids"]) == set(CANDIDATE_IDS.values())
    assert result["control_reuse"] == "CONTROL_REUSED_FROM_E01_READ_ONLY"
    assert result["optimizer"] == "SGD"
    assert result["optimizer_config"] == OPTIMIZER_CONFIG
    assert result["scheduler"] == "OFF"
    assert result["test_access"] == "NO"
    assert result["training_executed"] is False
    assert result["inference_executed"] is False


def test_e08_candidate_configs_are_exact_and_fresh():
    document = load_e08_config(project_root())
    specs = build_e08_candidates(document)
    assert len(specs) == 4
    assert len({spec.candidate_id for spec in specs}) == 4
    for spec in specs:
        training = spec.config["training"]
        assert training["optimizer_name"] == "SGD"
        assert training["optimizer_config"] == OPTIMIZER_CONFIG
        assert training["learning_rate"] in CHALLENGER_LRS
        assert training["weight_decay"] == 1e-3
        assert training["scheduler_name"] is None
        assert training["scheduler_config"] is None
        assert spec.config["model"].get("prediction_formulation", "DIRECT") == "DIRECT"


def test_e08_invalid_matrix_and_optimizer_contract_are_rejected():
    root = project_root()
    document = load_e08_config(root)
    document["training_candidates"] = document["training_candidates"][:-1]
    with pytest.raises(RuntimeError, match="exactly four"):
        validate_e08_document(document, root)

    document = load_e08_config(root)
    document["training_candidates"][0]["config"]["training"]["optimizer_config"]["momentum"] = 0.9
    with pytest.raises(RuntimeError, match="config mismatch"):
        validate_e08_document(document, root)


def test_e08_context_isolated_and_control_excluded():
    root = project_root()
    document = load_e08_config(root)
    context = build_e08_run_context(root, document)
    assert context.registry_namespace == REGISTRY_NAMESPACE
    assert context.execution_track == EXECUTION_TRACK
    assert len(context.candidate_specs) == 4
    assert all(spec.candidate_id != "TR_C2_ALT_LOOKBACK" for spec in context.candidate_specs)
    assert context.enable_stage_b_lr_replay is False
    assert context.seed_before_model_construction is True
    assert context.reuse_completed_runs is False
    assert_context_invariants(context)


def test_shared_cache_and_direct_shape_scope_are_exact():
    assert V2_SHARED_PRETEST_CACHE_KEYS[EXECUTION_TRACK] == "MODEL_IMPROVEMENT_V2_E08_SHARED_PRETEST"
    assert V2_DIRECT_METRIC_FLATTEN_TRACKS == frozenset({
        "MODEL_IMPROVEMENT_V2_E06",
        "MODEL_IMPROVEMENT_V2_E07",
        "MODEL_IMPROVEMENT_V2_E08",
        "MODEL_IMPROVEMENT_V2_E09",
    })
    for experiment in ("E01", "E03", "E04", "E05"):
        assert f"MODEL_IMPROVEMENT_V2_{experiment}" not in V2_DIRECT_METRIC_FLATTEN_TRACKS


def test_stage_a_and_stage_b_use_factory_with_no_scheduler_replay():
    from course_work.training.engine import TrainingEngine
    from course_work.rolling_origin.refit_engine import RefitEngine

    assert "build_optimizer" in inspect.getsource(TrainingEngine.train)
    assert "build_optimizer" in inspect.getsource(RefitEngine.refit)
    document = load_e08_config(project_root())
    for candidate in document["training_candidates"]:
        training = candidate["config"]["training"]
        first = build_optimizer(
            torch.nn.Linear(1, 1).parameters(),
            training["optimizer_name"],
            training["learning_rate"],
            training["weight_decay"],
            training["optimizer_config"],
        )
        second = build_optimizer(
            torch.nn.Linear(1, 1).parameters(),
            training["optimizer_name"],
            training["learning_rate"],
            training["weight_decay"],
            training["optimizer_config"],
        )
        assert first is not second
        assert first.param_groups[0]["lr"] == training["learning_rate"]


def test_finalization_policy_is_rmse_only_and_threshold_locked():
    document = load_e08_config(project_root())
    policy = document["selection_policy"]
    assert CONTROL_METRICS["rmse_wh"] == 59.85291570400546
    assert PROMOTION_THRESHOLD_RMSE_WH == 59.75291570400546
    assert policy["primary_metric"] == "pooled_rmse_wh"
    assert policy["mae_can_promote"] is False
    assert policy["fallback"] == "RETAIN_E01_ADAMW"


def test_cli_authorization_guards(capsys):
    assert main(["--experiment", "E08", "--mode", "preflight", "--seed", "42", "--authorize-training"]) == 2
    assert "valid only in official mode" in capsys.readouterr().err
    assert main(["--experiment", "E08", "--mode", "official", "--seed", "42"]) == 3
    assert "requires --authorize-training" in capsys.readouterr().err


def test_e08_registry_namespace_and_canonical_stage_ids(tmp_path):
    from course_work.experiments.registry import ExperimentRegistry, V2_NAMESPACE_IDS

    assert "V2_E06" in V2_NAMESPACE_IDS
    assert "V2_E07" in V2_NAMESPACE_IDS
    assert "V2_E08" in V2_NAMESPACE_IDS
    registry = ExperimentRegistry(
        project_root=tmp_path,
        registry_root=tmp_path / "registry",
        run_root=tmp_path / "runs",
        run_id_namespace="V2_E08",
        upstream_context_override={"lineage": {}, "feature_sets": {}},
    )
    stage_a = registry.allocate_run_id(
        model_family="TRANSFORMER_ENCODER",
        experiment_family="ROLLING_ORIGIN",
        config_fingerprint="a1b2c3d4" * 8,
        run_stage="RO1_A",
    )
    stage_b = registry.allocate_run_id(
        model_family="TRANSFORMER_ENCODER",
        experiment_family="ROLLING_ORIGIN",
        config_fingerprint="deadbeef" * 8,
        run_stage="RO1_B",
    )
    assert stage_a == "RUN_V2_TR_E08_RO1_A_0001_A1B2C3D4"
    assert stage_b == "RUN_V2_TR_E08_RO1_B_0001_DEADBEEF"


def test_v1_registry_namespace_and_run_id_format_remain_unchanged(tmp_path):
    from course_work.experiments.registry import ExperimentRegistry, V2_NAMESPACE_IDS

    assert V2_NAMESPACE_IDS == frozenset({
        "V2_E01", "V2_E02", "V2_E03", "V2_E04",
        "V2_E05", "V2_E06", "V2_E07", "V2_E08", "V2_E09",
    })
    registry = ExperimentRegistry.__new__(ExperimentRegistry)
    registry.run_id_namespace = None
    registry.run_root = tmp_path / "runs"
    registry._load_records = lambda: []
    registry._family = lambda _family_id: {"family_code": "ROB"}
    run_id = registry.allocate_run_id(
        model_family="TRANSFORMER_ENCODER",
        experiment_family="ROLLING_ORIGIN",
        config_fingerprint="12345678" * 8,
    )
    assert run_id == "RUN_TR_ROB_0001_12345678"


def test_arbitrary_registry_namespace_remains_rejected(tmp_path):
    from course_work.experiments.registry import ExperimentRegistry

    with pytest.raises(ValueError, match="Unsupported run_id_namespace"):
        ExperimentRegistry(
            project_root=tmp_path,
            registry_root=tmp_path / "registry",
            run_root=tmp_path / "runs",
            run_id_namespace="V2_ARBITRARY",
        )


def test_e08_official_context_can_initialize_registry_without_training(tmp_path):
    from course_work.experiments.registry import ExperimentRegistry

    root = project_root()
    context = build_e08_run_context(root, load_e08_config(root))
    registry = ExperimentRegistry(
        project_root=tmp_path,
        registry_root=tmp_path / "registry",
        run_root=tmp_path / "runs",
        run_id_namespace=context.registry_namespace,
        upstream_context_override=context.registry_upstream_context,
    )
    assert registry.run_id_namespace == "V2_E08"
    assert not list((tmp_path / "runs").glob("RUN_*"))


def test_e08_pretest_projection_matches_canonical_fs2_contract():
    from course_work.model_improvement_v2.pretest_adapter import build_v2_pretest_dataset

    root = project_root()
    document = load_e08_config(root)
    dataset, evidence, audit = build_v2_pretest_dataset(
        root,
        document["feature_order"],
        experiment_id="E08",
        feature_variant_id="FS2_TF1",
    )
    assert dataset.experiment_id == "E08"
    assert dataset.feature_variant_id == "FS2_TF1"
    assert tuple(dataset.feature_order) == tuple(document["feature_order"])
    assert dataset.window_records["target_id"].tolist() == list(
        evidence.train_ids + evidence.validation_ids
    )
    assert audit.population_status == "PASS"
    assert audit.feature_order_status == "PASS"
    assert audit.fold_fingerprint_status == {"RO1": "PASS", "RO2": "PASS", "RO3": "PASS"}
    assert audit.test_rows_read == 0
    assert audit.test_target_ids_seen == 0


def test_e08_rejects_unapproved_projection_pairs():
    from course_work.model_improvement_v2.pretest_adapter import build_v2_pretest_dataset

    root = project_root()
    document = load_e08_config(root)
    with pytest.raises(RuntimeError, match="Unsupported V2 pre-Test projection"):
        build_v2_pretest_dataset(
            root,
            document["feature_order"],
            experiment_id="E08",
            feature_variant_id="FS1_TF1",
        )
    with pytest.raises(RuntimeError, match="Unsupported V2 pre-Test projection"):
        build_v2_pretest_dataset(
            root,
            document["feature_order"],
            experiment_id="E99",
            feature_variant_id="FS2_TF1",
        )


def test_all_e08_candidates_share_one_logical_pretest_dataset():
    root = project_root()
    document = load_e08_config(root)
    context = build_e08_run_context(root, document)
    datasets = [context.dataset_factory(candidate, None) for candidate in context.candidate_specs]
    assert len(datasets) == 4
    assert len({id(dataset) for dataset in datasets}) == 1
    assert context.execution_track == "MODEL_IMPROVEMENT_V2_E08"
