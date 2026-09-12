import pytest

from course_work.experiments.registry import ExperimentRegistry
from course_work.model_improvement_v2.contracts import assert_one_primary_change
from course_work.model_improvement_v2.e16_runner import (
    CANDIDATE_ID,
    PARAMETER_COUNT,
    build_candidates,
    build_context,
    depth_config,
    incumbent_config,
    load_e16_config,
    main,
    project_root,
    run_preflight,
)
from course_work.rolling_origin.real_run import assert_context_invariants


def test_e15_rejection_retains_exact_e14_m1_control():
    preflight = run_preflight(project_root())
    assert preflight["e15_human_decision"] == "REJECT_E15"
    assert preflight["accepted_incumbent"] == "TR_C2_ALT_LOOKBACK_E14_M1"
    assert preflight["control_reuse_status"] == "PASS"


def test_depth_is_the_only_primary_change_and_post_ln_is_locked():
    root = project_root()
    base = incumbent_config(root)
    challenger = depth_config(root)
    assert set(assert_one_primary_change("E16", base, challenger)) == {"model.num_layers"}
    assert challenger["model"]["num_layers"] == 3
    assert challenger["model"]["norm_first"] is False
    assert challenger["model"]["d_model"] == 64
    assert challenger["model"]["num_heads"] == 4
    assert challenger["model"]["ffn_dim"] == 256
    assert challenger["training"] == base["training"]
    assert challenger["data"] == base["data"]


def test_parameter_population_scaler_and_test_firewall():
    preflight = run_preflight(project_root())
    assert preflight["parameter_count"] == PARAMETER_COUNT == 152193
    assert preflight["parameter_gate"] == "PASS"
    assert preflight["feature_variant"] == "FS2_TF1"
    assert preflight["feature_count"] == 33
    assert preflight["population"]["target_count"] == 16630
    assert preflight["population"]["status"] == "PASS"
    assert preflight["scaler_lineage"] == "FOLD_LOCAL_TRAIN_ONLY"
    assert preflight["test_rows_read"] == 0
    assert preflight["test_target_ids_seen"] == 0
    assert preflight["test_status"] == "NOT_ACCESSED"


def test_exact_one_candidate_and_v2_context():
    root = project_root()
    candidates = build_candidates(root)
    assert len(candidates) == 1
    assert candidates[0].candidate_id == CANDIDATE_ID
    context = build_context(root, load_e16_config(root))
    assert_context_invariants(context)
    assert context.registry_namespace == "V2_E16"
    assert context.execution_track == "MODEL_IMPROVEMENT_V2_E16"


def test_non_depth_change_is_rejected():
    root = project_root()
    base = incumbent_config(root)
    challenger = depth_config(root)
    challenger["training"]["learning_rate"] = 3e-4
    with pytest.raises(ValueError):
        assert_one_primary_change("E16", base, challenger)


def test_official_requires_separate_human_training_authorization():
    assert main(["--experiment", "E16", "--mode", "official", "--seed", "42"]) == 3


def test_preflight_scope_and_no_training():
    preflight = run_preflight(project_root())
    assert preflight["expected_new_training_runs"] == 6
    assert preflight["training_executed"] is False
    assert preflight["recovery_contract"] == "NO_RETRAIN_EXACT_COMPLETED_LEDGER_REQUIRED"


def test_no_train_resume_requires_exact_six_completed_identities():
    root = project_root()
    context = build_context(root, load_e16_config(root))
    context.reuse_completed_runs = True
    context.reuse_completed_run_ids = {
        f"{CANDIDATE_ID}:RO{fold}_{stage}": f"synthetic-{fold}-{stage}"
        for fold in (1, 2, 3)
        for stage in ("A", "B")
    }
    assert_context_invariants(context)
    context.reuse_completed_run_ids.pop(f"{CANDIDATE_ID}:RO3_B")
    with pytest.raises(RuntimeError, match="exact six locked"):
        assert_context_invariants(context)


def test_e16_registry_namespace_and_inherited_batch16(tmp_path):
    root = project_root()
    context = build_context(root, load_e16_config(root))
    config = depth_config(root)
    registry = ExperimentRegistry(
        project_root=root,
        registry_root=tmp_path / "registry",
        run_root=tmp_path / "runs",
        run_id_namespace="V2_E16",
        upstream_context_override=context.registry_upstream_context,
    )
    record = registry.register_run(
        config,
        experiment_family="ROLLING_ORIGIN",
        execution_type="ROBUSTNESS",
        candidate_id=CANDIDATE_ID,
        sweep_stage="RO1_A",
        rerun_reason="PHASE44_CORRECTIVE_RERUN",
    )
    assert record["run_id"].startswith("RUN_V2_TR_E16_RO1_A_")
    assert record["config"]["training"]["batch_size"] == 16
