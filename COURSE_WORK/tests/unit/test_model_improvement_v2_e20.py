from copy import deepcopy

import pytest

from course_work.experiments.registry import ExperimentRegistry
from course_work.model_improvement_v2.contracts import assert_one_primary_change
from course_work.model_improvement_v2.e20_runner import (
    ALL_SEEDS,
    BASE_FINGERPRINTS,
    CONTROL_ID,
    FINALIST_ID,
    NEW_SEEDS,
    SEEDED_FINGERPRINTS,
    audit_seed42_reuse,
    base_config,
    build_candidates,
    build_context,
    expected_new_run_identities,
    load_e20_config,
    main,
    project_root,
    run_preflight,
    seeded_config,
)
from course_work.rolling_origin.real_run import assert_context_invariants


@pytest.fixture(scope="module")
def preflight():
    return run_preflight(project_root())


def test_seed42_reuse_is_exact_and_read_only(preflight):
    audit = preflight["seed42_reuse"]
    assert audit["status"] == "PASS"
    assert audit["reused_runs"] == 12
    assert audit["retrain_allowed"] is False
    assert audit["control"]["run_count"] == 6
    assert audit["finalist"]["run_count"] == 6
    assert audit_seed42_reuse(project_root())["status"] == "PASS"


def test_only_registered_new_seeds_and_exact_fingerprints():
    root = project_root()
    assert ALL_SEEDS == (42, 123, 2026)
    assert NEW_SEEDS == (123, 2026)
    for role in ("CONTROL", "FINALIST"):
        assert base_config(root, role)
        for seed in NEW_SEEDS:
            config = seeded_config(root, role, seed)
            assert config["reproducibility"]["seed"] == seed
            assert set(assert_one_primary_change("E20", base_config(root, role), config)) == {
                "reproducibility.seed",
                "reproducibility.global_seed",
                "reproducibility.dataloader_seed",
            }
            from course_work.experiments.registry import compute_config_fingerprint

            assert compute_config_fingerprint(config) == SEEDED_FINGERPRINTS[seed][role]
    assert BASE_FINGERPRINTS["CONTROL"].startswith("585c5e79")
    assert BASE_FINGERPRINTS["FINALIST"].startswith("ff99dba3")


def test_locked_configs_do_not_drift_except_seed():
    root = project_root()
    for role in ("CONTROL", "FINALIST"):
        left = seeded_config(root, role, 123)
        right = seeded_config(root, role, 2026)
        for key in ("model", "training", "data", "runtime", "lineage"):
            assert left[key] == right[key]
        lrep = deepcopy(left["reproducibility"])
        rrep = deepcopy(right["reproducibility"])
        for key in ("seed", "global_seed", "dataloader_seed"):
            lrep.pop(key)
            rrep.pop(key)
        assert lrep == rrep


def test_contexts_are_seed_isolated_and_fresh_state():
    root = project_root()
    document = load_e20_config(root)
    contexts = [build_context(root, document, seed) for seed in NEW_SEEDS]
    for seed, context in zip(NEW_SEEDS, contexts):
        assert_context_invariants(context)
        assert context.seed == seed
        assert context.registry_namespace == "V2_E20"
        assert context.execution_track == "MODEL_IMPROVEMENT_V2_E20"
        assert context.seed_before_model_construction is True
        assert context.validated_registry_lifecycle is True
        assert context.reuse_completed_runs is False
        assert context.apply_fold_x_scaling is True
        assert {candidate.candidate_id for candidate in context.candidate_specs} == {
            CONTROL_ID,
            FINALIST_ID,
        }
    assert contexts[0].registry_root != contexts[1].registry_root
    assert contexts[0].run_root != contexts[1].run_root


def test_registry_allocates_exactly_24_e20_stage_ids_without_training(tmp_path):
    root = project_root()
    seen = []
    for seed in NEW_SEEDS:
        context = build_context(root, load_e20_config(root), seed)
        registry = ExperimentRegistry(
            project_root=root,
            registry_root=tmp_path / f"registry-{seed}",
            run_root=tmp_path / f"runs-{seed}",
            run_id_namespace="V2_E20",
            upstream_context_override=context.registry_upstream_context,
        )
        for candidate in build_candidates(root, seed):
            for fold in (1, 2, 3):
                for stage in "AB":
                    record = registry.register_run(
                        candidate.config,
                        experiment_family="ROLLING_ORIGIN",
                        execution_type="ROBUSTNESS",
                        candidate_id=candidate.candidate_id,
                        sweep_stage=f"RO{fold}_{stage}",
                        rerun_reason="PHASE44_CORRECTIVE_RERUN",
                    )
                    seen.append(record["run_id"])
    assert len(expected_new_run_identities()) == 24
    assert len(seen) == len(set(seen)) == 24
    assert all(run_id.startswith("RUN_V2_TR_E20_RO") for run_id in seen)
    context = build_context(root, load_e20_config(root), 123)
    with pytest.raises(ValueError, match="Unsupported run_id_namespace"):
        ExperimentRegistry(
            project_root=root,
            registry_root=tmp_path / "bad-registry",
            run_root=tmp_path / "bad-runs",
            run_id_namespace="V2_ARBITRARY",
            upstream_context_override=context.registry_upstream_context,
        )


def test_preflight_population_scaling_scope_and_test_firewall(preflight):
    assert preflight["status"] == "PASS"
    assert preflight["expected_total_evidence"] == 36
    assert preflight["expected_new_training_runs"] == 24
    assert preflight["new_seeds"] == [123, 2026]
    assert preflight["population"]["status"] == "PASS"
    assert preflight["population"]["target_count"] == 16630
    assert set(preflight["population"]["fold_fingerprints"]) == {"RO1", "RO2", "RO3"}
    assert preflight["scaler_lineage"] == "FOLD_LOCAL_TRAIN_ONLY"
    assert preflight["test_rows_read"] == 0
    assert preflight["test_target_ids_seen"] == 0
    assert preflight["test_status"] == "NOT_ACCESSED"
    assert preflight["training_executed"] is False


def test_recovery_contract_reuses_only_verified_completed_runs():
    root = project_root()
    context = build_context(root, load_e20_config(root), 123)
    context.reuse_completed_runs = True
    context.reuse_completed_run_ids = {}
    context.allow_partial_stage_recovery = True
    assert context.reuse_completed_runs is True
    assert context.allow_partial_stage_recovery is True
    assert_context_invariants(context)
    context.allow_partial_stage_recovery = False
    with pytest.raises(RuntimeError, match="exact 12 completed"):
        assert_context_invariants(context)


def test_human_gate_and_seed42_retraining_refusal():
    assert main(["--experiment", "E20", "--mode", "official", "--seeds", "123", "2026"]) == 3
    assert main([
        "--experiment", "E20", "--mode", "official", "--seeds", "42", "123", "2026",
        "--authorize-training",
    ]) == 2
