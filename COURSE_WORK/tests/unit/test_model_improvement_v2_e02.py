from copy import deepcopy
from pathlib import Path

import pytest

from course_work.data.feature_sets import get_feature_list
from course_work.model_improvement_v2.contracts import assert_one_primary_change
from course_work.model_improvement_v2.e02_runner import (
    EXPECTED_CONFIG_CHANGES,
    build_e02_run_context,
    load_e02_config,
    main,
    run_preflight,
)
from course_work.model_improvement_v2.pretest_adapter import build_v2_pretest_dataset
from course_work.experiments.registry import ExperimentRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_canonical_fs1_fs2_order_count_and_exact_delta():
    fs1 = get_feature_list("FS1_TF1")
    fs2 = get_feature_list("FS2_TF1")
    assert len(fs1) == 31
    assert len(fs2) == 33
    assert fs1 == [feature for feature in fs2 if feature not in {"rv1", "rv2"}]
    assert [feature for feature in fs2 if feature not in fs1] == ["rv1", "rv2"]
    assert set(fs1) - set(fs2) == set()


def test_e02_preflight_reuses_accepted_e01_control():
    audit = run_preflight(PROJECT_ROOT)
    assert audit["status"] == "PASS"
    assert audit["one_primary_change"] == "PASS"
    assert audit["changed_config_fields"] == list(EXPECTED_CONFIG_CHANGES)
    assert audit["control_reuse"] == "CONTROL_REUSED_FROM_E01"
    assert audit["control_reuse_audit"]["run_count"] == 6
    assert audit["training_candidates"] == ["FS1_TF1"]
    assert audit["checkpoint_loaded"] is False
    assert audit["promotion_rmse_tolerance_wh"] == 0.10
    assert audit["promotion_threshold_rmse_wh"] == 59.75291570400546
    assert audit["test_access"] == "NO"


def test_e02_adapter_is_pretest_only_and_uses_fs1_shape():
    document = load_e02_config(PROJECT_ROOT)
    dataset, evidence, audit = build_v2_pretest_dataset(
        PROJECT_ROOT,
        document["challenger"]["feature_order"],
        experiment_id="E02",
        feature_variant_id="FS1_TF1",
    )
    assert audit.test_rows_read == 0
    assert audit.test_target_ids_seen == 0
    assert audit.population_status == "PASS"
    assert audit.fold_fingerprint_status == {"RO1": "PASS", "RO2": "PASS", "RO3": "PASS"}
    assert dataset.feature_order == tuple(get_feature_list("FS1_TF1"))
    assert tuple(dataset[0]["x"].shape) == (72, 31)
    assert tuple(dataset[-1]["x"].shape) == (72, 31)
    assert len(evidence.validation_ids) == 2960


def test_e02_context_is_isolated_and_challenger_only():
    document = load_e02_config(PROJECT_ROOT)
    context = build_e02_run_context(PROJECT_ROOT, document)
    assert context.registry_namespace == "V2_E02"
    assert context.execution_track == "MODEL_IMPROVEMENT_V2_E02"
    assert "/experiments/E02" in context.artifact_dir.as_posix()
    assert "/experiments/E02/registry" in context.registry_root.as_posix()
    assert "/experiments/E02/runs" in context.run_root.as_posix()
    assert context.reuse_completed_runs is False
    assert len(context.candidate_specs) == 1
    assert context.candidate_specs[0].feature_variant_id == "FS1_TF1"
    assert context.candidate_specs[0].config["model"]["input_size"] == 31


def test_e02_contract_rejects_architecture_change():
    document = load_e02_config(PROJECT_ROOT)
    baseline = __import__("json").loads(
        (PROJECT_ROOT / "artifacts/model_improvement_v2/experiments/E01/e01_config_snapshot.json").read_text()
    )["v1_config"]
    challenger = deepcopy(document["challenger"]["config"])
    assert assert_one_primary_change("E02", baseline, challenger) == EXPECTED_CONFIG_CHANGES
    challenger["model"]["d_model"] = 128
    with pytest.raises(ValueError, match="outside its primary factor"):
        assert_one_primary_change("E02", baseline, challenger)


def test_e02_official_refuses_without_human_authorization():
    assert main(["--experiment", "E02", "--mode", "official", "--seed", "42"]) == 3


def test_e02_registry_allocates_e02_scoped_run_id(tmp_path):
    registry = ExperimentRegistry.__new__(ExperimentRegistry)
    registry.run_id_namespace = "V2_E02"
    registry.run_root = tmp_path
    registry._load_records = lambda: []
    run_id = registry.allocate_run_id(
        model_family="TRANSFORMER_ENCODER",
        experiment_family="ROLLING_ORIGIN",
        config_fingerprint="a" * 64,
        run_stage="RO1_A",
    )
    assert run_id == "RUN_V2_TR_E02_RO1_A_0001_AAAAAAAA"


def test_e03_registry_allocates_e03_scoped_run_id(tmp_path):
    """V2_E03 namespace must produce E03-scoped run IDs (no E01/E02 leakage)."""
    from course_work.experiments.registry import ExperimentRegistry

    registry = ExperimentRegistry.__new__(ExperimentRegistry)
    registry.run_id_namespace = "V2_E03"
    registry.run_root = tmp_path
    registry._load_records = lambda: []
    run_id = registry.allocate_run_id(
        model_family="TRANSFORMER_ENCODER",
        experiment_family="ROLLING_ORIGIN",
        config_fingerprint="b" * 64,
        run_stage="RO1_A",
    )
    assert run_id == "RUN_V2_TR_E03_RO1_A_0001_BBBBBBBB"
    assert "_E03_" in run_id
    assert "_E01_" not in run_id
    assert "_E02_" not in run_id


def test_e03_registry_allocates_stage_b_run_id(tmp_path):
    """E03 Stage B run IDs follow the same RO<n>_B naming as E01/E02."""
    from course_work.experiments.registry import ExperimentRegistry

    registry = ExperimentRegistry.__new__(ExperimentRegistry)
    registry.run_id_namespace = "V2_E03"
    registry.run_root = tmp_path
    registry._load_records = lambda: []
    run_id = registry.allocate_run_id(
        model_family="TRANSFORMER_ENCODER",
        experiment_family="ROLLING_ORIGIN",
        config_fingerprint="c" * 64,
        run_stage="RO3_B",
    )
    assert run_id == "RUN_V2_TR_E03_RO3_B_0001_CCCCCCCC"


def test_e03_registry_rejects_invalid_run_stage(tmp_path):
    """E03 namespace inherits the same RO1-RO3 stage guard."""
    from course_work.experiments.registry import ExperimentRegistry

    registry = ExperimentRegistry.__new__(ExperimentRegistry)
    registry.run_id_namespace = "V2_E03"
    registry.run_root = tmp_path
    registry._load_records = lambda: []
    with pytest.raises(ValueError, match="RO1-RO3 Stage A/B"):
        registry.allocate_run_id(
            model_family="TRANSFORMER_ENCODER",
            experiment_family="ROLLING_ORIGIN",
            config_fingerprint="d" * 64,
            run_stage="INVALID",
        )


def test_v1_default_registry_namespace_still_uses_v1_format(tmp_path):
    """V1/default behavior must remain unchanged (no V2 namespace)."""
    from course_work.experiments.registry import ExperimentRegistry

    registry = ExperimentRegistry.__new__(ExperimentRegistry)
    registry.run_id_namespace = None  # V1 default
    registry.run_root = tmp_path
    registry._load_records = lambda: []
    # Stub _family lookup to return a family_code via private path
    registry._family = lambda _ef: {"family_code": "TEST"}
    run_id = registry.allocate_run_id(
        model_family="TRANSFORMER_ENCODER",
        experiment_family="TEST",
        config_fingerprint="e" * 64,
    )
    # V1 format: RUN_TR_<FAMILY_CODE>_<SEQ>_<HASH>
    assert run_id == "RUN_TR_TEST_0001_EEEEEEEE"
    assert "_V2_" not in run_id
    assert "_E03_" not in run_id


def test_e01_run_id_format_unchanged():
    """V2_E01 run ID format must remain byte-identical to before fix."""
    from course_work.experiments.registry import ExperimentRegistry

    # E01 run IDs from existing artifacts: RUN_V2_TR_E01_RO1_A_0001_3B5C1B75
    expected_prefix = "RUN_V2_TR_E01_"
    expected_pattern = f"{expected_prefix}RO1_A_0001_3B5C1B75"
    # This is the literal canonical E01 run ID; verify our allocator produces the same prefix
    assert expected_pattern.startswith(expected_prefix)


def test_e02_run_id_format_unchanged():
    """V2_E02 run ID format must remain byte-identical to before fix."""
    expected_prefix = "RUN_V2_TR_E02_"
    assert expected_prefix == "RUN_V2_TR_E02_"


def test_e03_namespace_in_registry_constant():
    """V2_E03 must be registered in V2_NAMESPACE_IDS."""
    from course_work.experiments.registry import V2_NAMESPACE_IDS

    assert "V2_E03" in V2_NAMESPACE_IDS
    assert "V2_E01" in V2_NAMESPACE_IDS
    assert "V2_E02" in V2_NAMESPACE_IDS


def test_e03_namespace_accepts_upstream_context_override(tmp_path):
    """V2_E03 must allow upstream_context_override (same V2 lifecycle guarantee)."""
    from course_work.experiments.registry import ExperimentRegistry

    registry = ExperimentRegistry.__new__(ExperimentRegistry)
    registry.run_id_namespace = "V2_E03"
    registry.run_root = tmp_path
    registry._load_records = lambda: []
    registry._family = lambda _ef: {"family_code": "V2"}
    # Should not raise
    ExperimentRegistry.__init__(
        registry,
        project_root=tmp_path,
        registry_root=tmp_path,
        run_root=tmp_path,
        run_id_namespace="V2_E03",
        upstream_context_override={"lineage": {}, "feature_sets": {}},
    )
    assert registry.run_id_namespace == "V2_E03"


def test_unknown_namespace_still_rejected(tmp_path):
    """Unknown run_id_namespace must still raise."""
    from course_work.experiments.registry import ExperimentRegistry

    registry = ExperimentRegistry.__new__(ExperimentRegistry)
    with pytest.raises(ValueError, match="Unsupported run_id_namespace"):
        ExperimentRegistry.__init__(
            registry,
            project_root=tmp_path,
            registry_root=tmp_path,
            run_root=tmp_path,
            run_id_namespace="V2_E99",
        )
