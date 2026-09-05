import json
import platform
import sys
from pathlib import Path

import pytest

from course_work.experiments.phase_execution import (
    PhaseAction,
    PhaseState,
    get_sweep_phase_spec,
    inspect_phase_state,
    inspect_execution_readiness,
    plan_phase_resume,
    resolve_condition_values,
    resolve_phase_conditions,
    resolve_phase_action,
)
from course_work.sweeps.sweep_results import materialize_phase_23, validate_sweep_signoff
from course_work.utils.artifacts import sha256_file


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def create_phase_22_prerequisite(root: Path) -> None:
    write_json(
        root / "artifacts/learning_diagnostics/phase_22_signoff.json",
        {"status": "PASS", "output_paths": [], "output_checksums": {}},
    )


def create_valid_environment(root: Path) -> None:
    report_path = root / "artifacts/environment/environment_report.json"
    freeze_path = root / "artifacts/environment/requirements_freeze.txt"
    smoke_path = root / "artifacts/environment/smoke_test_report.json"
    write_json(
        report_path,
        {
            "python_version": platform.python_version(),
            "python_executable": str(Path(sys.executable).resolve()),
        },
    )
    freeze_path.write_text("test\n", encoding="utf-8")
    write_json(smoke_path, {"status": "PASS"})
    output_paths = [str(path.relative_to(root)) for path in (report_path, freeze_path, smoke_path)]
    write_json(
        root / "artifacts/environment/phase_1_signoff.json",
        {
            "status": "PASS",
            "output_paths": output_paths,
            "output_checksums": {path: sha256_file(root / path) for path in output_paths},
        },
    )


def create_completed_run(root: Path, run_id: str, family_id: str, config: dict, rmse_wh: float) -> dict:
    run_root = root / "artifacts/runs" / run_id
    config_payload = {
        "run_id": run_id,
        "config_fingerprint": f"fingerprint-{run_id}",
        "config": config,
    }
    write_json(run_root / "config.json", config_payload)
    write_json(run_root / "status.json", {"run_id": run_id, "status": "COMPLETED"})
    write_json(run_root / "metrics/best_validation_metrics.json", {"rmse_wh": rmse_wh})
    artifact_paths = (
        ("CONFIG", run_root / "config.json"),
        ("STATUS", run_root / "status.json"),
        ("METRICS", run_root / "metrics/best_validation_metrics.json"),
    )
    return {
        "run_id": run_id,
        "status": "COMPLETED",
        "experiment_family": family_id,
        "config_fingerprint": config_payload["config_fingerprint"],
        "artifacts": [
            {
                "artifact_type": artifact_type,
                "artifact_path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
                "required": True,
            }
            for artifact_type, path in artifact_paths
        ],
        "metrics": [
            {"split_id": "VALIDATION", "status": "PASS", "metric_name": "mae_wh", "metric_value": rmse_wh - 1.0},
            {"split_id": "VALIDATION", "status": "PASS", "metric_name": "rmse_wh", "metric_value": rmse_wh},
            {"split_id": "VALIDATION", "status": "PASS", "metric_name": "r2", "metric_value": 0.5},
        ],
    }


def write_registry(root: Path, records: list[dict]) -> None:
    path = root / "artifacts/experiments/experiment_registry.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(record, sort_keys=True)}\n" for record in records), encoding="utf-8")


def create_phase_23_registry(root: Path) -> None:
    records = []
    for index, condition in enumerate(("FS0_TF1", "FS1_TF1", "FS2_TF1"), start=1):
        family = "TRANSFORMER_BASELINE" if condition == "FS1_TF1" else "S1_FEATURE_SET"
        records.append(
            create_completed_run(
                root,
                f"RUN-{condition}",
                family,
                {"data": {"feature_variant_id": condition}},
                60.0 + index,
            )
        )
    write_registry(root, records)


def create_valid_phase_23(root: Path, include_log: bool = True) -> None:
    create_phase_22_prerequisite(root)
    create_phase_23_registry(root)
    spec = get_sweep_phase_spec(23)
    root_path = root / spec.artifact_root
    root_path.mkdir(parents=True, exist_ok=True)
    (root / spec.results_path).write_text("condition,val_rmse\nFS0_TF1,1.0\n", encoding="utf-8")
    write_json(root / spec.manifest_path, {"phase_id": 23})
    write_json(root / spec.winner_path, {"condition": "FS0_TF1"})
    write_json(root / spec.reference_path, {"condition": "FS0_TF1"})
    output_paths = [str(spec.manifest_path), str(spec.results_path), str(spec.winner_path), str(spec.reference_path)]
    output_checksums = {path: sha256_file(root / path) for path in output_paths}
    write_json(
        root / spec.signoff_path,
        {
            "phase_id": 23,
            "phase_version": "PHASE-23-v1",
            "status": "PASS",
            "input_paths": [],
            "input_checksums": {},
            "output_paths": output_paths,
            "output_checksums": output_checksums,
        },
    )
    if include_log:
        write_json(
            root / spec.processing_log_path,
            {
                "phase_id": 23,
                "status": "PASS",
                "source_artifacts": [
                    {
                        "path": str(spec.manifest_path),
                        "sha256": sha256_file(root / spec.manifest_path),
                    }
                ],
            },
        )


def test_resolve_phase_action() -> None:
    assert resolve_phase_action(PhaseState.VALID_REUSABLE) is PhaseAction.RENDER_ONLY
    assert resolve_phase_action("LOG_MISSING") is PhaseAction.REBUILD_LOG_ONLY
    assert resolve_phase_action(PhaseState.UPSTREAM_INVALID) is PhaseAction.BLOCK


def test_get_sweep_phase_spec_rejects_unsupported_phase() -> None:
    with pytest.raises(ValueError, match="Phase 23-37"):
        get_sweep_phase_spec(22)


def test_phase_31_spec_matches_weight_decay_contract() -> None:
    spec = get_sweep_phase_spec(31)
    assert spec.phase_name == "S9 Weight-Decay Sweep"
    assert spec.family_id == "S9_WEIGHT_DECAY"
    assert spec.condition_path == ("training", "weight_decay")
    assert spec.condition_values == (("WD0", 0.0), ("WD1", 0.0001), ("WD2", 0.001))
    assert spec.reference_condition == "WD1"
    assert str(spec.manifest_path) == "artifacts/sweeps/S9_weight_decay/s9_weight_decay_sweep_manifest.json"
    assert str(spec.results_path) == "artifacts/sweeps/S9_weight_decay/s9_weight_decay_metrics.csv"


def test_phase_32_spec_matches_dropout_contract() -> None:
    spec = get_sweep_phase_spec(32)
    assert spec.phase_name == "S10 Dropout Sweep"
    assert spec.family_id == "S10_DROPOUT"
    assert spec.condition_path == ("model", "dropout")
    assert spec.condition_values == (("DR01", 0.1), ("DR02", 0.2), ("DR03", 0.3))
    assert spec.reference_condition == "DR01"
    assert str(spec.manifest_path) == "artifacts/sweeps/S10_dropout/s10_dropout_sweep_manifest.json"
    assert str(spec.results_path) == "artifacts/sweeps/S10_dropout/s10_dropout_metrics.csv"


def test_phase_33_spec_matches_d_model_contract() -> None:
    spec = get_sweep_phase_spec(33)
    assert spec.phase_name == "S11 d_model Sweep"
    assert spec.family_id == "S11_D_MODEL"
    assert spec.condition_path == ("model", "d_model")
    assert spec.condition_values == (("D32", 32), ("D64", 64))
    assert spec.reference_condition == "D64"
    assert str(spec.manifest_path) == "artifacts/sweeps/S11_d_model/s11_d_model_sweep_manifest.json"
    assert str(spec.results_path) == "artifacts/sweeps/S11_d_model/s11_d_model_metrics.csv"


def test_phase_34_spec_matches_head_contract() -> None:
    spec = get_sweep_phase_spec(34)
    assert spec.phase_name == "S12 Head Sweep"
    assert spec.family_id == "S12_HEADS"
    assert spec.condition_path == ("model", "num_heads")
    assert spec.condition_values == (("H2", 2), ("H4", 4))
    assert spec.reference_condition == "H4"
    assert str(spec.manifest_path) == "artifacts/sweeps/S12_heads/s12_head_sweep_manifest.json"
    assert str(spec.results_path) == "artifacts/sweeps/S12_heads/s12_head_metrics.csv"


def test_phase_35_spec_matches_layer_contract() -> None:
    spec = get_sweep_phase_spec(35)
    assert spec.phase_name == "S13 Layer Sweep"
    assert spec.family_id == "S13_LAYERS"
    assert spec.condition_path == ("model", "num_layers")
    assert spec.condition_values == (("N1", 1), ("N2", 2))
    assert spec.reference_condition == "N2"
    assert str(spec.manifest_path) == "artifacts/sweeps/S13_layers/s13_layer_sweep_manifest.json"
    assert str(spec.results_path) == "artifacts/sweeps/S13_layers/s13_layer_metrics.csv"


def test_phase_36_spec_matches_ffn_contract() -> None:
    spec = get_sweep_phase_spec(36)
    assert spec.phase_name == "S14 FFN Sweep"
    assert spec.family_id == "S14_FFN"
    assert spec.condition_path == ("model", "ffn_dim")
    assert spec.condition_values == (("F64", 64), ("F128", 128), ("F256", 256))
    assert spec.reference_condition == "F128"
    assert str(spec.manifest_path) == "artifacts/sweeps/S14_ffn/s14_ffn_sweep_manifest.json"
    assert str(spec.results_path) == "artifacts/sweeps/S14_ffn/s14_ffn_metrics.csv"


def test_phase_37_spec_matches_loss_contract() -> None:
    spec = get_sweep_phase_spec(37)
    assert spec.phase_name == "S15 Loss Sweep"
    assert spec.family_id == "S15_LOSS"
    assert spec.condition_path == ("training", "loss_name")
    assert spec.condition_values == (("L0", "MSE"), ("L1", "HUBER"))
    assert spec.reference_condition == "L0"
    assert str(spec.manifest_path) == "artifacts/sweeps/S15_loss/s15_loss_sweep_manifest.json"
    assert str(spec.results_path) == "artifacts/sweeps/S15_loss/s15_loss_metrics.csv"


def test_phase_24_condition_values_follow_phase_23_winner(tmp_path: Path) -> None:
    run_id = "RUN-S1-WINNER"
    write_json(
        tmp_path / f"artifacts/runs/{run_id}/config.json",
        {
            "run_id": run_id,
            "config_fingerprint": "winner-fingerprint",
            "config": {"data": {"feature_variant_id": "FS2_TF1"}},
        },
    )
    write_json(
        tmp_path / "artifacts/sweeps/s1_feature_set/s1_reference_update.json",
        {"winner_run_id": run_id},
    )
    assert resolve_condition_values(24, tmp_path) == {
        "TF0": "FS2_TF0",
        "TF1": "FS2_TF1",
    }


def test_inspector_reports_upstream_invalid_without_mutation(tmp_path: Path) -> None:
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    result = inspect_phase_state(30, tmp_path)
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert result["state"] == "UPSTREAM_INVALID"
    assert result["action"] == "BLOCK"
    assert before == after


def test_phase_31_reports_exact_phase_30_block_reasons(tmp_path: Path) -> None:
    result = plan_phase_resume(31, tmp_path)
    assert result["state"] == "UPSTREAM_INVALID"
    assert result["effective_action"] == "BLOCK"
    assert result["reasons"] == [
        "artifacts/sweeps/s8_learning_rate/phase_30_signoff.json: MISSING",
        "artifacts/sweeps/s8_learning_rate/s8_learning_rate_winner.json: MISSING",
        "artifacts/sweeps/s8_learning_rate/s8_reference_update.json: MISSING",
        "artifacts/experiments/experiment_registry.jsonl: MISSING",
        "artifacts/environment/phase_1_signoff.json: MISSING",
        "artifacts/environment/environment_report.json: MISSING",
    ]


def test_inspector_reports_incomplete_conditions_before_missing_signoff(tmp_path: Path) -> None:
    create_phase_22_prerequisite(tmp_path)
    result = inspect_phase_state(23, tmp_path)
    assert result["state"] == "CONDITION_INCOMPLETE"
    assert result["signoff"]["status"] == "MISSING"


def test_inspector_reports_incomplete_conditions_before_stale_signoff(tmp_path: Path) -> None:
    create_phase_22_prerequisite(tmp_path)
    create_valid_environment(tmp_path)
    write_registry(
        tmp_path,
        [
            create_completed_run(
                tmp_path,
                "RUN-FS1",
                "TRANSFORMER_BASELINE",
                {"data": {"feature_variant_id": "FS1_TF1"}},
                60.0,
            )
        ],
    )
    spec = get_sweep_phase_spec(23)
    write_json(
        tmp_path / spec.signoff_path,
        {
            "status": "PASS",
            "output_paths": [str(spec.results_path)],
            "output_checksums": {str(spec.results_path): "stale"},
        },
    )
    inspection = inspect_phase_state(23, tmp_path)
    decision = plan_phase_resume(23, tmp_path, allow_execution=True)
    assert inspection["state"] == "CONDITION_INCOMPLETE"
    assert inspection["conditions"]["missing_conditions"] == ["FS0_TF1", "FS2_TF1"]
    assert decision["resolved_action"] == "EXECUTE_MISSING_ONLY"
    assert decision["effective_action"] == "EXECUTE_MISSING_ONLY"


def test_inspector_rejects_missing_declared_artifact(tmp_path: Path) -> None:
    create_valid_phase_23(tmp_path)
    (tmp_path / get_sweep_phase_spec(23).results_path).unlink()
    result = inspect_phase_state(23, tmp_path)
    assert result["state"] == "SIGNOFF_INVALID"
    assert any(item["reason"] == "MISSING" for item in result["signoff"]["issues"])


def test_inspector_reports_log_missing(tmp_path: Path) -> None:
    create_valid_phase_23(tmp_path, include_log=False)
    result = inspect_phase_state(23, tmp_path)
    assert result["state"] == "LOG_MISSING"
    assert result["action"] == "REBUILD_LOG_ONLY"


def test_inspector_reports_log_stale(tmp_path: Path) -> None:
    create_valid_phase_23(tmp_path)
    spec = get_sweep_phase_spec(23)
    manifest = root_manifest = tmp_path / spec.manifest_path
    manifest.write_text("{}\n", encoding="utf-8")
    result = inspect_phase_state(23, tmp_path)
    assert result["state"] == "SIGNOFF_INVALID"
    assert root_manifest.is_file()


def test_inspector_reports_valid_reusable(tmp_path: Path) -> None:
    create_valid_phase_23(tmp_path)
    result = inspect_phase_state(23, tmp_path)
    assert result["state"] == "VALID_REUSABLE"
    assert result["action"] == "RENDER_ONLY"


def test_resolver_returns_expected_minus_verified_conditions(tmp_path: Path) -> None:
    create_phase_22_prerequisite(tmp_path)
    records = [
        create_completed_run(
            tmp_path,
            "RUN-FS1",
            "TRANSFORMER_BASELINE",
            {"data": {"feature_variant_id": "FS1_TF1"}},
            60.0,
        )
    ]
    write_registry(tmp_path, records)
    result = resolve_phase_conditions(23, tmp_path)
    assert [item["condition_id"] for item in result["verified_conditions"]] == ["FS1_TF1"]
    assert result["missing_conditions"] == ["FS0_TF1", "FS2_TF1"]


def test_phase_30_rejects_lr3_attribution_from_lr2_config(tmp_path: Path) -> None:
    signoff_path = tmp_path / "artifacts/sweeps/s7_batch_size/phase_29_signoff.json"
    write_json(signoff_path, {"status": "PASS", "output_paths": [], "output_checksums": {}})
    write_json(tmp_path / "artifacts/sweeps/s7_batch_size/s7_batch_winner.json", {"winner_run_id": "RUN-LR2"})
    write_json(tmp_path / "artifacts/sweeps/s7_batch_size/s7_reference_update.json", {"winner_run_id": "RUN-LR2"})
    write_registry(
        tmp_path,
        [
            create_completed_run(
                tmp_path,
                "RUN-LR2",
                "S7_BATCH_SIZE",
                {"training": {"learning_rate": 0.0003}},
                60.0,
            )
        ],
    )
    result = resolve_phase_conditions(30, tmp_path)
    assert [item["condition_id"] for item in result["verified_conditions"]] == ["LR2"]
    assert result["missing_conditions"] == ["LR1", "LR3"]


def test_execution_readiness_requires_signed_current_environment(tmp_path: Path) -> None:
    create_phase_22_prerequisite(tmp_path)
    result = inspect_execution_readiness(23, tmp_path)
    assert result["ready"] is False
    assert result["state"] == "ENVIRONMENT_INVALID"
    create_valid_environment(tmp_path)
    result = inspect_execution_readiness(23, tmp_path)
    assert result["ready"] is True


def test_resume_plan_blocks_training_without_authorization(tmp_path: Path) -> None:
    create_phase_22_prerequisite(tmp_path)
    create_valid_environment(tmp_path)
    write_registry(tmp_path, [])
    result = plan_phase_resume(23, tmp_path, allow_execution=False)
    assert result["resolved_action"] == "EXECUTE_MISSING_ONLY"
    assert result["effective_action"] == "BLOCK"
    assert result["reasons"] == ["SCIENTIFIC_EXECUTION_NOT_AUTHORIZED"]


def test_condition_runner_has_no_upstream_materialization() -> None:
    project_root = Path(__file__).resolve().parents[2]
    source = (project_root / "scripts/run_single_condition.py").read_text(encoding="utf-8")
    assert "materialize_all" not in source
    assert "materialize_phase_0" not in source
    assert "environment_inventory" not in source
    assert "sys.path.insert" not in source


def test_validate_sweep_signoff_accepts_complete_declared_artifacts(tmp_path: Path) -> None:
    create_valid_phase_23(tmp_path)
    result = validate_sweep_signoff(23, tmp_path)
    assert result["valid"] is True
    assert result["issues"] == []


def test_validate_sweep_signoff_rejects_missing_declared_artifact(tmp_path: Path) -> None:
    create_valid_phase_23(tmp_path)
    (tmp_path / get_sweep_phase_spec(23).results_path).unlink()
    result = validate_sweep_signoff(23, tmp_path)
    assert result["valid"] is False
    assert any(item["reason"] == "MISSING" for item in result["issues"])


def test_validate_sweep_signoff_rejects_checksum_mismatch(tmp_path: Path) -> None:
    create_valid_phase_23(tmp_path)
    (tmp_path / get_sweep_phase_spec(23).results_path).write_text("changed\n", encoding="utf-8")
    result = validate_sweep_signoff(23, tmp_path)
    assert result["valid"] is False
    assert any(item["reason"] == "CHECKSUM_MISMATCH" for item in result["issues"])


def test_materializer_refuses_stale_pass_signoff(tmp_path: Path) -> None:
    create_valid_phase_23(tmp_path)
    (tmp_path / get_sweep_phase_spec(23).results_path).unlink()
    with pytest.raises(RuntimeError, match="not reusable"):
        materialize_phase_23(tmp_path)
