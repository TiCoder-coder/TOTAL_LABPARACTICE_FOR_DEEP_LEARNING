import json
from pathlib import Path

import pytest

from course_work.experiments.phase_execution import inspect_phase_state
from course_work.sweeps.sweep_results import finalize_verified_sweep, validate_sweep_signoff
from course_work.utils.artifacts import read_json, sha256_file


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def create_phase_22(root: Path) -> None:
    summary = root / "artifacts/learning_diagnostics/learning_diagnostics_summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("model,validation_rmse_wh\ntransformer,60\n", encoding="utf-8")
    relative_summary = str(summary.relative_to(root))
    write_json(
        root / "artifacts/learning_diagnostics/phase_22_signoff.json",
        {
            "status": "PASS",
            "output_paths": [relative_summary],
            "output_checksums": {relative_summary: sha256_file(summary)},
        },
    )


def create_run(root: Path, condition_id: str, rmse: float, include_test: bool = False) -> dict:
    run_id = f"RUN-{condition_id}"
    family = "TRANSFORMER_BASELINE" if condition_id == "FS1_TF1" else "S1_FEATURE_SET"
    run_root = root / "artifacts/runs" / run_id
    config_payload = {
        "run_id": run_id,
        "config_fingerprint": f"fingerprint-{condition_id}",
        "config": {
            "data": {
                "feature_variant_id": condition_id,
                "target_scaling_option": "YS1",
                "lookback_steps": 144,
            },
            "model": {"pooling": "LAST_STEP", "activation": "GELU"},
            "training": {
                "batch_size": 64,
                "optimizer_name": "AdamW",
                "learning_rate": 0.0003,
                "weight_decay": 0.0001,
            },
            "lineage": {
                "population_fingerprint": "population-fingerprint",
                "metric_version": "METRICS-v1",
            },
        },
    }
    write_json(run_root / "config.json", config_payload)
    write_json(run_root / "status.json", {"run_id": run_id, "status": "COMPLETED"})
    write_json(run_root / "metrics/best_validation_metrics.json", {"rmse_wh": rmse})
    artifact_paths = (
        ("CONFIG", run_root / "config.json"),
        ("STATUS", run_root / "status.json"),
        ("METRICS", run_root / "metrics/best_validation_metrics.json"),
    )
    metrics = [
        {"split_id": "VALIDATION", "status": "PASS", "metric_name": "mae_wh", "metric_value": rmse - 1.0},
        {"split_id": "VALIDATION", "status": "PASS", "metric_name": "rmse_wh", "metric_value": rmse},
        {"split_id": "VALIDATION", "status": "PASS", "metric_name": "r2", "metric_value": 0.5},
    ]
    if include_test:
        metrics.append({"split_id": "TEST", "status": "PASS", "metric_name": "rmse_wh", "metric_value": rmse})
    return {
        "run_id": run_id,
        "status": "COMPLETED",
        "experiment_family": family,
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
        "metrics": metrics,
    }


def create_registry(root: Path, include_test: bool = False) -> None:
    records = [
        create_run(root, "FS0_TF1", 60.0, include_test),
        create_run(root, "FS1_TF1", 60.0),
        create_run(root, "FS2_TF1", 62.0),
    ]
    path = root / "artifacts/experiments/experiment_registry.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(record, sort_keys=True)}\n" for record in records), encoding="utf-8")


def create_phase_31(root: Path) -> None:
    winner_path = root / "artifacts/sweeps/S9_weight_decay/s9_weight_decay_winner.json"
    reference_path = root / "artifacts/sweeps/S9_weight_decay/s9_reference_update.json"
    write_json(winner_path, {"winner_run_id": "RUN-DR01", "status": "PASS"})
    write_json(reference_path, {"winner_run_id": "RUN-DR01", "approved_for_phase32": True})
    output_paths = [str(winner_path.relative_to(root)), str(reference_path.relative_to(root))]
    write_json(
        root / "artifacts/sweeps/S9_weight_decay/phase_31_signoff.json",
        {
            "status": "PASS",
            "output_paths": output_paths,
            "output_checksums": {path: sha256_file(root / path) for path in output_paths},
        },
    )


def create_dropout_run(root: Path, condition_id: str, dropout: float, rmse: float) -> dict:
    run_id = f"RUN-{condition_id}"
    run_root = root / "artifacts/runs" / run_id
    config_payload = {
        "run_id": run_id,
        "config_fingerprint": f"fingerprint-{condition_id}",
        "config": {
            "data": {
                "feature_variant_id": "FS1_TF1",
                "target_scaling_option": "YS1",
                "lookback_steps": 144,
            },
            "model": {
                "input_size": 31,
                "d_model": 64,
                "num_heads": 4,
                "num_layers": 2,
                "ffn_dim": 128,
                "dropout": dropout,
                "pooling": "LAST_STEP",
                "activation": "GELU",
            },
            "training": {
                "batch_size": 64,
                "optimizer_name": "AdamW",
                "learning_rate": 0.0003,
                "weight_decay": 0.0001,
            },
            "lineage": {
                "population_fingerprint": "population-fingerprint",
                "metric_version": "METRICS-v1",
            },
        },
    }
    write_json(run_root / "config.json", config_payload)
    write_json(run_root / "status.json", {"run_id": run_id, "status": "COMPLETED"})
    write_json(run_root / "metrics/best_validation_metrics.json", {"rmse_wh": rmse})
    artifact_paths = (
        ("CONFIG", run_root / "config.json"),
        ("STATUS", run_root / "status.json"),
        ("METRICS", run_root / "metrics/best_validation_metrics.json"),
    )
    return {
        "run_id": run_id,
        "status": "COMPLETED",
        "experiment_family": "S9_WEIGHT_DECAY" if condition_id == "DR01" else "S10_DROPOUT",
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
            {"split_id": "VALIDATION", "status": "PASS", "metric_name": "mae_wh", "metric_value": rmse - 1.0},
            {"split_id": "VALIDATION", "status": "PASS", "metric_name": "rmse_wh", "metric_value": rmse},
            {"split_id": "VALIDATION", "status": "PASS", "metric_name": "r2", "metric_value": 0.5},
        ],
    }


def create_dropout_registry(root: Path) -> None:
    records = [
        create_dropout_run(root, "DR01", 0.1, 60.0),
        create_dropout_run(root, "DR02", 0.2, 60.0),
        create_dropout_run(root, "DR03", 0.3, 62.0),
    ]
    path = root / "artifacts/experiments/experiment_registry.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(record, sort_keys=True)}\n" for record in records), encoding="utf-8")


def create_phase_32(root: Path) -> None:
    winner_path = root / "artifacts/sweeps/S10_dropout/s10_dropout_winner.json"
    reference_path = root / "artifacts/sweeps/S10_dropout/s10_reference_update.json"
    write_json(winner_path, {"winner_run_id": "RUN-D64", "status": "PASS"})
    write_json(reference_path, {"winner_run_id": "RUN-D64", "approved_for_phase33": True})
    output_paths = [str(winner_path.relative_to(root)), str(reference_path.relative_to(root))]
    write_json(
        root / "artifacts/sweeps/S10_dropout/phase_32_signoff.json",
        {
            "status": "PASS",
            "output_paths": output_paths,
            "output_checksums": {path: sha256_file(root / path) for path in output_paths},
        },
    )


def create_d_model_run(root: Path, condition_id: str, d_model: int, rmse: float) -> dict:
    run_id = f"RUN-{condition_id}"
    run_root = root / "artifacts/runs" / run_id
    config_payload = {
        "run_id": run_id,
        "config_fingerprint": f"fingerprint-{condition_id}",
        "config": {
            "data": {
                "feature_variant_id": "FS1_TF1",
                "target_scaling_option": "YS1",
                "lookback_steps": 144,
            },
            "model": {
                "input_size": 31,
                "d_model": d_model,
                "num_heads": 4,
                "num_layers": 2,
                "ffn_dim": 128,
                "dropout": 0.1,
                "pooling": "LAST_STEP",
                "activation": "GELU",
            },
            "training": {
                "batch_size": 64,
                "optimizer_name": "AdamW",
                "learning_rate": 0.0003,
                "weight_decay": 0.001,
            },
            "lineage": {
                "population_fingerprint": "population-fingerprint",
                "metric_version": "METRICS-v1",
            },
        },
    }
    write_json(run_root / "config.json", config_payload)
    write_json(run_root / "status.json", {"run_id": run_id, "status": "COMPLETED"})
    write_json(run_root / "metrics/best_validation_metrics.json", {"rmse_wh": rmse})
    artifact_paths = (
        ("CONFIG", run_root / "config.json"),
        ("STATUS", run_root / "status.json"),
        ("METRICS", run_root / "metrics/best_validation_metrics.json"),
    )
    return {
        "run_id": run_id,
        "status": "COMPLETED",
        "experiment_family": "S11_D_MODEL" if condition_id == "D32" else "S10_DROPOUT",
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
            {"split_id": "VALIDATION", "status": "PASS", "metric_name": "mae_wh", "metric_value": rmse - 1.0},
            {"split_id": "VALIDATION", "status": "PASS", "metric_name": "rmse_wh", "metric_value": rmse},
            {"split_id": "VALIDATION", "status": "PASS", "metric_name": "r2", "metric_value": 0.5},
        ],
    }


def create_d_model_registry(root: Path) -> None:
    records = [
        create_d_model_run(root, "D32", 32, 60.0),
        create_d_model_run(root, "D64", 64, 60.0),
    ]
    path = root / "artifacts/experiments/experiment_registry.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(record, sort_keys=True)}\n" for record in records), encoding="utf-8")


def create_phase_33(root: Path) -> None:
    winner_path = root / "artifacts/sweeps/S11_d_model/s11_d_model_winner.json"
    reference_path = root / "artifacts/sweeps/S11_d_model/s11_reference_update.json"
    write_json(winner_path, {"winner_run_id": "RUN-H4", "status": "PASS"})
    write_json(reference_path, {"winner_run_id": "RUN-H4", "approved_for_phase34": True})
    output_paths = [str(winner_path.relative_to(root)), str(reference_path.relative_to(root))]
    write_json(
        root / "artifacts/sweeps/S11_d_model/phase_33_signoff.json",
        {
            "status": "PASS",
            "output_paths": output_paths,
            "output_checksums": {path: sha256_file(root / path) for path in output_paths},
        },
    )


def create_head_run(root: Path, condition_id: str, num_heads: int, rmse: float) -> dict:
    run_id = f"RUN-{condition_id}"
    run_root = root / "artifacts/runs" / run_id
    config_payload = {
        "run_id": run_id,
        "config_fingerprint": f"fingerprint-{condition_id}",
        "config": {
            "data": {
                "feature_variant_id": "FS1_TF1",
                "target_scaling_option": "YS1",
                "lookback_steps": 36,
            },
            "model": {
                "input_size": 31,
                "d_model": 64,
                "num_heads": num_heads,
                "num_layers": 2,
                "ffn_dim": 128,
                "dropout": 0.1,
                "pooling": "LAST_STEP",
                "activation": "GELU",
            },
            "training": {
                "batch_size": 32,
                "optimizer_name": "AdamW",
                "learning_rate": 0.0003,
                "weight_decay": 0.001,
            },
            "lineage": {
                "population_fingerprint": "population-fingerprint",
                "metric_version": "METRICS-v1",
            },
        },
    }
    write_json(run_root / "config.json", config_payload)
    write_json(run_root / "status.json", {"run_id": run_id, "status": "COMPLETED"})
    write_json(run_root / "metrics/best_validation_metrics.json", {"rmse_wh": rmse})
    artifact_paths = (
        ("CONFIG", run_root / "config.json"),
        ("STATUS", run_root / "status.json"),
        ("METRICS", run_root / "metrics/best_validation_metrics.json"),
    )
    return {
        "run_id": run_id,
        "status": "COMPLETED",
        "experiment_family": "S12_HEADS" if condition_id == "H2" else "S11_D_MODEL",
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
            {"split_id": "VALIDATION", "status": "PASS", "metric_name": "mae_wh", "metric_value": rmse - 1.0},
            {"split_id": "VALIDATION", "status": "PASS", "metric_name": "rmse_wh", "metric_value": rmse},
            {"split_id": "VALIDATION", "status": "PASS", "metric_name": "r2", "metric_value": 0.5},
        ],
    }


def create_head_registry(root: Path) -> None:
    records = [
        create_head_run(root, "H2", 2, 60.0),
        create_head_run(root, "H4", 4, 60.0),
    ]
    path = root / "artifacts/experiments/experiment_registry.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(record, sort_keys=True)}\n" for record in records), encoding="utf-8")


def test_finalizer_builds_verified_outputs_and_exact_tie_winner(tmp_path: Path) -> None:
    create_phase_22(tmp_path)
    create_registry(tmp_path)
    signoff = finalize_verified_sweep(23, tmp_path)
    winner = read_json(tmp_path / "artifacts/sweeps/s1_feature_set/s1_feature_set_winner.json")
    reference = read_json(tmp_path / "artifacts/sweeps/s1_feature_set/s1_reference_update.json")
    assert signoff["status"] == "PASS"
    assert signoff["test_status"] == "FORBIDDEN"
    assert winner["winner_variant_id"] == "FS0_TF1"
    assert winner["exact_tie_applied"] is True
    assert reference["approved_for_phase24"] is True
    assert validate_sweep_signoff(23, tmp_path)["valid"] is True
    assert inspect_phase_state(23, tmp_path)["state"] == "LOG_MISSING"


def test_finalizer_preserves_stale_outputs_in_history(tmp_path: Path) -> None:
    create_phase_22(tmp_path)
    create_registry(tmp_path)
    stale_root = tmp_path / "artifacts/sweeps/s1_feature_set"
    write_json(stale_root / "sweep_manifest.json", {"status": "STALE"})
    write_json(stale_root / "phase_23_signoff.json", {"status": "PASS"})
    signoff = finalize_verified_sweep(23, tmp_path, replace_stale=True)
    assert signoff["archived_stale_paths"]
    for relative_path in signoff["archived_stale_paths"]:
        assert (tmp_path / relative_path).is_file()


def test_finalizer_rejects_test_metrics(tmp_path: Path) -> None:
    create_phase_22(tmp_path)
    create_registry(tmp_path, include_test=True)
    with pytest.raises(RuntimeError, match="Test metric access is forbidden"):
        finalize_verified_sweep(23, tmp_path)
    assert not (tmp_path / "artifacts/sweeps/s1_feature_set/results.csv").exists()


def test_phase_32_finalizer_selects_lower_dropout_on_exact_tie(tmp_path: Path) -> None:
    create_phase_31(tmp_path)
    create_dropout_registry(tmp_path)
    signoff = finalize_verified_sweep(32, tmp_path)
    winner = read_json(tmp_path / "artifacts/sweeps/S10_dropout/s10_dropout_winner.json")
    reference = read_json(tmp_path / "artifacts/sweeps/S10_dropout/s10_reference_update.json")
    assert signoff["status"] == "PASS"
    assert signoff["test_status"] == "FORBIDDEN"
    assert winner["winner_dropout_id"] == "DR01"
    assert winner["winner_dropout_probability"] == 0.1
    assert winner["exact_tie_applied"] is True
    assert winner["dropout_scope_fingerprint"]
    assert reference["selected_dropout_id"] == "DR01"
    assert reference["selected_dropout_probability"] == 0.1
    assert reference["approved_for_phase33"] is True
    assert validate_sweep_signoff(32, tmp_path)["valid"] is True
    assert inspect_phase_state(32, tmp_path)["state"] == "LOG_MISSING"


def test_phase_33_finalizer_selects_d32_on_exact_tie_and_records_capacity(tmp_path: Path) -> None:
    create_phase_32(tmp_path)
    create_d_model_registry(tmp_path)
    signoff = finalize_verified_sweep(33, tmp_path)
    winner = read_json(tmp_path / "artifacts/sweeps/S11_d_model/s11_d_model_winner.json")
    reference = read_json(tmp_path / "artifacts/sweeps/S11_d_model/s11_reference_update.json")
    assert signoff["status"] == "PASS"
    assert signoff["test_status"] == "FORBIDDEN"
    assert signoff["d32_parameter_count"] < signoff["d64_parameter_count"]
    assert winner["winner_d_model_id"] == "D32"
    assert winner["winner_d_model"] == 32
    assert winner["exact_tie_applied"] is True
    assert winner["winner_trainable_parameters"] < winner["runner_up_trainable_parameters"]
    assert reference["selected_d_model_id"] == "D32"
    assert reference["selected_d_model"] == 32
    assert reference["approved_for_phase34"] is True
    assert validate_sweep_signoff(33, tmp_path)["valid"] is True
    assert inspect_phase_state(33, tmp_path)["state"] == "LOG_MISSING"


def test_phase_34_finalizer_selects_h2_on_exact_tie_and_records_geometry(tmp_path: Path) -> None:
    create_phase_33(tmp_path)
    create_head_registry(tmp_path)
    signoff = finalize_verified_sweep(34, tmp_path)
    winner = read_json(tmp_path / "artifacts/sweeps/S12_heads/s12_head_winner.json")
    reference = read_json(tmp_path / "artifacts/sweeps/S12_heads/s12_reference_update.json")
    assert signoff["status"] == "PASS"
    assert signoff["test_status"] == "FORBIDDEN"
    assert signoff["h2_parameter_count"] == signoff["h4_parameter_count"]
    assert winner["winner_head_id"] == "H2"
    assert winner["winner_num_heads"] == 2
    assert winner["winner_head_dim"] == 32
    assert winner["exact_tie_applied"] is True
    assert winner["parameter_schema_equal"] is True
    assert reference["selected_head_id"] == "H2"
    assert reference["selected_num_heads"] == 2
    assert reference["selected_head_dim"] == 32
    assert reference["approved_for_phase35"] is True
    required_outputs = {
        "s12_head_sweep_manifest.json",
        "s12_head_sweep_contract.json",
        "s12_head_preflight_audit.csv",
        "s12_run_matrix.csv",
        "s12_head_definition_audit.csv",
        "s12_mha_geometry_audit.csv",
        "s12_architecture_role_audit.csv",
        "s12_parameter_schema_audit.csv",
        "s12_parameter_count_audit.csv",
        "s12_config_delta_audit.csv",
        "s12_head_unit_tests.csv",
        "s12_common_data_audit.csv",
        "s12_head_training_audit.csv",
        "s12_initialization_audit.csv",
        "s12_sample_order_audit.csv",
        "s12_dropout_scope_audit.csv",
        "s12_attention_api_audit.csv",
        "s12_optimizer_budget_audit.csv",
        "s12_head_run_provenance.csv",
        "s12_head_metrics.csv",
        "s12_head_effect.csv",
        "s12_head_efficiency_context.csv",
        "s12_optimization_diagnostics.csv",
        "s12_convergence_diagnostics.csv",
        "s12_runtime_diagnostics.csv",
        "s12_hypothesis_outcomes.csv",
        "s12_head_findings.csv",
        "s12_head_winner.json",
        "s12_reference_update.json",
        "s12_head_sweep_tests.csv",
        "s12_head_discrepancies.json",
        "s12_head_sweep_summary.json",
        "s12_head_sweep_report.md",
        "README_S12_HEAD_SWEEP.md",
        "figures/S12_01_validation_rmse_by_epoch.png",
        "figures/S12_02_validation_mae_by_epoch.png",
        "figures/S12_03_train_loss_by_epoch.png",
        "figures/S12_04_gradient_clipping_fraction.png",
        "figures/S12_05_best_validation_metrics.png",
        "figures/S12_06_runtime_vs_rmse.png",
        "figures/S12_07_convergence_summary.png",
    }
    artifact_root = tmp_path / "artifacts/sweeps/S12_heads"
    assert all((artifact_root / path).is_file() for path in required_outputs)
    assert (artifact_root / "figures/S12_01_validation_rmse_by_epoch.png").read_bytes().startswith(b"\x89PNG")
    assert {str(Path(path).relative_to("artifacts/sweeps/S12_heads")) for path in signoff["output_paths"] if path.startswith("artifacts/sweeps/S12_heads/")} >= required_outputs
    assert validate_sweep_signoff(34, tmp_path)["valid"] is True
    assert inspect_phase_state(34, tmp_path)["state"] == "LOG_MISSING"
