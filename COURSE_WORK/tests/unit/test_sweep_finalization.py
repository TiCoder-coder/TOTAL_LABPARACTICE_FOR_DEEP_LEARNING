import json
from pathlib import Path

import pytest

from course_work.experiments.phase_execution import inspect_phase_state
from course_work.reporting.phase_summary import build_phase_resume_log
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


def create_phase_33(root: Path, winner_rmse: float = 60.0) -> None:
    winner_path = root / "artifacts/sweeps/S11_d_model/s11_d_model_winner.json"
    reference_path = root / "artifacts/sweeps/S11_d_model/s11_reference_update.json"
    shared = {
        "winner_run_id": "RUN-H4",
        "winner_config_fingerprint": "fingerprint-H4",
        "feature_variant_id": "FS1_TF1",
        "target_scaling_id": "YS1",
        "lookback_id": "L36",
        "pooling_id": "LAST_STEP",
        "activation_id": "GELU",
        "batch_id": "B32",
        "learning_rate": 0.0003,
        "weight_decay": 0.001,
        "dropout": 0.1,
        "d_model": 64,
        "num_heads": 4,
        "num_layers": 2,
        "ffn_dim": 128,
        "population_fingerprint": "population-fingerprint",
        "metric_version": "METRICS-v1",
        "test_status": "FORBIDDEN",
    }
    write_json(
        winner_path,
        {**shared, "status": "PASS", "winner_d_model": 64, "winner_rmse_wh": winner_rmse, "winner_mae_wh": winner_rmse - 1.0, "winner_r2": 0.5},
    )
    write_json(
        reference_path,
        {**shared, "approved_for_phase34": True, "selected_d_model": 64, "current_num_heads": 4, "current_head_dim": 16, "winner_rmse_wh": winner_rmse},
    )
    output_paths = [str(winner_path.relative_to(root)), str(reference_path.relative_to(root))]
    write_json(
        root / "artifacts/sweeps/S11_d_model/phase_33_signoff.json",
        {
            "status": "PASS",
            "approved_for_phase34": True,
            "test_status": "FORBIDDEN",
            "winner_run_id": "RUN-H4",
            "winner_d_model": 64,
            "winner_rmse_wh": winner_rmse,
            "population_fingerprint": "population-fingerprint",
            "metric_version": "METRICS-v1",
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
                "boundary_protocol": "WB0_CONTEXT_CARRY_OVER",
                "target_access_mode": "VALIDATION",
                "train_sample_count": 2,
                "validation_sample_count": 2,
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
                "loss_name": "MSE",
                "max_epochs": 50,
                "early_stopping_patience": 10,
                "early_stopping_min_delta": 0,
                "gradient_clipping_enabled": True,
                "gradient_clip_max_norm": 1.0,
                "scheduler_name": None,
            },
            "lineage": {
                "population_fingerprint": "population-fingerprint",
                "metric_version": "METRICS-v1",
            },
            "reproducibility": {"seed": 42},
        },
    }
    write_json(run_root / "config.json", config_payload)
    write_json(run_root / "status.json", {"run_id": run_id, "status": "COMPLETED", "best_epoch": 15, "best_validation_rmse_wh": rmse})
    write_json(
        run_root / "metrics/best_validation_metrics.json",
        {"metric_result": {"run_id": run_id, "rmse_wh": rmse, "mae_wh": rmse - 1.0, "r2": 0.5, "population_fingerprint": "population-fingerprint", "metric_version": "METRICS-v1", "split_id": "VALIDATION"}},
    )
    artifact_paths = (
        ("CONFIG", run_root / "config.json"),
        ("STATUS", run_root / "status.json"),
        ("METRICS", run_root / "metrics/best_validation_metrics.json"),
    )
    artifacts = [
        {
            "artifact_type": artifact_type,
            "artifact_path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
            "required": True,
        }
        for artifact_type, path in artifact_paths
    ]
    if condition_id == "H4":
        artifacts.extend(
            [
                {"artifact_type": "TRAIN_LOG", "artifact_path": f"artifacts/runs/{run_id}/training.log", "sha256": "f709354699e8fb103b226fbae64ac2d2611fd80854ae1342246e86c94415d5d7", "required": True},
                {"artifact_type": "BEST_CHECKPOINT", "artifact_path": f"artifacts/runs/{run_id}/checkpoints/best_checkpoint.pt", "sha256": "3ccf735488336340275a4dccf040fcd17b98a4fa4746d3e665cf14f97428d75d", "required": True},
                {"artifact_type": "PREDICTIONS", "artifact_path": f"artifacts/runs/{run_id}/predictions/best_validation_predictions.csv", "sha256": "9bb212edb469f8fac02cf9179a401fec04d27487131a1e17a5e14179b94a5a4e", "required": False},
            ]
        )
    return {
        "run_id": run_id,
        "status": "COMPLETED",
        "experiment_family": "S12_HEADS" if condition_id == "H2" else "S11_D_MODEL",
        "config_fingerprint": config_payload["config_fingerprint"],
        "config": config_payload["config"],
        "best_epoch": 15,
        "test_access_authorized": False,
        "artifacts": artifacts,
        "metrics": [
            {"split_id": "VALIDATION", "status": "PASS", "metric_name": "mae_wh", "metric_value": rmse - 1.0, "population_fingerprint": "population-fingerprint", "metric_version": "METRICS-v1", "epoch_or_checkpoint": "epoch_15"},
            {"split_id": "VALIDATION", "status": "PASS", "metric_name": "rmse_wh", "metric_value": rmse, "population_fingerprint": "population-fingerprint", "metric_version": "METRICS-v1", "epoch_or_checkpoint": "epoch_15"},
            {"split_id": "VALIDATION", "status": "PASS", "metric_name": "r2", "metric_value": 0.5, "population_fingerprint": "population-fingerprint", "metric_version": "METRICS-v1", "epoch_or_checkpoint": "epoch_15"},
        ],
    }


def create_head_registry(root: Path, h4_rmse: float = 60.0) -> None:
    records = [
        create_head_run(root, "H2", 2, 60.0),
        create_head_run(root, "H4", 4, h4_rmse),
    ]
    path = root / "artifacts/experiments/experiment_registry.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(record, sort_keys=True)}\n" for record in records), encoding="utf-8")


def create_layer_n1_run(root: Path, rmse: float = 61.0) -> None:
    reference = read_json(root / "artifacts/runs/RUN-H4/config.json")["config"]
    config = json.loads(json.dumps(reference))
    config["model"]["num_layers"] = 1
    run_id = "RUN-N1"
    run_root = root / "artifacts/runs" / run_id
    config_path = run_root / "config.json"
    status_path = run_root / "status.json"
    metrics_path = run_root / "metrics/best_validation_metrics.json"
    checkpoint_path = run_root / "checkpoints/best_checkpoint.pt"
    training_log_path = run_root / "training.log"
    history_path = run_root / "training_history.csv"
    predictions_path = run_root / "predictions/best_validation_predictions.csv"
    write_json(config_path, {"run_id": run_id, "config_fingerprint": "fingerprint-N1", "config": config})
    write_json(status_path, {"run_id": run_id, "status": "COMPLETED", "best_epoch": 15, "best_validation_rmse_wh": rmse})
    write_json(
        metrics_path,
        {
            "metric_result": {
                "run_id": run_id,
                "rmse_wh": rmse,
                "mae_wh": rmse - 1.0,
                "r2": 0.4,
                "n_samples": 2,
                "population_fingerprint": "population-fingerprint",
                "metric_version": "METRICS-v1",
                "split_id": "VALIDATION",
                "finite_status": "PASS",
            }
        },
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"phase35-n1-checkpoint")
    training_log_path.write_text("phase35 n1 complete\n", encoding="utf-8")
    history_path.write_text(
        "epoch,train_loss,validation_rmse_wh,validation_mae_wh,validation_r2,epoch_seconds\n"
        + "".join(f"{epoch},1.0,{rmse},{rmse - 1.0},0.4,1.0\n" for epoch in range(1, 26)),
        encoding="utf-8",
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(
        "run_id,sample_idx,y_true_wh,y_pred_wh,residual_wh\n"
        f"{run_id},2,10.0,9.0,1.0\n"
        f"{run_id},3,20.0,18.0,2.0\n",
        encoding="utf-8",
    )
    artifacts = []
    for artifact_type, path, required in (
        ("CONFIG", config_path, True), ("STATUS", status_path, True),
        ("METRICS", metrics_path, True), ("BEST_CHECKPOINT", checkpoint_path, True),
        ("TRAIN_LOG", training_log_path, True), ("PREDICTIONS", predictions_path, False),
    ):
        artifacts.append(
            {"artifact_type": artifact_type, "artifact_path": str(path.relative_to(root)), "sha256": sha256_file(path), "required": required}
        )
    record = {
        "run_id": run_id,
        "status": "COMPLETED",
        "experiment_family": "S13_LAYERS",
        "config_fingerprint": "fingerprint-N1",
        "config": config,
        "best_epoch": 15,
        "best_validation_rmse_wh": rmse,
        "test_access_authorized": False,
        "artifacts": artifacts,
        "metrics": [
            {"split_id": "VALIDATION", "status": "PASS", "metric_name": "mae_wh", "metric_value": rmse - 1.0, "population_fingerprint": "population-fingerprint", "metric_version": "METRICS-v1", "epoch_or_checkpoint": "epoch_15"},
            {"split_id": "VALIDATION", "status": "PASS", "metric_name": "rmse_wh", "metric_value": rmse, "population_fingerprint": "population-fingerprint", "metric_version": "METRICS-v1", "epoch_or_checkpoint": "epoch_15"},
            {"split_id": "VALIDATION", "status": "PASS", "metric_name": "r2", "metric_value": 0.4, "population_fingerprint": "population-fingerprint", "metric_version": "METRICS-v1", "epoch_or_checkpoint": "epoch_15"},
        ],
    }
    registry_path = root / "artifacts/experiments/experiment_registry.jsonl"
    with registry_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


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
    create_head_registry(tmp_path)
    create_phase_33(tmp_path)
    signoff = finalize_verified_sweep(34, tmp_path)
    winner = read_json(tmp_path / "artifacts/sweeps/S12_heads/s12_head_winner.json")
    reference = read_json(tmp_path / "artifacts/sweeps/S12_heads/s12_reference_update.json")
    assert signoff["status"] == "PASS_WITH_WARNING"
    assert signoff["overall_status"] == "PASS_WITH_WARNING"
    assert signoff["warnings"] == ["H4_SOURCE_ARTIFACT_RETENTION_INCOMPLETE"]
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
    assert reference["phase_34_status"] == "PASS_WITH_WARNING"
    assert reference["inherited_warnings"] == ["H4_SOURCE_ARTIFACT_RETENTION_INCOMPLETE"]
    assert len(reference["h4_missing_artifacts"]) == 4
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
    log = build_phase_resume_log(34, tmp_path)
    assert log["result"]["h2_run_id"] == "RUN-H2"
    assert log["result"]["h4_reference_run_id"] == "RUN-H4"
    assert log["result"]["winner"] == "H2"
    assert log["result"]["selected_num_heads"] == 2
    assert log["result"]["selected_head_dim"] == 32
    assert log["result"]["h2_best_epoch"] == 15
    assert log["result"]["h4_evidence_status"] == "PASS_WITH_WARNING"
    assert log["result"]["phase_34_final_status"] == "PASS_WITH_WARNING"
    assert log["result"]["test_access"] == "FORBIDDEN"
    assert log["result"]["approved_for_phase35"] is True


def test_phase_35_finalizer_selects_n2_and_preserves_historical_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from course_work.sweeps import sweep_results

    def verified_fixture(*_args: object, **_kwargs: object) -> dict:
        return {
            "epochs_executed": 25,
            "best_epoch": 15,
            "runtime_seconds": 25.0,
            "checkpoint_size_bytes": 1024,
            "order_status": "PASS",
            "strict_load_status": "PASS",
            "strict_validation_status": "PASS",
            "verification_tolerance": 1e-6,
            "recomputed_validation_rmse_wh": 61.0,
            "recomputed_validation_mae_wh": 60.0,
            "recomputed_validation_r2": 0.4,
            "stored_validation_rmse_wh": 61.0,
            "stored_validation_mae_wh": 60.0,
            "stored_validation_r2": 0.4,
            "best_checkpoint_sha256": "fixture-checkpoint",
            "history_sha256": "fixture-history",
            "metric_artifact_sha256": "fixture-metric",
            "prediction_artifact_sha256": "fixture-predictions",
        }

    monkeypatch.setattr(sweep_results, "_validate_phase_35_n1_evidence", verified_fixture)
    create_head_registry(tmp_path, h4_rmse=58.0)
    create_phase_33(tmp_path, winner_rmse=58.0)
    phase_34 = finalize_verified_sweep(34, tmp_path)
    assert phase_34["winner_head_id"] == "H4"
    create_layer_n1_run(tmp_path, rmse=61.0)

    signoff = finalize_verified_sweep(35, tmp_path)
    winner = read_json(tmp_path / "artifacts/sweeps/S13_layers/s13_layer_winner.json")
    reference = read_json(tmp_path / "artifacts/sweeps/S13_layers/s13_reference_update.json")

    assert signoff["status"] == "PASS_WITH_WARNING"
    assert signoff["overall_status"] == "PASS_WITH_WARNING"
    assert signoff["test_status"] == "FORBIDDEN"
    assert signoff["n1_run_id"] == "RUN-N1"
    assert signoff["n2_reference_run_id"] == "RUN-H4"
    assert signoff["n1_epochs_executed"] == 25
    assert signoff["n1_best_epoch"] == 15
    assert signoff["n1_parameter_count"] < signoff["n2_parameter_count"]
    assert signoff["layer_parameter_delta"] == 33472
    assert signoff["sweep_version"] == "SWEEP_S13_LAYERS-v1"
    assert signoff["phase"] == 35
    assert signoff["phase_name"] == "S13 Layer sweep"
    assert signoff["initialization_policy_audit_status"] == "PASS"
    assert signoff["sample_order_match_status"] == "PASS_WITH_WARNING"
    assert signoff["inherited_warnings"] == ["H4_SOURCE_ARTIFACT_RETENTION_INCOMPLETE"]
    assert signoff["n1_best_strict_load_status"] == "PASS"
    assert signoff["n1_best_validation_verification_status"] == "PASS"
    assert winner["winner_layer_id"] == "N2"
    assert winner["winner_num_layers"] == 2
    assert winner["winner_rmse_wh"] == 58.0
    assert winner["sweep_version"] == "SWEEP_S13_LAYERS-v1"
    assert winner["dropout_probability"] == 0.1
    assert winner["parameter_difference"] == 33472
    assert winner["exact_tie_applied"] is False
    assert winner["inherited_warnings"] == ["H4_SOURCE_ARTIFACT_RETENTION_INCOMPLETE"]
    assert reference["selected_layer_id"] == "N2"
    assert reference["selected_num_layers"] == 2
    assert reference["approved_for_phase36"] is True
    assert reference["phase_35_status"] == "PASS_WITH_WARNING"
    assert reference["dropout_probability"] == 0.1
    assert reference["current_ffn_dim"] == 128
    assert reference["phase36_condition_policy"] == {
        "F64": "TRAIN_NEW",
        "F128": "REUSE_REFERENCE",
        "F256": "TRAIN_NEW",
    }
    assert validate_sweep_signoff(35, tmp_path)["valid"] is True
    assert inspect_phase_state(35, tmp_path)["state"] == "LOG_MISSING"
    required = {
        "s13_layer_metrics.csv",
        "s13_layer_sweep_manifest.json",
        "s13_layer_winner.json",
        "s13_reference_update.json",
        "s13_layer_sweep_contract.json",
        "s13_layer_preflight_audit.csv",
        "s13_run_matrix.csv",
        "s13_layer_definition_audit.csv",
        "s13_encoder_stack_geometry_audit.csv",
        "s13_architecture_role_audit.csv",
        "s13_state_dict_key_delta_audit.csv",
        "s13_shared_parameter_schema_audit.csv",
        "s13_parameter_count_audit.csv",
        "s13_layer_delta_parameter_audit.csv",
        "s13_layer_independence_audit.csv",
        "s13_optimizer_coverage_audit.csv",
        "s13_config_delta_audit.csv",
        "s13_layer_unit_tests.csv",
        "s13_common_data_audit.csv",
        "s13_layer_training_audit.csv",
        "s13_initialization_policy_audit.csv",
        "s13_sample_order_audit.csv",
        "s13_dropout_depth_audit.csv",
        "s13_attention_api_audit.csv",
        "s13_optimizer_budget_audit.csv",
        "s13_layer_run_provenance.csv",
        "s13_layer_metrics.csv",
        "s13_layer_effect.csv",
        "s13_depth_efficiency_context.csv",
        "s13_depth_pareto_context.json",
        "s13_optimization_diagnostics.csv",
        "s13_convergence_diagnostics.csv",
        "s13_runtime_capacity_diagnostics.csv",
        "s13_hypothesis_outcomes.csv",
        "s13_layer_findings.csv",
        "s13_layer_winner.json",
        "s13_reference_update.json",
        "s13_layer_sweep_tests.csv",
        "s13_layer_discrepancies.json",
        "s13_layer_sweep_summary.json",
        "s13_layer_sweep_report.md",
        "README_S13_LAYER_SWEEP.md",
        "figures/S13_01_validation_rmse_by_epoch.png",
        "figures/S13_02_validation_mae_by_epoch.png",
        "figures/S13_03_train_loss_by_epoch.png",
        "figures/S13_04_gradient_clipping_fraction.png",
        "figures/S13_05_best_validation_metrics.png",
        "figures/S13_06_parameter_count_vs_rmse.png",
        "figures/S13_07_runtime_vs_rmse.png",
        "figures/S13_08_checkpoint_size_comparison.png",
        "figures/S13_09_convergence_summary.png",
    }
    artifact_root = tmp_path / "artifacts/sweeps/S13_layers"
    assert all((artifact_root / path).is_file() for path in required)
    expected_metric_columns = [
        "layer_id", "num_layers", "run_id", "source_type", "d_model", "num_heads", "head_dim",
        "ffn_dim", "trainable_parameters", "best_epoch", "epochs_completed", "total_optimizer_steps",
        "stop_reason", "validation_mae_wh", "validation_rmse_wh", "validation_r2", "rmse_rank",
        "is_empirical_winner", "population_fingerprint", "metric_version", "status",
    ]
    assert (artifact_root / "s13_layer_metrics.csv").read_text(encoding="utf-8").splitlines()[0].split(",") == expected_metric_columns
