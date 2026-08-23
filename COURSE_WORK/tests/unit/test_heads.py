import json
from pathlib import Path

import pytest

from course_work.sweeps.heads import (
    CONDITIONS,
    compare_head_geometry,
    inspect_head_geometry,
    inspect_phase_33_handoff,
)
from course_work.utils.artifacts import sha256_file


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def model_config(num_heads: int = 4) -> dict:
    return {
        "input_size": 33,
        "d_model": 64,
        "num_heads": num_heads,
        "num_layers": 2,
        "ffn_dim": 128,
        "dropout": 0.1,
        "pooling": "LAST_STEP",
        "activation": "GELU",
    }


def create_handoff(root: Path) -> None:
    run_id = "RUN-S11-WINNER"
    fingerprint = "fingerprint-s11-winner"
    population = "population-v1"
    shared = {
        "feature_variant_id": "FS2_TF1",
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
        "population_fingerprint": population,
        "winner_run_id": run_id,
        "winner_config_fingerprint": fingerprint,
        "test_status": "FORBIDDEN",
    }
    winner_path = root / "artifacts/sweeps/S11_d_model/s11_d_model_winner.json"
    reference_path = root / "artifacts/sweeps/S11_d_model/s11_reference_update.json"
    write_json(
        winner_path,
        {
            **shared,
            "status": "PASS",
            "winner_d_model": 64,
        },
    )
    write_json(
        reference_path,
        {
            **shared,
            "approved_for_phase34": True,
            "selected_d_model": 64,
            "current_num_heads": 4,
            "current_head_dim": 16,
        },
    )
    source_config = {
        "data": {
            "feature_variant_id": "FS2_TF1",
            "target_scaling_option": "YS1",
            "lookback_steps": 36,
            "boundary_protocol": "WB0_CONTEXT_CARRY_OVER",
        },
        "model": model_config(),
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
        "reproducibility": {"seed": 42},
        "lineage": {"population_fingerprint": population},
    }
    run_root = root / "artifacts/runs" / run_id
    config_path = run_root / "config.json"
    status_path = run_root / "status.json"
    checkpoint_path = run_root / "checkpoints/best_checkpoint.pt"
    metrics_path = run_root / "metrics/best_validation_metrics.json"
    write_json(config_path, {"run_id": run_id, "config_fingerprint": fingerprint, "config": source_config})
    write_json(status_path, {"run_id": run_id, "status": "COMPLETED"})
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint")
    write_json(metrics_path, {"run_id": run_id, "rmse_wh": 58.0})
    (run_root / "training_history.csv").write_text("epoch,rmse_wh\n1,58.0\n", encoding="utf-8")
    prediction_path = run_root / "predictions/best_validation_predictions.csv"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text("sample_idx,y_true_wh,y_pred_wh\n0,1,1\n", encoding="utf-8")
    artifact_specs = (
        (config_path, "CONFIG"),
        (status_path, "STATUS"),
        (checkpoint_path, "BEST_CHECKPOINT"),
        (metrics_path, "METRICS"),
    )
    artifacts = [
        {
            "artifact_path": path.relative_to(root).as_posix(),
            "artifact_type": artifact_type,
            "required": True,
            "sha256": sha256_file(path),
        }
        for path, artifact_type in artifact_specs
    ]
    metrics = [
        {
            "metric_name": name,
            "metric_value": value,
            "split_id": "VALIDATION",
            "status": "PASS",
            "population_fingerprint": population,
        }
        for name, value in (("rmse_wh", 58.0), ("mae_wh", 27.0), ("r2", 0.6))
    ]
    registry_path = root / "artifacts/experiments/experiment_registry.jsonl"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "COMPLETED",
                "config_fingerprint": fingerprint,
                "test_access_authorized": False,
                "artifacts": artifacts,
                "metrics": metrics,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_paths = [winner_path.relative_to(root).as_posix(), reference_path.relative_to(root).as_posix()]
    write_json(
        root / "artifacts/sweeps/S11_d_model/phase_33_signoff.json",
        {
            "status": "PASS",
            "approved_for_phase34": True,
            "test_status": "FORBIDDEN",
            "output_paths": output_paths,
            "output_checksums": {path: sha256_file(root / path) for path in output_paths},
        },
    )


def test_head_registry_is_exact() -> None:
    assert [(item.condition_id, item.num_heads, item.execution_mode) for item in CONDITIONS] == [
        ("H2", 2, "TRAIN_NEW"),
        ("H4", 4, "REUSE_REFERENCE"),
    ]


def test_head_geometry_preserves_parameter_schema_and_output_contract() -> None:
    comparison = compare_head_geometry(model_config())
    by_id = {item["condition_id"]: item for item in comparison["candidates"]}
    assert comparison["status"] == "PASS"
    assert comparison["parameter_schema_equal"] is True
    assert comparison["parameter_count_equal"] is True
    assert comparison["trainable_parameter_count_equal"] is True
    assert comparison["initial_state_equal"] is True
    assert comparison["config_delta"] == ["num_heads"]
    assert by_id["H2"]["head_dim"] == 32
    assert by_id["H4"]["head_dim"] == 16
    assert by_id["H2"]["attention_shapes"] == [[2, 2, 36, 36], [2, 2, 36, 36]]
    assert by_id["H4"]["attention_shapes"] == [[2, 4, 36, 36], [2, 4, 36, 36]]


def test_invalid_head_divisibility_is_rejected() -> None:
    with pytest.raises(ValueError, match="divisible"):
        inspect_head_geometry({**model_config(), "d_model": 62})


def test_phase_33_handoff_resolves_dynamic_d_model_and_reference(tmp_path: Path) -> None:
    create_handoff(tmp_path)
    result = inspect_phase_33_handoff(tmp_path)
    assert result["valid"] is True
    assert result["selected_d_model"] == 64
    assert result["winner_run_id"] == "RUN-S11-WINNER"
    assert result["frozen_configuration"]["num_heads"] == 4
    assert result["geometry_comparison"]["status"] == "PASS"
    assert result["reference_evidence"]["metrics"]["rmse_wh"] == 58.0
    assert result["processing_log_observation"]["authoritative"] is False


def test_phase_33_handoff_rejects_reference_head_drift(tmp_path: Path) -> None:
    create_handoff(tmp_path)
    path = tmp_path / "artifacts/sweeps/S11_d_model/s11_reference_update.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["current_num_heads"] = 8
    write_json(path, value)
    result = inspect_phase_33_handoff(tmp_path)
    assert result["valid"] is False
    assert {item["reason"] for item in result["issues"]} >= {
        "CHECKSUM_MISMATCH",
        "REFERENCE_HEAD_COUNT_MISMATCH",
    }
