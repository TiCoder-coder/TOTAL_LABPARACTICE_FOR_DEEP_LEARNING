import json
from pathlib import Path

import pytest

from course_work.sweeps.weight_decay import (
    CONDITIONS,
    build_phase_31_preflight,
    inspect_phase_30_handoff,
    prepare_phase_31_condition,
)
from course_work.utils.artifacts import sha256_file


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def create_handoff(root: Path, learning_rate: float = 0.0007) -> None:
    shared = {
        "feature_variant_id": "FS1_TF1",
        "target_scaling_id": "YS1",
        "lookback_id": "L144",
        "pooling_id": "LAST_STEP",
        "activation_id": "GELU",
        "batch_id": "B64",
        "population_fingerprint": "population-v1",
    }
    write_json(
        root / "artifacts/sweeps/s8_learning_rate/phase_30_signoff.json",
        {
            "status": "PASS",
            "approved_for_phase31": True,
            "test_status": "LOCKED_UNTIL_PHASE_47",
        },
    )
    write_json(
        root / "artifacts/sweeps/s8_learning_rate/s8_learning_rate_winner.json",
        {
            **shared,
            "status": "PASS",
            "test_status": "FORBIDDEN",
            "winner_run_id": "RUN-S8-WINNER",
            "winner_learning_rate": learning_rate,
        },
    )
    write_json(
        root / "artifacts/sweeps/s8_learning_rate/s8_reference_update.json",
        {
            **shared,
            "approved_for_phase31": True,
            "weight_decay_state": "WD1_1E-4",
            "winner_run_id": "RUN-S8-WINNER",
            "selected_learning_rate": learning_rate,
            "winner_config_fingerprint": "fingerprint-s8-winner",
            "winner_optimizer_config_fingerprint": "optimizer-fingerprint-s8-winner",
        },
    )
    write_json(
        root / "artifacts/runs/RUN-S8-WINNER/config.json",
        {
            "run_id": "RUN-S8-WINNER",
            "config_fingerprint": "fingerprint-s8-winner",
            "config": {
                "data": {
                    "feature_variant_id": "FS1_TF1",
                    "target_scaling_option": "YS1",
                    "lookback_steps": 144,
                },
                "model": {
                    "pooling": "LAST_STEP",
                    "activation": "GELU",
                    "dropout": 0.1,
                },
                "training": {
                    "batch_size": 64,
                    "optimizer_name": "AdamW",
                    "learning_rate": learning_rate,
                    "weight_decay": 0.0001,
                    "loss_name": "MSE",
                    "max_epochs": 50,
                    "early_stopping_patience": 10,
                    "gradient_clipping_enabled": True,
                    "gradient_clip_max_norm": 1.0,
                    "scheduler_name": None,
                },
                "reproducibility": {"seed": 42},
                "lineage": {"population_fingerprint": "population-v1"},
            },
        },
    )


def create_reference_evidence(root: Path) -> None:
    run_id = "RUN-S8-WINNER"
    run_root = root / "artifacts/runs" / run_id
    write_json(run_root / "status.json", {"run_id": run_id, "status": "COMPLETED"})
    write_json(
        run_root / "metrics/best_validation_metrics.json",
        {"mae_wh": 25.0, "rmse_wh": 60.0, "r2": 0.5},
    )
    artifacts = []
    for artifact_type, path in (
        ("CONFIG", run_root / "config.json"),
        ("STATUS", run_root / "status.json"),
        ("METRICS", run_root / "metrics/best_validation_metrics.json"),
    ):
        artifacts.append(
            {
                "artifact_type": artifact_type,
                "artifact_path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
                "required": True,
            }
        )
    registry_path = root / "artifacts/experiments/experiment_registry.jsonl"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "COMPLETED",
                "experiment_family": "S8_LEARNING_RATE",
                "config_fingerprint": "fingerprint-s8-winner",
                "artifacts": artifacts,
                "metrics": [
                    {"split_id": "VALIDATION", "status": "PASS", "metric_name": "mae_wh", "metric_value": 25.0},
                    {"split_id": "VALIDATION", "status": "PASS", "metric_name": "rmse_wh", "metric_value": 60.0},
                    {"split_id": "VALIDATION", "status": "PASS", "metric_name": "r2", "metric_value": 0.5},
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_weight_decay_registry_is_exact() -> None:
    assert [(item.condition_id, item.weight_decay, item.execution_mode) for item in CONDITIONS] == [
        ("WD0", 0.0, "TRAIN_NEW"),
        ("WD1", 0.0001, "REUSE_REFERENCE"),
        ("WD2", 0.001, "TRAIN_NEW"),
    ]


def test_handoff_never_falls_back_to_hard_coded_learning_rate(tmp_path: Path) -> None:
    create_handoff(tmp_path, learning_rate=0.0007)
    result = inspect_phase_30_handoff(tmp_path)
    assert result["valid"] is True
    assert result["selected_learning_rate"] == 0.0007
    assert result["frozen_configuration"]["learning_rate"] == 0.0007


def test_missing_phase_30_evidence_blocks_preflight_without_writes(tmp_path: Path) -> None:
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    result = build_phase_31_preflight(tmp_path)
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert result["ready"] is False
    assert result["decision"]["effective_action"] == "BLOCK"
    assert before == after


def test_wd1_is_reuse_only(tmp_path: Path) -> None:
    create_handoff(tmp_path)
    create_reference_evidence(tmp_path)
    result = prepare_phase_31_condition("WD1", tmp_path)
    assert result["execution_mode"] == "REUSE_REFERENCE"
    assert result["reference_run_id"] == "RUN-S8-WINNER"
    assert result["weight_decay"] == 0.0001


def test_wd1_refuses_unverified_reference_run(tmp_path: Path) -> None:
    create_handoff(tmp_path)
    with pytest.raises(RuntimeError, match="reference evidence is invalid"):
        prepare_phase_31_condition("WD1", tmp_path)


def test_wd0_refuses_training_when_selective_gate_is_not_ready(tmp_path: Path) -> None:
    create_handoff(tmp_path)
    with pytest.raises(RuntimeError, match="Condition execution blocked"):
        prepare_phase_31_condition("WD0", tmp_path)


def test_unknown_weight_decay_condition_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not registered"):
        prepare_phase_31_condition("WD9", tmp_path)
