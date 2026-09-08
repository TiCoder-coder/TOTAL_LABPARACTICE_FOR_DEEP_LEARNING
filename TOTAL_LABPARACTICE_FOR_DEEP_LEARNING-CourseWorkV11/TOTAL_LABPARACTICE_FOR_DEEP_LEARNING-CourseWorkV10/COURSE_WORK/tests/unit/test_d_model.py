import json
from pathlib import Path

import pytest
import torch

from course_work.models.transformer_regressor import TransformerRegressor
from course_work.sweeps.d_model import (
    CONDITIONS,
    build_phase_33_preflight,
    compare_d_model_geometry,
    inspect_d_model_geometry,
    inspect_phase_32_handoff,
    prepare_phase_33_condition,
)
from course_work.utils.artifacts import sha256_file


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def model_config(d_model: int = 64) -> dict:
    return {
        "input_size": 33,
        "d_model": d_model,
        "num_heads": 4,
        "num_layers": 2,
        "ffn_dim": 128,
        "dropout": 0.1,
        "pooling": "LAST_STEP",
        "activation": "GELU",
    }


def create_handoff(root: Path) -> None:
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
        "population_fingerprint": "population-v1",
    }
    winner_path = root / "artifacts/sweeps/S10_dropout/s10_dropout_winner.json"
    reference_path = root / "artifacts/sweeps/S10_dropout/s10_reference_update.json"
    write_json(
        winner_path,
        {
            **shared,
            "status": "PASS",
            "test_status": "FORBIDDEN",
            "winner_run_id": "RUN-S10-WINNER",
            "winner_dropout_probability": 0.1,
        },
    )
    write_json(
        reference_path,
        {
            **shared,
            "approved_for_phase33": True,
            "test_status": "FORBIDDEN",
            "d_model_state": "D64",
            "selected_dropout_probability": 0.1,
            "winner_run_id": "RUN-S10-WINNER",
            "winner_config_fingerprint": "fingerprint-s10-winner",
        },
    )
    write_json(
        root / "artifacts/runs/RUN-S10-WINNER/config.json",
        {
            "run_id": "RUN-S10-WINNER",
            "config_fingerprint": "fingerprint-s10-winner",
            "config": {
                "data": {
                    "feature_variant_id": "FS2_TF1",
                    "target_scaling_option": "YS1",
                    "lookback_steps": 36,
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
                    "gradient_clipping_enabled": True,
                    "gradient_clip_max_norm": 1.0,
                    "scheduler_name": None,
                },
                "reproducibility": {"seed": 42},
                "lineage": {"population_fingerprint": "population-v1"},
            },
        },
    )
    output_paths = [str(winner_path.relative_to(root)), str(reference_path.relative_to(root))]
    write_json(
        root / "artifacts/sweeps/S10_dropout/phase_32_signoff.json",
        {
            "status": "PASS",
            "approved_for_phase33": True,
            "test_status": "FORBIDDEN",
            "output_paths": output_paths,
            "output_checksums": {path: sha256_file(root / path) for path in output_paths},
        },
    )


def test_d_model_registry_is_exact() -> None:
    assert [(item.condition_id, item.d_model, item.execution_mode) for item in CONDITIONS] == [
        ("D32", 32, "TRAIN_NEW"),
        ("D64", 64, "REUSE_REFERENCE"),
    ]


def test_d_model_geometry_preserves_external_output_contract() -> None:
    d32 = inspect_d_model_geometry(model_config(32))
    d64 = inspect_d_model_geometry(model_config(64))
    comparison = compare_d_model_geometry(model_config())
    assert d32["status"] == "PASS"
    assert d64["status"] == "PASS"
    assert d32["head_dim"] == 8
    assert d64["head_dim"] == 16
    assert d64["parameter_count"] > d32["parameter_count"]
    assert comparison["external_output_contract_equal"] is True
    assert comparison["parameter_count_delta"] > 0


def test_wrong_width_checkpoint_load_is_rejected() -> None:
    d64 = TransformerRegressor(model_config(64))
    d32 = TransformerRegressor(model_config(32))
    with pytest.raises(RuntimeError):
        d32.load_state_dict(d64.state_dict(), strict=True)


def test_phase_32_handoff_preserves_frozen_configuration(tmp_path: Path) -> None:
    create_handoff(tmp_path)
    result = inspect_phase_32_handoff(tmp_path)
    assert result["valid"] is True
    assert result["selected_dropout"] == 0.1
    assert result["winner_run_id"] == "RUN-S10-WINNER"
    assert result["frozen_configuration"]["weight_decay"] == 0.001
    assert result["capacity_comparison"]["status"] == "PASS"
    assert result["processing_log_observation"]["authoritative"] is False


def test_missing_phase_32_evidence_blocks_preflight_without_writes(tmp_path: Path) -> None:
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    result = build_phase_33_preflight(tmp_path)
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert result["ready"] is False
    assert result["decision"]["effective_action"] == "BLOCK"
    assert before == after


def test_unknown_d_model_condition_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not registered"):
        prepare_phase_33_condition("D128", tmp_path)
