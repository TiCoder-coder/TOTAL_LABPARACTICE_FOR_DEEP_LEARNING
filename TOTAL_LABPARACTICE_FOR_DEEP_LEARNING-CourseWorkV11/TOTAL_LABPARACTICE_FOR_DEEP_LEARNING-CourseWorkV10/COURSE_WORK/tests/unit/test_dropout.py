import json
from pathlib import Path

import pytest

from course_work.sweeps.dropout import (
    CONDITIONS,
    build_phase_32_preflight,
    inspect_dropout_scope,
    inspect_phase_31_handoff,
    prepare_phase_32_condition,
    verify_dropout_mode_semantics,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def create_handoff(root: Path) -> None:
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
        root / "artifacts/sweeps/S9_weight_decay/phase_31_signoff.json",
        {
            "status": "PASS",
            "approved_for_phase32": True,
            "test_status": "LOCKED_UNTIL_PHASE_47",
        },
    )
    write_json(
        root / "artifacts/sweeps/S9_weight_decay/s9_weight_decay_winner.json",
        {
            **shared,
            "status": "PASS",
            "test_status": "FORBIDDEN",
            "winner_run_id": "RUN-S9-WINNER",
            "winner_learning_rate": 0.0007,
            "winner_weight_decay": 0.0001,
        },
    )
    write_json(
        root / "artifacts/sweeps/S9_weight_decay/s9_reference_update.json",
        {
            **shared,
            "approved_for_phase32": True,
            "dropout_state": "DR01_0P1",
            "selected_dropout": 0.1,
            "selected_weight_decay": 0.0001,
            "winner_run_id": "RUN-S9-WINNER",
            "winner_config_fingerprint": "fingerprint-s9-winner",
        },
    )
    write_json(
        root / "artifacts/runs/RUN-S9-WINNER/config.json",
        {
            "run_id": "RUN-S9-WINNER",
            "config_fingerprint": "fingerprint-s9-winner",
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
                    "dropout": 0.1,
                    "pooling": "LAST_STEP",
                    "activation": "GELU",
                },
                "training": {
                    "batch_size": 64,
                    "optimizer_name": "AdamW",
                    "learning_rate": 0.0007,
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


def test_dropout_registry_is_exact() -> None:
    assert [(item.condition_id, item.dropout_probability, item.execution_mode) for item in CONDITIONS] == [
        ("DR01", 0.1, "REUSE_REFERENCE"),
        ("DR02", 0.2, "TRAIN_NEW"),
        ("DR03", 0.3, "TRAIN_NEW"),
    ]


def test_dropout_scope_covers_all_encoder_sites() -> None:
    result = inspect_dropout_scope({"input_size": 31, "dropout": 0.2})
    assert result["status"] == "PASS"
    assert result["site_count"] == 8
    assert {item["semantic_location"] for item in result["sites"]} == {
        "ATTENTION_WEIGHT",
        "ATTENTION_RESIDUAL",
        "FFN_HIDDEN",
        "FFN_RESIDUAL",
    }
    assert all(item["dropout_probability"] == 0.2 for item in result["sites"])
    assert result["input_dropout_present"] is False
    assert result["head_dropout_present"] is False
    assert result["pooling_dropout_present"] is False


@pytest.mark.parametrize("probability", [0.1, 0.2, 0.3])
def test_dropout_mode_semantics(probability: float) -> None:
    result = verify_dropout_mode_semantics(probability)
    assert result["status"] == "PASS"
    assert result["train_stochastic"] is True
    assert result["eval_repeatable"] is True
    assert result["eval_identity"] is True
    assert result["mc_dropout"] is False


def test_phase_31_handoff_preserves_frozen_configuration(tmp_path: Path) -> None:
    create_handoff(tmp_path)
    result = inspect_phase_31_handoff(tmp_path)
    assert result["valid"] is True
    assert result["selected_weight_decay"] == 0.0001
    assert result["winner_run_id"] == "RUN-S9-WINNER"
    assert result["frozen_configuration"]["learning_rate"] == 0.0007
    assert result["frozen_configuration"]["dropout_site_count"] == 8
    assert result["processing_log_observation"]["authoritative"] is False


def test_missing_phase_31_evidence_blocks_preflight_without_writes(tmp_path: Path) -> None:
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    result = build_phase_32_preflight(tmp_path)
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert result["ready"] is False
    assert result["decision"]["effective_action"] == "BLOCK"
    assert before == after


def test_unknown_dropout_condition_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not registered"):
        prepare_phase_32_condition("DR09", tmp_path)
