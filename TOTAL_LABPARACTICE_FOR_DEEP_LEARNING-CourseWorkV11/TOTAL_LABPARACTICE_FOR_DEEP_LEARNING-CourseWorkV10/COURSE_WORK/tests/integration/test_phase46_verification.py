"""Tests for Phase 46 verification functions using disposable checkpoints.

These tests use synthetic small checkpoints to verify that strict_load,
forward_sanity_probe, and attention_compatibility_probe work correctly
without requiring real training to have happened.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from course_work.training.engine import build_model_from_run_config
from course_work.verification.phase46_verification import (
    attention_compatibility_probe,
    cross_seed_consistency,
    forward_sanity_probe,
    strict_load_checkpoint,
)


TRANSFORMER_RUN_CONFIG = {
    "model": {
        "model_family": "TRANSFORMER_ENCODER",
        "input_size": 30,
        "d_model": 32,
        "num_heads": 4,
        "num_layers": 2,
        "ffn_dim": 64,
        "dropout": 0.0,
        "pooling": "MEAN",
        "output_size": 1,
        "revin_enabled": False,
    },
    "training": {
        "optimizer_name": "AdamW",
        "loss_name": "MSE",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "gradient_clip_max_norm": 1.0,
        "gradient_clipping_enabled": True,
        "revin_enabled": False,
        "revin_eps": 1e-5,
        "max_epochs": 50,
        "batch_size": 32,
    },
    "data": {
        "feature_variant_id": "FS2_TF1",
        "lookback_steps": 36,
        "horizon_steps": 1,
        "boundary_protocol": "WB0_CONTEXT_CARRY_OVER",
        "target_scaling_option": "YS1",
        "input_size": 30,
        "sampling_interval_minutes": 10,
    },
    "lineage": {
        "population_fingerprint": "abc123",
        "feature_fingerprint": "def456",
    },
}


def _make_disposable_checkpoint(
    tmp_path: Path, seed: int, official_epoch: int = 50
) -> tuple[Path, str]:
    """Create a small disposable checkpoint for testing."""
    import hashlib
    torch.manual_seed(seed)
    model = build_model_from_run_config(TRANSFORMER_RUN_CONFIG)
    sd = model.state_dict()
    ckpt_path = tmp_path / f"disposable_seed_{seed}.pt"
    payload = {
        "model_state_dict": sd,
        "seed": seed,
        "run_id": f"DISPOSABLE_{seed}",
        "checkpoint_type": "FINAL_REFIT",
        "official_epoch": official_epoch,
        "FINAL_REFIT_EPOCHS": 50,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "state_schema_fingerprint": hashlib.sha256(
            "|".join(sorted(sd.keys())).encode()
        ).hexdigest(),
        "x_scaler_sha256": "FAKE_X_SHA",
        "y_scaler_sha256": "FAKE_Y_SHA",
        "config_sha256": "FAKE_CONFIG_SHA",
    }
    torch.save(payload, ckpt_path)
    import io
    file_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    return ckpt_path, file_sha


def test_strict_load_checkpoint_success(tmp_path):
    ckpt_path, file_sha = _make_disposable_checkpoint(tmp_path, seed=42)
    result = strict_load_checkpoint(
        checkpoint_path=ckpt_path,
        run_config=TRANSFORMER_RUN_CONFIG,
        expected_sha256=file_sha,
        expected_official_epoch=50,
    )
    assert result["status"] == "PASS", result
    assert result["checks"]["physical_file_exists"] is True
    assert result["checks"]["torch_load_succeeds"] is True
    assert result["checks"]["checkpoint_type_is_FINAL_REFIT"] is True
    assert result["checks"]["official_epoch_matches"] is True
    assert result["checks"]["strict_load_missing_keys_zero"] is True
    assert result["checks"]["strict_load_unexpected_keys_zero"] is True
    assert result["checks"]["all_params_finite"] is True
    assert result["missing_keys"] == []
    assert result["unexpected_keys"] == []


def test_strict_load_checkpoint_sha_mismatch(tmp_path):
    ckpt_path, _ = _make_disposable_checkpoint(tmp_path, seed=42)
    result = strict_load_checkpoint(
        checkpoint_path=ckpt_path,
        run_config=TRANSFORMER_RUN_CONFIG,
        expected_sha256="DEADBEEF" * 8,
        expected_official_epoch=50,
    )
    assert result["status"] == "FAIL"
    assert "SHA mismatch" in result["error"]


def test_strict_load_checkpoint_epoch_mismatch(tmp_path):
    ckpt_path, file_sha = _make_disposable_checkpoint(tmp_path, seed=42, official_epoch=49)
    result = strict_load_checkpoint(
        checkpoint_path=ckpt_path,
        run_config=TRANSFORMER_RUN_CONFIG,
        expected_sha256=file_sha,
        expected_official_epoch=50,
    )
    assert result["status"] == "FAIL"
    assert "official_epoch" in result["error"]


def test_forward_sanity_probe(tmp_path):
    ckpt_path, file_sha = _make_disposable_checkpoint(tmp_path, seed=42)
    result = forward_sanity_probe(
        checkpoint_path=ckpt_path,
        run_config=TRANSFORMER_RUN_CONFIG,
        seed=42,
        probe_batch_size=8,
    )
    assert result["status"] == "PASS", result
    assert result["checks"]["prediction_shape_is_B_1"] is True
    assert result["checks"]["prediction_finite"] is True
    assert result["checks"]["eval_repeatable"] is True


def test_attention_compatibility_probe(tmp_path):
    ckpt_path, file_sha = _make_disposable_checkpoint(tmp_path, seed=42)
    result = attention_compatibility_probe(
        checkpoint_path=ckpt_path,
        run_config=TRANSFORMER_RUN_CONFIG,
        seed=42,
        probe_batch_size=8,
    )
    assert result["status"] == "PASS", result
    assert result["checks"]["predictions_allclose"] is True
    assert result["checks"]["layer_count_correct"] is True
    assert result["checks"]["attention_shape_B_H_L_L"] is True
    assert result["checks"]["attention_finite"] is True
    assert result["layer_count_observed"] == 2  # num_layers=2
    assert result["attention_shape_observed"] == [8, 4, 36, 36]  # B, H, L, L


def test_cross_seed_consistency_pass():
    seeds = [42, 123, 2026]
    run_records = [
        {
            "seed": s,
            "run_id": f"RUN_{s}",
            "config_sha256": "SAME_CFG",
            "recipe_sha256": "SAME_RECIPE",
            "final_lock_sha256": "SAME_LOCK",
            "population_fingerprint": "SAME_POP",
            "x_scaler_sha256": "SAME_X",
            "y_scaler_sha256": "SAME_Y",
            "epochs": 50,
            "parameter_count": 12345,
            "state_schema_fingerprint": "SAME_SCHEMA",
            "attention_compatibility": "PASS",
        }
        for s in seeds
    ]
    result = cross_seed_consistency(
        seeds=seeds,
        run_records=run_records,
        lock_sha="SAME_LOCK",
        recipe_sha="SAME_RECIPE",
        final_dev_fp="SAME_POP",
        x_scaler_sha="SAME_X",
        y_scaler_sha="SAME_Y",
        epochs=50,
    )
    assert result["overall"] == "PASS", result
    for dim in ["config", "recipe", "lock", "population", "scalers", "epochs", "parameter_count", "state_schema"]:
        assert result["dimensions"][dim]["status"] == "PASS"


def test_cross_seed_consistency_fail_on_lock_mismatch():
    seeds = [42, 123, 2026]
    run_records = [
        {
            "seed": s,
            "run_id": f"RUN_{s}",
            "config_sha256": "SAME",
            "recipe_sha256": "SAME",
            "final_lock_sha256": "LOCK_A" if s == 42 else "LOCK_B",
            "population_fingerprint": "SAME",
            "x_scaler_sha256": "SAME",
            "y_scaler_sha256": "SAME",
            "epochs": 50,
            "parameter_count": 12345,
            "state_schema_fingerprint": "SAME",
            "attention_compatibility": "PASS",
        }
        for s in seeds
    ]
    result = cross_seed_consistency(
        seeds=seeds,
        run_records=run_records,
        lock_sha="LOCK_A",
        recipe_sha="SAME",
        final_dev_fp="SAME",
        x_scaler_sha="SAME",
        y_scaler_sha="SAME",
        epochs=50,
    )
    assert result["overall"] == "FAIL"
    assert result["dimensions"]["lock"]["status"] == "FAIL"


def test_phase47_release_excludes_historical_ids():
    """Test that build_phase47_release rejects historical run IDs."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib

    # Import the script as a module
    spec = importlib.util.spec_from_file_location(
        "phase46", ROOT / "scripts" / "phase46_three_seed_runs.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase46"] = mod
    spec.loader.exec_module(mod)

    EXCLUDED = [
        "RUN_TR_FSD_0153_B15A19DC",
        "RUN_TR_FSD_0154_DD82D743",
        "RUN_TR_FSD_0155_59A50ADD",
    ]

    # Release with historical IDs leaked → must FAIL
    bad_records = [{"seed": s, "run_id": rid, "checkpoint_sha256": "FAKE"} for s, rid in zip([42, 123, 2026], EXCLUDED)]
    release = mod._build_phase47_release(
        all_seeds_completed=True,
        all_checkpoints_verified=True,
        lock_hash_match=True, config_match=True, recipe_match=True,
        population_match=True, scalers_match=True, epochs_match=True,
        schema_match=True, test_not_accessed=True,
        seeds=[42, 123, 2026],
        run_records=bad_records,
        x_scaler_sha="X", y_scaler_sha="Y", final_dev_fp="FP",
    )
    assert release["released"] is False
    assert release["gates"]["no_historical_run_id_leaked"] is False
    assert release["historical_run_ids_exclusion_status"] == "VIOLATED"

    # Release with corrected IDs → must PASS
    good_records = [{"seed": s, "run_id": f"RUN_NEW_{s}", "checkpoint_sha256": "FAKE"} for s in [42, 123, 2026]]
    release = mod._build_phase47_release(
        all_seeds_completed=True,
        all_checkpoints_verified=True,
        lock_hash_match=True, config_match=True, recipe_match=True,
        population_match=True, scalers_match=True, epochs_match=True,
        schema_match=True, test_not_accessed=True,
        seeds=[42, 123, 2026],
        run_records=good_records,
        x_scaler_sha="X", y_scaler_sha="Y", final_dev_fp="FP",
    )
    assert release["released"] is True
    assert release["gates"]["no_historical_run_id_leaked"] is True
    assert release["historical_run_ids_exclusion_status"] == "ENFORCED"
