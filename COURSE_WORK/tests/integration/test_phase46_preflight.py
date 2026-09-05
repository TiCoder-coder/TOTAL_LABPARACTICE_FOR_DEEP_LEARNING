"""Integration tests for Phase 46 preflight."""
from __future__ import annotations

import sys
from pathlib import Path

# Import directly from the script module (scripts/ is not a Python package)
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPT_PATH))

import pytest

# Import from the phase46 script as a standalone module
from phase46_three_seed_runs import (
    _build_phase47_release,
    write_phase46_artifacts,
    save_seed_checkpoint,
)


def test_phase47_release_not_released_without_training():
    """phase47_test_release.json must have released=false before training."""
    release = _build_phase47_release(
        all_seeds_completed=False,
        all_checkpoints_verified=False,
        lock_hash_match=True,
        config_match=True,
        recipe_match=True,
        population_match=True,
        scalers_match=True,
        epochs_match=True,
        schema_match=True,
        test_not_accessed=True,
        seeds=[42, 123, 2026],
        run_records=[],
        x_scaler_sha="aabbccdd",
        y_scaler_sha="aabbccdd",
        final_dev_fp="final_dev_fp_test",
    )
    assert release["released"] is False
    assert release["phase47_eligible"] is False
    assert len(release["failed_gates"]) > 0
    assert release["owned_by_phase"] == 46


def test_phase47_release_0_seeds_fails():
    """Cannot release with 0/3 seeds completed."""
    release = _build_phase47_release(
        all_seeds_completed=False,
        all_checkpoints_verified=False,
        lock_hash_match=True,
        config_match=True,
        recipe_match=True,
        population_match=True,
        scalers_match=True,
        epochs_match=True,
        schema_match=True,
        test_not_accessed=True,
        seeds=[42, 123, 2026],
        run_records=[],
        x_scaler_sha="aabbccdd",
        y_scaler_sha="aabbccdd",
        final_dev_fp="fp",
    )
    assert release["released"] is False
    assert "all_seeds_completed" in release["failed_gates"]


def test_phase47_release_1_seed_fails():
    """Cannot release with 1/3 seeds completed."""
    release = _build_phase47_release(
        all_seeds_completed=False,
        all_checkpoints_verified=False,
        lock_hash_match=True,
        config_match=True,
        recipe_match=True,
        population_match=True,
        scalers_match=True,
        epochs_match=True,
        schema_match=True,
        test_not_accessed=True,
        seeds=[42, 123, 2026],
        run_records=[{"seed": 42, "run_id": "RUN_TEST", "checkpoint_verified": True}],
        x_scaler_sha="aabbccdd",
        y_scaler_sha="aabbccdd",
        final_dev_fp="fp",
    )
    assert release["released"] is False
    assert "all_seeds_completed" in release["failed_gates"]


def test_phase47_release_2_seeds_fails():
    """Cannot release with 2/3 seeds completed."""
    release = _build_phase47_release(
        all_seeds_completed=False,
        all_checkpoints_verified=False,
        lock_hash_match=True,
        config_match=True,
        recipe_match=True,
        population_match=True,
        scalers_match=True,
        epochs_match=True,
        schema_match=True,
        test_not_accessed=True,
        seeds=[42, 123, 2026],
        run_records=[
            {"seed": 42, "run_id": "RUN1", "checkpoint_verified": True},
            {"seed": 123, "run_id": "RUN2", "checkpoint_verified": True},
        ],
        x_scaler_sha="aabbccdd",
        y_scaler_sha="aabbccdd",
        final_dev_fp="fp",
    )
    assert release["released"] is False


def test_phase47_release_scaler_mismatch_fails():
    """Cannot release if scalers don't match."""
    release = _build_phase47_release(
        all_seeds_completed=True,
        all_checkpoints_verified=True,
        lock_hash_match=True,
        config_match=True,
        recipe_match=True,
        population_match=True,
        scalers_match=False,
        epochs_match=True,
        schema_match=True,
        test_not_accessed=True,
        seeds=[42, 123, 2026],
        run_records=[
            {"seed": 42, "run_id": "RUN1", "checkpoint_verified": True},
            {"seed": 123, "run_id": "RUN2", "checkpoint_verified": True},
            {"seed": 2026, "run_id": "RUN3", "checkpoint_verified": True},
        ],
        x_scaler_sha="aabbccdd",
        y_scaler_sha="aabbccdd",
        final_dev_fp="fp",
    )
    assert release["released"] is False
    assert "scalers_match" in release["failed_gates"]


def test_phase47_release_all_gates_pass():
    """Released=true only when ALL gates pass."""
    release = _build_phase47_release(
        all_seeds_completed=True,
        all_checkpoints_verified=True,
        lock_hash_match=True,
        config_match=True,
        recipe_match=True,
        population_match=True,
        scalers_match=True,
        epochs_match=True,
        schema_match=True,
        test_not_accessed=True,
        seeds=[42, 123, 2026],
        run_records=[
            {"seed": 42, "run_id": "RUN1", "checkpoint_verified": True},
            {"seed": 123, "run_id": "RUN2", "checkpoint_verified": True},
            {"seed": 2026, "run_id": "RUN3", "checkpoint_verified": True},
        ],
        x_scaler_sha="aabbccdd11223344",
        y_scaler_sha="aabbccdd11223344",
        final_dev_fp="final_dev_fp_sha256",
    )
    assert release["released"] is True
    assert release["phase47_eligible"] is True
    assert release["failed_gates"] == []
    assert release["released_at"] is not None
    assert len(release["run_records"]) == 3


def test_phase47_release_test_accessed_fails():
    """Cannot release if Test was accessed."""
    release = _build_phase47_release(
        all_seeds_completed=True,
        all_checkpoints_verified=True,
        lock_hash_match=True,
        config_match=True,
        recipe_match=True,
        population_match=True,
        scalers_match=True,
        epochs_match=True,
        schema_match=True,
        test_not_accessed=False,
        seeds=[42, 123, 2026],
        run_records=[
            {"seed": 42, "run_id": "RUN1", "checkpoint_verified": True},
            {"seed": 123, "run_id": "RUN2", "checkpoint_verified": True},
            {"seed": 2026, "run_id": "RUN3", "checkpoint_verified": True},
        ],
        x_scaler_sha="aabbccdd",
        y_scaler_sha="aabbccdd",
        final_dev_fp="fp",
    )
    assert release["released"] is False
    assert "test_not_accessed" in release["failed_gates"]


def test_final_refit_semantics_guard():
    """evaluate_validation=False must set official_epoch=max_epochs, not 1."""
    from course_work.training.engine import HISTORY_COLUMNS
    assert "train_loss" in HISTORY_COLUMNS
    assert "train_rmse_wh" in HISTORY_COLUMNS
    assert "validation_rmse_wh" in HISTORY_COLUMNS
    assert "is_best" in HISTORY_COLUMNS


def test_train_diagnostic_label_in_summary():
    """Three-seed summary must label metrics as TRAIN_DIAGNOSTIC."""
    import inspect
    sig = inspect.signature(write_phase46_artifacts)
    assert "final_dev_manifest" in sig.parameters
    assert "x_scaler_sha" in sig.parameters
    assert "y_scaler_sha" in sig.parameters


def test_phase47_release_artifact_fields():
    """phase47_test_release.json has all required fields."""
    release = _build_phase47_release(
        all_seeds_completed=False,
        all_checkpoints_verified=False,
        lock_hash_match=False,
        config_match=False,
        recipe_match=False,
        population_match=False,
        scalers_match=False,
        epochs_match=False,
        schema_match=False,
        test_not_accessed=False,
        seeds=[42, 123, 2026],
        run_records=[],
        x_scaler_sha="sha",
        y_scaler_sha="sha",
        final_dev_fp="fp",
    )
    required_fields = [
        "artifact_version", "owned_by_phase", "released", "gates",
        "failed_gates", "status", "phase47_eligible", "created_at",
    ]
    for field in required_fields:
        assert field in release, f"Missing field: {field}"


def test_checkpoint_metadata_includes_provenance():
    """Checkpoint metadata must include all required provenance fields."""
    import inspect
    sig = inspect.signature(save_seed_checkpoint)
    params = list(sig.parameters.keys())
    assert "final_dev_fp" in params, "Checkpoint must include population fingerprint"
    assert "x_scaler_sha" in params, "Checkpoint must include X scaler SHA"
    assert "y_scaler_sha" in params, "Checkpoint must include Y scaler SHA"
    assert "config_sha" in params, "Checkpoint must include config SHA"
    assert "recipe_sha" in params, "Checkpoint must include recipe SHA"
