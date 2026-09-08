"""Phase 47 — Transformer Checkpoint Loader.

Strict loading and verification of the three FINAL_REFIT Transformer checkpoints.
All checkpoints are verified against Phase46 handoff SHA256 and config fingerprints before use.

Guard: This module NEVER allows loading BEST, LAST, or Phase44 checkpoints.
Guard: This module NEVER allows loading checkpoints that don't match Phase46 lock fingerprints.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from course_work.final_test_evaluation import (
    LOCKED_BOUNDARY_PROTOCOL,
    LOCKED_CANDIDATE,
    LOCKED_CONFIG_FP,
    LOCKED_FEATURES,
    LOCKED_LOOKBACK,
    LSTM_TUNED_DEV,
    OFFICIAL_RUNS,
)
from course_work.utils.artifacts import get_project_root


@dataclass(frozen=True)
class VerifiedCheckpoint:
    """A verified FINAL_REFIT Transformer checkpoint."""
    seed: int
    run_id: str
    checkpoint_path: Path
    checkpoint_sha256: str
    state_dict: dict[str, torch.Tensor]
    model_config: dict[str, Any]
    optimizer_config: dict[str, Any]
    official_epoch: int
    final_refit_epochs: int
    checkpoint_type: str
    final_lock_sha256: str
    config_sha256: str
    recipe_sha256: str
    population_fingerprint: str
    x_scaler_sha256: str
    y_scaler_sha256: str


class Phase47CheckpointError(Exception):
    """Raised when a checkpoint fails verification."""
    pass


def _compute_checkpoint_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_checkpoint_sha256(path: Path, expected: str) -> None:
    """Verify a checkpoint file's SHA256 matches expected value."""
    observed = _compute_checkpoint_sha256(path)
    if observed != expected:
        raise Phase47CheckpointError(
            f"Checkpoint SHA256 mismatch for {path.name}: "
            f"expected {expected}, got {observed}"
        )


def load_verified_transformer_checkpoint(
    seed: int,
    project_root: Path | None = None,
    strict: bool = True,
) -> VerifiedCheckpoint:
    """Load and verify a FINAL_REFIT Transformer checkpoint for a given seed.

    Args:
        seed: The seed (42, 123, or 2026).
        project_root: COURSE_WORK root.
        strict: If True, enforce all verification checks.

    Returns:
        VerifiedCheckpoint with loaded state_dict.

    Raises:
        Phase47CheckpointError: If any verification fails.
    """
    if seed not in OFFICIAL_RUNS:
        raise Phase47CheckpointError(
            f"Seed {seed} is not a canonical Phase47 seed. "
            f"Canonical seeds: {list(OFFICIAL_RUNS.keys())}"
        )

    root = project_root or get_project_root()
    run_info = OFFICIAL_RUNS[seed]
    ckpt_path = root / run_info["checkpoint_path"]

    if not ckpt_path.exists():
        raise Phase47CheckpointError(
            f"FINAL_REFIT checkpoint not found: {ckpt_path}"
        )

    if strict:
        verify_checkpoint_sha256(ckpt_path, run_info["checkpoint_sha256"])

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    ckpt_type = ckpt.get("checkpoint_type", "UNKNOWN")
    if strict and ckpt_type != "FINAL_REFIT":
        raise Phase47CheckpointError(
            f"Checkpoint type mismatch for seed {seed}: "
            f"expected FINAL_REFIT, got {ckpt_type}"
        )

    official_epoch = ckpt.get("official_epoch", 0)
    if strict and official_epoch != 30:
        raise Phase47CheckpointError(
            f"Official epoch mismatch for seed {seed}: "
            f"expected 30, got {official_epoch}"
        )

    final_refit_epochs = ckpt.get("FINAL_REFIT_EPOCHS", 0)
    if strict and final_refit_epochs != 30:
        raise Phase47CheckpointError(
            f"FINAL_REFIT_EPOCHS mismatch for seed {seed}: "
            f"expected 30, got {final_refit_epochs}"
        )

    final_lock_sha = ckpt.get("final_lock_sha256", "")
    if strict and final_lock_sha != LOCKED_CONFIG_FP:
        raise Phase47CheckpointError(
            f"final_lock_sha256 mismatch for seed {seed}: "
            f"expected {LOCKED_CONFIG_FP}, got {final_lock_sha}"
        )

    config_sha = ckpt.get("config_sha256", "")
    if strict and config_sha != LOCKED_CONFIG_FP:
        raise Phase47CheckpointError(
            f"config_sha256 mismatch for seed {seed}: "
            f"expected {LOCKED_CONFIG_FP}, got {config_sha}"
        )

    ckpt_seed = ckpt.get("seed")
    if strict and ckpt_seed != seed:
        raise Phase47CheckpointError(
            f"Checkpoint embedded seed mismatch: "
            f"expected {seed}, got {ckpt_seed}"
        )

    ckpt_run_id = ckpt.get("run_id", "")
    if strict and ckpt_run_id != run_info["run_id"]:
        raise Phase47CheckpointError(
            f"Checkpoint run_id mismatch: "
            f"expected {run_info['run_id']}, got {ckpt_run_id}"
        )

    x_sha = ckpt.get("x_scaler_sha256", "")
    y_sha = ckpt.get("y_scaler_sha256", "")
    if strict:
        from course_work.final_test_evaluation.scaler_loader import FINAL_SCALING_CHECKSUMS
        if x_sha != FINAL_SCALING_CHECKSUMS["x_bundle"]:
            raise Phase47CheckpointError(
                f"x_scaler_sha256 mismatch for seed {seed}: "
                f"expected {FINAL_SCALING_CHECKSUMS['x_bundle']}, got {x_sha}"
            )
        if y_sha != FINAL_SCALING_CHECKSUMS["y_bundle"]:
            raise Phase47CheckpointError(
                f"y_scaler_sha256 mismatch for seed {seed}: "
                f"expected {FINAL_SCALING_CHECKSUMS['y_bundle']}, got {y_sha}"
            )

    return VerifiedCheckpoint(
        seed=seed,
        run_id=run_info["run_id"],
        checkpoint_path=ckpt_path,
        checkpoint_sha256=run_info["checkpoint_sha256"],
        state_dict=ckpt["model_state_dict"],
        model_config=ckpt.get("model_config", {}),
        optimizer_config=ckpt.get("optimizer_config", {}),
        official_epoch=official_epoch,
        final_refit_epochs=final_refit_epochs,
        checkpoint_type=ckpt_type,
        final_lock_sha256=final_lock_sha,
        config_sha256=config_sha,
        recipe_sha256=ckpt.get("recipe_sha256", ""),
        population_fingerprint=ckpt.get("population_fingerprint", ""),
        x_scaler_sha256=x_sha,
        y_scaler_sha256=y_sha,
    )


def load_all_three_checkpoints(
    project_root: Path | None = None,
    strict: bool = True,
) -> dict[int, VerifiedCheckpoint]:
    """Load all 3 verified FINAL_REFIT Transformer checkpoints.

    Args:
        project_root: COURSE_WORK root.
        strict: If True, enforce all verification checks.

    Returns:
        Dict mapping seed (42, 123, 2026) -> VerifiedCheckpoint.
    """
    checkpoints = {}
    for seed in [42, 123, 2026]:
        checkpoints[seed] = load_verified_transformer_checkpoint(seed, project_root, strict)
    return checkpoints


def build_transformer_model_from_checkpoint(
    checkpoint: VerifiedCheckpoint,
) -> torch.nn.Module:
    """Reconstruct the exact Transformer model from a verified checkpoint.

    Args:
        checkpoint: VerifiedCheckpoint from load_verified_transformer_checkpoint.

    Returns:
        The TransformerRegressor model with weights loaded.
    """
    from course_work.models.transformer_regressor import (
        TransformerModelConfig,
        TransformerRegressor,
        validate_transformer_config,
    )

    model_config = checkpoint.model_config
    if not model_config:
        raise Phase47CheckpointError(
            f"No model_config found in checkpoint for seed {checkpoint.seed}"
        )

    model_config_with_input = dict(model_config)
    if "input_size" not in model_config_with_input:
        model_config_with_input["input_size"] = LOCKED_FEATURES

    cfg = validate_transformer_config(model_config_with_input)

    if cfg.input_size != LOCKED_FEATURES:
        raise Phase47CheckpointError(
            f"input_size mismatch for seed {checkpoint.seed}: "
            f"expected {LOCKED_FEATURES} (LOCKED_FEATURES=33, FS2_TF1), "
            f"got {cfg.input_size}"
        )

    model = TransformerRegressor(config=cfg)

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(checkpoint.state_dict.keys())

    if model_keys != ckpt_keys:
        missing = model_keys - ckpt_keys
        unexpected = ckpt_keys - model_keys
        raise Phase47CheckpointError(
            f"State dict key mismatch for seed {checkpoint.seed}: "
            f"missing in checkpoint: {missing}, unexpected in checkpoint: {unexpected}"
        )

    model.load_state_dict(checkpoint.state_dict, strict=True)

    return model


def verify_lstm_checkpoint(
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Verify the LSTM_TUNED_DEV frozen checkpoint eligibility for Phase47.

    Args:
        project_root: COURSE_WORK root.

    Returns:
        Dict with eligibility decision and details.
    """
    root = project_root or get_project_root()
    lstm_info = LSTM_TUNED_DEV
    ckpt_path = root / lstm_info["checkpoint_path"]

    result = {
        "model_id": lstm_info["model_id"],
        "source_phase": lstm_info["source_phase"],
        "run_id": lstm_info["run_id"],
        "checkpoint_exists": ckpt_path.exists(),
        "checkpoint_frozen_pretest": False,
        "test_previously_accessed": lstm_info["test_status"] == "NOT_ACCESSED",
        "config_valid": False,
        "scaler_artifacts_available": False,
        "FINAL_TEST_POP_supported": False,  
        "common_target_comparison_possible": False,
        "training_protocol_symmetric_with_transformer": False,
        "eligible": False,
        "eligibility_status": "NOT_EVALUATED_BY_PROTOCOL",
        "reason": "",
    }

    if not ckpt_path.exists():
        result["eligibility_status"] = "NOT_ELIGIBLE_CHECKPOINT_MISSING"
        result["reason"] = f"Checkpoint not found at {ckpt_path}"
        return result

    if lstm_info["test_status"] != "NOT_ACCESSED":
        result["eligibility_status"] = "NOT_ELIGIBLE_CONFIG_MISMATCH"
        result["reason"] = "LSTM checkpoint has already accessed Test"
        return result

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config = ckpt.get("model_config", {})
        if config.get("lookback_steps") == 36:
            result["config_valid"] = True
        else:
            result["eligibility_status"] = "NOT_ELIGIBLE_CONFIG_MISMATCH"
            result["reason"] = (
                f"LSTM uses lookback={config.get('lookback_steps')}, "
                f"but final Transformer uses lookback=72"
            )
            return result
    except Exception as e:
        result["eligibility_status"] = "NOT_ELIGIBLE_CONFIG_MISMATCH"
        result["reason"] = f"Cannot load LSTM checkpoint config: {e}"
        return result

    result["FINAL_TEST_POP_supported"] = False
    result["eligibility_status"] = "NOT_EVALUATED_BY_PROTOCOL"
    result["reason"] = (
        "LSTM uses L36 lookback, which generates a different Test window population "
        "than the FINAL_TEST_POP-v1 (L72). Direct per-target comparison on identical "
        "target IDs requires a separate L36 Test loader. Per Phase47 plan §141, "
        "direct comparison on FINAL_TEST_POP-v1 is not supported for LSTM. "
        "This is reported as NOT_EVALUATED_BY_PROTOCOL with fairness caveat."
    )

    return result
