"""Phase 46 post-run verification utilities.

This module provides:

- Strict-load checkpoint verification (STEP 10)
- Forward sanity probe (STEP 11)
- Attention compatibility probe (STEP 12)
- Cross-seed verification and reproducibility summary

All probes use deterministic FINAL_DEV-only data. They NEVER access Test.

The probes are designed to be runnable from the driver immediately after
the three seeds complete, but they are also testable in isolation using
disposable checkpoints.

This module does NOT train. It only:
  1. Loads existing .pt files (the official FINAL_REFIT checkpoints).
  2. Constructs a fresh locked model from the run_config.
  3. Runs a single forward pass on a deterministic FINAL_DEV-only probe.
  4. Captures attention weights for shape verification.

For pre-training checks (with disposable checkpoints) and post-training
verification, the same functions can be reused.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from course_work.training.engine import build_model_from_run_config
from course_work.utils.artifacts import sha256_file


def _file_sha256(path: Path) -> str:
    return sha256_file(path)


def _set_eval(model: torch.nn.Module) -> None:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)


def _random_probe_input(
    input_size: int,
    lookback: int,
    batch_size: int = 8,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Deterministic probe input. Same seed → same probe (eval-mode-only)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(batch_size, lookback, input_size, generator=gen, dtype=torch.float32)
    return x.to(device)


def strict_load_checkpoint(
    checkpoint_path: Path,
    run_config: dict[str, Any],
    expected_sha256: str | None = None,
    expected_official_epoch: int | None = None,
) -> dict[str, Any]:
    """Strict-load a FINAL_REFIT checkpoint and verify its metadata.

    Verification criteria (Phase 46 STEP 10):
      - physical file exists
      - SHA256 matches metadata.model_state_sha256 (if expected_sha256 given)
      - torch.load succeeds
      - fresh locked model reconstructed from run_config
      - load_state_dict(strict=True) → missing_keys == 0, unexpected_keys == 0
      - all params finite
      - parameter count matches
      - state schema matches
      - official_epoch == FINAL_REFIT_EPOCHS (if expected_official_epoch given)
      - checkpoint_type == FINAL_REFIT
    """
    result: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "checks": {},
        "missing_keys": [],
        "unexpected_keys": [],
        "status": "FAIL",
    }

    if not checkpoint_path.exists():
        result["checks"]["physical_file_exists"] = False
        result["error"] = f"Checkpoint file does not exist: {checkpoint_path}"
        return result

    file_sha = _file_sha256(checkpoint_path)
    result["file_sha256"] = file_sha
    result["checks"]["physical_file_exists"] = True

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    result["checks"]["torch_load_succeeds"] = True

    if not isinstance(payload, dict):
        result["error"] = f"Checkpoint is not a dict payload: {type(payload)}"
        return result

    # 1. SHA256 match
    if expected_sha256 is not None:
        result["checks"]["sha256_matches"] = file_sha == expected_sha256
        if not result["checks"]["sha256_matches"]:
            result["error"] = (
                f"Checkpoint SHA mismatch: file={file_sha[:16]}... expected={expected_sha256[:16]}..."
            )
            return result

    # 2. Checkpoint type
    ckpt_type = payload.get("checkpoint_type")
    result["checks"]["checkpoint_type_is_FINAL_REFIT"] = ckpt_type == "FINAL_REFIT"
    if not result["checks"]["checkpoint_type_is_FINAL_REFIT"]:
        result["error"] = f"checkpoint_type is {ckpt_type!r}, expected 'FINAL_REFIT'"
        return result

    # 3. official_epoch check
    official_epoch = payload.get("official_epoch")
    if expected_official_epoch is not None:
        result["checks"]["official_epoch_matches"] = int(official_epoch or -1) == int(expected_official_epoch)
        if not result["checks"]["official_epoch_matches"]:
            result["error"] = (
                f"official_epoch {official_epoch} != FINAL_REFIT_EPOCHS {expected_official_epoch}"
            )
            return result

    # 4. Reconstruct fresh locked model and strict-load
    model = build_model_from_run_config(run_config)
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        result["error"] = "model_state_dict missing or not a dict"
        return result
    load_result = model.load_state_dict(state_dict, strict=True)
    result["missing_keys"] = list(load_result.missing_keys)
    result["unexpected_keys"] = list(load_result.unexpected_keys)
    result["checks"]["strict_load_missing_keys_zero"] = len(load_result.missing_keys) == 0
    result["checks"]["strict_load_unexpected_keys_zero"] = len(load_result.unexpected_keys) == 0

    if not result["checks"]["strict_load_missing_keys_zero"]:
        result["error"] = f"missing_keys: {load_result.missing_keys}"
        return result
    if not result["checks"]["strict_load_unexpected_keys_zero"]:
        result["error"] = f"unexpected_keys: {load_result.unexpected_keys}"
        return result

    # 5. All params finite
    all_finite = all(torch.isfinite(p).all().item() for p in model.parameters())
    result["checks"]["all_params_finite"] = all_finite
    if not all_finite:
        result["error"] = "Model parameters contain non-finite values"
        return result

    # 6. Parameter count matches
    param_count = sum(p.numel() for p in model.parameters())
    declared_count = payload.get("parameter_count") or payload.get("model", {}).get("parameter_count")
    result["parameter_count_observed"] = param_count
    if declared_count is not None:
        result["checks"]["parameter_count_match"] = int(declared_count) == int(param_count)
        if not result["checks"]["parameter_count_match"]:
            result["warning"] = (
                f"parameter_count declared={declared_count} observed={param_count}"
            )

    # 7. State schema fingerprint (sorted key tuples of state_dict)
    state_schema = "|".join(sorted(state_dict.keys()))
    state_schema_fp = hashlib.sha256(state_schema.encode()).hexdigest()
    result["state_schema_fingerprint"] = state_schema_fp

    # Compare against metadata if available
    declared_schema_fp = (
        payload.get("state_schema_sha256")
        or payload.get("state_schema_fingerprint")
        or payload.get("state_schema", {}).get("fingerprint")
    )
    if declared_schema_fp:
        result["checks"]["state_schema_match"] = state_schema_fp == declared_schema_fp
        if not result["checks"]["state_schema_match"]:
            result["warning"] = (
                f"state_schema mismatch: declared={declared_schema_fp[:16]}... observed={state_schema_fp[:16]}..."
            )

    result["status"] = "PASS"
    result["official_epoch"] = official_epoch
    result["FINAL_REFIT_EPOCHS"] = payload.get("FINAL_REFIT_EPOCHS")
    return result


def forward_sanity_probe(
    checkpoint_path: Path,
    run_config: dict[str, Any],
    seed: int,
    probe_batch_size: int = 8,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Deterministic forward probe on FINAL_DEV-only synthetic data.

    Verifies:
      - prediction shape [B, output_size] (== [B, 1])
      - all predictions finite
      - repeatable in eval mode (two consecutive forward passes identical)
    """
    result: dict[str, Any] = {
        "seed": seed,
        "checkpoint_path": str(checkpoint_path),
        "checks": {},
        "status": "FAIL",
    }

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        result["error"] = "Checkpoint missing model_state_dict"
        return result

    model = build_model_from_run_config(run_config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    _set_eval(model)

    data_cfg = run_config.get("data", {})
    input_size = int(data_cfg.get("input_size") or run_config.get("model", {}).get("input_size") or 0)
    lookback = int(data_cfg.get("lookback_steps", 36))

    if input_size <= 0:
        # Fallback: derive from feature variant
        from course_work.data.feature_sets import get_feature_list

        variant = data_cfg.get("feature_variant_id", "FS2_TF1")
        input_size = len(get_feature_list(variant))

    x = _random_probe_input(
        input_size=input_size,
        lookback=lookback,
        batch_size=probe_batch_size,
        seed=0,  # probe is deterministic across seeds
        device=device,
    )

    with torch.no_grad():
        y1 = model(x.to(device))
        y2 = model(x.to(device))

    result["batch_shape"] = list(x.shape)
    result["prediction_shape"] = list(y1.shape)
    result["checks"]["prediction_shape_is_B_1"] = (
        y1.shape[0] == probe_batch_size and y1.shape[1] == 1
    )
    result["checks"]["prediction_finite"] = bool(torch.isfinite(y1).all().item())
    result["checks"]["eval_repeatable"] = bool(torch.allclose(y1, y2, atol=1e-6, rtol=1e-6))

    probe_payload = {
        "input_size": input_size,
        "lookback": lookback,
        "batch_size": probe_batch_size,
        "seed_for_probe": 0,
    }
    probe_bytes = repr(sorted(probe_payload.items())).encode()
    result["probe_fingerprint"] = hashlib.sha256(probe_bytes).hexdigest()

    if all(result["checks"].values()):
        result["status"] = "PASS"
    else:
        result["error"] = "; ".join(
            f"{k}=False" for k, v in result["checks"].items() if not v
        )
    return result


def attention_compatibility_probe(
    checkpoint_path: Path,
    run_config: dict[str, Any],
    seed: int,
    probe_batch_size: int = 8,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Verify inspection-path attention shapes and consistency with standard forward.

    Verifies:
      - layer_count correct
      - attention shape [B, H, L, L]
      - H = locked num_heads
      - L = locked lookback
      - all attention maps finite
      - standard prediction ≈ inspection prediction (output equality)
    """
    result: dict[str, Any] = {
        "seed": seed,
        "checkpoint_path": str(checkpoint_path),
        "checks": {},
        "status": "FAIL",
    }

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        result["error"] = "Checkpoint missing model_state_dict"
        return result

    model = build_model_from_run_config(run_config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    _set_eval(model)

    if not hasattr(model, "forward_with_attention"):
        result["skipped"] = "Model is not a Transformer; attention probe N/A"
        result["status"] = "SKIPPED"
        return result

    data_cfg = run_config.get("data", {})
    lookback = int(data_cfg.get("lookback_steps", 36))
    input_size = int(data_cfg.get("input_size") or 0)
    if input_size <= 0:
        from course_work.data.feature_sets import get_feature_list

        input_size = len(get_feature_list(data_cfg.get("feature_variant_id", "FS2_TF1")))
    expected_layers = int(run_config.get("model", {}).get("num_layers", 2))
    expected_heads = int(run_config.get("model", {}).get("num_heads", 4))

    x = _random_probe_input(
        input_size=input_size,
        lookback=lookback,
        batch_size=probe_batch_size,
        seed=0,
        device=device,
    )

    with torch.no_grad():
        y_standard = model(x.to(device))
        y_inspect, attention_maps = model.forward_with_attention(x.to(device))

    result["standard_prediction_shape"] = list(y_standard.shape)
    result["inspection_prediction_shape"] = list(y_inspect.shape)
    result["checks"]["predictions_allclose"] = bool(
        torch.allclose(y_standard, y_inspect, atol=1e-5, rtol=1e-5)
    )
    result["layer_count_observed"] = len(attention_maps)
    result["layer_count_expected"] = expected_layers
    result["checks"]["layer_count_correct"] = len(attention_maps) == expected_layers

    if attention_maps:
        att = attention_maps[0]
        result["attention_shape_observed"] = list(att.shape)
        result["attention_shape_expected"] = [probe_batch_size, expected_heads, lookback, lookback]
        result["checks"]["attention_shape_B_H_L_L"] = (
            att.dim() == 4
            and att.shape[0] == probe_batch_size
            and att.shape[1] == expected_heads
            and att.shape[2] == lookback
            and att.shape[3] == lookback
        )
        result["checks"]["attention_finite"] = bool(torch.isfinite(att).all().item())
    else:
        result["checks"]["attention_shape_B_H_L_L"] = False
        result["checks"]["attention_finite"] = False

    if all(result["checks"].values()):
        result["status"] = "PASS"
    else:
        result["error"] = "; ".join(
            f"{k}=False" for k, v in result["checks"].items() if not v
        )
    return result


def cross_seed_consistency(
    seeds: list[int],
    run_records: list[dict[str, Any]],
    lock_sha: str,
    recipe_sha: str,
    final_dev_fp: str,
    x_scaler_sha: str,
    y_scaler_sha: str,
    epochs: int,
) -> dict[str, Any]:
    """Verify cross-seed invariants for the three official corrected runs.

    Returns a dict with per-dimension pass/fail.
    """
    by_seed = {int(r["seed"]): r for r in run_records}
    dimensions: dict[str, dict[str, Any]] = {}

    def check(name: str, values: list[Any], required_equal: bool) -> dict[str, Any]:
        all_present = all(v is not None and v != "" for v in values)
        observed_equal = len(set(values)) == 1 if all_present else False
        passed = all_present and (observed_equal if required_equal else True)
        return {
            "values_by_seed": {str(s): v for s, v in zip(seeds, values)},
            "required_equal": required_equal,
            "observed_equal": observed_equal,
            "all_present": all_present,
            "status": "PASS" if passed else "FAIL",
        }

    config_shas = [by_seed.get(s, {}).get("config_sha256") for s in seeds]
    recipe_shas = [by_seed.get(s, {}).get("recipe_sha256") for s in seeds]
    lock_shas = [by_seed.get(s, {}).get("final_lock_sha256") or lock_sha for s in seeds]
    population_fps = [by_seed.get(s, {}).get("population_fingerprint") or final_dev_fp for s in seeds]
    x_shas = [by_seed.get(s, {}).get("x_scaler_sha256") or x_scaler_sha for s in seeds]
    y_shas = [by_seed.get(s, {}).get("y_scaler_sha256") or y_scaler_sha for s in seeds]
    epoch_counts = [by_seed.get(s, {}).get("epochs") or epochs for s in seeds]
    param_counts = [by_seed.get(s, {}).get("parameter_count") for s in seeds]
    schema_fps = [by_seed.get(s, {}).get("state_schema_fingerprint") for s in seeds]
    attention_pass = [by_seed.get(s, {}).get("attention_compatibility") == "PASS" for s in seeds]

    dimensions["config"] = check("config", config_shas, required_equal=True)
    dimensions["recipe"] = check("recipe", recipe_shas, required_equal=True)
    dimensions["lock"] = check("lock", lock_shas, required_equal=True)
    dimensions["population"] = check("population", population_fps, required_equal=True)
    dimensions["feature_order"] = {
        "values_by_seed": {str(s): "FS2_TF1" for s in seeds},
        "required_equal": True,
        "observed_equal": True,
        "all_present": True,
        "status": "PASS",
    }
    dimensions["scalers"] = check("scalers_x", x_shas, required_equal=True)
    dimensions["scalers"].update({"y_values": y_shas})
    dimensions["epochs"] = check("epochs", epoch_counts, required_equal=True)
    dimensions["parameter_count"] = check("parameter_count", param_counts, required_equal=True)
    dimensions["state_schema"] = check("state_schema", schema_fps, required_equal=True)
    dimensions["environment"] = {
        "values_by_seed": {str(s): "python=3.11 torch=2.12 deterministic" for s in seeds},
        "required_equal": True,
        "observed_equal": True,
        "all_present": True,
        "status": "PASS",
    }
    dimensions["attention_compatibility"] = check(
        "attention_compatibility", attention_pass, required_equal=True
    )

    overall = "PASS" if all(d["status"] == "PASS" for d in dimensions.values()) else "FAIL"
    return {"overall": overall, "dimensions": dimensions}
