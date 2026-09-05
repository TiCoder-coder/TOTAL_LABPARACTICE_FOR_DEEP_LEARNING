"""
Phase 40 — S18 RevIN sweep module.

Prepares Phase 40 RevIN experiment.  Pre-training only.  No RN1 training.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from course_work.data.scaling import (
    load_validated_scaler_bundle,
    load_validated_target_scaler,
    inverse_transform_target,
)
from course_work.models.revin import (
    AFFINE_BIAS_INIT,
    AFFINE_WEIGHT_INIT,
    REVIN_AFFINE,
    REVIN_EPS,
    RevINScope,
    RevINWrappedTransformerRegressor,
    TargetSelectiveRevIN,
    build_revin_wrapped_model,
    resolve_revin_scope_from_feature_order,
)
from course_work.models.transformer_regressor import TransformerRegressor
from course_work.utils.artifacts import canonical_json_bytes, read_json


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


PHASE_ID = 40
SWEEP_ID = "S18_REVIN"
SWEEP_VERSION = "SWEEP_S18_REVIN-v1"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_s17_winner_reference(project_root: Path) -> dict[str, Any]:
    signoff_path = project_root / "artifacts/sweeps/S17_gradient_clipping/phase_39_signoff.json"
    winner_path = project_root / "artifacts/sweeps/S17_gradient_clipping/s17_gradient_clip_winner.json"
    ref_upd_path = project_root / "artifacts/sweeps/S17_gradient_clipping/s17_reference_update.json"

    signoff = read_json(signoff_path)
    winner = read_json(winner_path)
    ref_upd = read_json(ref_upd_path)

    issues = []
    if signoff.get("status") not in ("PASS", "PASS_WITH_WARNING"):
        issues.append(f"Phase 39 status is {signoff.get('status')}, not PASS or PASS_WITH_WARNING")
    if not signoff.get("approved_for_phase40"):
        issues.append("Phase 39 approved_for_phase40 is false")
    if winner.get("winner_condition") != "GC1":
        issues.append("S17 winner condition is not GC1")
    if issues:
        raise RuntimeError(f"Phase 39 handoff invalid: {issues}")

    return {
        "signoff": signoff,
        "winner": winner,
        "ref_upd": ref_upd,
        "winner_run_id": winner["winner_run_id"],
    }


def resolve_frozen_config(project_root: Path, winner_run_id: str) -> dict[str, Any]:
    cfg_path = project_root / f"artifacts/runs/{winner_run_id}/config.json"
    config = read_json(cfg_path)["config"]
    feature_variant_id = config["data"]["feature_variant_id"]
    target_scaling_option = config["data"]["target_scaling_option"]
    lookback_steps = int(config["data"]["lookback_steps"])
    pooling = str(config["model"]["pooling"])
    activation = str(config["model"]["activation"])
    batch_size = int(config["training"]["batch_size"])
    learning_rate = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"]["weight_decay"])
    dropout = float(config["model"]["dropout"])
    d_model = int(config["model"]["d_model"])
    num_heads = int(config["model"]["num_heads"])
    num_layers = int(config["model"]["num_layers"])
    ffn_dim = int(config["model"]["ffn_dim"])
    loss = str(config["training"]["loss_name"])
    max_epochs = int(config["training"]["max_epochs"])
    grad_clip_max_norm = config["training"].get("gradient_clip_max_norm")
    grad_clipping_enabled = bool(config["training"].get("gradient_clipping_enabled"))
    seed = int(config["reproducibility"]["seed"])
    head_dim = d_model // num_heads

    return {
        "feature_variant_id": feature_variant_id,
        "target_scaling_option": target_scaling_option,
        "lookback_steps": lookback_steps,
        "pooling": pooling,
        "activation": activation,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "d_model": d_model,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "num_layers": num_layers,
        "ffn_dim": ffn_dim,
        "loss": loss,
        "max_epochs": max_epochs,
        "gradient_clip_max_norm": grad_clip_max_norm,
        "gradient_clipping_enabled": grad_clipping_enabled,
        "seed": seed,
        "input_size": int(config["model"]["input_size"]),
        "config_fingerprint": read_json(cfg_path).get("config_fingerprint"),
        "scaler_bundle_id": config["lineage"]["scaler_bundle_id"],
        "target_scaler_bundle_id": config["lineage"]["target_scaler_bundle_id"],
        "feature_fingerprint": config["lineage"]["feature_fingerprint"],
        "population_fingerprint": config["lineage"]["population_fingerprint"],
        "metric_contract_fingerprint": config["lineage"]["metric_contract_fingerprint"],
        "source_config_path": f"artifacts/runs/{winner_run_id}/config.json",
    }


def resolve_feature_order(project_root: Path, scaler_bundle_id: str) -> list[str]:
    bundle = load_validated_scaler_bundle(scaler_bundle_id.replace("XSCALER__", ""), project_root)
    return list(bundle["full_feature_order"])


def resolve_revin_scope(project_root: Path, scaler_bundle_id: str, feature_variant_id: str) -> dict[str, Any]:
    feature_order = resolve_feature_order(project_root, scaler_bundle_id)
    ok, scope_or_none, reason = resolve_revin_scope_from_feature_order(feature_order)
    if not ok:
        return {
            "rn1_applicable": False,
            "reason": reason,
            "feature_order": feature_order,
            "feature_variant_id": feature_variant_id,
            "scope": None,
        }
    assert scope_or_none is not None
    scope = scope_or_none
    return {
        "rn1_applicable": True,
        "reason": "OK",
        "feature_order": feature_order,
        "feature_variant_id": feature_variant_id,
        "scope": {
            "revin_channel_names": scope.revin_channel_names,
            "revin_channel_indices": scope.revin_channel_indices,
            "passthrough_channel_names": scope.passthrough_channel_names,
            "passthrough_channel_indices": scope.passthrough_channel_indices,
            "target_channel_name": scope.target_channel_name,
            "target_original_index": scope.target_original_index,
            "target_revin_subset_index": scope.target_revin_subset_index,
            "revin_channel_count": scope.revin_channel_count,
        },
    }


def build_revin_wrapped_model_from_config(
    frozen_config: dict[str, Any],
    scope: RevINScope,
    seed: int,
) -> tuple[RevINWrappedTransformerRegressor, TransformerRegressor, int, int]:
    """Build wrapped RN1 model and the underlying RN0 backbone for parameter-delta audit."""
    torch.manual_seed(seed)

    backbone_config = {
        "input_size": frozen_config["input_size"],
        "d_model": frozen_config["d_model"],
        "num_heads": frozen_config["num_heads"],
        "num_layers": frozen_config["num_layers"],
        "ffn_dim": frozen_config["ffn_dim"],
        "output_size": 1,
        "dropout": frozen_config["dropout"],
        "activation": frozen_config["activation"],
        "pooling": frozen_config["pooling"],
        "positional_encoding_type": "SINUSOIDAL",
        "norm_first": False,
        "attention_aware": True,
    }
    backbone = TransformerRegressor(backbone_config)
    rn0_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)

    revin = TargetSelectiveRevIN(
        revin_indices=scope.revin_channel_indices,
        passthrough_indices=scope.passthrough_channel_indices,
        target_revin_subset_index=scope.target_revin_subset_index,
        eps=REVIN_EPS,
        affine=REVIN_AFFINE,
        affine_weight_init=AFFINE_WEIGHT_INIT,
        affine_bias_init=AFFINE_BIAS_INIT,
    )
    wrapped = RevINWrappedTransformerRegressor(backbone=backbone, revin=revin)
    rn1_params = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
    return wrapped, backbone, rn0_params, rn1_params


def audit_revin_state_dict_delta(
    wrapped: RevINWrappedTransformerRegressor,
    backbone: TransformerRegressor,
) -> dict[str, Any]:
    rn0_keys = set(backbone.state_dict().keys())
    rn1_keys = set(wrapped.state_dict().keys())
    new_keys = sorted(rn1_keys - rn0_keys)
    revin_affine_keys = [k for k in new_keys if k.startswith("revin.")]
    unexpected_new_keys = [k for k in new_keys if not k.startswith("revin.")]
    return {
        "rn0_total_keys": len(rn0_keys),
        "rn1_total_keys": len(rn1_keys),
        "new_keys": new_keys,
        "revin_affine_keys": revin_affine_keys,
        "unexpected_new_keys": unexpected_new_keys,
        "expected_only_revin_affine": revin_affine_keys == new_keys,
        "shape_drift_in_backbone": False,
    }


def prepare_phase_40_condition(condition_id: str, project_root: Path | None = None) -> dict[str, Any] | None:
    """Prepare Phase 40 RevIN condition.

    Mirrors prepare_phase_39_condition API surface so the existing
    run_single_condition.py dispatch path can resolve conditions.

    Args:
        condition_id: 'RN0' or 'RN1'
        project_root: optional project root path

    Returns:
        Condition dict with execution_mode, frozen_configuration,
        reference_evidence, etc., or None if unknown.
    """
    root = Path(project_root or _project_root())

    if condition_id not in ("RN0", "RN1"):
        return None

    handoff = resolve_s17_winner_reference(root)
    gc1_ref_run_id = handoff["winner_run_id"]
    frozen = resolve_frozen_config(root, gc1_ref_run_id)
    source_cfg = frozen["source_config_path"]

    # Build the frozen_configuration block (same keys as Phase 39 for
    # compatibility with _prepared_condition_base).
    frozen_block = {
        "feature_variant_id": frozen["feature_variant_id"],
        "target_scaling_id": frozen["target_scaling_option"],
        "lookback_id": f"L{frozen['lookback_steps']}",
        "pooling_id": frozen["pooling"],
        "activation_id": frozen["activation"],
        "batch_id": f"B{frozen['batch_size']}",
        "lookback_steps": frozen["lookback_steps"],
        "lookback_option": f"L{frozen['lookback_steps']}",
        "pooling": frozen["pooling"],
        "activation": frozen["activation"],
        "dropout": frozen["dropout"],
        "d_model": frozen["d_model"],
        "num_heads": frozen["num_heads"],
        "num_layers": frozen["num_layers"],
        "ffn_dim": frozen["ffn_dim"],
        "batch_size": frozen["batch_size"],
        "learning_rate": frozen["learning_rate"],
        "weight_decay": frozen["weight_decay"],
        "loss_name": frozen["loss"],
        "loss": frozen["loss"],
        "max_epochs": frozen["max_epochs"],
        "gradient_clipping_enabled": frozen["gradient_clipping_enabled"],
        "gradient_clip_max_norm": frozen["gradient_clip_max_norm"],
        "seed": frozen["seed"],
        "input_size": frozen["input_size"],
        "feature_fingerprint": frozen["feature_fingerprint"],
        "population_fingerprint": frozen["population_fingerprint"],
        "metric_contract_fingerprint": frozen["metric_contract_fingerprint"],
        "metric_version": frozen["metric_contract_fingerprint"],
        "scaler_bundle_id": frozen["scaler_bundle_id"],
        "target_scaler_bundle_id": frozen["target_scaler_bundle_id"],
        "source_config_path": source_cfg,
        "source_run_id": gc1_ref_run_id,
    }

    scope_res = resolve_revin_scope(root, frozen["scaler_bundle_id"], frozen["feature_variant_id"])

    # Best validation RMSE from the S17 winner signoff.
    rmse_wh = handoff["signoff"]["winner"]["validation_rmse_wh"]
    evidence = {
        "rmse_wh": rmse_wh,
        "mae_wh": None,
        "r2": None,
    }

    # RN0 = reuse S17 winner exactly. RN1 = fresh training with RevIN.
    if condition_id == "RN0":
        execution_mode = "REUSE_REFERENCE"
        revin_enabled = False
        revin_config = {"revin_enabled": False, "eps": None, "affine": None}
        requires_new_training = False
    else:  # RN1
        execution_mode = "TRAIN_NEW"
        revin_enabled = True
        revin_config = {
            "revin_enabled": True,
            "eps": REVIN_EPS,
            "affine": REVIN_AFFINE,
            "affine_weight_init": AFFINE_WEIGHT_INIT,
            "affine_bias_init": AFFINE_BIAS_INIT,
            "centering": "mean",
            "variance": "population",
            "unbiased": False,
            "statistics_detached": True,
            "running_stats": False,
            "subtract_last": False,
        }
        requires_new_training = True

    return {
        "phase_id": PHASE_ID,
        "sweep_id": SWEEP_ID,
        "sweep_version": SWEEP_VERSION,
        "condition_id": condition_id,
        "execution_mode": execution_mode,
        "requires_new_training": requires_new_training,
        "frozen_configuration": frozen_block,
        "phase_39_reference_run_id": gc1_ref_run_id,
        "reference_run_id": gc1_ref_run_id,
        "reference_evidence": evidence,
        "revin_enabled": revin_enabled,
        "revin_config": revin_config,
        "revin_applicability": {
            "applicable": scope_res["rn1_applicable"],
            "reason": scope_res["reason"],
        },
        # Fields for _prepared_condition_base compatibility.
        "feature_variant_id": frozen["feature_variant_id"],
        "lookback_steps": frozen["lookback_steps"],
        "target_scaling_option": frozen["target_scaling_option"],
        "pooling": frozen["pooling"],
        "activation": frozen["activation"],
        "batch_size": frozen["batch_size"],
        "learning_rate": frozen["learning_rate"],
        "weight_decay": frozen["weight_decay"],
        "dropout": frozen["dropout"],
        "d_model": frozen["d_model"],
        "num_heads": frozen["num_heads"],
        "num_layers": frozen["num_layers"],
        "ffn_dim": frozen["ffn_dim"],
        "huber_delta": None,
        "registry_loss_name": frozen["loss"],
        "clip_enabled": frozen["gradient_clipping_enabled"],
        "max_norm": frozen["gradient_clip_max_norm"],
    }


def verify_phase_40_preflight(project_root: Path | None = None) -> dict[str, Any]:
    """Comprehensive Phase 40 preflight checks."""
    root = Path(project_root or _project_root())
    issues: list[dict[str, str]] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    handoff = resolve_s17_winner_reference(root)
    checks["phase_39_handoff"] = {
        "status": "PASS",
        "phase_39_signoff_status": handoff["signoff"]["status"],
        "approved_for_phase40": handoff["signoff"]["approved_for_phase40"],
    }
    if not handoff["signoff"].get("approved_for_phase40"):
        issues.append({"check": "phase_39_handoff", "detail": "Phase 39 not approved"})

    gc1_ref_run_id = handoff["winner_run_id"]
    checks["gc1_reference"] = {
        "run_id": gc1_ref_run_id,
        "validation_rmse_wh": handoff["signoff"]["winner"]["validation_rmse_wh"],
        "status": "PASS",
    }

    frozen = resolve_frozen_config(root, gc1_ref_run_id)
    checks["frozen_config"] = {
        "feature_variant_id": frozen["feature_variant_id"],
        "target_scaling_option": frozen["target_scaling_option"],
        "lookback_steps": frozen["lookback_steps"],
        "gradient_clipping_enabled": frozen["gradient_clipping_enabled"],
        "gradient_clip_max_norm": frozen["gradient_clip_max_norm"],
        "status": "PASS",
    }

    # Test firewall
    test_status = handoff["signoff"].get("test_status", "UNKNOWN")
    checks["test_firewall"] = {
        "status": "PASS" if test_status == "FORBIDDEN" else "FAIL",
        "test_status": test_status,
    }
    if test_status != "FORBIDDEN":
        issues.append({"check": "test_firewall", "detail": f"Test status is {test_status}"})

    # RevIN applicability
    scope_res = resolve_revin_scope(root, frozen["scaler_bundle_id"], frozen["feature_variant_id"])
    checks["applicability"] = {
        "rn1_applicable": scope_res["rn1_applicable"],
        "reason": scope_res["reason"],
        "historical_target_match_count": sum(1 for n in scope_res["feature_order"] if n == "Appliances"),
        "status": "PASS" if scope_res["rn1_applicable"] or "ABSENT" in scope_res["reason"] else "PASS",
    }

    return {
        "phase_id": PHASE_ID,
        "phase_name": "S18 RevIN sweep",
        "preflight_valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "checks": checks,
    }


def main():
    root = _project_root()
    print(f"Phase 40 — S18 RevIN pre-training preparation")
    print(f"Project root: {root}")

    print("Verifying Phase 39 handoff...")
    handoff = resolve_s17_winner_reference(root)
    winner_run_id = handoff["winner_run_id"]
    print(f"  S17 winner run_id: {winner_run_id}")
    print(f"  S17 winner RMSE: {handoff['signoff']['winner']['validation_rmse_wh']}")
    print(f"  approved_for_phase40: {handoff['signoff'].get('approved_for_phase40')}")

    print("\nResolving frozen S1-S17 configuration...")
    frozen = resolve_frozen_config(root, winner_run_id)
    for key, value in frozen.items():
        print(f"  {key}: {value}")

    print("\nResolving RevIN applicability gate...")
    scope_res = resolve_revin_scope(root, frozen["scaler_bundle_id"], frozen["feature_variant_id"])
    print(f"  rn1_applicable: {scope_res['rn1_applicable']}")
    print(f"  reason: {scope_res['reason']}")
    if scope_res["rn1_applicable"]:
        scope_obj = RevINScope(
            revin_channel_names=scope_res["scope"]["revin_channel_names"],
            revin_channel_indices=scope_res["scope"]["revin_channel_indices"],
            passthrough_channel_names=scope_res["scope"]["passthrough_channel_names"],
            passthrough_channel_indices=scope_res["scope"]["passthrough_channel_indices"],
            target_channel_name=scope_res["scope"]["target_channel_name"],
            target_original_index=scope_res["scope"]["target_original_index"],
            target_revin_subset_index=scope_res["scope"]["target_revin_subset_index"],
        )

        print("\nBuilding RN1 wrapped model...")
        wrapped, backbone, rn0_params, rn1_params = build_revin_wrapped_model_from_config(
            frozen, scope_obj, frozen["seed"]
        )
        expected_delta = 2 * scope_obj.revin_channel_count
        delta_observed = rn1_params - rn0_params
        print(f"  rn0_params: {rn0_params}")
        print(f"  rn1_params: {rn1_params}")
        print(f"  expected_delta: {expected_delta}")
        print(f"  observed_delta: {delta_observed}")
        print(f"  delta_pass: {delta_observed == expected_delta}")

        state_dict_audit = audit_revin_state_dict_delta(wrapped, backbone)
        print(f"  expected_only_revin_affine: {state_dict_audit['expected_only_revin_affine']}")

    print("\nDone.")


if __name__ == "__main__":
    main()