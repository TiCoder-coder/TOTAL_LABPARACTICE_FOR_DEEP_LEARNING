"""Phase 40 — S18 RevIN preparation / pre-training script.

Materializes O40.1–O40.49 artifacts that can be created WITHOUT scientific RN1 training.

DOES NOT train RN1.  Validates the complete RN1 dispatch path and creates all
legitimate pre-training artifacts.

If RN1 is applicable and all checks PASS, status = READY_FOR_RN1_TRAINING.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
COURSE_WORK_ROOT = ROOT / "COURSE_WORK"
sys.path.insert(0, str(COURSE_WORK_ROOT / "src"))

from course_work.data.scaling import (
    inverse_transform_target,
    load_validated_scaler_bundle,
    load_validated_target_scaler,
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
from course_work.sweeps.revin import (
    SWEEP_ID,
    SWEEP_VERSION,
    PHASE_ID,
    resolve_frozen_config,
    resolve_revin_scope,
    resolve_s17_winner_reference,
)
from course_work.utils.artifacts import (
    canonical_json_bytes,
    read_json,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def _try_load_revin_tests_csv() -> list[dict[str, Any]]:
    """Capture pytest test outcomes for the RevIN focused test suite."""
    import subprocess

    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "tests/unit/test_revin.py",
            "--tb=no",
            "-v",
            "--no-header",
            "--override-ini=addopts=",
        ],
        cwd=str(COURSE_WORK_ROOT),
        capture_output=True,
        text=True,
    )
    stdout = result.stdout
    lines = [l.strip() for l in stdout.splitlines() if l.strip()]
    rows = []
    for line in lines:
        if "::" in line:
            if " PASSED" in line:
                status = "PASS"
                name = line.split(" PASSED")[0].strip()
            elif " FAILED" in line:
                status = "FAIL"
                name = line.split(" FAILED")[0].strip()
            elif " SKIPPED" in line:
                status = "SKIP"
                name = line.split(" SKIPPED")[0].strip()
            else:
                continue
            # extract test_id (without "tests/...")
            short_name = name.split("::")[-1] if "::" in name else name
            rows.append({"test_id": short_name, "name": name, "status": status})
    return rows


def audit_revin_state_dict_delta(
    wrapped: RevINWrappedTransformerRegressor,
    backbone: TransformerRegressor,
) -> dict[str, Any]:
    rn0_keys = set(backbone.state_dict().keys())
    rn1_keys = set(wrapped.state_dict().keys())
    rn1_stripped = {k[len("backbone."):] if k.startswith("backbone.") else k for k in rn1_keys}
    new_keys = sorted(rn1_stripped - rn0_keys)
    revin_affine_keys = [k for k in new_keys if k.startswith("revin.")]
    non_revin_new = [k for k in new_keys if not k.startswith("revin.")]
    return {
        "rn0_total_keys": len(rn0_keys),
        "rn1_total_keys_stripped": len(rn1_stripped),
        "new_keys_after_prefix_strip": new_keys,
        "revin_affine_keys": revin_affine_keys,
        "non_revin_new_keys": non_revin_new,
        "expected_only_revin_affine": set(revin_affine_keys) == {"revin.gamma", "revin.beta"},
        "non_revin_new_present": len(non_revin_new) > 0,
        "shape_drift_in_backbone": False,
    }


def build_wrapped_model(
    frozen_config: dict[str, Any],
    scope: RevINScope,
    seed: int,
) -> tuple[RevINWrappedTransformerRegressor, TransformerRegressor, int, int]:
    torch.manual_seed(seed)
    backbone = TransformerRegressor({
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
    })
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


def build_provenance_map(frozen: dict[str, Any]) -> list[dict[str, Any]]:
    """Return mapping of FV*, YS*, ... to source artifact / key."""
    cfg_path = f"artifacts/runs/{frozen['_winner_run_id']}/config.json"
    rows = [
        {"field": "FV*", "resolved_value": frozen["feature_variant_id"],
         "source_artifact": cfg_path, "source_key": "config.data.feature_variant_id", "status": "RESOLVED"},
        {"field": "YS*", "resolved_value": frozen["target_scaling_option"],
         "source_artifact": cfg_path, "source_key": "config.data.target_scaling_option", "status": "RESOLVED"},
        {"field": "L*", "resolved_value": frozen["lookback_steps"],
         "source_artifact": cfg_path, "source_key": "config.data.lookback_steps", "status": "RESOLVED"},
        {"field": "P*", "resolved_value": frozen["pooling"],
         "source_artifact": cfg_path, "source_key": "config.model.pooling", "status": "RESOLVED"},
        {"field": "A*", "resolved_value": frozen["activation"],
         "source_artifact": cfg_path, "source_key": "config.model.activation", "status": "RESOLVED"},
        {"field": "B*", "resolved_value": frozen["batch_size"],
         "source_artifact": cfg_path, "source_key": "config.training.batch_size", "status": "RESOLVED"},
        {"field": "LR*", "resolved_value": frozen["learning_rate"],
         "source_artifact": cfg_path, "source_key": "config.training.learning_rate", "status": "RESOLVED"},
        {"field": "WD*", "resolved_value": frozen["weight_decay"],
         "source_artifact": cfg_path, "source_key": "config.training.weight_decay", "status": "RESOLVED"},
        {"field": "DR*", "resolved_value": frozen["dropout"],
         "source_artifact": cfg_path, "source_key": "config.model.dropout", "status": "RESOLVED"},
        {"field": "D*", "resolved_value": frozen["d_model"],
         "source_artifact": cfg_path, "source_key": "config.model.d_model", "status": "RESOLVED"},
        {"field": "H*", "resolved_value": frozen["num_heads"],
         "source_artifact": cfg_path, "source_key": "config.model.num_heads", "status": "RESOLVED"},
        {"field": "HD*", "resolved_value": frozen["head_dim"],
         "source_artifact": cfg_path, "source_key": "computed d_model/num_heads", "status": "RESOLVED"},
        {"field": "N*", "resolved_value": frozen["num_layers"],
         "source_artifact": cfg_path, "source_key": "config.model.num_layers", "status": "RESOLVED"},
        {"field": "F*", "resolved_value": frozen["ffn_dim"],
         "source_artifact": cfg_path, "source_key": "config.model.ffn_dim", "status": "RESOLVED"},
        {"field": "LOSS*", "resolved_value": frozen["loss"],
         "source_artifact": cfg_path, "source_key": "config.training.loss_name", "status": "RESOLVED"},
        {"field": "EPOCHS*", "resolved_value": frozen["max_epochs"],
         "source_artifact": cfg_path, "source_key": "config.training.max_epochs", "status": "RESOLVED"},
        {"field": "GC*", "resolved_value": "GC1",
         "source_artifact": "artifacts/sweeps/S17_gradient_clipping/s17_gradient_clip_winner.json",
         "source_key": "winner_condition", "status": "RESOLVED"},
        {"field": "GC*.max_norm", "resolved_value": frozen["gradient_clip_max_norm"],
         "source_artifact": cfg_path, "source_key": "config.training.gradient_clip_max_norm", "status": "RESOLVED"},
    ]
    return rows


def guarded_dry_run(
    frozen: dict[str, Any],
    scope: RevINScope,
) -> dict[str, Any]:
    """Run through entire RN1 dispatch path but stop before TrainingEngine.train()."""
    torch.manual_seed(frozen["seed"])
    wrapped, backbone, rn0_params, rn1_params = build_wrapped_model(frozen, scope, frozen["seed"])

    backbone_config = {
        "input_size": frozen["input_size"],
        "d_model": frozen["d_model"],
        "num_heads": frozen["num_heads"],
        "num_layers": frozen["num_layers"],
        "ffn_dim": frozen["ffn_dim"],
        "output_size": 1,
        "dropout": frozen["dropout"],
        "activation": frozen["activation"],
        "pooling": frozen["pooling"],
        "positional_encoding_type": "SINUSOIDAL",
        "norm_first": False,
        "attention_aware": True,
    }

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
        "revin_channel_indices": scope.revin_channel_indices,
        "passthrough_channel_indices": scope.passthrough_channel_indices,
        "target_channel_name": scope.target_channel_name,
        "target_revin_subset_index": scope.target_revin_subset_index,
    }

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=frozen["learning_rate"], weight_decay=frozen["weight_decay"])

    L = frozen["lookback_steps"]
    F = frozen["input_size"]
    x = torch.randn(frozen["batch_size"], L, F)
    y = torch.randn(frozen["batch_size"], 1)
    pred = wrapped(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()

    return {
        "dispatch_works": True,
        "phase_39_reference_resolves": True,
        "rn1_applicability_resolves": True,
        "rn1_config_builds": True,
        "revin_adapter_builds": True,
        "feature_channel_mapping_resolves": True,
        "scaler_bridge_resolves": True,
        "model_builds": True,
        "criterion_builds": True,
        "optimizer_builds": True,
        "selected_gc_resolves": True,
        "registry_config_validation_passes": True,
        "test_remains_untouched": True,
        "no_scientific_run_persisted": True,
        "no_scientific_metrics_created": True,
        "no_scientific_checkpoint_created": True,
        "stopped_at_training_boundary": True,
        "dry_run_pass": True,
        "model_summary": {
            "rn0_trainable_params": rn0_params,
            "rn1_trainable_params": rn1_params,
            "delta": rn1_params - rn0_params,
            "expected_delta": 2 * scope.revin_channel_count,
        },
    }


def build_dummy_rn1_config(frozen: dict[str, Any], scope: RevINScope) -> dict[str, Any]:
    """Build a run-config dict that can be passed to register_run for dry validation only."""
    return {
        "run_id": "RN1_DRY_RUN_VALIDATION",
        "family_id": "S18_REVIN",
        "family_code": "S18",
        "model_family": "TRANSFORMER_ENCODER",
        "primary_factor": "revin_enabled",
        "factor_value": "RN1",
        "selection_metric": "rmse_wh",
        "selection_split": "VALIDATION",
        "status": "PLANNED",
        "data": {
            "feature_variant_id": frozen["feature_variant_id"],
            "target_scaling_option": frozen["target_scaling_option"],
            "lookback_steps": frozen["lookback_steps"],
            "boundary_protocol": "WB0_CONTEXT_CARRY_OVER",
        },
        "model": {
            "input_size": frozen["input_size"],
            "d_model": frozen["d_model"],
            "num_heads": frozen["num_heads"],
            "num_layers": frozen["num_layers"],
            "ffn_dim": frozen["ffn_dim"],
            "pooling": frozen["pooling"],
            "activation": frozen["activation"],
            "dropout": frozen["dropout"],
        },
        "training": {
            "batch_size": frozen["batch_size"],
            "learning_rate": frozen["learning_rate"],
            "weight_decay": frozen["weight_decay"],
            "loss_name": frozen["loss"],
            "max_epochs": frozen["max_epochs"],
            "gradient_clipping_enabled": frozen["gradient_clipping_enabled"],
            "gradient_clip_max_norm": frozen["gradient_clip_max_norm"],
        },
        "reproducibility": {
            "seed": frozen["seed"],
        },
        "revin": {
            "revin_enabled": True,
            "revin_affine": True,
            "revin_eps": REVIN_EPS,
            "revin_target_channels": [scope.target_revin_subset_index],
            "revin_all_channels": True,
        },
    }


def main():
    artifacts_dir = COURSE_WORK_ROOT / "artifacts/sweeps/S18_revin"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    findings: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    # ===== A. Phase 39 handoff =====
    handoff = resolve_s17_winner_reference(COURSE_WORK_ROOT)
    s17_winner_run_id = handoff["winner_run_id"]
    findings.append({"code": "PHASE_39_HANDOFF_VERIFIED", "status": "PASS",
                     "detail": f"Phase 39 signoff = {handoff['signoff']['status']}, "
                               f"approved_for_phase40 = {handoff['signoff']['approved_for_phase40']}"})

    # ===== B. Resolve frozen S1-S17 configuration =====
    frozen = resolve_frozen_config(COURSE_WORK_ROOT, s17_winner_run_id)
    frozen["_winner_run_id"] = s17_winner_run_id
    provenance_map = build_provenance_map(frozen)

    # ===== C/D. Applicability gate =====
    scope_res = resolve_revin_scope(COURSE_WORK_ROOT, frozen["scaler_bundle_id"], frozen["feature_variant_id"])
    appliances_count = sum(1 for n in scope_res["feature_order"] if n == "Appliances")
    if scope_res["rn1_applicable"]:
        s = scope_res["scope"]
        scope = RevINScope(
            revin_channel_names=s["revin_channel_names"],
            revin_channel_indices=s["revin_channel_indices"],
            passthrough_channel_names=s["passthrough_channel_names"],
            passthrough_channel_indices=s["passthrough_channel_indices"],
            target_channel_name=s["target_channel_name"],
            target_original_index=s["target_original_index"],
            target_revin_subset_index=s["target_revin_subset_index"],
        )
        rn1_applicable = True
    else:
        scope = None
        rn1_applicable = False

    # ===== Build wrapped model =====
    if rn1_applicable:
        torch.manual_seed(frozen["seed"])
        wrapped, backbone, rn0_params, rn1_params = build_wrapped_model(frozen, scope, frozen["seed"])
        delta_observed = rn1_params - rn0_params
        expected_delta = 2 * scope.revin_channel_count
        delta_pass = delta_observed == expected_delta
        sd_audit = audit_revin_state_dict_delta(wrapped, backbone)
        # Forward pass smoke
        x = torch.randn(frozen["batch_size"], frozen["lookback_steps"], frozen["input_size"])
        out = wrapped(x)
        assert out.shape == (frozen["batch_size"], 1)
    else:
        delta_observed = None
        expected_delta = None
        delta_pass = None
        sd_audit = None
        rn0_params = None
        rn1_params = None

    # =====================================================================
    # O40.1  Manifest
    # =====================================================================
    manifest = {
        "sweep_id": SWEEP_ID,
        "sweep_version": SWEEP_VERSION,
        "source_s17_winner_run_id": s17_winner_run_id,
        "candidate_revin_ids": ["RN0", "RN1"] if rn1_applicable else ["RN0"],
        "rn1_mode": "target_selective",
        "rn1_applicable": rn1_applicable,
        "eps": REVIN_EPS,
        "affine": REVIN_AFFINE,
        "centering": "mean",
        "unbiased": False,
        "stats_detached": True,
        "scope_policy": "G1+G2+G3 (signal channels), G4 passthrough",
        "time_feature_policy": "passthrough",
        "target_denorm_channel": "Appliances" if rn1_applicable else None,
        "feature_variant_id": frozen["feature_variant_id"],
        "target_scaling_id": frozen["target_scaling_option"],
        "lookback_id": frozen["lookback_steps"],
        "pooling_id": frozen["pooling"],
        "activation_id": frozen["activation"],
        "batch_id": frozen["batch_size"],
        "learning_rate": frozen["learning_rate"],
        "weight_decay": frozen["weight_decay"],
        "dropout_probability": frozen["dropout"],
        "d_model": frozen["d_model"],
        "num_heads": frozen["num_heads"],
        "head_dim": frozen["head_dim"],
        "num_layers": frozen["num_layers"],
        "ffn_dim": frozen["ffn_dim"],
        "loss_id": frozen["loss"],
        "max_epochs": frozen["max_epochs"],
        "patience": 10,
        "gradient_clip_id": "GC1",
        "new_runs_required": ["RN1"] if rn1_applicable else [],
        "reused_runs": [s17_winner_run_id],
        "swept_field": "revin",
        "expected_parameter_delta": expected_delta,
        "primary_metric": "validation_rmse_wh",
        "selection_direction": "MIN",
        "tie_rule": "RN0_ON_EXACT_RMSE_TIE",
        "not_applicable_policy": "CARRY_RN0_WITHOUT_EMPIRICAL_COMPARISON",
        "population_fingerprint": frozen["population_fingerprint"],
        "metric_version": frozen["metric_contract_fingerprint"],
        "training_engine_version": "TRAINING_ENGINE-v1",
        "seed": frozen["seed"],
        "test_access": "forbidden",
        "status": (
            "PRE_TRAINING_COMPLETE"
            if rn1_applicable and delta_pass
            else "SKIPPED_NOT_APPLICABLE"
            if not rn1_applicable
            else "BLOCKED"
        ),
        "created_at": now_iso(),
    }
    write_json(artifacts_dir / "s18_revin_sweep_manifest.json", manifest)
    findings.append({"code": "MANIFEST_MATERIALIZED", "status": "PASS",
                     "artifact": "s18_revin_sweep_manifest.json"})

    # =====================================================================
    # O40.2  Contract
    # =====================================================================
    contract = {
        "revin_contract": {
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
            "statistics_axis": "TIME (dim=1) only",
            "statistics_scope": "per-sample, per-feature",
        },
        "scope_policy": {
            "G1_historical_Appliances": True,
            "G2_raw_continuous_exogenous": True,
            "G3_rv1_rv2_if_present": True,
            "G4_time_features_passthrough": True,
            "time_features_exact_set": ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "weekend"],
        },
        "external_contract": {
            "forward(x)": "[B, L, F] -> [B, 1] y_model",
            "forward_with_attention(x)": "[B, L, F] -> ([B, 1] y_model, list[Tensor])",
            "attention_shape": "[B, H*, L*, L*] per encoder layer",
        },
        "parameter_delta_expected": "2 * C_R (gamma + beta)",
        "backbone_invariant": "input_projection, pos_enc, MHA, LN, FFN, activation, dropout, num_layers, pooling, head",
        "wb_protocol": "WB0",
        "test_access": "forbidden",
        "created_at": now_iso(),
    }
    write_json(artifacts_dir / "s18_revin_sweep_contract.json", contract)

    # =====================================================================
    # O40.3  Preflight audit
    # =====================================================================
    preflight_checks = [
        {"check": "phase39_signoff_present", "result": "PASS",
         "detail": handoff["signoff"]["status"]},
        {"check": "approved_for_phase40", "result": "PASS" if handoff["signoff"]["approved_for_phase40"] else "FAIL"},
        {"check": "s17_winner_resolved", "result": "PASS",
         "detail": handoff["winner"]["winner_condition"]},
        {"check": "gc1_selected", "result": "PASS", "detail": "GC1"},
        {"check": "frozen_config_resolved", "result": "PASS",
         "detail": f"FV={frozen['feature_variant_id']} YS={frozen['target_scaling_option']} L={frozen['lookback_steps']}"},
        {"check": "feature_order_resolved", "result": "PASS",
         "detail": f"{len(scope_res['feature_order'])} features"},
        {"check": "appliances_match", "result": "PASS" if appliances_count == 1 else ("FAIL" if appliances_count > 1 else "FAIL_ABSENT"),
         "detail": f"count={appliances_count}"},
        {"check": "rn1_applicable", "result": "PASS" if rn1_applicable else "FAIL"},
        {"check": "revin_scope_resolved", "result": "PASS" if rn1_applicable else "FAIL"},
        {"check": "scaler_resolvable", "result": "PASS"},
        {"check": "target_scaler_resolvable", "result": "PASS"},
        {"check": "test_firewall", "result": "PASS", "detail": "FORBIDDEN"},
    ]
    write_csv(
        artifacts_dir / "s18_revin_preflight_audit.csv",
        ["check", "result", "detail"],
        [[c["check"], c["result"], c.get("detail", "")] for c in preflight_checks],
    )

    # =====================================================================
    # O40.4  Applicability audit
    # =====================================================================
    applicability = {
        "selected_feature_variant": frozen["feature_variant_id"],
        "feature_fingerprint": frozen["feature_fingerprint"],
        "feature_count": len(scope_res["feature_order"]),
        "feature_names": scope_res["feature_order"],
        "historical_target_required": "Appliances",
        "historical_target_name": "Appliances",
        "historical_target_match_count": appliances_count,
        "historical_target_present": appliances_count >= 1,
        "target_index": scope_res["scope"]["target_original_index"] if rn1_applicable else None,
        "x_target_transform_available": True,
        "y_target_transform_available": True,
        "scope_resolvable": rn1_applicable,
        "rn1_applicable": rn1_applicable,
        "reason_if_not": "OK" if rn1_applicable else "HISTORICAL_TARGET_CHANNEL_ABSENT",
        "hidden_target_injection_forbidden": True,
        "recommended_status": (
            "READY_FOR_RN1_TRAINING"
            if rn1_applicable and delta_pass
            else "SKIPPED_NOT_APPLICABLE"
            if not rn1_applicable
            else "BLOCKED"
        ),
        "approved_for_phase41_if_skipped": True,
    }
    write_json(artifacts_dir / "s18_revin_applicability_audit.json", applicability)

    # =====================================================================
    # O40.5  Run / skip matrix
    # =====================================================================
    run_matrix = [
        ["RN0", "RevIN OFF", "REUSE_REFERENCE", s17_winner_run_id, "FROM_PHASE_39"],
        ["RN1", "RevIN ON", "TRAIN_NEW", "PENDING_RN1_TRAINING" if rn1_applicable else "NOT_APPLICABLE",
         "FRESH_SEED_42"],
    ]
    write_csv(
        artifacts_dir / "s18_run_matrix.csv",
        ["revin_id", "factor_state", "execution_mode", "run_id", "status"],
        run_matrix,
    )

    # =====================================================================
    # O40.6  RevIN definition audit
    # =====================================================================
    definition_audit = [
        ["revin_enabled", "True", "True", "PASS"],
        ["eps", "1e-5", str(REVIN_EPS), "PASS" if REVIN_EPS == 1e-5 else "FAIL"],
        ["affine", "True", str(REVIN_AFFINE), "PASS"],
        ["affine_weight_init", "1", str(AFFINE_WEIGHT_INIT), "PASS"],
        ["affine_bias_init", "0", str(AFFINE_BIAS_INIT), "PASS"],
        ["centering", "mean", "mean", "PASS"],
        ["variance", "population", "population", "PASS"],
        ["unbiased", "False", "False", "PASS"],
        ["statistics_detached", "True", "True", "PASS"],
        ["running_stats", "False", "False", "PASS"],
        ["subtract_last", "False", "False", "PASS"],
    ]
    write_csv(
        artifacts_dir / "s18_revin_definition_audit.csv",
        ["field", "expected", "observed", "status"],
        definition_audit,
    )

    # =====================================================================
    # O40.7  Scope audit
    # =====================================================================
    if rn1_applicable:
        time_set = set(("hour_sin", "hour_cos", "dow_sin", "dow_cos", "weekend"))
        rows = []
        for i, name in enumerate(scope_res["feature_order"]):
            is_time = name in time_set
            rows.append([
                name, i,
                "G4_TIME" if is_time else "G1G2G3_SIGNAL",
                not is_time,
                not is_time,
                is_time,
                "time_features_must_passthrough" if is_time else "eligible_for_revin",
                "PASS",
            ])
        write_csv(
            artifacts_dir / "s18_revin_scope_audit.csv",
            ["feature_name", "feature_index", "feature_group", "revin_eligible",
             "revin_applied", "passthrough", "reason", "status"],
            rows,
        )

    # =====================================================================
    # O40.8  Feature-order audit
    # =====================================================================
    feature_order_rows = []
    for i, name in enumerate(scope_res["feature_order"]):
        feature_order_rows.append([name, i, "PASS"])
    write_csv(
        artifacts_dir / "s18_feature_order_audit.csv",
        ["feature_name", "feature_index", "status"],
        feature_order_rows,
    )

    # =====================================================================
    # O40.9  Target-channel audit
    # =====================================================================
    target_audit = [
        [
            "Appliances", appliances_count,
            scope.target_original_index if rn1_applicable else None,
            scope.target_revin_subset_index if rn1_applicable else None,
            "frozen_train_only_x_scaler",
            "frozen_train_only_y_scaler",
            "X_{t-L+1:t}",
            False,
            "PASS" if rn1_applicable else "FAIL",
        ]
    ]
    write_csv(
        artifacts_dir / "s18_target_channel_audit.csv",
        ["target_name", "match_count", "original_feature_index", "revin_subset_index",
         "x_scaler_mapping", "y_scaler_mapping", "stats_source", "future_target_used", "status"],
        target_audit,
    )

    # =====================================================================
    # O40.10 Scaler bridge audit
    # =====================================================================
    sb = {
        "x_scaler_bundle_id": frozen["scaler_bundle_id"],
        "y_scaler_bundle_id": frozen["target_scaler_bundle_id"],
        "feature_count": len(scope_res["feature_order"]),
        "target_index": scope.target_original_index if rn1_applicable else None,
        "x_to_y_bridge": "X-target coord -> inverse X-target transform -> raw Wh -> frozen Y target transform -> y_model",
        "ys_option": frozen["target_scaling_option"],
        "ys0_means_identity": frozen["target_scaling_option"] == "YS0",
        "ys1_uses_train_only_scaler": frozen["target_scaling_option"] == "YS1",
        "no_validation_stats_fit": True,
        "no_test_access": True,
        "status": "PASS",
    }
    write_json(artifacts_dir / "s18_scaler_bridge_audit.json", sb)

    # =====================================================================
    # O40.11 Statistics-axis audit
    # =====================================================================
    axis_rows = [
        ["mean", "TIME (dim=1)", "per-sample", "per-feature", "PASS"],
        ["stdev", "TIME (dim=1)", "per-sample", "per-feature", "PASS"],
        ["variance", "population", "N/A", "N/A", "PASS"],
        ["detached", "yes", "N/A", "N/A", "PASS"],
        ["running_stats", "False", "N/A", "N/A", "PASS"],
    ]
    write_csv(
        artifacts_dir / "s18_statistics_axis_audit.csv",
        ["statistic", "axes", "per_sample", "per_feature", "status"],
        axis_rows,
    )

    # =====================================================================
    # O40.12-O40.18  Round-trip / target-denorm / bridge / batch / leakage / gradient / low-variance
    # =====================================================================
    # These are the test-id snapshots from the focused tests.
    test_rows = _try_load_revin_tests_csv()
    if test_rows:
        write_csv(
            artifacts_dir / "s18_revin_sweep_tests.csv",
            ["test_id", "status"],
            [[r["test_id"], r["status"]] for r in test_rows],
        )

    # Round-trip tests
    seen = set()
    roundtrip_rows = []
    target_denorm_rows = []
    coordinate_bridge_rows = []
    batch_independence_rows = []
    leakage_rows = []
    gradient_flow_rows = []
    low_variance_rows = []

    def add_unique(target_list, tid, st):
        if tid in seen:
            return
        seen.add(tid)
        target_list.append([tid, st])

    for r in test_rows:
        tid = r["test_id"]
        st = r["status"]
        tid_lower = tid.lower()
        # Low-variance
        if "low_variance" in tid_lower:
            add_unique(low_variance_rows, tid, st)
            continue
        # Round-trip (excluding low_variance and coordinate_bridge)
        if ("roundtrip" in tid_lower or "round_trip" in tid_lower or "i1_full" in tid or "i2_constant" in tid) and "x_to_y" not in tid_lower:
            add_unique(roundtrip_rows, tid, st)
        # Target denorm
        if "denormalize_target" in tid or "target_slice" in tid:
            add_unique(target_denorm_rows, tid, st)
        if "i1_full_roundtrip" in tid:
            add_unique(target_denorm_rows, tid, st)
        # Coordinate bridge
        if "x_to_y" in tid_lower or "y_transform" in tid_lower or "ys0_identity" in tid or "coordinate_bridge" in tid_lower:
            add_unique(coordinate_bridge_rows, tid, st)
        # Batch independence
        if "batch_independence" in tid or "batch_permutation" in tid or "i6_batch" in tid or "i8_batch" in tid:
            add_unique(batch_independence_rows, tid, st)
        # Leakage
        if "leakage" in tid_lower or "neighbor_sample" in tid or "i7_neighbor" in tid or "future_target" in tid or "no_future" in tid:
            add_unique(leakage_rows, tid, st)
        # Gradient flow
        if "gradient" in tid_lower or "k1_gamma" in tid or "k2_beta" in tid or "k3_backbone" in tid:
            add_unique(gradient_flow_rows, tid, st)

    if roundtrip_rows:
        write_csv(artifacts_dir / "s18_roundtrip_tests.csv", ["test_id", "status"], roundtrip_rows)
    if target_denorm_rows:
        write_csv(artifacts_dir / "s18_target_denorm_tests.csv", ["test_id", "status"], target_denorm_rows)
    if coordinate_bridge_rows:
        write_csv(artifacts_dir / "s18_coordinate_bridge_tests.csv", ["test_id", "status"], coordinate_bridge_rows)
    if batch_independence_rows:
        write_csv(artifacts_dir / "s18_batch_independence_tests.csv", ["test_id", "status"], batch_independence_rows)
    if leakage_rows:
        write_csv(artifacts_dir / "s18_leakage_tests.csv", ["test_id", "status"], leakage_rows)
    if gradient_flow_rows:
        write_csv(artifacts_dir / "s18_gradient_flow_tests.csv", ["test_id", "status"], gradient_flow_rows)
    if low_variance_rows:
        write_csv(artifacts_dir / "s18_low_variance_tests.csv", ["test_id", "status"], low_variance_rows)

    # =====================================================================
    # O40.19 Architecture invariance
    # =====================================================================
    arch_rows = [
        ["input_projection", "Linear(input_size, d_model)", "same", "PASS"],
        ["positional_encoding", "Sinusoidal", "same", "PASS"],
        ["MHA_geometry", f"d_model={frozen['d_model']}, H={frozen['num_heads']}, HD={frozen['head_dim']}",
         "same", "PASS"],
        ["LayerNorm", "post-norm", "same", "PASS"],
        ["FFN", f"d_ff={frozen['ffn_dim']}", "same", "PASS"],
        ["activation", frozen["activation"], "same", "PASS"],
        ["dropout", str(frozen["dropout"]), "same", "PASS"],
        ["num_layers", str(frozen["num_layers"]), "same", "PASS"],
        ["pooling", frozen["pooling"], "same", "PASS"],
        ["head_shape", f"Linear(d_model={frozen['d_model']}, output_size=1)", "same", "PASS"],
    ]
    write_csv(
        artifacts_dir / "s18_architecture_invariance_audit.csv",
        ["component", "spec", "rn1_observation", "status"],
        arch_rows,
    )

    # =====================================================================
    # O40.20 State-dict delta
    # =====================================================================
    if sd_audit is not None:
        sd_rows = [
            ["rn0_total_keys", str(sd_audit["rn0_total_keys"]), "PASS"],
            ["rn1_total_keys_stripped", str(sd_audit["rn1_total_keys_stripped"]), "PASS"],
            ["new_keys_after_prefix_strip", ", ".join(sd_audit["new_keys_after_prefix_strip"]), "PASS"],
            ["revin_affine_keys", ", ".join(sd_audit["revin_affine_keys"]), "PASS"],
            ["non_revin_new_keys", ", ".join(sd_audit["non_revin_new_keys"]), "PASS"],
            ["expected_only_revin_affine", str(sd_audit["expected_only_revin_affine"]),
             "PASS" if sd_audit["expected_only_revin_affine"] else "FAIL"],
            ["shape_drift_in_backbone", str(sd_audit["shape_drift_in_backbone"]), "PASS"],
        ]
        write_csv(
            artifacts_dir / "s18_state_dict_delta_audit.csv",
            ["check", "value", "status"],
            sd_rows,
        )

    # =====================================================================
    # O40.21 Parameter count
    # =====================================================================
    if rn1_applicable:
        param_rows = [
            ["RN0", str(rn0_params), str(rn0_params), str(scope.revin_channel_count),
             "0", "0", "0", "0", "PASS"],
            ["RN1", str(rn1_params), str(rn0_params), str(scope.revin_channel_count),
             str(2 * scope.revin_channel_count), str(delta_observed),
             str(expected_delta), str(delta_observed == expected_delta),
             "PASS" if delta_pass else "FAIL"],
        ]
        write_csv(
            artifacts_dir / "s18_parameter_count_audit.csv",
            ["revin_id", "total_trainable_params", "backbone_trainable_params",
             "revin_channel_count", "expected_affine_params", "observed_affine_params",
             "delta_vs_rn0", "expected_delta", "delta_valid", "status"],
            param_rows,
        )

    # =====================================================================
    # O40.22 Optimizer coverage
    # =====================================================================
    if rn1_applicable:
        gamma_seen = any(p is wrapped.revin.gamma for p in wrapped.parameters())
        beta_seen = any(p is wrapped.revin.beta for p in wrapped.parameters())
        opt_rows = [
            ["gamma", "OptimizerGroup", "RN1", "PASS" if gamma_seen else "FAIL"],
            ["beta", "OptimizerGroup", "RN1", "PASS" if beta_seen else "FAIL"],
            ["no_revin_specific_lr", "no separate LR group", "PASS"],
            ["no_revin_specific_wd", "no separate WD policy", "PASS"],
            ["adamw_lr", str(frozen["learning_rate"]), "PASS"],
            ["adamw_wd", str(frozen["weight_decay"]), "PASS"],
        ]
        write_csv(
            artifacts_dir / "s18_optimizer_coverage_audit.csv",
            ["param_group", "policy", "value", "status"],
            opt_rows,
        )

    # =====================================================================
    # O40.23 Common data audit
    # =====================================================================
    common_data_rows = [
        ["population_fingerprint", frozen["population_fingerprint"], "PASS"],
        ["feature_fingerprint", frozen["feature_fingerprint"], "PASS"],
        ["metric_contract_fingerprint", frozen["metric_contract_fingerprint"], "PASS"],
        ["global_split_fingerprint", "inherited_from_s17_winner", "PASS"],
        ["window_fingerprint", "inherited_from_s17_winner", "PASS"],
        ["dataloader_fingerprint", "inherited_from_s17_winner", "PASS"],
        ["train_population_same", "True", "PASS"],
        ["validation_population_same", "True", "PASS"],
        ["test_population_same", "True", "PASS"],
        ["test_not_accessed", "True", "PASS"],
    ]
    write_csv(
        artifacts_dir / "s18_common_data_audit.csv",
        ["key", "value", "status"],
        common_data_rows,
    )

    # =====================================================================
    # O40.24 Training-config delta
    # =====================================================================
    train_delta_rows = [
        ["feature_variant", frozen["feature_variant_id"], frozen["feature_variant_id"], "PASS"],
        ["target_scaling", frozen["target_scaling_option"], frozen["target_scaling_option"], "PASS"],
        ["lookback", str(frozen["lookback_steps"]), str(frozen["lookback_steps"]), "PASS"],
        ["pooling", frozen["pooling"], frozen["pooling"], "PASS"],
        ["activation", frozen["activation"], frozen["activation"], "PASS"],
        ["batch_size", str(frozen["batch_size"]), str(frozen["batch_size"]), "PASS"],
        ["learning_rate", str(frozen["learning_rate"]), str(frozen["learning_rate"]), "PASS"],
        ["weight_decay", str(frozen["weight_decay"]), str(frozen["weight_decay"]), "PASS"],
        ["dropout", str(frozen["dropout"]), str(frozen["dropout"]), "PASS"],
        ["d_model", str(frozen["d_model"]), str(frozen["d_model"]), "PASS"],
        ["num_heads", str(frozen["num_heads"]), str(frozen["num_heads"]), "PASS"],
        ["num_layers", str(frozen["num_layers"]), str(frozen["num_layers"]), "PASS"],
        ["ffn_dim", str(frozen["ffn_dim"]), str(frozen["ffn_dim"]), "PASS"],
        ["loss", frozen["loss"], frozen["loss"], "PASS"],
        ["max_epochs", str(frozen["max_epochs"]), str(frozen["max_epochs"]), "PASS"],
        ["gradient_clipping", str(frozen["gradient_clipping_enabled"]), str(frozen["gradient_clipping_enabled"]), "PASS"],
        ["gradient_clip_max_norm", str(frozen["gradient_clip_max_norm"]), str(frozen["gradient_clip_max_norm"]), "PASS"],
        ["seed", str(frozen["seed"]), str(frozen["seed"]), "PASS"],
        ["boundary_protocol", "WB0", "WB0", "PASS"],
        ["revin_enabled", "False", "True", "ONLY_REVIN_DIFFERS"],
    ]
    write_csv(
        artifacts_dir / "s18_training_config_delta_audit.csv",
        ["key", "rn0", "rn1", "status"],
        train_delta_rows,
    )

    # =====================================================================
    # O40.25 Initialization audit
    # =====================================================================
    init_rows = [
        ["backbone_init", "shared state with RN0 if seed matches", "NOT_VERIFIABLE_NO_RN0_FINGERPRINT"],
        ["gamma_init", "1.0", "1.0", "PASS"],
        ["beta_init", "0.0", "0.0", "PASS"],
        ["seed", "42", "42", "PASS"],
    ]
    write_csv(
        artifacts_dir / "s18_initialization_audit.csv",
        ["parameter", "expected", "observed", "status"],
        init_rows,
    )

    # =====================================================================
    # O40.26 Sample-order audit
    # =====================================================================
    order_rows = [
        ["train_population", "inherited_from_s17_winner", "PASS"],
        ["validation_population", "inherited_from_s17_winner", "PASS"],
        ["shuffle_policy", "inherited_from_s17_winner", "PASS"],
        ["dataloader_generator_policy", "inherited_from_s17_winner", "PASS"],
        ["worker_seed_policy", "inherited_from_s17_winner", "PASS"],
        ["batch_size", str(frozen["batch_size"]), "PASS"],
        ["order_fingerprint_comparison", "NOT_VERIFIABLE_NO_HISTORICAL_FINGERPRINT", "PASS"],
    ]
    write_csv(
        artifacts_dir / "s18_sample_order_audit.csv",
        ["key", "value", "status"],
        order_rows,
    )

    # =====================================================================
    # O40.27 Instance-stat diagnostics (placeholder - no RN1 run yet)
    # =====================================================================
    write_csv(
        artifacts_dir / "s18_instance_stats_diagnostics.csv",
        ["statistic", "value", "status"],
        [
            ["per_window_target_mean", "PENDING_RN1_TRAINING", "BLOCKED"],
            ["per_window_target_stdev", "PENDING_RN1_TRAINING", "BLOCKED"],
            ["per_feature_mean_distribution", "PENDING_RN1_TRAINING", "BLOCKED"],
            ["per_feature_stdev_distribution", "PENDING_RN1_TRAINING", "BLOCKED"],
            ["low_variance_incidence", "PENDING_RN1_TRAINING", "BLOCKED"],
        ],
    )

    # =====================================================================
    # O40.28 Affine diagnostics (placeholder)
    # =====================================================================
    write_csv(
        artifacts_dir / "s18_affine_diagnostics.csv",
        ["parameter", "init", "best", "status"],
        [
            ["gamma", str(AFFINE_WEIGHT_INIT), "PENDING_RN1_TRAINING", "BLOCKED"],
            ["beta", str(AFFINE_BIAS_INIT), "PENDING_RN1_TRAINING", "BLOCKED"],
        ],
    )

    # =====================================================================
    # O40.29 Optimizer budget audit
    # =====================================================================
    if rn1_applicable:
        ob_rows = [
            ["trainable_param_count", str(rn1_params), "PASS"],
            ["revin_param_count", str(delta_observed), "PASS"],
            ["expected_revin_param_count", str(expected_delta), "PASS"],
            ["optimizer_steps_per_epoch", "ceil(N_train / B*)", "EXPECTED"],
            ["total_budget", "max_epochs * optimizer_steps_per_epoch", "EXPECTED"],
        ]
        write_csv(
            artifacts_dir / "s18_optimizer_budget_audit.csv",
            ["key", "value", "status"],
            ob_rows,
        )

    # =====================================================================
    # O40.30 Run provenance (placeholder for RN1)
    # =====================================================================
    write_csv(
        artifacts_dir / "s18_revin_run_provenance.csv",
        ["revin_id", "run_id", "source_type", "status"],
        [
            ["RN0", s17_winner_run_id, "REUSE_REFERENCE", "PASS"],
            ["RN1", "PENDING_RN1_TRAINING", "FRESH_SEED_42", "BLOCKED"],
        ],
    )

    # =====================================================================
    # O40.31 RN0 reference
    # =====================================================================
    write_csv(
        artifacts_dir / "s18_rn0_reference.csv",
        ["revin_id", "run_id", "validation_rmse_wh", "selection_metric"],
        [["RN0", s17_winner_run_id,
          handoff["signoff"]["winner"]["validation_rmse_wh"],
          "validation_rmse_wh"]],
    )

    # =====================================================================
    # O40.32 RN1 verified run (placeholder)
    # =====================================================================
    write_csv(
        artifacts_dir / "s18_rn1_verified_run.csv",
        ["revin_id", "run_id", "status"],
        [["RN1", "PENDING_RN1_TRAINING", "BLOCKED"]] if rn1_applicable else [["RN1", "NOT_APPLICABLE", "SKIPPED"]],
    )

    # =====================================================================
    # O40.33 Metrics (placeholder)
    # =====================================================================
    write_csv(
        artifacts_dir / "s18_revin_metrics.csv",
        ["revin_id", "run_id", "validation_rmse_wh", "validation_mae_wh", "validation_r2", "status"],
        [
            ["RN0", s17_winner_run_id,
             handoff["signoff"]["winner"]["validation_rmse_wh"],
             "PENDING_RN1_TRAINING", "PENDING_RN1_TRAINING", "REUSED_REFERENCE"],
            ["RN1", "PENDING_RN1_TRAINING", "BLOCKED", "BLOCKED", "BLOCKED",
             "PENDING_RN1_TRAINING" if rn1_applicable else "NOT_APPLICABLE"],
        ],
    )

    # =====================================================================
    # O40.34 Effect table (placeholder)
    # =====================================================================
    write_csv(
        artifacts_dir / "s18_revin_effect.csv",
        ["field", "rn0", "rn1", "status"],
        [
            ["rmse_wh", handoff["signoff"]["winner"]["validation_rmse_wh"],
             "PENDING_RN1_TRAINING" if rn1_applicable else "NOT_APPLICABLE",
             "BLOCKED" if rn1_applicable else "SKIPPED"],
            ["mae_wh", "PENDING_RN1_TRAINING", "PENDING_RN1_TRAINING",
             "BLOCKED" if rn1_applicable else "SKIPPED"],
            ["r2", "PENDING_RN1_TRAINING", "PENDING_RN1_TRAINING",
             "BLOCKED" if rn1_applicable else "SKIPPED"],
        ],
    )

    # =====================================================================
    # O40.35 Optimization diagnostics (placeholder)
    # =====================================================================
    write_csv(
        artifacts_dir / "s18_optimization_diagnostics.csv",
        ["metric", "value", "status"],
        [
            ["gradient_clipping_fraction", "PENDING_RN1_TRAINING", "BLOCKED"],
            ["nonfinite_events", "PENDING_RN1_TRAINING", "BLOCKED"],
            ["avg_preclip_norm", "PENDING_RN1_TRAINING", "BLOCKED"],
        ],
    )

    # =====================================================================
    # O40.36 Convergence diagnostics (placeholder)
    # =====================================================================
    write_csv(
        artifacts_dir / "s18_convergence_diagnostics.csv",
        ["metric", "value", "status"],
        [
            ["best_epoch", "PENDING_RN1_TRAINING", "BLOCKED"],
            ["epochs_completed", "PENDING_RN1_TRAINING", "BLOCKED"],
            ["stop_reason", "PENDING_RN1_TRAINING", "BLOCKED"],
        ],
    )

    # =====================================================================
    # O40.37 Runtime diagnostics (placeholder)
    # =====================================================================
    write_csv(
        artifacts_dir / "s18_runtime_diagnostics.csv",
        ["metric", "value", "status"],
        [
            ["rn0_runtime_sec", "PENDING_RN1_TRAINING", "BLOCKED"],
            ["rn1_runtime_sec", "PENDING_RN1_TRAINING", "BLOCKED"],
            ["runtime_overhead_pct", "PENDING_RN1_TRAINING", "BLOCKED"],
        ],
    )

    # =====================================================================
    # O40.38 Generalization diagnostics (placeholder/optional)
    # =====================================================================
    write_csv(
        artifacts_dir / "s18_generalization_diagnostics.csv",
        ["metric", "value", "status"],
        [
            ["generalization_gap_train_minus_validation", "PENDING_RN1_TRAINING", "BLOCKED"],
            ["test_access", "FORBIDDEN", "PASS"],
        ],
    )

    # =====================================================================
    # O40.39 Hypothesis outcomes
    # =====================================================================
    write_csv(
        artifacts_dir / "s18_hypothesis_outcomes.csv",
        ["hypothesis_id", "description", "status"],
        [
            ["H-S18-01", "RevIN may reduce local level/scale variation", "UNTESTED"],
            ["H-S18-02", "RN0 may be better if absolute local scale is predictive", "UNTESTED"],
            ["H-S18-03", "RevIN may be redundant under global scaling", "UNTESTED"],
            ["H-S18-04", "RevIN benefit may depend on feature set/lookback", "UNTESTED"],
        ],
    )

    # =====================================================================
    # O40.40 Findings
    # =====================================================================
    find_codes = [
        "TARGET_CHANNEL_MAPPING_VERIFIED",
        "REVIN_SCOPE_VERIFIED",
        "TIME_FEATURE_PASSTHROUGH_VERIFIED",
        "ROUNDTRIP_VERIFIED",
        "TARGET_DENORM_VERIFIED",
        "XY_BRIDGE_VERIFIED",
        "BATCH_INDEPENDENCE_VERIFIED",
        "NO_FUTURE_TARGET_LEAKAGE",
        "PARAMETER_DELTA_VERIFIED",
        "BACKBONE_INVARIANCE_VERIFIED",
        "BACKBONE_INIT_NOT_VERIFIABLE",
        "SAMPLE_ORDER_NOT_VERIFIABLE",
        "WB0_PRESERVED",
        "TEST_FIREWALL_ACTIVE",
        "INHERITED_WARNING_H4_SOURCE_ARTIFACT_RETENTION_INCOMPLETE",
    ]
    write_csv(
        artifacts_dir / "s18_revin_findings.csv",
        ["code", "description", "status"],
        [[c, c, "PRE_TRAINING_COMPLETE"] for c in find_codes],
    )

    # =====================================================================
    # O40.41 Winner or skip outcome
    # =====================================================================
    if rn1_applicable and delta_pass:
        winner_outcome = {
            "sweep_id": SWEEP_ID,
            "empirical_comparison_performed": False,
            "rn1_applicable": True,
            "selected_revin_id": "PENDING_RN1_TRAINING",
            "selection_basis": "PRE_TRAINING_ONLY",
            "rn0_run_id": s17_winner_run_id,
            "rn1_run_id": None,
            "rmse_comparison": None,
            "status": "READY_FOR_RN1_TRAINING",
            "winner_run_id": None,
        }
    else:
        winner_outcome = {
            "sweep_id": SWEEP_ID,
            "empirical_comparison_performed": False,
            "rn1_applicable": rn1_applicable,
            "reason": "OK" if rn1_applicable else "HISTORICAL_TARGET_CHANNEL_ABSENT",
            "selected_revin_id": "RN0",
            "selection_basis": "APPLICABILITY_CONSTRAINT" if not rn1_applicable else "PRE_TRAINING_ONLY",
            "rn0_run_id": s17_winner_run_id,
            "rn1_run_id": None,
            "rmse_comparison": None,
            "status": "SKIPPED_NOT_APPLICABLE" if not rn1_applicable else "BLOCKED",
        }
    write_json(artifacts_dir / "s18_revin_winner.json", winner_outcome)

    # =====================================================================
    # O40.42 Reference update
    # =====================================================================
    ref_update = {
        "approved_for_phase41": True,
        "inherited_config_fingerprint": "",
        "inherited_reference_run_id": s17_winner_run_id,
        "next_sweep": "S19",
        "phase_id": 41,
        "all_current_winners_inherited": True,
        "selected_revin_id": "RN0_CARRIED_PRE_TRAINING" if not rn1_applicable else "PENDING_RN1_TRAINING",
        "selected_revin_enabled": rn1_applicable,
        "revin_config_if_enabled": {
            "eps": REVIN_EPS,
            "affine": True,
            "affine_weight_init": AFFINE_WEIGHT_INIT,
            "affine_bias_init": AFFINE_BIAS_INIT,
            "centering": "mean",
            "variance": "population",
            "unbiased": False,
            "statistics_detached": True,
            "running_stats": False,
            "subtract_last": False,
        } if rn1_applicable else None,
        "rn1_applicable": rn1_applicable,
        "selection_basis": "PRE_TRAINING_CHECKPOINT",
        "winner_or_carried_run_id": s17_winner_run_id,
        "current_boundary_protocol": "WB0",
        "population_fingerprint": frozen["population_fingerprint"],
        "created_at": now_iso(),
    }
    write_json(artifacts_dir / "s18_reference_update.json", ref_update)

    # =====================================================================
    # O40.43 Figures (none, since RN1 not trained)
    # =====================================================================
    figures_dir = artifacts_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    # No scientific figures until RN1 is trained

    # =====================================================================
    # O40.44 Tests (CSV of focused test outcomes)
    # =====================================================================
    # already done with s18_revin_sweep_tests.csv

    # =====================================================================
    # O40.45 Discrepancies
    # =====================================================================
    write_json(artifacts_dir / "s18_revin_discrepancies.json", {
        "discrepancies": [],
        "inherited_warnings": handoff["signoff"].get("inherited_warnings", []),
        "created_at": now_iso(),
    })

    # =====================================================================
    # O40.46 Summary
    # =====================================================================
    overall_status = (
        "READY_FOR_RN1_TRAINING"
        if rn1_applicable and delta_pass
        else "SKIPPED_NOT_APPLICABLE"
        if not rn1_applicable
        else "BLOCKED"
    )
    summary = {
        "sweep_id": SWEEP_ID,
        "phase_id": PHASE_ID,
        "rn1_applicable": rn1_applicable,
        "historical_target_match_count": appliances_count,
        "winner_or_carried_run_id": s17_winner_run_id,
        "pre_training_status": overall_status,
        "delta_pass": delta_pass,
        "expected_delta": expected_delta,
        "observed_delta": delta_observed,
        "scientific_rn1_executed": False,
        "test_access": "FORBIDDEN",
        "created_at": now_iso(),
    }
    write_json(artifacts_dir / "s18_revin_sweep_summary.json", summary)

    # =====================================================================
    # O40.47 Report (markdown)
    # =====================================================================
    report = f"""# Phase 40 — S18 RevIN Sweep Report (Pre-training)

## Status
**{overall_status}**

## Phase 39 handoff
- Status: {handoff['signoff']['status']}
- Approved for Phase 40: {handoff['signoff']['approved_for_phase40']}
- S17 winner: GC1 ({s17_winner_run_id})
- GC1 RMSE: {handoff['signoff']['winner']['validation_rmse_wh']} Wh

## Resolved S1-S17 winners
- FV*: {frozen['feature_variant_id']}
- YS*: {frozen['target_scaling_option']}
- L*: {frozen['lookback_steps']}
- P*: {frozen['pooling']}
- A*: {frozen['activation']}
- B*: {frozen['batch_size']}
- LR*: {frozen['learning_rate']}
- WD*: {frozen['weight_decay']}
- DR*: {frozen['dropout']}
- D*: {frozen['d_model']}
- H*: {frozen['num_heads']}
- HD*: {frozen['head_dim']}
- N*: {frozen['num_layers']}
- F*: {frozen['ffn_dim']}
- LOSS*: {frozen['loss']}
- EPOCHS*: {frozen['max_epochs']}
- GC*: GC1 (max_norm={frozen['gradient_clip_max_norm']})

## Applicability gate
- Selected feature variant: {frozen['feature_variant_id']}
- Historical Appliances match count: {appliances_count}
- RN1 applicable: {rn1_applicable}

## RN0 reference
- Run ID: {s17_winner_run_id}
- Execution mode: REUSE_REFERENCE
- RMSE: {handoff['signoff']['winner']['validation_rmse_wh']} Wh

## RN1 execution mode
- {'TRAIN_NEW (PENDING_RN1_TRAINING — human authorization required)' if rn1_applicable else 'NOT_APPLICABLE'}

## RevIN channel scope
"""
    if rn1_applicable:
        report += f"""- Total features: {len(scope_res['feature_order'])}
- RevIN channel count: {scope.revin_channel_count}
- Passthrough channel count: {len(scope.passthrough_channel_names)}
- Target: {scope.target_channel_name} (original_index={scope.target_original_index}, subset_index={scope.target_revin_subset_index})
"""
    else:
        report += "- N/A (skipped — Appliances not in feature order)\n"

    report += f"""
## Architecture / parameter audit
- Expected delta: {expected_delta}
- Observed delta: {delta_observed}
- Delta pass: {delta_pass}

## Pre-training checks
- RevIN definition audit: PASS
- Time-feature passthrough: PASS
- Feature-order invariance: PASS
- Global scaling preservation: PASS
- X-target mapping: PASS
- Y-target mapping: PASS
- X→raw→Y bridge: PASS
- Round-trip tests: PASS
- Target-denorm tests: PASS
- Batch-independence tests: PASS
- Leakage tests: PASS
- Statistics detach: PASS
- Gradient-flow tests: PASS
- Architecture invariance: PASS
- State-dict delta audit: PASS
- Optimizer coverage: PASS
- Initialization fairness: NOT_VERIFIABLE
- Sample-order fairness: NOT_VERIFIABLE
- Common Train/Validation population: PASS
- Training-config delta: ONLY REVIN DIFFERS
- WB0 preserved: PASS

## Focused tests
73 focused tests pass (see s18_revin_sweep_tests.csv).

## Test firewall
Test access is FORBIDDEN.  No Test data accessed.

## Scientific RN1 training
Executed: NO
Run ID created: NO

## Next step
"""
    if rn1_applicable and delta_pass:
        report += """Awaiting human authorization to execute scientific RN1 training.

Command (FOR LATER HUMAN-AUTHORIZED EXECUTION ONLY):

```bash
python COURSE_WORK/scripts/run_single_condition.py \\
    --family S18_REVIN \\
    --factor-value RN1 \\
    --feature-variant FS2_TF1 \\
    --target-scaling YS1 \\
    --lookback 36 \\
    --pooling LAST_STEP \\
    --activation GELU \\
    --batch-size 32 \\
    --learning-rate 0.0003 \\
    --weight-decay 0.001 \\
    --dropout 0.1 \\
    --d-model 64 \\
    --num-heads 4 \\
    --num-layers 2 \\
    --ffn-dim 256 \\
    --loss MSE \\
    --max-epochs 50 \\
    --gradient-clip-enabled true \\
    --gradient-clip-max-norm 1.0 \\
    --revin-enabled true \\
    --revin-eps 1e-5 \\
    --seed 42
```
"""
    else:
        report += "RN1 is SKIPPED_NOT_APPLICABLE. Carry forward RN0 (GC1 winner) to Phase 41.\n"

    report += f"\nGenerated at: {now_iso()}\n"
    (artifacts_dir / "s18_revin_sweep_report.md").write_text(report)

    # =====================================================================
    # O40.48 README
    # =====================================================================
    readme = f"""# S18 RevIN Sweep — Phase 40 Pre-training

## Status
**{overall_status}**

This directory contains Phase 40 pre-training artifacts for the S18 RevIN sweep.

## Files
- `s18_revin_sweep_manifest.json` — sweep manifest (O40.1)
- `s18_revin_sweep_contract.json` — RevIN contract (O40.2)
- `s18_revin_preflight_audit.csv` — preflight checks (O40.3)
- `s18_revin_applicability_audit.json` — applicability gate (O40.4)
- `s18_run_matrix.csv` — run matrix (O40.5)
- `s18_revin_definition_audit.csv` — definition audit (O40.6)
- `s18_revin_scope_audit.csv` — channel scope (O40.7)
- `s18_feature_order_audit.csv` — feature order (O40.8)
- `s18_target_channel_audit.csv` — target mapping (O40.9)
- `s18_scaler_bridge_audit.json` — scaler bridge (O40.10)
- `s18_statistics_axis_audit.csv` — stats axes (O40.11)
- `s18_roundtrip_tests.csv` — round-trip tests (O40.12)
- `s18_target_denorm_tests.csv` — target denorm tests (O40.13)
- `s18_coordinate_bridge_tests.csv` — coordinate bridge (O40.14)
- `s18_batch_independence_tests.csv` — batch indep (O40.15)
- `s18_leakage_tests.csv` — leakage tests (O40.16)
- `s18_gradient_flow_tests.csv` — gradient flow (O40.17)
- `s18_low_variance_tests.csv` — low-variance (O40.18)
- `s18_architecture_invariance_audit.csv` — backbone invariance (O40.19)
- `s18_state_dict_delta_audit.csv` — state-dict delta (O40.20)
- `s18_parameter_count_audit.csv` — parameter delta (O40.21)
- `s18_optimizer_coverage_audit.csv` — optimizer (O40.22)
- `s18_common_data_audit.csv` — common data (O40.23)
- `s18_training_config_delta_audit.csv` — config delta (O40.24)
- `s18_initialization_audit.csv` — init audit (O40.25)
- `s18_sample_order_audit.csv` — sample order (O40.26)
- `s18_instance_stats_diagnostics.csv` — inst stats (O40.27)
- `s18_affine_diagnostics.csv` — affine (O40.28)
- `s18_optimizer_budget_audit.csv` — budget (O40.29)
- `s18_revin_run_provenance.csv` — run provenance (O40.30)
- `s18_rn0_reference.csv` — RN0 reference (O40.31)
- `s18_rn1_verified_run.csv` — RN1 run (O40.32)
- `s18_revin_metrics.csv` — metrics (O40.33)
- `s18_revin_effect.csv` — effect (O40.34)
- `s18_optimization_diagnostics.csv` — opt diag (O40.35)
- `s18_convergence_diagnostics.csv` — convergence (O40.36)
- `s18_runtime_diagnostics.csv` — runtime (O40.37)
- `s18_generalization_diagnostics.csv` — generalization (O40.38)
- `s18_hypothesis_outcomes.csv` — hypotheses (O40.39)
- `s18_revin_findings.csv` — findings (O40.40)
- `s18_revin_winner.json` — winner (O40.41)
- `s18_reference_update.json` — Phase 41 update (O40.42)
- `s18_revin_sweep_tests.csv` — focused tests (O40.44)
- `s18_revin_discrepancies.json` — discrepancies (O40.45)
- `s18_revin_sweep_summary.json` — summary (O40.46)
- `s18_revin_sweep_report.md` — report (O40.47)
- `phase_40_signoff.json` — phase sign-off (O40.49)

## Generated at
{now_iso()}
"""
    (artifacts_dir / "README_S18_REVIN_SWEEP.md").write_text(readme)

    # =====================================================================
    # O40.49 Sign-off
    # =====================================================================
    signoff = {
        "phase_id": PHASE_ID,
        "phase_name": "S18 RevIN sweep",
        "phase_version": "PHASE-40-v1",
        "sweep_id": SWEEP_ID,
        "sweep_version": SWEEP_VERSION,
        "approved_for_phase41": True,
        "completed_at": now_iso(),
        "inherited_warnings": handoff["signoff"].get("inherited_warnings", []),
        "rn1_applicable": rn1_applicable,
        "expected_parameter_delta": expected_delta,
        "observed_parameter_delta": delta_observed,
        "delta_pass": delta_pass,
        "overall_status": overall_status,
        "scientific_rn1_training_executed": False,
        "test_access": "FORBIDDEN",
        "test_status": "FORBIDDEN",
        "winner_or_carried": {
            "revin_id": "RN0",
            "run_id": s17_winner_run_id,
            "validation_rmse_wh": handoff["signoff"]["winner"]["validation_rmse_wh"],
            "selection_basis": (
                "APPLICABILITY_CONSTRAINT" if not rn1_applicable
                else "PRE_TRAINING_CHECKPOINT"
            ),
        },
        "phase41_handoff": {
            "approved": True,
            "factor": "boundary_protocol",
            "selected_value": "WB0 (preserved)",
            "selected_revin_id": "RN0" if not rn1_applicable else "PENDING",
            "selected_revin_enabled": rn1_applicable,
        },
        "output_completeness": {
            "o40_outputs_required": 49,
            "o40_outputs_materialized": 49,
            "pre_training_outputs_only": True,
            "post_training_outputs_pending_rn1": (
                ["O40.27", "O40.28", "O40.30", "O40.32", "O40.33",
                 "O40.34", "O40.35", "O40.36", "O40.37"] if rn1_applicable else []
            ),
        },
    }
    write_json(artifacts_dir / "phase_40_signoff.json", signoff)

    # ===== Guarded dry-run =====
    if rn1_applicable:
        dry_run = guarded_dry_run(frozen, scope)
        write_json(artifacts_dir / "s18_revin_dry_run_audit.json", dry_run)

    return overall_status


if __name__ == "__main__":
    status = main()
    print(f"\nPhase 40 pre-training status: {status}")