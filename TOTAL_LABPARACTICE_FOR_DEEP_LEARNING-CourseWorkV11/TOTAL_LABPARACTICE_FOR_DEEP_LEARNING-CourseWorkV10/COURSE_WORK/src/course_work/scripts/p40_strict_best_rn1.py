"""Phase 40 Strict BEST Verification for RN1 — Option A (MPS required, 1e-9 absolute).

Per Human-approved plan:
  COURSE_WORK/docs/plan-doc/plan_before_process/phase_40_s18_revin_strict_best_corrective_plan.md

Policy:
  - strict-BEST verification must run on the recorded scientific device (MPS)
  - CPU fallback is FORBIDDEN
  - NO silent CPU fallback
  - absolute tolerance = 1e-9
  - NO relative tolerance arm
  - if MPS is unavailable on the verification host:
      * if an authoritative external-report file is present and validates, use it as PASS
        evidence (Human-performed same-device MPS verification)
      * otherwise, emit NOT_VERIFIABLE_ON_CURRENT_DEVICE (no CPU recomputation)
  - if any metric delta > 1e-9, strict BEST is FAIL

External-report contract:
  - file path: artifacts/sweeps/S18_revin/rn1_external_strict_best_report.json
  - must declare recorded_device == mps (matches RN1 recorded device)
  - must declare cpu_fallback == DISABLED, comparison_rule == absolute_only
  - tolerance must equal 1e-9 (canonical)
  - rel_tolerance must be None
  - stored_validation_metrics.{rmse_wh,mae_wh,r2} must equal the canonical RN1 stored metrics
  - metric_delta.{rmse_wh,mae_wh,r2} must each be <= 1e-9
  - status must be PASS
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

COURSE_WORK_ROOT = Path("/Users/vientu/Deep Learning/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK")
sys.path.insert(0, str(COURSE_WORK_ROOT / "src"))

from course_work.data.datasets import build_train_validation_loaders  
from course_work.data.scaling import (  
    load_validated_scaler_bundle,
    load_validated_target_scaler,
)
from course_work.evaluation.metrics import EvaluationMode, compute_regression_metrics 
from course_work.models.revin import (  
    AFFINE_BIAS_INIT,
    AFFINE_WEIGHT_INIT,
    REVIN_AFFINE,
    REVIN_EPS,
    RevINWrappedTransformerRegressor,
    TargetSelectiveRevIN,
    resolve_revin_scope_from_feature_order,
)
from course_work.models.transformer_regressor import TransformerRegressor   
from course_work.utils.artifacts import read_json  

ROOT = COURSE_WORK_ROOT
RUN_ID = "RUN_TR_S18_0031_A711A9B8"
SEED = 42
CANONICAL_TOLERANCE = 1e-9
RECORDED_DEVICE_REQUIRED = "mps"
PLAN_AUTHORITY = (
    "COURSE_WORK/docs/plan-doc/plan_before_process/"
    "phase_40_s18_revin_strict_best_corrective_plan.md"
)
EXTERNAL_REPORT_PATH = ROOT / "artifacts/sweeps/S18_revin/rn1_external_strict_best_report.json"


def _build_backbone_config(config: dict) -> dict:
    m = config["model"]
    return {
        "input_size": int(m["input_size"]),
        "d_model": int(m["d_model"]),
        "num_heads": int(m["num_heads"]),
        "num_layers": int(m["num_layers"]),
        "ffn_dim": int(m["ffn_dim"]),
        "output_size": int(m["output_size"]),
        "dropout": float(m["dropout"]),
        "activation": str(m["activation"]),
        "pooling": str(m["pooling"]),
        "positional_encoding_type": str(m["positional_encoding_type"]),
        "norm_first": bool(m.get("norm_first", False)),
        "attention_aware": bool(m.get("attention_aware", True)),
    }


def _check_recorded_device_available(config: dict) -> tuple[bool, str, str]:
    """Return (available, recorded_device, reason_if_unavailable).

    Does NOT raise. The caller decides whether to gracefully emit
    NOT_VERIFIABLE_ON_CURRENT_DEVICE or proceed with strict BEST.
    """
    recorded = config["runtime"]["device_type"]
    if recorded not in ("cpu", "cuda", "mps"):
        return False, recorded, f"unsupported recorded device: {recorded}"
    if recorded == "mps" and not torch.backends.mps.is_available():
        return False, recorded, "recorded device MPS is not available on this host"
    if recorded == "cuda" and not torch.cuda.is_available():
        return False, recorded, "recorded device CUDA is not available on this host"
    return True, recorded, ""


def _try_load_external_report(config: dict) -> tuple[dict | None, str]:
    """Validate and return the Human-authored external strict BEST report.

    Returns (payload, reason) where reason is empty on success.

    The external report is the Human-authoritative PASS evidence produced by
    running the same-device MPS verification on a host that exposes MPS. The
    report must satisfy all of:
      - status == PASS
      - recorded_device matches RN1's recorded device
      - cpu_fallback == DISABLED
      - comparison_rule == absolute_only
      - tolerance == CANONICAL_TOLERANCE (1e-9)
      - rel_tolerance is None
      - stored_validation_metrics exactly match RN1's canonical stored metrics
      - metric_delta.{rmse_wh,mae_wh,r2} each <= CANONICAL_TOLERANCE
      - test_access == FORBIDDEN
    """
    if not EXTERNAL_REPORT_PATH.is_file():
        return None, "external report file not present"
    try:
        report = read_json(EXTERNAL_REPORT_PATH)
    except Exception as exc:
        return None, f"external report unreadable: {exc}"

    recorded = config["runtime"]["device_type"]
    expected_stored_path = ROOT / "artifacts/runs" / RUN_ID / "metrics/best_validation_metrics.json"
    try:
        canonical_metrics = read_json(expected_stored_path)["metric_result"]
    except Exception as exc:
        return None, f"canonical RN1 metrics unreadable: {exc}"

    canonical_stored = {
        "rmse_wh": float(canonical_metrics["rmse_wh"]),
        "mae_wh": float(canonical_metrics["mae_wh"]),
        "r2": float(canonical_metrics["r2"]),
    }

    issues: list[str] = []
    if str(report.get("status", "")) != "PASS":
        issues.append(f"status != PASS ({report.get('status')})")
    if str(report.get("recorded_device", "")) != recorded:
        issues.append(
            f"recorded_device mismatch (report={report.get('recorded_device')}, "
            f"canonical={recorded})"
        )
    if str(report.get("cpu_fallback", "")) != "DISABLED":
        issues.append(f"cpu_fallback != DISABLED ({report.get('cpu_fallback')})")
    if str(report.get("comparison_rule", "")) != "absolute_only":
        issues.append(f"comparison_rule != absolute_only ({report.get('comparison_rule')})")
    if report.get("tolerance") != CANONICAL_TOLERANCE:
        issues.append(
            f"tolerance != {CANONICAL_TOLERANCE} (got={report.get('tolerance')})"
        )
    if report.get("rel_tolerance") is not None:
        issues.append(f"rel_tolerance must be None (got={report.get('rel_tolerance')})")

    stored = report.get("stored_validation_metrics") or {}
    for field, value in canonical_stored.items():
        if float(stored.get(field, float("nan"))) != value:
            issues.append(f"stored_validation_metrics.{field} != canonical")

    delta = report.get("metric_delta") or {}
    for field in ("rmse_wh", "mae_wh", "r2"):
        if abs(float(delta.get(field, float("inf")))) > CANONICAL_TOLERANCE:
            issues.append(f"metric_delta.{field} > {CANONICAL_TOLERANCE}")

    if str(report.get("test_access", "")) != "FORBIDDEN":
        issues.append(f"test_access != FORBIDDEN ({report.get('test_access')})")

    if issues:
        return None, "external report validation failed: " + "; ".join(issues)
    return report, ""


def _build_external_accepted_payload(config: dict, report: dict) -> dict:
    """Build a strict-BEST PASS payload from the validated external report.

    Reuses canonical RN1 evidence (feature variant, lookback, seed, model arch,
    parameter count, RevIN channel counts) and folds in the externally-verified
    metric deltas. Mirrors the shape of the on-host PASS payload so the
    finalize script can consume it identically.
    """
    run_root = ROOT / "artifacts/runs" / RUN_ID
    config_payload = read_json(run_root / "config.json")
    config = config_payload["config"]
    status = read_json(run_root / "status.json")
    history = pd.read_csv(run_root / "training_history.csv")
    metrics = read_json(run_root / "metrics/best_validation_metrics.json")["metric_result"]
    fv = config["data"]["feature_variant_id"]
    bundle = load_validated_scaler_bundle(fv, ROOT)
    feat_order = list(bundle["full_feature_order"])
    ok, scope_or_none, _ = resolve_revin_scope_from_feature_order(feat_order)
    assert ok and scope_or_none is not None
    scope = scope_or_none

    backbone = TransformerRegressor(_build_backbone_config(config))
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
    parameter_count = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)

    stored = report["stored_validation_metrics"]
    rec = report["recomputed_validation_metrics"]
    delta = report["metric_delta"]

    return {
        "run_id": RUN_ID,
        "experiment_family": "S18_REVIN",
        "status": "PASS",
        "strict_load_verification": "PASS",
        "strict_best_verification": "PASS",
        "recorded_device": str(report["recorded_device"]),
        "verification_device": str(report["verification_device"]),
        "verification_device_used_at_runtime": str(
            report.get("verification_device_used_at_runtime", report["verification_device"])
        ),
        "verification_device_warning": None,
        "cpu_fallback": "DISABLED",
        "external_report_used": True,
        "external_report_path": str(EXTERNAL_REPORT_PATH.relative_to(ROOT)),
        "external_report_authority": str(report.get("_verified_by", "Human reviewer")),
        "external_report_verified_at": str(report.get("_verified_at", "")),
        "plan_authority": PLAN_AUTHORITY,
        "stored_validation_metrics": {
            "rmse_wh": float(stored["rmse_wh"]),
            "mae_wh": float(stored["mae_wh"]),
            "r2": float(stored["r2"]),
            "n_samples": int(stored["n_samples"]),
        },
        "recomputed_validation_metrics": {
            "rmse_wh": float(rec["rmse_wh"]),
            "mae_wh": float(rec["mae_wh"]),
            "r2": float(rec["r2"]),
            "n_samples": int(rec["n_samples"]),
        },
        "metric_delta": {
            "rmse_wh": float(delta["rmse_wh"]),
            "mae_wh": float(delta["mae_wh"]),
            "r2": float(delta["r2"]),
        },
        "comparison_rule": "absolute_only",
        "canonical_tolerance": CANONICAL_TOLERANCE,
        "tolerance": CANONICAL_TOLERANCE,
        "abs_tolerance": CANONICAL_TOLERANCE,
        "rel_tolerance": None,
        "per_metric_pass": {
            "rmse": bool(report["per_metric_pass"]["rmse"]),
            "mae": bool(report["per_metric_pass"]["mae"]),
            "r2": bool(report["per_metric_pass"]["r2"]),
        },
        "population_match": bool(report.get("population_match", True)),
        "sample_order_match": bool(report.get("sample_order_match", True)),
        "parameter_count": parameter_count,
        "revin_channel_count": scope.revin_channel_count,
        "passthrough_channel_count": len(scope.passthrough_channel_indices),
        "target_channel_name": scope.target_channel_name,
        "target_original_index": scope.target_original_index,
        "target_revin_subset_index": scope.target_revin_subset_index,
        "revin_eps": REVIN_EPS,
        "revin_affine": REVIN_AFFINE,
        "best_epoch": int(report.get("best_epoch", status["best_epoch"])),
        "epochs_executed": int(history["epoch"].max()),
        "test_access": "FORBIDDEN",
        "feature_variant_id": fv,
        "feature_count": len(feat_order),
        "target_scaling_option": config["data"]["target_scaling_option"],
        "lookback_steps": int(config["data"]["lookback_steps"]),
        "seed": int(config["reproducibility"]["seed"]),
        "max_epochs": int(config["training"]["max_epochs"]),
        "gradient_clip_max_norm": float(config["training"]["gradient_clip_max_norm"]),
        "loss_name": str(config["training"]["loss_name"]),
        "model_architecture": _build_backbone_config(config),
        "_kept_metric_consistency_check": {
            "stored_matches_canonical": (
                float(metrics["rmse_wh"]) == float(stored["rmse_wh"])
                and float(metrics["mae_wh"]) == float(stored["mae_wh"])
                and float(metrics["r2"]) == float(stored["r2"])
            )
        },
    }


def _build_not_verifiable_payload(config: dict, recorded_device: str, reason: str) -> dict:
    """Emit a structured NOT_VERIFIABLE_ON_CURRENT_DEVICE payload.

    NO recomputation is performed. NO CPU fallback is performed. The payload
    preserves all stored RN1 metrics and the canonical tolerance/device
    policy so that Phase 40 is correctly recorded as BLOCKED.
    """
    run_root = ROOT / "artifacts/runs" / RUN_ID
    metrics_path = run_root / "metrics/best_validation_metrics.json"
    metrics = read_json(metrics_path)["metric_result"]
    status_path = run_root / "status.json"
    status = read_json(status_path)

    return {
        "run_id": RUN_ID,
        "experiment_family": "S18_REVIN",
        "status": "NOT_VERIFIABLE_ON_CURRENT_DEVICE",
        "strict_load_verification": "NOT_VERIFIABLE_ON_CURRENT_DEVICE",
        "strict_best_verification": "NOT_VERIFIABLE_ON_CURRENT_DEVICE",
        "recorded_device": recorded_device,
        "verification_device": recorded_device,
        "verification_device_used_at_runtime": (
            f"{recorded_device} (intended); verifier refused to run because "
            f"torch backend reports device unavailable on this host"
        ),
        "verification_device_warning": None,
        "cpu_fallback": "DISABLED",
        "not_verifiable_reason": reason,
        "plan_authority": PLAN_AUTHORITY,
        "stored_validation_metrics": {
            "rmse_wh": float(metrics["rmse_wh"]),
            "mae_wh": float(metrics["mae_wh"]),
            "r2": float(metrics["r2"]),
            "n_samples": int(metrics["n_samples"]),
        },
        "recomputed_validation_metrics": None,
        "metric_delta": None,
        "comparison_rule": "absolute_only",
        "canonical_tolerance": CANONICAL_TOLERANCE,
        "tolerance": CANONICAL_TOLERANCE,
        "abs_tolerance": CANONICAL_TOLERANCE,
        "rel_tolerance": None,
        "per_metric_pass": None,
        "population_match": None,
        "sample_order_match": None,
        "parameter_count": None,
        "revin_channel_count": None,
        "passthrough_channel_count": None,
        "target_channel_name": None,
        "target_original_index": None,
        "target_revin_subset_index": None,
        "revin_eps": REVIN_EPS,
        "revin_affine": REVIN_AFFINE,
        "best_epoch": int(status["best_epoch"]),
        "epochs_executed": None,
        "test_access": "FORBIDDEN",
        "feature_variant_id": None,
        "feature_count": None,
        "target_scaling_option": None,
        "lookback_steps": None,
        "seed": None,
        "max_epochs": None,
        "gradient_clip_max_norm": None,
        "loss_name": None,
        "model_architecture": None,
    }


def main():
    run_root = ROOT / "artifacts/runs" / RUN_ID
    paths = {
        "config": run_root / "config.json",
        "status": run_root / "status.json",
        "checkpoint": run_root / "checkpoints/best_checkpoint.pt",
        "history": run_root / "training_history.csv",
        "metrics": run_root / "metrics/best_validation_metrics.json",
        "predictions": run_root / "predictions/best_validation_predictions.csv",
    }
    missing = [str(p.relative_to(ROOT)) for p in paths.values() if not p.is_file()]
    if missing:
        raise RuntimeError(f"Phase 40 RN1 evidence missing: {missing}")

    config_payload = read_json(paths["config"])
    config = config_payload["config"]
    recorded_device = config["runtime"]["device_type"]

    available, recorded, reason = _check_recorded_device_available(config)
    if not available:
        ext_report, ext_reason = _try_load_external_report(config)
        if ext_report is not None:
            payload = _build_external_accepted_payload(config, ext_report)
            return "PASS", payload
        payload = _build_not_verifiable_payload(config, recorded, reason)
        return "NOT_VERIFIABLE_ON_CURRENT_DEVICE", payload
    status = read_json(paths["status"])
    metrics = read_json(paths["metrics"])["metric_result"]
    history = pd.read_csv(paths["history"])
    predictions = pd.read_csv(paths["predictions"])

    fv = config["data"]["feature_variant_id"]
    bundle = load_validated_scaler_bundle(fv, ROOT)
    feat_order = list(bundle["full_feature_order"])
    ok, scope_or_none, reason = resolve_revin_scope_from_feature_order(feat_order)
    assert ok and scope_or_none is not None, f"RevIN scope resolution failed: {reason}"
    scope = scope_or_none

    backbone = TransformerRegressor(_build_backbone_config(config))
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

    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    incompatible = wrapped.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys, (
        f"incompatible keys: missing={incompatible.missing_keys}, "
        f"unexpected={incompatible.unexpected_keys}"
    )
    wrapped.eval()
    parameter_count = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)

    verification_device = torch.device(recorded_device)
    device_used = recorded_device
    wrapped = wrapped.to(verification_device)

    loaders = build_train_validation_loaders(
        project_root=ROOT,
        variant_id=config["data"]["feature_variant_id"],
        lookback=int(config["data"]["lookback_steps"]),
        target_option=config["data"]["target_scaling_option"],
        batch_size=int(config["training"]["batch_size"]),
        seed=int(config["reproducibility"]["seed"]),
        num_workers=0,
        device_type=device_used,
    )
    validation_loader = loaders["VALIDATION"][0]
    target_scaler = load_validated_target_scaler(ROOT)

    recomputed_sample_ids: list[int] = []
    recomputed_y_true: list[float] = []
    recomputed_y_pred: list[float] = []
    with torch.no_grad():
        for batch in validation_loader:
            x = batch["x"].to(verification_device)
            y_model = batch["y_model"].to(verification_device)
            prediction = wrapped(x)
            assert prediction.shape == y_model.shape, (
                f"prediction shape {prediction.shape} != target shape {y_model.shape}"
            )
            prediction_wh = inverse_transform_target(
                prediction.cpu().numpy().reshape(-1, 1), "YS1", target_scaler
            ).reshape(-1)
            recomputed_sample_ids.extend(batch["sample_idx"].cpu().numpy().astype(int).tolist())
            recomputed_y_true.extend(batch["y_raw_wh"].cpu().numpy().reshape(-1).astype(float).tolist())
            recomputed_y_pred.extend(prediction_wh.astype(float).tolist())

    recomputed = compute_regression_metrics(
        np.asarray(recomputed_y_true, dtype=np.float64),
        np.asarray(recomputed_y_pred, dtype=np.float64),
        np.asarray(recomputed_sample_ids, dtype=np.int64),
        "VALIDATION",
        EvaluationMode.VALIDATION.value,
        config["lineage"]["population_fingerprint"],
        RUN_ID,
        config["model"].get("model_name", config["model"]["model_family"]),
        lookback_steps=int(config["data"]["lookback_steps"]),
        horizon_steps=int(config["data"]["horizon_steps"]),
        target_scaling_option=config["data"]["target_scaling_option"],
        project_root=ROOT,
    )

    stored_rmse = float(metrics["rmse_wh"])
    stored_mae = float(metrics["mae_wh"])
    stored_r2 = float(metrics["r2"])
    stored_n = int(metrics["n_samples"])
    stored_pop = metrics["population_fingerprint"]

    rmse_delta = abs(recomputed.rmse_wh - stored_rmse)
    mae_delta = abs(recomputed.mae_wh - stored_mae)
    r2_delta = abs(recomputed.r2 - stored_r2)

    rmse_ok = rmse_delta <= CANONICAL_TOLERANCE
    mae_ok = mae_delta <= CANONICAL_TOLERANCE
    r2_ok = r2_delta <= CANONICAL_TOLERANCE
    n_ok = int(recomputed.n_samples) == stored_n
    pop_ok = recomputed.population_fingerprint == stored_pop
    sample_ok = np.array_equal(
        np.asarray(recomputed_sample_ids, dtype=np.int64),
        predictions["sample_idx"].astype(int).to_numpy(),
    )

    overall = "PASS" if (rmse_ok and mae_ok and r2_ok and n_ok and pop_ok and sample_ok) else "FAIL"

    out = {
        "run_id": RUN_ID,
        "experiment_family": "S18_REVIN",
        "status": overall,
        "strict_load_verification": (
            "PASS" if (not incompatible.missing_keys and not incompatible.unexpected_keys) else "FAIL"
        ),
        "strict_best_verification": overall,
        "recorded_device": recorded_device,
        "verification_device": device_used,
        "verification_device_warning": None,
        "cpu_fallback": "DISABLED",
        "plan_authority": PLAN_AUTHORITY,
        "stored_validation_metrics": {
            "rmse_wh": stored_rmse,
            "mae_wh": stored_mae,
            "r2": stored_r2,
            "n_samples": stored_n,
        },
        "recomputed_validation_metrics": {
            "rmse_wh": float(recomputed.rmse_wh),
            "mae_wh": float(recomputed.mae_wh),
            "r2": float(recomputed.r2),
            "n_samples": int(recomputed.n_samples),
        },
        "metric_delta": {
            "rmse_wh": rmse_delta,
            "mae_wh": mae_delta,
            "r2": r2_delta,
        },
        "comparison_rule": "absolute_only",
        "canonical_tolerance": CANONICAL_TOLERANCE,
        "tolerance": CANONICAL_TOLERANCE,
        "abs_tolerance": CANONICAL_TOLERANCE,
        "rel_tolerance": None,
        "per_metric_pass": {
            "rmse": rmse_ok,
            "mae": mae_ok,
            "r2": r2_ok,
        },
        "population_match": pop_ok,
        "sample_order_match": sample_ok,
        "parameter_count": parameter_count,
        "revin_channel_count": scope.revin_channel_count,
        "passthrough_channel_count": len(scope.passthrough_channel_indices),
        "target_channel_name": scope.target_channel_name,
        "target_original_index": scope.target_original_index,
        "target_revin_subset_index": scope.target_revin_subset_index,
        "revin_eps": REVIN_EPS,
        "revin_affine": REVIN_AFFINE,
        "best_epoch": int(status["best_epoch"]),
        "epochs_executed": int(history["epoch"].max()),
        "test_access": "FORBIDDEN",
        "feature_variant_id": fv,
        "feature_count": len(feat_order),
        "target_scaling_option": config["data"]["target_scaling_option"],
        "lookback_steps": int(config["data"]["lookback_steps"]),
        "seed": int(config["reproducibility"]["seed"]),
        "max_epochs": int(config["training"]["max_epochs"]),
        "gradient_clip_max_norm": float(config["training"]["gradient_clip_max_norm"]),
        "loss_name": str(config["training"]["loss_name"]),
        "model_architecture": _build_backbone_config(config),
    }
    return overall, out


if __name__ == "__main__":
    overall, payload = main()
    print(json.dumps({"status": overall, **payload}, indent=2))
