"""
Phase 39 — S17 Gradient-Clipping Sweep Implementation

Sweep Factor: gradient clipping
Conditions: GC0 (OFF), GC1 (global norm, max_norm=1.0)

Reference: Phase 38 (S16 Epoch-cap sweep)
Factor from S16: max_epochs=50, loss=MSE

This sweep tests whether gradient clipping (max_norm=1.0) helps under the
frozen configuration with patience=10 early stopping.

Key distinction:
- GC0: clipping OFF, finite-gradient guard ON
- GC1: global L2 norm clipping, max_norm=1.0, after backward, before optimizer.step()
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from course_work.data.scaling import load_validated_target_scaler
from course_work.experiments.registry import compute_config_fingerprint
from course_work.models.transformer_regressor import TransformerRegressor, validate_transformer_config
from course_work.sweeps.sweep_results import validate_sweep_signoff
from course_work.training.losses import build_training_criterion
from course_work.utils.artifacts import (
    canonical_json_bytes,
    get_project_root,
    read_json,
    sha256_bytes,
    sha256_file,
)
from course_work.utils.reproducibility import set_seed


PHASE_ID = 39
PHASE_VERSION = "PHASE-39-v1"
SWEEP_ID = "S17_GRADIENT_CLIPPING"
SWEEP_VERSION = "SWEEP_S17_GRADIENTCLIP-v1"
REFERENCE_CONDITION_ID = "GC1"
SEED = 42
TEST_ACCESS = "FORBIDDEN"

PHASE_38_ROOT = Path("artifacts/sweeps/S16_epoch_cap")
PHASE_38_SIGNOFF_PATH = PHASE_38_ROOT / "phase_38_signoff.json"
PHASE_38_WINNER_PATH = PHASE_38_ROOT / "s16_epoch_cap_winner.json"
PHASE_38_REFERENCE_PATH = PHASE_38_ROOT / "s16_reference_update.json"

PROCESSING_LOG_PATH = Path("docs/save_log_in_processing/phase_39_s17_gradient_clip_log.json")

INHERITED_WARNING = "H4_SOURCE_ARTIFACT_RETENTION_INCOMPLETE"


@dataclass(frozen=True)
class GradientClipCondition:
    """Gradient clipping condition specification."""
    condition_id: str
    clip_enabled: bool
    clip_method: str | None  
    max_norm: float | None
    norm_type: int | None
    execution_mode: str

CONDITIONS = (
    GradientClipCondition("GC0", False, None, None, None, "TRAIN_NEW"),
    GradientClipCondition("GC1", True, "global_norm", 1.0, 2, "REUSE_REFERENCE"),
)


def _load_object(root: Path, relative_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load a JSON file, returning (payload, None) on success or (None, error) on failure."""
    path = root / relative_path
    if not path.is_file():
        return None, "MISSING"
    try:
        payload = read_json(path)
    except (OSError, TypeError, ValueError):
        return None, "INVALID_JSON"
    return (payload, None) if isinstance(payload, dict) else (None, "INVALID_OBJECT")


def _add_issue(issues: list[dict[str, str]], path: str | Path, reason: str) -> None:
    """Add an issue to the issues list if not already present."""
    issue = {"path": str(path), "reason": reason}
    if issue not in issues:
        issues.append(issue)


def _test_locked(value: Any) -> bool:
    """Check if a value indicates Test is locked/forbidden."""
    return str(value).upper() in {"FORBIDDEN", "LOCKED", "NOT_ACCESSED", "UNTOUCHED"}


def _same_number(left: Any, right: Any) -> bool:
    """Compare two numbers with tight tolerance."""
    return (
        isinstance(left, (int, float))
        and isinstance(right, (int, float))
        and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    )


def get_phase_38_handoff() -> dict[str, Any]:
    """
    Load and verify Phase 38 (S16 Epoch-cap sweep) handoff artifacts.
    
    Returns:
        dict with keys:
        - valid: bool
        - signoff: dict or None
        - winner: dict or None
        - reference: dict or None
        - issues: list of issues
    """
    root = Path(get_project_root())
    issues: list[dict[str, str]] = []
    
    signoff, signoff_error = _load_object(root, PHASE_38_SIGNOFF_PATH)
    if signoff_error:
        _add_issue(issues, PHASE_38_SIGNOFF_PATH, signoff_error)
        return {"valid": False, "signoff": None, "winner": None, "reference": None, "issues": issues}
    
    winner, winner_error = _load_object(root, PHASE_38_WINNER_PATH)
    if winner_error:
        _add_issue(issues, PHASE_38_WINNER_PATH, winner_error)
        return {"valid": False, "signoff": signoff, "winner": None, "reference": None, "issues": issues}
    
    reference, reference_error = _load_object(root, PHASE_38_REFERENCE_PATH)
    if reference_error:
        _add_issue(issues, PHASE_38_REFERENCE_PATH, reference_error)
        return {"valid": False, "signoff": signoff, "winner": winner, "reference": None, "issues": issues}
    
    if signoff.get("status") not in {"PASS", "PASS_WITH_WARNING"}:
        _add_issue(issues, PHASE_38_SIGNOFF_PATH, f"Phase38 status is {signoff.get('status')}")
    
    if not signoff.get("approved_for_phase39"):
        _add_issue(issues, PHASE_38_SIGNOFF_PATH, "Phase38 did not approve Phase39")
    
    winner_run = winner.get("winner_run_id")
    if not winner_run:
        _add_issue(issues, PHASE_38_WINNER_PATH, "No winner_run_id found")
    
    return {
        "valid": len(issues) == 0,
        "signoff": signoff,
        "winner": winner,
        "reference": reference,
        "issues": issues,
    }


def verify_gc1_reference(root: Path) -> dict[str, Any]:
    """
    Verify the GC1 reference run (S16 winner).
    
    GC1 = global norm clipping with max_norm=1.0.
    The S16 winner already has clipping enabled with max_norm=1.0.
    """
    issues: list[dict[str, str]] = []
    
    handoff = get_phase_38_handoff()
    if not handoff["valid"]:
        return {"valid": False, "run_id": None, "issues": handoff["issues"]}
    
    winner = handoff["winner"]
    gc1_run_id = winner.get("winner_run_id")
    
    if not gc1_run_id:
        _add_issue(issues, PHASE_38_WINNER_PATH, "No GC1 run_id")
        return {"valid": False, "run_id": None, "issues": issues}
    
    config_path = root / "artifacts/runs" / gc1_run_id / "config.json"
    config, config_error = _load_object(root, config_path)
    if config_error:
        _add_issue(issues, config_path, config_error)
        return {"valid": False, "run_id": gc1_run_id, "config": None, "issues": issues}
    
    status_path = root / "artifacts/runs" / gc1_run_id / "status.json"
    status, status_error = _load_object(root, status_path)
    if status_error:
        _add_issue(issues, status_path, status_error)
    elif status.get("status") != "COMPLETED":
        _add_issue(issues, status_path, f"Run status is {status.get('status')}")
    
    checkpoint_path = root / "artifacts/runs" / gc1_run_id / "checkpoints/best_checkpoint.pt"
    if not checkpoint_path.is_file():
        _add_issue(issues, checkpoint_path, "BEST checkpoint missing")
    
    cfg = config.get("config", {})
    training = cfg.get("training", {})
    model = cfg.get("model", {})
    data = cfg.get("data", {})
    lineage = cfg.get("lineage", {})
    
    grad_clip = training.get("gradient_clip_max_norm")
    if grad_clip != 1.0:
        _add_issue(issues, config_path, f"GC1 expected gradient_clip_max_norm=1.0, got {grad_clip}")
    
    verification = {
        "run_id": gc1_run_id,
        "config": config,
        "status": status,
        "issues": issues,
        "config_summary": {
            "max_epochs": training.get("max_epochs"),
            "loss": training.get("loss_name"),
            "learning_rate": training.get("learning_rate"),
            "weight_decay": training.get("weight_decay"),
            "gradient_clip_max_norm": grad_clip,
            "batch_size": training.get("batch_size"),
            "d_model": model.get("d_model"),
            "num_heads": model.get("num_heads"),
            "num_layers": model.get("num_layers"),
            "ffn_dim": model.get("ffn_dim"),
            "feature_variant": data.get("feature_variant_id"),
            "target_scaling": data.get("target_scaling_option"),
            "lookback": data.get("lookback_steps"),
            "population_fingerprint": lineage.get("population_fingerprint"),
        }
    }
    
    verification["issues"] = issues
    verification["valid"] = len(issues) == 0
    
    return verification


def get_condition_spec(condition_id: str) -> GradientClipCondition | None:
    """Get the specification for a condition ID."""
    for cond in CONDITIONS:
        if cond.condition_id == condition_id:
            return cond
    return None


def get_phase_39_frozen_config(root: Path) -> dict[str, Any]:
    """
    Get the frozen configuration from Phase 38 winner.
    
    This config is shared between GC0 and GC1, except for gradient clipping.
    """
    gc1_ref = verify_gc1_reference(root)
    if not gc1_ref["valid"]:
        raise ValueError(f"GC1 reference invalid: {gc1_ref['issues']}")
    
    config = gc1_ref["config"]["config"]
    
    training = config["training"]
    model = config["model"]
    data = config["data"]
    
    return {
        "source_config_path": f"artifacts/runs/{gc1_ref['run_id']}/config.json",
        "feature_variant_id": data["feature_variant_id"],
        "lookback_id": f"L{data['lookback_steps']}",
        "target_scaling_id": data["target_scaling_option"],
        "pooling_id": model["pooling"],
        "activation_id": model["activation"],
        "batch_id": f"B{training['batch_size']}",
        "learning_rate": training["learning_rate"],
        "weight_decay": training["weight_decay"],
        "dropout": model["dropout"],
        "d_model": model["d_model"],
        "num_heads": model["num_heads"],
        "num_layers": model["num_layers"],
        "ffn_dim": model["ffn_dim"],
        "loss": training["loss_name"],
        "max_epochs": training["max_epochs"],
        "huber_delta": training.get("huber_delta"),
        "population_fingerprint": config["lineage"]["population_fingerprint"],
        "metric_version": config["lineage"]["metric_version"],
    }


def verify_phase_39_preflight(root: Path) -> dict[str, Any]:
    """
    Run the complete Phase 39 pre-flight verification.
    
    This must PASS before any GC0 training is authorized.
    """
    issues: list[dict[str, str]] = []
    warnings: list[str] = []
    checks: dict[str, dict[str, Any]] = {}
    
    handoff = get_phase_38_handoff()
    checks["phase_38_handoff"] = {
        "valid": handoff["valid"],
        "status": "PASS" if handoff["valid"] else "FAIL",
    }
    if not handoff["valid"]:
        issues.extend(handoff["issues"])
    
    gc1_ref = verify_gc1_reference(root)
    checks["gc1_reference"] = {
        "valid": gc1_ref["valid"],
        "run_id": gc1_ref["run_id"],
        "status": "PASS" if gc1_ref["valid"] else "FAIL",
    }
    if not gc1_ref["valid"]:
        issues.extend(gc1_ref["issues"])
    
    if gc1_ref.get("config"):
        cfg = gc1_ref["config"].get("config", {})
        training = cfg.get("training", {})
        grad_clip = training.get("gradient_clip_max_norm")
        checks["gc1_clipping"] = {
            "valid": grad_clip == 1.0,
            "gradient_clip_max_norm": grad_clip,
            "status": "PASS" if grad_clip == 1.0 else "FAIL",
        }
    
    phase38_test_status = handoff.get("signoff", {}).get("test_status", "UNKNOWN")
    if not _test_locked(phase38_test_status):
        _add_issue(issues, "test_firewall", f"Test status is not locked: {phase38_test_status}")
        checks["test_firewall"] = {"valid": False, "status": "FAIL", "test_status": phase38_test_status}
    else:
        checks["test_firewall"] = {"valid": True, "status": "PASS", "test_status": phase38_test_status}
    
    phase38_warnings = handoff.get("signoff", {}).get("inherited_warnings", [])
    if INHERITED_WARNING in phase38_warnings:
        warnings.append(INHERITED_WARNING)
    
    checks["warnings"] = {"warnings": warnings}
    
    preflight_valid = len(issues) == 0
    
    return {
        "phase_id": PHASE_ID,
        "phase_name": "S17 Gradient-clipping sweep",
        "preflight_valid": preflight_valid,
        "issues": issues,
        "warnings": warnings,
        "checks": checks,
        "gc1_reference": {
            "run_id": gc1_ref.get("run_id"),
            "config_summary": gc1_ref.get("config_summary"),
        } if gc1_ref else None,
    }


def create_phase_39_manifest(root: Path) -> dict[str, Any]:
    """
    Create the Phase 39 sweep manifest.
    
    This defines the complete contract for Phase 39.
    """
    handoff = get_phase_38_handoff()
    winner = handoff.get("winner", {})
    gc1_ref = verify_gc1_reference(root)
    frozen_config = get_phase_39_frozen_config(root) if gc1_ref["valid"] else {}
    
    manifest = {
        "sweep_id": SWEEP_ID,
        "sweep_version": SWEEP_VERSION,
        "phase_id": PHASE_ID,
        "artifact_version": f"SWEEP_S17_GRADIENTCLIP-MANIFEST-v1",
        "created_at": "2026-08-24T12:00:00.000000+00:00",
        "source_phase": 38,
        "source_sweep_id": "S16_EPOCH_CAP",
        "source_sweep_version": "SWEEP_S16_EPOCHCAP-v1",
        "source_s16_winner_run_id": winner.get("winner_run_id"),
        "factor": "gradient_clipping",
        "factor_description": "Global L2 norm clipping before optimizer.step()",
        "candidate_clip_configs": ["GC0", "GC1"],
        "clip_config_aliases": {
            "GC0": {"clip_enabled": False, "method": None, "max_norm": None},
            "GC1": {"clip_enabled": True, "method": "global_norm", "max_norm": 1.0, "norm_type": 2},
        },
        "selection_metric": "validation_rmse_wh",
        "selection_direction": "MIN",
        "tie_rule": "GC0_ON_EXACT_RMSE_TIE",
        "selected_model_config": {
            "feature_variant_id": frozen_config.get("feature_variant_id"),
            "target_scaling_id": frozen_config.get("target_scaling_option"),
            "lookback_steps": frozen_config.get("lookback_steps"),
            "pooling": frozen_config.get("pooling"),
            "activation": frozen_config.get("activation"),
            "d_model": frozen_config.get("d_model"),
            "num_heads": frozen_config.get("num_heads"),
            "num_layers": frozen_config.get("num_layers"),
            "ffn_dim": frozen_config.get("ffn_dim"),
            "dropout": frozen_config.get("dropout"),
        },
        "selected_training_config": {
            "batch_size": frozen_config.get("batch_size"),
            "learning_rate": frozen_config.get("learning_rate"),
            "weight_decay": frozen_config.get("weight_decay"),
            "loss_name": frozen_config.get("loss_name"),
            "max_epochs": frozen_config.get("max_epochs"),
        },
        "gradient_clipping_config": {
            "gc0_clip_enabled": False,
            "gc0_method": None,
            "gc0_max_norm": None,
            "gc0_finite_guard": True,
            "gc1_clip_enabled": True,
            "gc1_method": "global_norm",
            "gc1_max_norm": 1.0,
            "gc1_norm_type": 2,
            "gc1_finite_guard": True,
        },
        "early_stopping_config": {
            "patience": 10,
            "min_delta": 0,
            "monitor": "validation_rmse_wh",
            "mode": "MIN",
        },
        "population": {
            "fingerprint": frozen_config.get("population_fingerprint"),
        },
        "metric_version": frozen_config.get("metric_version"),
        "seed": SEED,
        "test_access": TEST_ACCESS,
        "inherited_warnings": [INHERITED_WARNING],
        "conditions": [
            {
                "condition_id": "GC0",
                "clip_enabled": False,
                "method": None,
                "max_norm": None,
                "source_type": "TRAIN_NEW",
                "requires_new_training": True,
            },
            {
                "condition_id": "GC1",
                "clip_enabled": True,
                "method": "global_norm",
                "max_norm": 1.0,
                "source_type": "REUSE_REFERENCE",
                "source_run_id": winner.get("winner_run_id"),
                "requires_new_training": False,
            },
        ],
        "new_runs_required": 1,  
        "reused_runs": 1, 
        "key_semantic_distinction": "GC0 has no gradient rescaling; GC1 rescales gradients when global L2 norm > 1.0",
        "gc1_telemetry_available": True,
        "gc1_clipping_fraction": 0.929375531011045,  
    }
    
    return manifest


def prepare_phase_39_condition(condition_id: str, project_root: Path | None = None) -> dict[str, Any] | None:
    """
    Prepare Phase 39 gradient-clipping condition.
    
    Args:
        condition_id: Either "GC0" or "GC1"
        project_root: Optional project root path
        
    Returns:
        Condition dict if valid, None if condition not found
    """
    root = Path(project_root or get_project_root())
    
    condition = get_condition_spec(condition_id)
    if condition is None:
        return None
    
    gc1_ref = verify_gc1_reference(root)
    if not gc1_ref["valid"]:
        raise RuntimeError(f"GC1 reference invalid: {gc1_ref['issues']}")
    
    frozen_config = get_phase_39_frozen_config(root)
    
    result: dict[str, Any] = {
        "phase_id": PHASE_ID,
        "sweep_id": SWEEP_ID,
        "sweep_version": SWEEP_VERSION,
        "condition_id": condition_id,
        "clip_enabled": condition.clip_enabled,
        "clip_method": condition.clip_method,
        "max_norm": condition.max_norm,
        "norm_type": condition.norm_type,
        "execution_mode": condition.execution_mode,
        "frozen_configuration": frozen_config,
        "phase_38_reference_run_id": gc1_ref["run_id"],
        "feature_variant_id": frozen_config["feature_variant_id"],
        "lookback_steps": int(frozen_config["lookback_id"].removeprefix("L")),
        "target_scaling_option": frozen_config["target_scaling_id"],
        "pooling": frozen_config["pooling_id"],
        "activation": frozen_config["activation_id"],
        "batch_size": int(frozen_config["batch_id"].removeprefix("B")),
        "learning_rate": frozen_config["learning_rate"],
        "weight_decay": frozen_config["weight_decay"],
        "dropout": frozen_config["dropout"],
        "d_model": frozen_config["d_model"],
        "num_heads": frozen_config["num_heads"],
        "num_layers": frozen_config["num_layers"],
        "ffn_dim": frozen_config["ffn_dim"],
        "loss_name": frozen_config["loss"],
        "registry_loss_name": frozen_config["loss"],  
        "huber_delta": frozen_config.get("huber_delta"),
        "max_epochs": frozen_config["max_epochs"],
    }
    
    if condition.execution_mode == "REUSE_REFERENCE":
        result["execution_mode"] = "REUSE_REFERENCE"
        result["reference_run_id"] = gc1_ref["run_id"]
        
        gc1_metrics_path = root / f"artifacts/runs/{gc1_ref['run_id']}/metrics/best_validation_metrics.json"
        if gc1_metrics_path.exists():
            gc1_metrics = json.loads(gc1_metrics_path.read_text())
            result["reference_evidence"] = {
                "rmse_wh": gc1_metrics["metric_result"]["rmse_wh"],
                "mae_wh": gc1_metrics["metric_result"]["mae_wh"],
                "r2": gc1_metrics["metric_result"]["r2"],
            }
    
    return result


def run_phase_39_preflight_only() -> dict[str, Any]:
    """
    Run Phase 39 preflight checks only.
    
    This function does NOT authorize training.
    It only verifies readiness for potential Phase 39 execution.
    """
    root = Path(get_project_root())
    
    preflight = verify_phase_39_preflight(root)
    
    return {
        "phase_id": PHASE_ID,
        "phase_name": "S17 Gradient-clipping sweep",
        "execution_authorized": False,
        "preflight": preflight,
        "message": "Preflight complete. Training NOT authorized.",
    }


if __name__ == "__main__":
    result = run_phase_39_preflight_only()
    print(json.dumps(result, indent=2, default=str))
