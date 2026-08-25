"""
Phase 38 — S16 Epoch-cap Sweep Implementation

Sweep Factor: training.max_epochs
Conditions: E50 (max_epochs=50), E100 (max_epochs=100)

Reference: Phase 37 (S15 Loss sweep)
Factor from S15: Loss = MSE

This sweep tests whether extending the maximum training epoch cap from 50 to 100
improves Validation RMSE under the frozen configuration with patience=10 early stopping.

Key semantic distinction:
- max_epochs is a CEILING, not a guaranteed budget
- Early stopping may terminate training before the cap
- E100 does NOT need to execute 100 epochs if early stopping triggers earlier
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from course_work.data.scaling import inverse_transform_target, load_validated_target_scaler
from course_work.experiments.registry import compute_config_fingerprint
from course_work.models.transformer_regressor import TransformerRegressor, validate_transformer_config
from course_work.sweeps.sweep_results import validate_sweep_signoff
from course_work.training.losses import (
    HUBER_DELTA_MODEL_SPACE,
    build_training_criterion,
    criterion_config,
    huber_regime_diagnostics,
    validate_criterion_inputs,
)
from course_work.utils.artifacts import (
    canonical_json_bytes,
    get_project_root,
    read_json,
    sha256_bytes,
    sha256_file,
)
from course_work.utils.reproducibility import set_seed


# Phase 38 Constants
PHASE_ID = 38
PHASE_VERSION = "PHASE-38-v1"
SWEEP_ID = "S16_EPOCH_CAP"
SWEEP_VERSION = "SWEEP_S16_EPOCHCAP-v1"
REFERENCE_CONDITION_ID = "E50"
SEED = 42
TEST_ACCESS = "FORBIDDEN"

# Upstream Phase 37 paths
PHASE_37_ROOT = Path("artifacts/sweeps/S15_loss")
PHASE_37_SIGNOFF_PATH = PHASE_37_ROOT / "phase_37_signoff.json"
PHASE_37_WINNER_PATH = PHASE_37_ROOT / "s15_loss_winner.json"
PHASE_37_REFERENCE_PATH = PHASE_37_ROOT / "s15_reference_update.json"

# Processing log
PROCESSING_LOG_PATH = Path("docs/save_log_in_processing/phase_38_s16_epoch_cap_log.json")

# Inherited warning from upstream phases
INHERITED_WARNING = "H4_SOURCE_ARTIFACT_RETENTION_INCOMPLETE"


@dataclass(frozen=True)
class EpochCapCondition:
    """Epoch cap condition specification."""
    condition_id: str
    max_epochs: int
    execution_mode: str


# Phase 38 Conditions
# E50: max_epochs=50, REUSE_REFERENCE (from Phase 37 winner)
# E100: max_epochs=100, TRAIN_NEW
CONDITIONS = (
    EpochCapCondition("E50", 50, "REUSE_REFERENCE"),
    EpochCapCondition("E100", 100, "TRAIN_NEW"),
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


def get_phase_37_handoff() -> dict[str, Any]:
    """
    Load and verify Phase 37 (S15 Loss sweep) handoff artifacts.
    
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
    
    signoff, signoff_error = _load_object(root, PHASE_37_SIGNOFF_PATH)
    if signoff_error:
        _add_issue(issues, PHASE_37_SIGNOFF_PATH, signoff_error)
        return {"valid": False, "signoff": None, "winner": None, "reference": None, "issues": issues}
    
    winner, winner_error = _load_object(root, PHASE_37_WINNER_PATH)
    if winner_error:
        _add_issue(issues, PHASE_37_WINNER_PATH, winner_error)
        return {"valid": False, "signoff": signoff, "winner": None, "reference": None, "issues": issues}
    
    reference, reference_error = _load_object(root, PHASE_37_REFERENCE_PATH)
    if reference_error:
        _add_issue(issues, PHASE_37_REFERENCE_PATH, reference_error)
        return {"valid": False, "signoff": signoff, "winner": winner, "reference": None, "issues": issues}
    
    # Verify Phase 37 status
    if signoff.get("overall_status") not in {"PASS", "PASS_WITH_WARNING"}:
        _add_issue(issues, PHASE_37_SIGNOFF_PATH, f"Phase37 status is {signoff.get('overall_status')}")
    
    # Verify approved_for_phase38 flag
    if not signoff.get("approved_for_phase38"):
        _add_issue(issues, PHASE_37_SIGNOFF_PATH, "Phase37 did not approve Phase38")
    
    # Verify Phase 37 winner is MSE
    if winner.get("winner_loss_name") != "MSE":
        _add_issue(issues, PHASE_37_WINNER_PATH, f"Expected MSE winner, got {winner.get('winner_loss_name')}")
    
    # Verify Phase 37 winner run
    expected_ref_run = winner.get("winner_run_id")
    if expected_ref_run != "RUN_TR_S14_0023_A711A9B8":
        _add_issue(issues, PHASE_37_WINNER_PATH, f"Expected reference run RUN_TR_S14_0023_A711A9B8, got {expected_ref_run}")
    
    return {
        "valid": len(issues) == 0,
        "signoff": signoff,
        "winner": winner,
        "reference": reference,
        "issues": issues,
    }


def verify_e50_reference(root: Path) -> dict[str, Any]:
    """
    Verify the E50 reference run (RUN_TR_S14_0023_A711A9B8).
    
    This is a critical verification since E50 is REUSE_REFERENCE.
    """
    issues: list[dict[str, str]] = []
    run_id = "RUN_TR_S14_0023_A711A9B8"
    
    # Load run config
    config_path = root / "artifacts/runs" / run_id / "config.json"
    config, config_error = _load_object(root, config_path)
    if config_error:
        _add_issue(issues, config_path, config_error)
        return {"valid": False, "run_id": run_id, "config": None, "issues": issues}
    
    # Verify run status
    status_path = root / "artifacts/runs" / run_id / "status.json"
    status, status_error = _load_object(root, status_path)
    if status_error:
        _add_issue(issues, status_path, status_error)
    elif status.get("status") != "COMPLETED":
        _add_issue(issues, status_path, f"Run status is {status.get('status')}")
    
    # Verify best epoch and metrics
    metrics_path = root / "artifacts/runs" / run_id / "metrics/best_validation_metrics.json"
    metrics, metrics_error = _load_object(root, metrics_path)
    if metrics_error:
        _add_issue(issues, metrics_path, metrics_error)
    
    # Verify BEST checkpoint exists
    checkpoint_path = root / "artifacts/runs" / run_id / "checkpoints/best_checkpoint.pt"
    if not checkpoint_path.is_file():
        _add_issue(issues, checkpoint_path, "BEST checkpoint missing")
    
    # Verify training history
    history_path = root / "artifacts/runs" / run_id / "training_history.csv"
    if not history_path.is_file():
        _add_issue(issues, history_path, "Training history missing")
    
    # Extract configuration details
    cfg = config.get("config", {})
    training = cfg.get("training", {})
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    lineage = cfg.get("lineage", {})
    
    # Verify frozen configuration from Phase 37
    verification = {
        "run_id": run_id,
        "config": config,
        "status": status,
        "metrics": metrics,
        "issues": issues,
        "config_summary": {
            "max_epochs": training.get("max_epochs"),
            "patience": training.get("early_stopping_patience"),
            "min_delta": 0,  # hardcoded in training engine
            "loss": training.get("loss_name"),
            "learning_rate": training.get("learning_rate"),
            "weight_decay": training.get("weight_decay"),
            "gradient_clip": training.get("gradient_clip_max_norm"),
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
    
    # Check expected values
    if training.get("max_epochs") != 50:
        _add_issue(issues, config_path, f"Expected max_epochs=50, got {training.get('max_epochs')}")
    
    if training.get("early_stopping_patience") != 10:
        _add_issue(issues, config_path, f"Expected patience=10, got {training.get('early_stopping_patience')}")
    
    if training.get("loss_name") != "MSE":
        _add_issue(issues, config_path, f"Expected MSE loss, got {training.get('loss_name')}")
    
    # Verify Validation RMSE matches expected
    if metrics:
        actual_rmse = metrics.get("metric_result", {}).get("rmse_wh")
        expected_rmse = 57.69679988114431
        if actual_rmse is not None and not _same_number(actual_rmse, expected_rmse):
            _add_issue(issues, metrics_path, f"Expected RMSE {expected_rmse}, got {actual_rmse}")
    
    verification["issues"] = issues
    verification["valid"] = len(issues) == 0
    
    return verification


def get_condition_spec(condition_id: str) -> EpochCapCondition | None:
    """Get the specification for a condition ID."""
    for cond in CONDITIONS:
        if cond.condition_id == condition_id:
            return cond
    return None


def get_e100_config_overrides(root: Path) -> dict[str, Any]:
    """
    Generate configuration overrides for E100 training.
    
    E100 differs from E50 only in max_epochs.
    All other settings must be identical to the frozen configuration.
    """
    # Load E50 reference config
    e50_config_path = root / "artifacts/runs/RUN_TR_S14_0023_A711A9B8/config.json"
    e50_config, error = _load_object(root, e50_config_path)
    
    if error or e50_config is None:
        raise ValueError(f"Cannot load E50 reference config: {error}")
    
    # Generate E100 overrides
    # Only max_epochs changes from 50 to 100
    e100_overrides = {
        "max_epochs": 100,
    }
    
    return e100_overrides


def verify_phase_38_preflight(root: Path) -> dict[str, Any]:
    """
    Run the complete Phase 38 pre-flight verification.
    
    This must PASS before any E100 training is authorized.
    """
    issues: list[dict[str, str]] = []
    warnings: list[str] = []
    checks: dict[str, dict[str, Any]] = {}
    
    # 1. Verify Phase 37 handoff
    handoff = get_phase_37_handoff()
    checks["phase_37_handoff"] = {
        "valid": handoff["valid"],
        "status": "PASS" if handoff["valid"] else "FAIL",
    }
    if not handoff["valid"]:
        issues.extend(handoff["issues"])
    
    # 2. Verify E50 reference run
    e50_ref = verify_e50_reference(root)
    checks["e50_reference"] = {
        "valid": e50_ref["valid"],
        "run_id": e50_ref["run_id"],
        "status": "PASS" if e50_ref["valid"] else "FAIL",
    }
    if not e50_ref["valid"]:
        issues.extend(e50_ref["issues"])
    
    # 3. Verify inherited configuration is frozen
    if e50_ref.get("config"):
        cfg = e50_ref["config"].get("config", {})
        training = cfg.get("training", {})
        model = cfg.get("model", {})
        data = cfg.get("data", {})
        lineage = cfg.get("lineage", {})
        
        # Check that configuration matches Phase 38 expectations
        expected_config = {
            "max_epochs": 50,
            "early_stopping_patience": 10,
            "loss_name": "MSE",
            "learning_rate": 0.0003,
            "weight_decay": 0.001,
            "gradient_clip_max_norm": 1.0,
            "batch_size": 32,
            "d_model": 64,
            "num_heads": 4,
            "num_layers": 2,
            "ffn_dim": 256,
            "feature_variant_id": "FS2_TF1",
            "target_scaling_option": "YS1",
            "lookback_steps": 36,
        }
        
        config_issues = []
        for key, expected in expected_config.items():
            actual = training.get(key) if key in training else model.get(key) if key in model else data.get(key)
            if not _same_number(actual, expected) and actual != expected:
                config_issues.append(f"{key}: expected {expected}, got {actual}")
        
        checks["frozen_configuration"] = {
            "valid": len(config_issues) == 0,
            "status": "PASS" if len(config_issues) == 0 else "FAIL",
            "details": config_issues if config_issues else "All frozen fields match",
        }
        for issue in config_issues:
            _add_issue(issues, "frozen_config", issue)
        
        # Check population fingerprint
        population_fp = lineage.get("population_fingerprint")
        expected_fp = "a40ded8802e90008535d268720bad9e9dcca5eee1436ddb359deea3ad39a1987"
        if population_fp != expected_fp:
            _add_issue(issues, "population", f"Population fingerprint mismatch: {population_fp} != {expected_fp}")
            checks["population"] = {"valid": False, "status": "FAIL"}
        else:
            checks["population"] = {"valid": True, "status": "PASS"}
    
    # 4. Verify Test firewall
    phase37_test_status = handoff.get("signoff", {}).get("test_status", "UNKNOWN")
    if not _test_locked(phase37_test_status):
        _add_issue(issues, "test_firewall", f"Test status is not locked: {phase37_test_status}")
        checks["test_firewall"] = {"valid": False, "status": "FAIL", "test_status": phase37_test_status}
    else:
        checks["test_firewall"] = {"valid": True, "status": "PASS", "test_status": phase37_test_status}
    
    # 5. Check inherited warnings
    phase37_warnings = handoff.get("signoff", {}).get("inherited_warnings", [])
    if INHERITED_WARNING in phase37_warnings:
        warnings.append(INHERITED_WARNING)
    
    # Collect warnings
    checks["warnings"] = {"warnings": warnings}
    
    # Determine overall preflight status
    preflight_valid = len(issues) == 0
    
    return {
        "phase_id": PHASE_ID,
        "phase_name": "S16 Epoch-cap sweep",
        "preflight_valid": preflight_valid,
        "issues": issues,
        "warnings": warnings,
        "checks": checks,
        "e50_reference": {
            "run_id": e50_ref.get("run_id"),
            "config_summary": e50_ref.get("config_summary"),
        } if e50_ref else None,
    }


def create_phase_38_manifest(root: Path) -> dict[str, Any]:
    """
    Create the Phase 38 sweep manifest.
    
    This defines the complete contract for Phase 38.
    """
    # Get Phase 37 handoff info
    handoff = get_phase_37_handoff()
    winner = handoff.get("winner", {})
    reference = handoff.get("reference", {})
    
    # Get E50 reference info
    e50_ref = verify_e50_reference(root)
    e50_config = e50_ref.get("config", {}).get("config", {}) if e50_ref.get("config") else {}
    e50_training = e50_config.get("training", {})
    e50_model = e50_config.get("model", {})
    e50_data = e50_config.get("data", {})
    e50_lineage = e50_config.get("lineage", {})
    
    manifest = {
        "sweep_id": SWEEP_ID,
        "sweep_version": SWEEP_VERSION,
        "phase_id": PHASE_ID,
        "artifact_version": f"SWEEP_S16_EPOCHCAP-MANIFEST-v1",
        "created_at": "2026-08-24T08:50:00.000000+00:00",  # Placeholder, updated at write time
        "source_phase": 37,
        "source_sweep_id": "S15_LOSS",
        "source_sweep_version": "SWEEP_S15_LOSS-v1",
        "source_s15_winner_run_id": winner.get("winner_run_id"),
        "factor": "training.max_epochs",
        "factor_description": "Maximum training epoch cap (ceiling, not budget)",
        "candidate_epoch_caps": [50, 100],
        "epoch_cap_aliases": {"E50": 50, "E100": 100},
        "selection_metric": "validation_rmse_wh",
        "selection_direction": "MIN",
        "tie_rule": "E50_ON_EXACT_RMSE_TIE",
        "selected_model_config": {
            "feature_variant_id": e50_data.get("feature_variant_id"),
            "target_scaling_id": e50_data.get("target_scaling_option"),
            "lookback_steps": e50_data.get("lookback_steps"),
            "pooling": e50_model.get("pooling"),
            "activation": e50_model.get("activation"),
            "d_model": e50_model.get("d_model"),
            "num_heads": e50_model.get("num_heads"),
            "num_layers": e50_model.get("num_layers"),
            "ffn_dim": e50_model.get("ffn_dim"),
            "dropout": e50_model.get("dropout"),
        },
        "selected_training_config": {
            "batch_size": e50_training.get("batch_size"),
            "learning_rate": e50_training.get("learning_rate"),
            "weight_decay": e50_training.get("weight_decay"),
            "gradient_clip_max_norm": e50_training.get("gradient_clip_max_norm"),
            "loss_name": e50_training.get("loss_name"),
            "scheduler": e50_training.get("scheduler_name"),
            "warmup": None,
        },
        "early_stopping_config": {
            "patience": e50_training.get("early_stopping_patience"),
            "min_delta": 0,
            "monitor": "validation_rmse_wh",
            "mode": "MIN",
        },
        "frozen_config": {
            "patience": 10,
            "min_delta": 0,
            "early_stopping_metric": "validation_rmse_wh",
            "best_metric": "validation_rmse_wh",
            "scheduler": None,
            "warmup": None,
            "gradient_clip": 1.0,
            "optimizer": "AdamW",
        },
        "population": {
            "fingerprint": e50_lineage.get("population_fingerprint"),
            "version": e50_lineage.get("population_version"),
            "validation_samples": e50_data.get("validation_sample_count"),
        },
        "metric_version": e50_lineage.get("metric_version"),
        "training_engine_version": "TRAINING_ENGINE-v1",
        "seed": SEED,
        "test_access": TEST_ACCESS,
        "inherited_warnings": [INHERITED_WARNING],
        "conditions": [
            {
                "condition_id": "E50",
                "max_epochs": 50,
                "source_type": "REUSE_REFERENCE",
                "source_run_id": "RUN_TR_S14_0023_A711A9B8",
                "requires_new_training": False,
            },
            {
                "condition_id": "E100",
                "max_epochs": 100,
                "source_type": "TRAIN_NEW",
                "requires_new_training": True,
            },
        ],
        "new_runs_required": 1,  # Only E100
        "reused_runs": 1,  # E50
        "key_semantic_distinction": "max_epochs is a CEILING, not a guaranteed budget. Early stopping may terminate training before the cap.",
        "prefix_equivalence_expected": True,
        "prefix_equivalence_note": "Under scheduler=None and full reproducibility, E50 and E100 should share identical trajectories before epoch 50",
    }
    
    return manifest


def validate_phase_38_artifacts(root: Path) -> dict[str, Any]:
    """
    Validate Phase 38 required artifacts exist and are properly structured.
    """
    issues: list[dict[str, str]] = []
    
    sweep_root = root / "artifacts/sweeps/S16_epoch_cap"
    
    # Required artifacts
    required = [
        "s16_epoch_cap_sweep_manifest.json",
        "s16_epoch_cap_sweep_contract.json",
        "s16_run_matrix.csv",
        "s16_epoch_cap_metrics.csv",
        "s16_epoch_cap_effect.csv",
        "s16_epoch_cap_winner.json",
        "s16_reference_update.json",
        "phase_38_signoff.json",
        "README_S16_EPOCH_CAP_SWEEP.md",
    ]
    
    existing = []
    missing = []
    
    for artifact in required:
        path = sweep_root / artifact
        if path.is_file():
            existing.append(artifact)
        else:
            missing.append(artifact)
            _add_issue(issues, path, "MISSING")
    
    return {
        "valid": len(issues) == 0,
        "existing": existing,
        "missing": missing,
        "issues": issues,
    }


def prepare_phase_38_condition(condition_id: str, project_root: Path | None = None) -> dict[str, Any] | None:
    """
    Prepare Phase 38 epoch-cap condition.
    
    This follows the same pattern as prepare_phase_37_condition in loss.py.
    
    Args:
        condition_id: Either "E50" or "E100"
        project_root: Optional project root path
        
    Returns:
        Condition dict if valid, None if condition not found
    """
    root = Path(project_root or get_project_root())
    
    # Verify Phase 37 handoff first
    handoff = get_phase_37_handoff()
    if not handoff["valid"]:
        raise RuntimeError(f"Phase 37 handoff invalid: {handoff['issues']}")
    
    # Verify E50 reference
    e50_ref = verify_e50_reference(root)
    if not e50_ref["valid"]:
        raise RuntimeError(f"E50 reference invalid: {e50_ref['issues']}")
    
    # Get condition spec
    condition = get_condition_spec(condition_id)
    if condition is None:
        return None
    
    # Get E50 config for frozen configuration
    e50_config = e50_ref.get("config", {}).get("config", {}) if e50_ref.get("config") else {}
    training = e50_config.get("training", {})
    model = e50_config.get("model", {})
    data = e50_config.get("data", {})
    lineage = e50_config.get("lineage", {})
    
    # Build frozen configuration
    frozen_configuration = {
        "source_config_path": f"artifacts/runs/RUN_TR_S14_0023_A711A9B8/config.json",
        "feature_variant_id": data.get("feature_variant_id"),
        "lookback_id": f"L{data.get('lookback_steps')}",
        "target_scaling_id": data.get("target_scaling_option"),
        "pooling_id": model.get("pooling"),
        "activation_id": model.get("activation"),
        "batch_id": f"B{training.get('batch_size')}",
        "learning_rate": training.get("learning_rate"),
        "weight_decay": training.get("weight_decay"),
        "dropout": model.get("dropout"),
        "d_model": model.get("d_model"),
        "num_heads": model.get("num_heads"),
        "num_layers": model.get("num_layers"),
        "ffn_dim": model.get("ffn_dim"),
        "loss": training.get("loss_name"),
    }
    
    # Build condition result
    result: dict[str, Any] = {
        "phase_id": PHASE_ID,
        "sweep_id": SWEEP_ID,
        "sweep_version": SWEEP_VERSION,
        "condition_id": condition_id,
        "max_epochs": condition.max_epochs,
        "execution_mode": condition.execution_mode,
        "frozen_configuration": frozen_configuration,
        "phase_37_reference_run_id": "RUN_TR_S14_0023_A711A9B8",
        "population_fingerprint": lineage.get("population_fingerprint"),
        "metric_version": lineage.get("metric_version"),
    }
    
    # Add reference evidence for REUSE_REFERENCE conditions
    if condition.execution_mode == "REUSE_REFERENCE":
        result["execution_mode"] = "REUSE_REFERENCE"
        result["reference_run_id"] = "RUN_TR_S14_0023_A711A9B8"
        result["reference_evidence"] = {
            "rmse_wh": 57.69679988114431,
            "mae_wh": 27.029547974302105,
            "r2": 0.6087328858180434,
            "best_epoch": 12,
        }
    
    return result


def run_phase_38_preflight_only() -> dict[str, Any]:
    """
    Run Phase 38 preflight checks only.
    
    This function does NOT authorize training.
    It only verifies readiness for potential Phase 38 execution.
    """
    root = Path(get_project_root())
    
    # Run preflight
    preflight = verify_phase_38_preflight(root)
    
    return {
        "phase_id": PHASE_ID,
        "phase_name": "S16 Epoch-cap sweep",
        "execution_authorized": False,
        "preflight": preflight,
        "message": "Preflight complete. Training NOT authorized.",
    }


if __name__ == "__main__":
    result = run_phase_38_preflight_only()
    print(json.dumps(result, indent=2, default=str))
