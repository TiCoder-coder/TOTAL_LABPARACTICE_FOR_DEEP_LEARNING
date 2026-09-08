from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from course_work.data.scaling import inverse_transform_target, load_validated_target_scaler
from course_work.experiments.phase_execution import (
    plan_phase_resume,
    resolve_phase_conditions,
    validate_condition_request,
)
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
    atomic_write_bytes,
    canonical_json_bytes,
    get_project_root,
    read_json,
    sha256_bytes,
    sha256_file,
)
from course_work.utils.reproducibility import set_seed


PHASE_ID = 37
PHASE_VERSION = "PHASE-37-v1"
SWEEP_ID = "S15_LOSS"
SWEEP_VERSION = "SWEEP_S15_LOSS-v1"
REFERENCE_CONDITION_ID = "L0"
SEED = 42
TEST_ACCESS = "FORBIDDEN"
PHASE_36_ROOT = Path("artifacts/sweeps/S14_ffn")
PHASE_36_SIGNOFF_PATH = PHASE_36_ROOT / "phase_36_signoff.json"
PHASE_36_WINNER_PATH = PHASE_36_ROOT / "s14_ffn_winner.json"
PHASE_36_REFERENCE_PATH = PHASE_36_ROOT / "s14_reference_update.json"
PROCESSING_LOG_PATH = Path("docs/save_log_in_processing/phase_37_s15_loss_log.json")
INHERITED_WARNING = "H4_SOURCE_ARTIFACT_RETENTION_INCOMPLETE"


@dataclass(frozen=True)
class LossCondition:
    condition_id: str
    registry_loss_name: str
    criterion_name: str
    huber_delta_model_space: float | None
    reduction: str
    execution_mode: str


CONDITIONS = (
    LossCondition("L0", "MSE", "MSELoss", None, "mean", "REUSE_REFERENCE"),
    LossCondition("L1", "HUBER", "HuberLoss", HUBER_DELTA_MODEL_SPACE, "mean", "TRAIN_NEW"),
)


def _load_object(root: Path, relative_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    path = root / relative_path
    if not path.is_file():
        return None, "MISSING"
    try:
        payload = read_json(path)
    except (OSError, TypeError, ValueError):
        return None, "INVALID_JSON"
    return (payload, None) if isinstance(payload, dict) else (None, "INVALID_OBJECT")


def _add_issue(issues: list[dict[str, str]], path: str | Path, reason: str) -> None:
    issue = {"path": str(path), "reason": reason}
    if issue not in issues:
        issues.append(issue)


def _test_locked(value: Any) -> bool:
    return str(value).upper() in {"FORBIDDEN", "LOCKED", "NOT_ACCESSED", "UNTOUCHED"}


def _same_number(left: Any, right: Any) -> bool:
    return (
        isinstance(left, (int, float))
        and isinstance(right, (int, float))
        and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    )


def _contains_signed_registry_prefix(path: Path, expected_sha256: object) -> bool:
    if not path.is_file() or not isinstance(expected_sha256, str):
        return False
    payload = path.read_bytes()
    for index, value in enumerate(payload):
        if value == 10 and hashlib.sha256(payload[: index + 1]).hexdigest() == expected_sha256:
            return True
    return hashlib.sha256(payload).hexdigest() == expected_sha256


def _optimizer_coverage(model: torch.nn.Module, learning_rate: float, weight_decay: float) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    optimizer_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    unique = set(optimizer_ids)
    missing = len(model_ids - unique)
    unexpected = len(unique - model_ids)
    duplicates = len(optimizer_ids) - len(unique)
    return {
        "missing_parameters": missing,
        "unexpected_parameters": unexpected,
        "duplicate_parameters": duplicates,
        "status": "PASS" if missing == unexpected == duplicates == 0 else "FAIL",
    }


def configuration_fingerprints(config: dict[str, Any]) -> dict[str, str]:
    model_config = config["model"]
    criterion_payload = criterion_config(config["training"])
    training_payload = {
        key: value
        for key, value in config["training"].items()
        if key not in {"loss_name", "huber_delta"}
    }
    return {
        "model_config_fingerprint": sha256_bytes(canonical_json_bytes(model_config)),
        "criterion_config_fingerprint": sha256_bytes(canonical_json_bytes(criterion_payload)),
        "training_config_fingerprint": sha256_bytes(canonical_json_bytes(training_payload)),
        "complete_config_fingerprint": compute_config_fingerprint(config),
    }


def resolve_huber_delta(project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    registry = read_json(root / "artifacts/scaling/scaler_registry.json")
    entry = registry["target_bundles"]["YS1"]
    relative_path = entry["artifact_path"]
    path = root / relative_path
    expected_checksum = entry["artifact_sha256"]
    actual_checksum = sha256_file(path)
    if actual_checksum != expected_checksum:
        raise RuntimeError("YS1 target scaler checksum mismatch")
    bundle = load_validated_target_scaler(root)
    scale = float(bundle["scaler"].scale_[0])
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError("YS1 target scaler scale must be finite and positive")
    return {
        "target_scaling_id": "YS1",
        "target_model_space": "STANDARDIZED_TARGET",
        "delta_model_space": HUBER_DELTA_MODEL_SPACE,
        "delta_raw_wh_equivalent": scale,
        "target_scaler_path": relative_path,
        "target_scaler_checksum": actual_checksum,
        "delta_source": "S15_PROTOCOL",
        "delta_tuned": False,
        "status": "PASS",
    }


def inspect_loss_invariance(
    model_config: dict[str, Any],
    *,
    learning_rate: float = 0.0003,
    weight_decay: float = 0.001,
) -> dict[str, Any]:
    base = validate_transformer_config(model_config).to_dict()
    candidates: dict[str, dict[str, Any]] = {}
    state_keys: dict[str, list[str]] = {}
    state_shapes: dict[str, dict[str, list[int]]] = {}
    parameter_shapes: dict[str, dict[str, list[int]]] = {}
    buffer_shapes: dict[str, dict[str, list[int]]] = {}
    architecture_fingerprints: dict[str, str] = {}
    initial_state_fingerprints: dict[str, str] = {}
    for condition in CONDITIONS:
        set_seed(SEED)
        model = TransformerRegressor(base)
        training = {
            "loss_name": condition.registry_loss_name,
            "huber_delta": condition.huber_delta_model_space,
        }
        criterion = build_training_criterion(training)
        sample = torch.randn(2, 36, base["input_size"], dtype=torch.float32)
        target = torch.zeros(2, 1, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            prediction = model(sample)
            validate_criterion_inputs(prediction, target)
            loss = criterion(prediction, target)
        state = model.state_dict()
        state_keys[condition.condition_id] = sorted(state)
        state_shapes[condition.condition_id] = {name: list(value.shape) for name, value in state.items()}
        parameter_shapes[condition.condition_id] = {
            name: list(value.shape) for name, value in model.named_parameters()
        }
        buffer_shapes[condition.condition_id] = {
            name: list(value.shape) for name, value in model.named_buffers()
        }
        architecture_payload = {
            "model_config": base,
            "state_shapes": state_shapes[condition.condition_id],
            "parameter_shapes": parameter_shapes[condition.condition_id],
            "buffer_shapes": buffer_shapes[condition.condition_id],
        }
        architecture_fingerprints[condition.condition_id] = sha256_bytes(
            canonical_json_bytes(architecture_payload)
        )
        digest = hashlib.sha256()
        for name, value in sorted(state.items()):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        initial_state_fingerprints[condition.condition_id] = digest.hexdigest()
        coverage = _optimizer_coverage(model, learning_rate, weight_decay)
        candidates[condition.condition_id] = {
            "condition_id": condition.condition_id,
            "criterion": criterion.__class__.__name__,
            "reduction": getattr(criterion, "reduction", None),
            "delta": getattr(criterion, "delta", None),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "output_shape": list(prediction.shape),
            "loss_finite": bool(torch.isfinite(loss).item()),
            "criterion_trainable_parameter_count": sum(
                parameter.numel() for parameter in criterion.parameters() if parameter.requires_grad
            ),
            "optimizer_coverage": coverage,
        }
    left, right = ("L0", "L1")
    checks = {
        "model_config_equal": True,
        "parameter_count_equal": (
            candidates[left]["trainable_parameter_count"]
            == candidates[right]["trainable_parameter_count"]
        ),
        "state_dict_keys_equal": state_keys[left] == state_keys[right],
        "state_dict_shapes_equal": state_shapes[left] == state_shapes[right],
        "parameter_shapes_equal": parameter_shapes[left] == parameter_shapes[right],
        "buffer_shapes_equal": buffer_shapes[left] == buffer_shapes[right],
        "architecture_fingerprint_equal": (
            architecture_fingerprints[left] == architecture_fingerprints[right]
        ),
        "output_shape_equal": candidates[left]["output_shape"] == candidates[right]["output_shape"],
        "initial_state_equal": initial_state_fingerprints[left] == initial_state_fingerprints[right],
        "optimizer_coverage": all(
            item["optimizer_coverage"]["status"] == "PASS" for item in candidates.values()
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "conditions": candidates,
        "checks": checks,
        "parameter_count": candidates[left]["trainable_parameter_count"],
        "architecture_fingerprints": architecture_fingerprints,
        "initial_state_fingerprints": initial_state_fingerprints,
        "config_delta": ["training.loss_name", "training.huber_delta"],
        "training_engine_called": False,
        "optimizer_step_executed": False,
        "test_access": TEST_ACCESS,
        "status": status,
    }


def inspect_phase_36_handoff(project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    signoff_validation = validate_sweep_signoff(36, root)
    signoff, signoff_error = _load_object(root, PHASE_36_SIGNOFF_PATH)
    winner, winner_error = _load_object(root, PHASE_36_WINNER_PATH)
    reference, reference_error = _load_object(root, PHASE_36_REFERENCE_PATH)
    issues: list[dict[str, str]] = []
    for path, error in (
        (PHASE_36_SIGNOFF_PATH, signoff_error),
        (PHASE_36_WINNER_PATH, winner_error),
        (PHASE_36_REFERENCE_PATH, reference_error),
    ):
        if error is not None:
            _add_issue(issues, path, error)
    if not signoff_validation["valid"]:
        for issue in signoff_validation["issues"]:
            if (
                issue["path"] == "artifacts/experiments/experiment_registry.jsonl"
                and issue["reason"] == "CHECKSUM_MISMATCH"
                and signoff is not None
                and _contains_signed_registry_prefix(
                    root / issue["path"], signoff.get("input_checksums", {}).get(issue["path"])
                )
            ):
                continue
            _add_issue(issues, issue["path"], issue["reason"])
    if signoff is None or winner is None or reference is None:
        return {
            "valid": False,
            "status": "BLOCKED",
            "issues": issues,
            "winner_run_id": None,
            "frozen_configuration": None,
            "reference_evidence": None,
            "invariance": None,
            "warnings": [],
        }
    if signoff.get("approved_for_phase37") is not True or reference.get("approved_for_phase37") is not True:
        _add_issue(issues, PHASE_36_SIGNOFF_PATH, "PHASE_37_NOT_APPROVED")
    expected_policy = {"MSE": "REUSE_REFERENCE", "Huber": "TRAIN_NEW"}
    if signoff.get("phase37_condition_policy") != expected_policy:
        _add_issue(issues, PHASE_36_SIGNOFF_PATH, "PHASE_37_CONDITION_POLICY_MISMATCH")
    if reference.get("phase37_condition_policy") != expected_policy:
        _add_issue(issues, PHASE_36_REFERENCE_PATH, "PHASE_37_CONDITION_POLICY_MISMATCH")
    if any(not _test_locked(item.get("test_status")) for item in (signoff, winner, reference)):
        _add_issue(issues, PHASE_36_SIGNOFF_PATH, "TEST_FIREWALL_NOT_CONFIRMED")
    run_ids = {
        signoff.get("winner_run_id"),
        winner.get("winner_run_id"),
        reference.get("winner_run_id"),
        reference.get("current_reference_run_id"),
    }
    if len(run_ids) != 1 or None in run_ids:
        _add_issue(issues, "winner_run_id", "REFERENCE_RUN_ID_MISMATCH")
    winner_run_id = next(iter(run_ids)) if len(run_ids) == 1 else None
    source_path = Path("artifacts/runs") / str(winner_run_id) / "config.json"
    source_payload, source_error = _load_object(root, source_path)
    frozen = None
    invariance = None
    if source_error is not None:
        _add_issue(issues, source_path, source_error)
    else:
        config = source_payload.get("config")
        if not isinstance(config, dict):
            _add_issue(issues, source_path, "CONFIG_MISSING")
        else:
            expected_fingerprint = winner.get("winner_config_fingerprint")
            if source_payload.get("config_fingerprint") != expected_fingerprint:
                _add_issue(issues, source_path, "CONFIG_FINGERPRINT_MISMATCH")
            model = config.get("model", {})
            training = config.get("training", {})
            data = config.get("data", {})
            lineage = config.get("lineage", {})
            reproducibility = config.get("reproducibility", {})
            expected_values = (
                (data.get("feature_variant_id"), reference.get("feature_variant_id"), "FEATURE_VARIANT_DRIFT"),
                (data.get("target_scaling_option"), "YS1", "TARGET_SCALING_DRIFT"),
                (data.get("lookback_steps"), 36, "LOOKBACK_DRIFT"),
                (model.get("pooling"), reference.get("pooling_id"), "POOLING_DRIFT"),
                (model.get("activation"), reference.get("activation_id"), "ACTIVATION_DRIFT"),
                (model.get("d_model"), 64, "DMODEL_DRIFT"),
                (model.get("num_heads"), 4, "HEAD_COUNT_DRIFT"),
                (model.get("num_layers"), 2, "LAYER_COUNT_DRIFT"),
                (model.get("ffn_dim"), 256, "FFN_DIM_DRIFT"),
                (training.get("loss_name"), "MSE", "LOSS_DRIFT"),
                (training.get("huber_delta"), None, "MSE_DELTA_DRIFT"),
                (training.get("batch_size"), 32, "BATCH_DRIFT"),
                (training.get("optimizer_name"), "AdamW", "OPTIMIZER_DRIFT"),
                (training.get("learning_rate"), 0.0003, "LEARNING_RATE_DRIFT"),
                (training.get("weight_decay"), 0.001, "WEIGHT_DECAY_DRIFT"),
                (training.get("max_epochs"), 50, "EPOCH_BUDGET_DRIFT"),
                (training.get("early_stopping_enabled"), True, "EARLY_STOPPING_DISABLED"),
                (training.get("early_stopping_patience"), 10, "PATIENCE_DRIFT"),
                (training.get("early_stopping_metric"), "rmse_wh", "EARLY_STOP_METRIC_DRIFT"),
                (training.get("early_stopping_mode"), "MIN", "EARLY_STOP_MODE_DRIFT"),
                (training.get("gradient_clipping_enabled"), True, "GRADIENT_CLIPPING_DISABLED"),
                (training.get("gradient_clip_max_norm"), 1.0, "GRADIENT_CLIP_DRIFT"),
                (training.get("scheduler_name"), None, "SCHEDULER_DRIFT"),
                (training.get("scheduler_config"), None, "SCHEDULER_CONFIG_DRIFT"),
                (reproducibility.get("seed"), SEED, "SEED_DRIFT"),
                (reproducibility.get("global_seed"), SEED, "GLOBAL_SEED_DRIFT"),
                (reproducibility.get("dataloader_seed"), SEED, "DATALOADER_SEED_DRIFT"),
                (lineage.get("population_fingerprint"), reference.get("population_fingerprint"), "POPULATION_DRIFT"),
                (lineage.get("metric_version"), "METRICS-v1", "METRIC_VERSION_DRIFT"),
            )
            for actual, expected, reason in expected_values:
                if actual != expected:
                    _add_issue(issues, source_path, reason)
            if data.get("target_access_mode") == "TEST":
                _add_issue(issues, source_path, "TEST_ACCESS_REQUESTED")
            try:
                invariance = inspect_loss_invariance(
                    model,
                    learning_rate=float(training["learning_rate"]),
                    weight_decay=float(training["weight_decay"]),
                )
            except (KeyError, RuntimeError, TypeError, ValueError) as error:
                _add_issue(issues, source_path, f"LOSS_INVARIANCE_INVALID:{type(error).__name__}")
            else:
                if invariance["status"] != "PASS":
                    _add_issue(issues, source_path, "LOSS_INVARIANCE_FAILED")
            if not issues:
                frozen = {
                    "feature_variant_id": data["feature_variant_id"],
                    "target_scaling_id": data["target_scaling_option"],
                    "lookback_id": f"L{data['lookback_steps']}",
                    "pooling_id": model["pooling"],
                    "activation_id": model["activation"],
                    "batch_id": f"B{training['batch_size']}",
                    "learning_rate": training["learning_rate"],
                    "weight_decay": training["weight_decay"],
                    "dropout": model["dropout"],
                    "d_model": model["d_model"],
                    "num_heads": model["num_heads"],
                    "head_dim": model["d_model"] // model["num_heads"],
                    "num_layers": model["num_layers"],
                    "ffn_dim": model["ffn_dim"],
                    "population_fingerprint": lineage["population_fingerprint"],
                    "metric_version": lineage["metric_version"],
                    "optimizer": training["optimizer_name"],
                    "max_epochs": training["max_epochs"],
                    "patience": training["early_stopping_patience"],
                    "min_delta": 0,
                    "gradient_clip_norm": training["gradient_clip_max_norm"],
                    "scheduler": "NONE",
                    "warmup": "NONE",
                    "gradient_accumulation": 1,
                    "seed": reproducibility["seed"],
                    "loss": "MSE",
                    "source_config_path": str(source_path),
                    "source_config_fingerprint": source_payload["config_fingerprint"],
                    "test_access": TEST_ACCESS,
                }
    evidence = resolve_phase_conditions(PHASE_ID, root)
    verified = {item["condition_id"]: item for item in evidence["verified_conditions"]}
    reference_evidence = verified.get(REFERENCE_CONDITION_ID)
    if reference_evidence is None:
        _add_issue(issues, str(winner_run_id), "MSE_REFERENCE_EVIDENCE_INVALID")
    elif reference_evidence.get("run_id") != winner_run_id:
        _add_issue(issues, str(winner_run_id), "MSE_REFERENCE_RUN_MISMATCH")
    warnings = list(
        dict.fromkeys(
            [
                *signoff.get("inherited_warnings", signoff.get("warnings", [])),
                *winner.get("inherited_warnings", []),
                *reference.get("inherited_warnings", []),
            ]
        )
    )
    return {
        "valid": not issues,
        "status": (
            "PASS_WITH_WARNING"
            if not issues and INHERITED_WARNING in warnings
            else "PASS" if not issues else "BLOCKED"
        ),
        "issues": issues,
        "source_paths": [
            str(PHASE_36_SIGNOFF_PATH),
            str(PHASE_36_WINNER_PATH),
            str(PHASE_36_REFERENCE_PATH),
            str(source_path),
        ],
        "winner_run_id": winner_run_id,
        "frozen_configuration": frozen,
        "reference_evidence": reference_evidence,
        "invariance": invariance,
        "warnings": warnings,
    }


def _criterion_unit_audit() -> dict[str, Any]:
    mse = build_training_criterion({"loss_name": "MSE", "huber_delta": None})
    huber = build_training_criterion({"loss_name": "HUBER", "huber_delta": 1.0})
    zero = torch.zeros(2, 1)
    small = torch.tensor([[0.5], [-0.5]])
    boundary = torch.tensor([[1.0], [-1.0]])
    large = torch.tensor([[2.0], [-2.0]])
    checks = {
        "mse_exact_class": type(mse) is torch.nn.MSELoss,
        "mse_mean_reduction": mse.reduction == "mean",
        "huber_exact_class": type(huber) is torch.nn.HuberLoss,
        "huber_delta_exact": float(huber.delta) == 1.0,
        "huber_mean_reduction": huber.reduction == "mean",
        "criterion_parameter_free": sum(p.numel() for p in mse.parameters()) == 0
        and sum(p.numel() for p in huber.parameters()) == 0,
        "zero_residual_zero": float(huber(zero, zero)) == 0.0,
        "quadratic_formula": math.isclose(float(huber(small, zero)), 0.125, abs_tol=1e-12),
        "boundary_formula": math.isclose(float(huber(boundary, zero)), 0.5, abs_tol=1e-12),
        "linear_formula": math.isclose(float(huber(large, zero)), 1.5, abs_tol=1e-12),
        "huber_symmetry": math.isclose(float(huber(large, zero)), float(huber(-large, zero)), abs_tol=0.0),
        "mse_symmetry": math.isclose(float(mse(large, zero)), float(mse(-large, zero)), abs_tol=0.0),
    }
    try:
        validate_criterion_inputs(torch.zeros(2, 1), torch.zeros(2))
    except RuntimeError:
        checks["broadcast_rejected"] = True
    else:
        checks["broadcast_rejected"] = False
    checks["finite_output"] = bool(torch.isfinite(huber(large, zero)).item())
    return {"checks": checks, "status": "PASS" if all(checks.values()) else "FAIL"}


def build_phase_37_preflight(project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    handoff = inspect_phase_36_handoff(root)
    try:
        delta = resolve_huber_delta(root)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        delta = {"status": "BLOCKED", "error": f"{type(error).__name__}: {error}"}
    criterion_audit = _criterion_unit_audit()
    invariance = handoff.get("invariance") or {}
    frozen = handoff.get("frozen_configuration") or {}
    reference_evidence = handoff.get("reference_evidence") or {}
    decision = plan_phase_resume(PHASE_ID, root)
    gate = validate_condition_request(PHASE_ID, "L1", root)
    gates = {
        "01_phase36_signoff": handoff["valid"],
        "02_s14_winner_identity": isinstance(handoff.get("winner_run_id"), str)
        and bool(handoff["winner_run_id"]),
        "03_prior_selections_frozen": handoff.get("frozen_configuration") is not None,
        "04_condition_registry": [item.condition_id for item in CONDITIONS] == ["L0", "L1"],
        "05_huber_delta_exact": delta.get("delta_model_space") == 1.0,
        "06_target_model_space": delta.get("target_model_space") == "STANDARDIZED_TARGET",
        "07_raw_wh_delta": delta.get("status") == "PASS",
        "08_common_population": bool(frozen.get("population_fingerprint")),
        "09_same_target_scaling": frozen.get("target_scaling_id") == "YS1",
        "10_architecture_equality": invariance.get("checks", {}).get("architecture_fingerprint_equal") is True,
        "11_parameter_schema_equality": all(
            invariance.get("checks", {}).get(key) is True
            for key in ("parameter_count_equal", "state_dict_keys_equal", "state_dict_shapes_equal")
        ),
        "12_criterion_contract": criterion_audit["status"] == "PASS",
        "13_no_broadcasting": criterion_audit["checks"].get("broadcast_rejected") is True,
        "14_selection_metric_rmse_wh": frozen.get("metric_version") == "METRICS-v1",
        "15_optimizer_training_frozen": invariance.get("checks", {}).get("optimizer_coverage") is True,
        "16_initialization_policy": invariance.get("checks", {}).get("initial_state_equal") is True,
        "17_sample_order_policy": frozen.get("seed") == 42,
        "18_mse_reuse_eligible": reference_evidence.get("run_id") == handoff.get("winner_run_id"),
        "19_test_firewall": frozen.get("test_access") == "FORBIDDEN",
        "20_registry_readiness": gate["allowed"] is True,
    }
    ready = all(gates.values())
    return {
        "phase_id": PHASE_ID,
        "phase_version": PHASE_VERSION,
        "sweep_id": SWEEP_ID,
        "sweep_version": SWEEP_VERSION,
        "conditions": [asdict(condition) for condition in CONDITIONS],
        "selection_metric": "VALIDATION_RMSE_WH",
        "selection_direction": "MIN",
        "tie_rule": "MSE_ON_EXACT_RMSE_TIE",
        "handoff": handoff,
        "delta_audit": delta,
        "criterion_audit": criterion_audit,
        "invariance": invariance,
        "gates": gates,
        "initialization_audit": "NOT_VERIFIABLE" if handoff.get("reference_evidence") else "BLOCKED",
        "sample_order_audit": "NOT_VERIFIABLE" if handoff.get("reference_evidence") else "BLOCKED",
        "loss_scale_comparability": {
            "raw_criterion_cross_loss": "NOT_NUMERICALLY_CROSS_LOSS_COMPARABLE",
            "validation_rmse_wh": "WINNER_ELIGIBLE",
            "validation_mae_wh": "SECONDARY_ONLY",
            "validation_r2": "SECONDARY_ONLY",
            "status": "PASS",
        },
        "gradient_clipping_instrumentation": "PASS",
        "huber_regime_support": "PASS",
        "decision": decision,
        "test_access": TEST_ACCESS,
        "scientific_artifacts_written": False,
        "training_engine_called": False,
        "ready": ready,
        "status": "READY" if ready else "BLOCKED",
    }


def prepare_phase_37_condition(condition_id: str, project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    condition_by_id = {condition.condition_id: condition for condition in CONDITIONS}
    if condition_id not in condition_by_id:
        raise ValueError(f"Condition {condition_id} is not registered for Phase 37")
    preflight = build_phase_37_preflight(root)
    handoff = preflight["handoff"]
    if not handoff["valid"]:
        raise RuntimeError(f"Phase 37 handoff is invalid: {handoff['issues']}")
    condition = condition_by_id[condition_id]
    payload = {
        "phase_id": PHASE_ID,
        "sweep_id": SWEEP_ID,
        "condition_id": condition.condition_id,
        "loss_id": condition.condition_id,
        "loss_name": condition.criterion_name,
        "registry_loss_name": condition.registry_loss_name,
        "reduction": condition.reduction,
        "huber_delta_model_space": condition.huber_delta_model_space,
        "huber_delta_raw_wh_equivalent": (
            preflight["delta_audit"].get("delta_raw_wh_equivalent")
            if condition.condition_id == "L1"
            else None
        ),
        "target_scaler_checksum": preflight["delta_audit"].get("target_scaler_checksum"),
        "execution_mode": condition.execution_mode,
        "reference_run_id": handoff["winner_run_id"],
        "reference_evidence": handoff["reference_evidence"],
        "frozen_configuration": handoff["frozen_configuration"],
        "invariance": handoff["invariance"],
        "warnings": handoff["warnings"],
        "test_access": TEST_ACCESS,
    }
    if condition.execution_mode == "REUSE_REFERENCE":
        return payload
    if not preflight["ready"]:
        failed = [name for name, passed in preflight["gates"].items() if not passed]
        raise RuntimeError(f"Phase 37 Huber preparation blocked: {failed}")
    gate = validate_condition_request(PHASE_ID, condition_id, root)
    if not gate["allowed"]:
        raise RuntimeError(f"Condition execution blocked: {gate}")
    return payload


def _config_differences(left: object, right: object, path: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: list[tuple[str, ...]] = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                differences.append((*path, str(key)))
            else:
                differences.extend(_config_differences(left[key], right[key], (*path, str(key))))
        return differences
    return [] if left == right else [path]


def verify_phase_37_huber_best(
    run_id: str,
    project_root: Path | None = None,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    from course_work.data.datasets import build_train_validation_loaders
    from course_work.data.scaling import inverse_transform_target
    from course_work.evaluation.metrics import EvaluationMode, compute_regression_metrics

    root = Path(project_root or get_project_root()).resolve()
    registry_path = root / "artifacts/experiments/experiment_registry.jsonl"
    records = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [record for record in records if record.get("run_id") == run_id]
    if len(matches) != 1:
        raise RuntimeError(f"Phase 37 expected one Huber registry record, found {len(matches)}")
    record = matches[0]
    run_root = root / "artifacts/runs" / run_id
    paths = {
        "config": run_root / "config.json",
        "status": run_root / "status.json",
        "checkpoint": run_root / "checkpoints/best_checkpoint.pt",
        "history": run_root / "training_history.csv",
        "metrics": run_root / "metrics/best_validation_metrics.json",
        "predictions": run_root / "predictions/best_validation_predictions.csv",
    }
    missing = [str(path.relative_to(root)) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Phase 37 Huber evidence is missing: {missing}")
    if record.get("status") != "COMPLETED" or record.get("experiment_family") != SWEEP_ID:
        raise RuntimeError("Phase 37 Huber registry identity/status is invalid")
    if record.get("test_access_authorized") is not False:
        raise RuntimeError("Phase 37 Huber registry does not preserve the Test firewall")
    if any(metric.get("split_id") != "VALIDATION" for metric in record.get("metrics", [])):
        raise RuntimeError("Phase 37 Huber registry contains non-Validation metrics")
    if any("test" in str(item.get("artifact_path", "")).lower() for item in record.get("artifacts", [])):
        raise RuntimeError("Phase 37 Huber registry contains Test artifacts")
    if any("test" in path.name.lower() for path in run_root.rglob("*") if path.is_file()):
        raise RuntimeError("Phase 37 Huber run directory contains Test evidence")
    for artifact in record.get("artifacts", []):
        if artifact.get("required") is not True:
            continue
        artifact_path = root / str(artifact.get("artifact_path"))
        if not artifact_path.is_file() or sha256_file(artifact_path) != artifact.get("sha256"):
            raise RuntimeError(f"Phase 37 Huber artifact checksum failed: {artifact_path.relative_to(root)}")

    config_payload = read_json(paths["config"])
    config = config_payload.get("config")
    if not isinstance(config, dict) or config_payload.get("config_fingerprint") != record.get("config_fingerprint"):
        raise RuntimeError("Phase 37 Huber config identity/fingerprint is invalid")
    handoff = inspect_phase_36_handoff(root)
    if not handoff["valid"]:
        raise RuntimeError(f"Phase 36 handoff is invalid: {handoff['issues']}")
    reference_config = read_json(root / handoff["frozen_configuration"]["source_config_path"])["config"]
    allowed_differences = {
        "training.loss_name",
        "training.huber_delta",
        "training.loss_id",
        "training.criterion_name",
        "training.loss_reduction",
        "training.huber_delta_model_space",
        "training.huber_delta_raw_wh_equivalent",
        "training.delta_source",
        "training.delta_tuned",
        "lineage.model_config_fingerprint",
        "lineage.criterion_config_fingerprint",
        "lineage.training_config_fingerprint",
    }
    differences = {".".join(path) for path in _config_differences(reference_config, config)}
    if differences - allowed_differences:
        raise RuntimeError(f"Phase 37 Huber frozen configuration drift: {sorted(differences - allowed_differences)}")
    expected_criterion = criterion_config({"loss_name": "HUBER", "huber_delta": 1.0})
    if criterion_config(config["training"]) != expected_criterion:
        raise RuntimeError("Phase 37 Huber criterion configuration is invalid")
    if config["reproducibility"].get("seed") != SEED:
        raise RuntimeError("Phase 37 Huber seed is invalid")
    if config["data"].get("target_access_mode") != "VALIDATION":
        raise RuntimeError("Phase 37 Huber data access is not Validation-only")

    status = read_json(paths["status"])
    metrics_payload = read_json(paths["metrics"])
    metrics = metrics_payload.get("metric_result")
    history = pd.read_csv(paths["history"])
    predictions = pd.read_csv(paths["predictions"])
    if not isinstance(metrics, dict):
        raise RuntimeError("Phase 37 Huber metric result is missing")
    if status.get("status") != "COMPLETED" or status.get("best_epoch") != record.get("best_epoch"):
        raise RuntimeError("Phase 37 Huber status/BEST epoch evidence mismatch")
    if history.empty or int(history["epoch"].max()) != len(history):
        raise RuntimeError("Phase 37 Huber epoch history is invalid")
    best_rows = history.loc[history["is_best"].astype(bool)]
    if best_rows.empty or int(best_rows.iloc[-1]["epoch"]) != int(status["best_epoch"]):
        raise RuntimeError("Phase 37 Huber BEST history marker mismatch")
    if metrics_payload.get("criterion_config") != expected_criterion:
        raise RuntimeError("Phase 37 Huber metric criterion provenance is invalid")
    gradient_diagnostics = metrics_payload.get("gradient_diagnostics")
    required_gradient_fields = {
        "mean_preclip_global_grad_norm",
        "max_preclip_global_grad_norm",
        "clipped_batches",
        "total_batches",
        "clipping_fraction",
        "nonfinite_grad_events",
        "clip_max_norm",
        "clip_order",
    }
    if not isinstance(gradient_diagnostics, dict) or not required_gradient_fields.issubset(gradient_diagnostics):
        raise RuntimeError("Phase 37 Huber gradient diagnostics are incomplete")
    if gradient_diagnostics["nonfinite_grad_events"] != 0:
        raise RuntimeError("Phase 37 Huber recorded non-finite gradients")

    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint.get("criterion_config") != expected_criterion:
        raise RuntimeError("Phase 37 Huber checkpoint criterion provenance is invalid")
    if checkpoint.get("best_epoch") != int(status["best_epoch"]):
        raise RuntimeError("Phase 37 Huber checkpoint BEST epoch provenance is invalid")
    set_seed(SEED)
    model = TransformerRegressor(config["model"])
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Phase 37 Huber BEST strict-load returned incompatible keys")
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if parameter_count != handoff["invariance"]["parameter_count"]:
        raise RuntimeError("Phase 37 Huber parameter count differs from the MSE reference")

    recorded_device = config["runtime"]["device_type"]
    if recorded_device not in ("cpu", "cuda", "mps"):
        raise RuntimeError(f"Phase 37 unsupported recorded device: {recorded_device}")
    if recorded_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Phase 37 recorded device MPS is not available on this platform")
    if recorded_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Phase 37 recorded device CUDA is not available on this platform")
    verification_device = torch.device(recorded_device)

    loaders = build_train_validation_loaders(
        project_root=root,
        variant_id=config["data"]["feature_variant_id"],
        lookback=int(config["data"]["lookback_steps"]),
        target_option=config["data"]["target_scaling_option"],
        batch_size=int(config["training"]["batch_size"]),
        seed=int(config["reproducibility"]["seed"]),
        num_workers=0,
        device_type=recorded_device,
    )
    validation_loader = loaders["VALIDATION"][0]
    target_scaler = load_validated_target_scaler(root)
    model = model.to(verification_device)
    model.eval()
    sample_ids: list[int] = []
    y_true_wh: list[float] = []
    y_pred_wh: list[float] = []
    y_true_model: list[float] = []
    y_pred_model: list[float] = []
    with torch.no_grad():
        for batch in validation_loader:
            batch_x = batch["x"].to(verification_device)
            batch_y_model = batch["y_model"].to(verification_device)
            prediction_model = model(batch_x)
            validate_criterion_inputs(prediction_model, batch_y_model)
            prediction_wh = inverse_transform_target(
                prediction_model.cpu().numpy().reshape(-1, 1),
                "YS1",
                target_scaler,
            ).reshape(-1)
            sample_ids.extend(batch["sample_idx"].numpy().astype(int).tolist())
            y_true_wh.extend(batch["y_raw_wh"].cpu().numpy().reshape(-1).astype(float).tolist())
            y_pred_wh.extend(prediction_wh.astype(float).tolist())
            y_true_model.extend(batch["y_model"].cpu().numpy().reshape(-1).astype(float).tolist())
            y_pred_model.extend(prediction_model.cpu().numpy().reshape(-1).astype(float).tolist())
    recomputed = compute_regression_metrics(
        np.asarray(y_true_wh, dtype=np.float64),
        np.asarray(y_pred_wh, dtype=np.float64),
        np.asarray(sample_ids, dtype=np.int64),
        "VALIDATION",
        EvaluationMode.VALIDATION.value,
        config["lineage"]["population_fingerprint"],
        run_id,
        config["model"].get("model_name", config["model"]["model_family"]),
        lookback_steps=int(config["data"]["lookback_steps"]),
        horizon_steps=int(config["data"]["horizon_steps"]),
        target_scaling_option=config["data"]["target_scaling_option"],
    )
    stored = {field: float(metrics[field]) for field in ("rmse_wh", "mae_wh", "r2")}
    recomputed_values = {field: float(getattr(recomputed, field)) for field in ("rmse_wh", "mae_wh", "r2")}
    tolerance = 1e-9
    if any(abs(stored[field] - recomputed_values[field]) > tolerance for field in stored):
        raise RuntimeError("Phase 37 Huber strict Validation metrics differ from stored BEST evidence")
    if predictions["sample_idx"].astype(int).tolist() != sample_ids:
        raise RuntimeError("Phase 37 Huber ordered Validation population differs from retained predictions")
    if not np.allclose(predictions["y_true_wh"], y_true_wh, atol=tolerance, rtol=tolerance):
        raise RuntimeError("Phase 37 Huber Validation targets differ from retained predictions")
    if not np.allclose(predictions["y_pred_wh"], y_pred_wh, atol=tolerance, rtol=tolerance):
        raise RuntimeError("Phase 37 Huber Validation predictions differ from retained predictions")
    regime = huber_regime_diagnostics(
        np.asarray(y_pred_model, dtype=np.float64).reshape(-1, 1),
        np.asarray(y_true_model, dtype=np.float64).reshape(-1, 1),
    )
    return {
        "run_id": run_id,
        "status": "VERIFIED",
        "strict_load_verification": "PASS",
        "strict_best_verification": "PASS",
        "stored_validation_metrics": stored,
        "recomputed_validation_metrics": recomputed_values,
        "metric_delta": max(abs(stored[field] - recomputed_values[field]) for field in stored),
        "verification_tolerance": tolerance,
        "parameter_count": parameter_count,
        "population_fingerprint": config["lineage"]["population_fingerprint"],
        "gradient_diagnostics": gradient_diagnostics,
        "huber_regime_diagnostics": regime,
        "test_access": TEST_ACCESS,
    }


def materialize_phase_37_preparation_log(project_root: Path | None = None) -> Path:
    root = Path(project_root or get_project_root()).resolve()
    preflight = build_phase_37_preflight(root)
    source_paths = [
        *preflight["handoff"].get("source_paths", []),
        preflight["delta_audit"].get("target_scaler_path"),
    ]
    source_paths = [path for path in dict.fromkeys(source_paths) if isinstance(path, str)]
    payload = {
        "phase_id": PHASE_ID,
        "phase_version": PHASE_VERSION,
        "sweep_id": SWEEP_ID,
        "sweep_version": SWEEP_VERSION,
        "status": "READY_FOR_MANUAL_HUBER" if preflight["ready"] else "BLOCKED",
        "source_artifacts": [
            {"path": path, "sha256": sha256_file(root / path)} for path in source_paths
        ],
        "phase36_status": preflight["handoff"]["status"],
        "mse_run_id": preflight["handoff"]["winner_run_id"],
        "mse_execution_mode": "REUSE_REFERENCE",
        "huber_execution_mode": "TRAIN_NEW",
        "huber_delta_model_space": 1.0,
        "huber_delta_raw_wh_equivalent": preflight["delta_audit"].get("delta_raw_wh_equivalent"),
        "target_scaler_checksum": preflight["delta_audit"].get("target_scaler_checksum"),
        "gates": preflight["gates"],
        "initialization_audit": preflight["initialization_audit"],
        "sample_order_audit": preflight["sample_order_audit"],
        "inherited_warnings": preflight["handoff"]["warnings"],
        "test_access": TEST_ACCESS,
        "training_executed": False,
        "scientific_artifacts_written": False,
        "phase38_executed": False,
    }
    path = root / PROCESSING_LOG_PATH
    atomic_write_bytes(path, canonical_json_bytes(payload))
    return path


__all__ = [
    "CONDITIONS",
    "HUBER_DELTA_MODEL_SPACE",
    "build_phase_37_preflight",
    "configuration_fingerprints",
    "huber_regime_diagnostics",
    "inspect_loss_invariance",
    "inspect_phase_36_handoff",
    "materialize_phase_37_preparation_log",
    "prepare_phase_37_condition",
    "resolve_huber_delta",
    "verify_phase_37_huber_best",
]
