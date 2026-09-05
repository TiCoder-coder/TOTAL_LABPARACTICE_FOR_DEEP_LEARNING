from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from course_work.experiments.phase_execution import (
    plan_phase_resume,
    resolve_phase_conditions,
    validate_condition_request,
)
from course_work.models.transformer_regressor import TransformerRegressor, validate_transformer_config
from course_work.utils.artifacts import canonical_json_bytes, get_project_root, read_json, sha256_bytes, sha256_file


PHASE_ID = 33
PHASE_VERSION = "PHASE-33-v1"
SWEEP_ID = "S11_D_MODEL"
SWEEP_VERSION = "SWEEP_S11_DMODEL-v1"
REFERENCE_CONDITION_ID = "D64"
SEED = 42
TEST_ACCESS = "FORBIDDEN"
PHASE_32_ROOT = Path("artifacts/sweeps/S10_dropout")
PHASE_32_SIGNOFF_PATH = PHASE_32_ROOT / "phase_32_signoff.json"
PHASE_32_WINNER_PATH = PHASE_32_ROOT / "s10_dropout_winner.json"
PHASE_32_REFERENCE_PATH = PHASE_32_ROOT / "s10_reference_update.json"
PHASE_32_LOG_PATH = Path("docs/save_log_in_processing/phase_32_s10_dropout_log.json")


@dataclass(frozen=True)
class DModelCondition:
    condition_id: str
    d_model: int
    execution_mode: str


CONDITIONS = (
    DModelCondition("D32", 32, "TRAIN_NEW"),
    DModelCondition("D64", 64, "REUSE_REFERENCE"),
)


def _load_object(root: Path, relative_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    path = root / relative_path
    if not path.is_file():
        return None, "MISSING"
    try:
        value = read_json(path)
    except (OSError, TypeError, ValueError):
        return None, "INVALID_JSON"
    if not isinstance(value, dict):
        return None, "INVALID_OBJECT"
    return value, None


def _status(record: dict[str, Any]) -> str | None:
    value = record.get("status", record.get("overall_status"))
    return str(value).upper() if value is not None else None


def _is_pass(record: dict[str, Any]) -> bool:
    return _status(record) in {"PASS", "PASS_WITH_WARNING"}


def _test_is_locked(value: Any) -> bool:
    return value is not None and str(value).upper() in {
        "FORBIDDEN",
        "LOCKED",
        "NOT_ACCESSED",
        "UNTOUCHED",
        "LOCKED_UNTIL_PHASE_47",
    }


def _number(record: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _string(record: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _integer_id(value: Any, prefix: str) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.startswith(prefix) and value[len(prefix):].isdigit():
        return int(value[len(prefix):])
    return None


def _add_issue(issues: list[dict[str, str]], path: Path | str, reason: str) -> None:
    issue = {"path": str(path), "reason": reason}
    if issue not in issues:
        issues.append(issue)


def _field_matches(
    issues: list[dict[str, str]],
    left: dict[str, Any],
    right: dict[str, Any],
    field: str,
) -> None:
    left_value = left.get(field)
    right_value = right.get(field)
    if left_value is None or right_value is None:
        _add_issue(issues, field, "FROZEN_FIELD_MISSING")
    elif left_value != right_value:
        _add_issue(issues, field, "FROZEN_FIELD_MISMATCH")


def _validate_declared_outputs(root: Path, signoff: dict[str, Any], issues: list[dict[str, str]]) -> None:
    output_paths = signoff.get("output_paths")
    output_checksums = signoff.get("output_checksums")
    if not isinstance(output_paths, list) or not isinstance(output_checksums, dict):
        _add_issue(issues, PHASE_32_SIGNOFF_PATH, "INVALID_OUTPUT_DECLARATION")
        return
    for relative_path in output_paths:
        if not isinstance(relative_path, str):
            _add_issue(issues, str(relative_path), "INVALID_PATH")
            continue
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            _add_issue(issues, relative_path, "PATH_OUTSIDE_PROJECT")
            continue
        if not path.is_file():
            _add_issue(issues, relative_path, "MISSING")
            continue
        expected = output_checksums.get(relative_path)
        if not isinstance(expected, str):
            _add_issue(issues, relative_path, "CHECKSUM_MISSING")
        elif sha256_file(path) != expected:
            _add_issue(issues, relative_path, "CHECKSUM_MISMATCH")


def inspect_d_model_geometry(model_config: dict[str, Any]) -> dict[str, Any]:
    config = validate_transformer_config(model_config)
    model = TransformerRegressor(config)
    sample = torch.zeros(2, 36, config.input_size, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        output = model(sample)
    state_shapes = {name: list(value.shape) for name, value in model.state_dict().items()}
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    geometry = {
        "d_model": config.d_model,
        "num_heads": config.num_heads,
        "head_dim": config.d_model // config.num_heads,
        "num_layers": config.num_layers,
        "ffn_dim": config.ffn_dim,
        "ffn_ratio": config.ffn_dim / config.d_model,
        "input_projection_shape": list(model.input_projection.weight.shape),
        "head_shape": list(model.head.weight.shape),
        "output_shape": list(output.shape),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "state_shapes": state_shapes,
    }
    return {
        "geometry_version": "DMODEL-GEOMETRY-v1",
        **geometry,
        "geometry_fingerprint": sha256_bytes(canonical_json_bytes(geometry)),
        "divisible_by_heads": config.d_model % config.num_heads == 0,
        "status": "PASS" if output.shape == (2, 1) else "FAIL",
    }


def compare_d_model_geometry(model_config: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for condition in CONDITIONS:
        candidate_config = dict(model_config)
        candidate_config["d_model"] = condition.d_model
        geometry = inspect_d_model_geometry(candidate_config)
        candidates.append({"condition_id": condition.condition_id, **geometry})
    by_id = {item["condition_id"]: item for item in candidates}
    d32 = by_id["D32"]
    d64 = by_id["D64"]
    allowed_shape_changes = sorted(
        name for name in d32["state_shapes"] if d32["state_shapes"][name] != d64["state_shapes"][name]
    )
    unchanged_shape_count = sum(
        d32["state_shapes"][name] == d64["state_shapes"][name] for name in d32["state_shapes"]
    )
    return {
        "comparison_version": "DMODEL-CAPACITY-v1",
        "candidates": candidates,
        "parameter_count_delta": d64["parameter_count"] - d32["parameter_count"],
        "parameter_count_ratio_d64_to_d32": d64["parameter_count"] / d32["parameter_count"],
        "changed_shape_parameters": allowed_shape_changes,
        "unchanged_shape_count": unchanged_shape_count,
        "external_output_contract_equal": d32["output_shape"] == d64["output_shape"] == [2, 1],
        "status": "PASS" if all(item["status"] == "PASS" for item in candidates) else "FAIL",
    }


def inspect_phase_32_handoff(project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    signoff, signoff_error = _load_object(root, PHASE_32_SIGNOFF_PATH)
    winner, winner_error = _load_object(root, PHASE_32_WINNER_PATH)
    reference, reference_error = _load_object(root, PHASE_32_REFERENCE_PATH)
    processing_log, processing_log_error = _load_object(root, PHASE_32_LOG_PATH)
    issues: list[dict[str, str]] = []
    for path, error in (
        (PHASE_32_SIGNOFF_PATH, signoff_error),
        (PHASE_32_WINNER_PATH, winner_error),
        (PHASE_32_REFERENCE_PATH, reference_error),
    ):
        if error is not None:
            _add_issue(issues, path, error)
    if signoff is not None:
        if not _is_pass(signoff):
            _add_issue(issues, PHASE_32_SIGNOFF_PATH, "SIGNOFF_NOT_PASS")
        if signoff.get("approved_for_phase33") is not True:
            _add_issue(issues, PHASE_32_SIGNOFF_PATH, "PHASE_33_NOT_APPROVED")
        if not _test_is_locked(signoff.get("test_status")):
            _add_issue(issues, PHASE_32_SIGNOFF_PATH, "TEST_FIREWALL_NOT_CONFIRMED")
        _validate_declared_outputs(root, signoff, issues)
    if winner is not None:
        if not _is_pass(winner):
            _add_issue(issues, PHASE_32_WINNER_PATH, "WINNER_NOT_PASS")
        if not _test_is_locked(winner.get("test_status")):
            _add_issue(issues, PHASE_32_WINNER_PATH, "TEST_FIREWALL_NOT_CONFIRMED")
    if reference is not None:
        if reference.get("approved_for_phase33") is not True:
            _add_issue(issues, PHASE_32_REFERENCE_PATH, "PHASE_33_NOT_APPROVED")
        if not _test_is_locked(reference.get("test_status")):
            _add_issue(issues, PHASE_32_REFERENCE_PATH, "TEST_FIREWALL_NOT_CONFIRMED")

    selected_dropout = None
    winner_run_id = None
    source_config_path = None
    source_config_fingerprint = None
    capacity_comparison = None
    if winner is not None and reference is not None:
        winner_run_id = _string(winner, "winner_run_id")
        reference_run_id = _string(reference, "winner_run_id", "current_reference_run_id")
        if winner_run_id is None or reference_run_id is None:
            _add_issue(issues, "winner_run_id", "REFERENCE_RUN_ID_MISSING")
        elif winner_run_id != reference_run_id:
            _add_issue(issues, "winner_run_id", "REFERENCE_RUN_ID_MISMATCH")
        winner_dropout = _number(winner, "winner_dropout_probability", "dropout")
        reference_dropout = _number(reference, "selected_dropout_probability", "dropout")
        if winner_dropout is None or reference_dropout is None:
            _add_issue(issues, "dropout", "SELECTED_DROPOUT_MISSING")
        elif not math.isclose(winner_dropout, reference_dropout, rel_tol=0.0, abs_tol=1e-12):
            _add_issue(issues, "dropout", "SELECTED_DROPOUT_MISMATCH")
        else:
            selected_dropout = winner_dropout
        for field in (
            "feature_variant_id",
            "target_scaling_id",
            "lookback_id",
            "pooling_id",
            "activation_id",
            "batch_id",
            "learning_rate",
            "weight_decay",
            "d_model",
            "num_heads",
            "num_layers",
            "ffn_dim",
            "population_fingerprint",
        ):
            _field_matches(issues, winner, reference, field)
        if winner.get("d_model") != 64 or reference.get("d_model") != 64:
            _add_issue(issues, "d_model", "REFERENCE_DMODEL_MISMATCH")
        if reference.get("d_model_state") not in {None, "D64"}:
            _add_issue(issues, PHASE_32_REFERENCE_PATH, "REFERENCE_DMODEL_STATE_MISMATCH")
        if winner_run_id is not None:
            source_config_path = Path("artifacts/runs") / winner_run_id / "config.json"
            source_payload, source_error = _load_object(root, source_config_path)
            if source_error is not None:
                _add_issue(issues, source_config_path, source_error)
            elif source_payload is not None:
                source_config_fingerprint = _string(source_payload, "config_fingerprint")
                if source_payload.get("run_id") != winner_run_id:
                    _add_issue(issues, source_config_path, "RUN_ID_MISMATCH")
                if source_config_fingerprint != _string(reference, "winner_config_fingerprint"):
                    _add_issue(issues, source_config_path, "CONFIG_FINGERPRINT_MISMATCH")
                config = source_payload.get("config")
                if not isinstance(config, dict):
                    _add_issue(issues, source_config_path, "CONFIG_MISSING")
                else:
                    data = config.get("data")
                    model = config.get("model")
                    training = config.get("training")
                    reproducibility = config.get("reproducibility")
                    lineage = config.get("lineage")
                    if not all(isinstance(value, dict) for value in (data, model, training, reproducibility, lineage)):
                        _add_issue(issues, source_config_path, "CONFIG_STRUCTURE_INVALID")
                    else:
                        expected_lookback = _integer_id(winner.get("lookback_id"), "L")
                        expected_batch = _integer_id(winner.get("batch_id"), "B")
                        checks = (
                            (data.get("feature_variant_id"), winner.get("feature_variant_id"), "FEATURE_VARIANT_DRIFT"),
                            (data.get("target_scaling_option"), winner.get("target_scaling_id"), "TARGET_SCALING_DRIFT"),
                            (data.get("lookback_steps"), expected_lookback, "LOOKBACK_DRIFT"),
                            (model.get("pooling"), winner.get("pooling_id"), "POOLING_DRIFT"),
                            (model.get("activation"), winner.get("activation_id"), "ACTIVATION_DRIFT"),
                            (model.get("d_model"), 64, "DMODEL_DRIFT"),
                            (model.get("num_heads"), 4, "HEAD_COUNT_DRIFT"),
                            (model.get("num_layers"), 2, "LAYER_COUNT_DRIFT"),
                            (model.get("ffn_dim"), 128, "FFN_DIM_DRIFT"),
                            (training.get("batch_size"), expected_batch, "BATCH_DRIFT"),
                            (training.get("optimizer_name"), "AdamW", "OPTIMIZER_IDENTITY_DRIFT"),
                            (training.get("loss_name"), "MSE", "LOSS_DRIFT"),
                            (training.get("max_epochs"), 50, "EPOCH_BUDGET_DRIFT"),
                            (training.get("early_stopping_patience"), 10, "PATIENCE_DRIFT"),
                            (training.get("gradient_clipping_enabled"), True, "GRADIENT_CLIP_DRIFT"),
                            (training.get("gradient_clip_max_norm"), 1.0, "GRADIENT_CLIP_DRIFT"),
                            (training.get("scheduler_name"), None, "SCHEDULER_DRIFT"),
                            (reproducibility.get("seed"), SEED, "SEED_DRIFT"),
                            (lineage.get("population_fingerprint"), winner.get("population_fingerprint"), "POPULATION_MISMATCH"),
                        )
                        for actual, expected, reason in checks:
                            if actual != expected:
                                _add_issue(issues, source_config_path, reason)
                        for actual, expected, reason in (
                            (model.get("dropout"), selected_dropout, "SELECTED_DROPOUT_MISMATCH"),
                            (training.get("learning_rate"), winner.get("learning_rate"), "LEARNING_RATE_DRIFT"),
                            (training.get("weight_decay"), winner.get("weight_decay"), "WEIGHT_DECAY_DRIFT"),
                        ):
                            if not isinstance(actual, (int, float)) or not isinstance(expected, (int, float)):
                                _add_issue(issues, source_config_path, reason)
                            elif not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
                                _add_issue(issues, source_config_path, reason)
                        try:
                            capacity_comparison = compare_d_model_geometry(model)
                        except (TypeError, ValueError, RuntimeError):
                            _add_issue(issues, source_config_path, "DMODEL_GEOMETRY_INVALID")
                        else:
                            if capacity_comparison["status"] != "PASS":
                                _add_issue(issues, source_config_path, "DMODEL_GEOMETRY_INVALID")

    frozen_configuration = None
    if winner is not None and reference is not None and not issues:
        frozen_configuration = {
            "feature_variant_id": winner["feature_variant_id"],
            "target_scaling_id": winner["target_scaling_id"],
            "lookback_id": winner["lookback_id"],
            "pooling_id": winner["pooling_id"],
            "activation_id": winner["activation_id"],
            "batch_id": winner["batch_id"],
            "learning_rate": winner["learning_rate"],
            "weight_decay": winner["weight_decay"],
            "dropout": selected_dropout,
            "population_fingerprint": winner["population_fingerprint"],
            "optimizer": "AdamW",
            "num_heads": 4,
            "num_layers": 2,
            "ffn_dim": 128,
            "max_epochs": 50,
            "patience": 10,
            "gradient_clip_norm": 1.0,
            "seed": SEED,
            "scheduler": "NONE",
            "warmup": "NONE",
            "gradient_accumulation": 1,
            "loss": "MSE",
            "test_access": TEST_ACCESS,
            "source_config_path": str(source_config_path),
            "source_config_fingerprint": source_config_fingerprint,
        }

    return {
        "valid": not issues,
        "issues": issues,
        "source_paths": [
            str(PHASE_32_SIGNOFF_PATH),
            str(PHASE_32_WINNER_PATH),
            str(PHASE_32_REFERENCE_PATH),
        ],
        "selected_dropout": selected_dropout,
        "winner_run_id": winner_run_id,
        "frozen_configuration": frozen_configuration,
        "capacity_comparison": capacity_comparison,
        "processing_log_observation": {
            "path": str(PHASE_32_LOG_PATH),
            "available": processing_log_error is None,
            "status": processing_log.get("status") if processing_log is not None else processing_log_error,
            "authoritative": False,
        },
    }


def build_phase_33_preflight(project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    decision = plan_phase_resume(PHASE_ID, root)
    handoff = inspect_phase_32_handoff(root)
    return {
        "phase_id": PHASE_ID,
        "phase_version": PHASE_VERSION,
        "sweep_id": SWEEP_ID,
        "sweep_version": SWEEP_VERSION,
        "conditions": [asdict(condition) for condition in CONDITIONS],
        "reference_condition_id": REFERENCE_CONDITION_ID,
        "selected_dropout": handoff["selected_dropout"],
        "test_access": TEST_ACCESS,
        "handoff": handoff,
        "decision": decision,
        "ready": handoff["valid"] and decision["readiness"]["ready"],
    }


def prepare_phase_33_condition(
    condition_id: str,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    condition_by_id = {condition.condition_id: condition for condition in CONDITIONS}
    if condition_id not in condition_by_id:
        raise ValueError(f"Condition {condition_id} is not registered for Phase 33")
    handoff = inspect_phase_32_handoff(root)
    if not handoff["valid"]:
        raise RuntimeError(f"Phase 33 handoff is invalid: {handoff['issues']}")
    condition = condition_by_id[condition_id]
    if condition.execution_mode == "REUSE_REFERENCE":
        evidence = resolve_phase_conditions(PHASE_ID, root)
        verified = {item["condition_id"]: item for item in evidence["verified_conditions"]}
        if condition_id not in verified:
            raise RuntimeError(f"Phase 33 reference evidence is invalid: {evidence}")
        return {
            "phase_id": PHASE_ID,
            "sweep_id": SWEEP_ID,
            "condition_id": condition.condition_id,
            "execution_mode": condition.execution_mode,
            "d_model": condition.d_model,
            "reference_run_id": handoff["winner_run_id"],
            "reference_evidence": verified[condition_id],
            "frozen_configuration": handoff["frozen_configuration"],
            "capacity_comparison": handoff["capacity_comparison"],
            "test_access": TEST_ACCESS,
        }
    gate = validate_condition_request(PHASE_ID, condition_id, root)
    if not gate["allowed"]:
        raise RuntimeError(f"Condition execution blocked: {gate}")
    return {
        "phase_id": PHASE_ID,
        "sweep_id": SWEEP_ID,
        "condition_id": condition.condition_id,
        "execution_mode": condition.execution_mode,
        "d_model": condition.d_model,
        "reference_run_id": handoff["winner_run_id"],
        "frozen_configuration": handoff["frozen_configuration"],
        "capacity_comparison": handoff["capacity_comparison"],
        "test_access": TEST_ACCESS,
    }
