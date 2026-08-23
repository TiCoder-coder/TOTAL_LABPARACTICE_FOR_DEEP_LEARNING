from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from course_work.experiments.phase_execution import (
    plan_phase_resume,
    resolve_phase_conditions,
    validate_condition_request,
)
from course_work.utils.artifacts import get_project_root, read_json


PHASE_ID = 31
PHASE_VERSION = "PHASE-31-v1"
SWEEP_ID = "S9_WEIGHT_DECAY"
SWEEP_VERSION = "SWEEP_S9_WEIGHTDECAY-v1"
REFERENCE_CONDITION_ID = "WD1"
REFERENCE_WEIGHT_DECAY = 0.0001
SEED = 42
TEST_ACCESS = "FORBIDDEN"
PHASE_30_SIGNOFF_PATH = Path("artifacts/sweeps/s8_learning_rate/phase_30_signoff.json")
PHASE_30_WINNER_PATH = Path("artifacts/sweeps/s8_learning_rate/s8_learning_rate_winner.json")
PHASE_30_REFERENCE_PATH = Path("artifacts/sweeps/s8_learning_rate/s8_reference_update.json")
PHASE_30_LOG_PATH = Path("docs/save_log_in_processing/phase_30_s8_learning_rate_log.json")


@dataclass(frozen=True)
class WeightDecayCondition:
    condition_id: str
    weight_decay: float
    execution_mode: str


CONDITIONS = (
    WeightDecayCondition("WD0", 0.0, "TRAIN_NEW"),
    WeightDecayCondition("WD1", REFERENCE_WEIGHT_DECAY, "REUSE_REFERENCE"),
    WeightDecayCondition("WD2", 0.001, "TRAIN_NEW"),
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
    if value is None:
        return False
    token = str(value).upper()
    return token in {
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
    if isinstance(value, str) and value.startswith(prefix):
        suffix = value[len(prefix):]
        if suffix.isdigit():
            return int(suffix)
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


def inspect_phase_30_handoff(project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    signoff, signoff_error = _load_object(root, PHASE_30_SIGNOFF_PATH)
    winner, winner_error = _load_object(root, PHASE_30_WINNER_PATH)
    reference, reference_error = _load_object(root, PHASE_30_REFERENCE_PATH)
    processing_log, processing_log_error = _load_object(root, PHASE_30_LOG_PATH)
    issues: list[dict[str, str]] = []

    for path, error in (
        (PHASE_30_SIGNOFF_PATH, signoff_error),
        (PHASE_30_WINNER_PATH, winner_error),
        (PHASE_30_REFERENCE_PATH, reference_error),
    ):
        if error is not None:
            _add_issue(issues, path, error)

    if signoff is not None:
        if not _is_pass(signoff):
            _add_issue(issues, PHASE_30_SIGNOFF_PATH, "SIGNOFF_NOT_PASS")
        if signoff.get("approved_for_phase31") is not True:
            _add_issue(issues, PHASE_30_SIGNOFF_PATH, "PHASE_31_NOT_APPROVED")
        if not _test_is_locked(signoff.get("test_status")):
            _add_issue(issues, PHASE_30_SIGNOFF_PATH, "TEST_FIREWALL_NOT_CONFIRMED")

    if winner is not None:
        if not _is_pass(winner):
            _add_issue(issues, PHASE_30_WINNER_PATH, "WINNER_NOT_PASS")
        if not _test_is_locked(winner.get("test_status")):
            _add_issue(issues, PHASE_30_WINNER_PATH, "TEST_FIREWALL_NOT_CONFIRMED")

    if reference is not None:
        if reference.get("approved_for_phase31") is not True:
            _add_issue(issues, PHASE_30_REFERENCE_PATH, "PHASE_31_NOT_APPROVED")
        if reference.get("weight_decay_state") != "WD1_1E-4":
            _add_issue(issues, PHASE_30_REFERENCE_PATH, "REFERENCE_WEIGHT_DECAY_MISMATCH")

    selected_learning_rate = None
    winner_run_id = None
    source_config_path = None
    source_config_fingerprint = None
    if winner is not None and reference is not None:
        winner_run_id = _string(winner, "winner_run_id")
        reference_run_id = _string(reference, "winner_run_id", "current_reference_run_id")
        if winner_run_id is None or reference_run_id is None:
            _add_issue(issues, "winner_run_id", "REFERENCE_RUN_ID_MISSING")
        elif winner_run_id != reference_run_id:
            _add_issue(issues, "winner_run_id", "REFERENCE_RUN_ID_MISMATCH")

        winner_learning_rate = _number(winner, "winner_learning_rate", "learning_rate")
        reference_learning_rate = _number(reference, "selected_learning_rate", "learning_rate")
        if winner_learning_rate is None or reference_learning_rate is None:
            _add_issue(issues, "learning_rate", "SELECTED_LEARNING_RATE_MISSING")
        elif not math.isclose(winner_learning_rate, reference_learning_rate, rel_tol=0.0, abs_tol=1e-12):
            _add_issue(issues, "learning_rate", "SELECTED_LEARNING_RATE_MISMATCH")
        else:
            selected_learning_rate = winner_learning_rate

        for field in (
            "feature_variant_id",
            "target_scaling_id",
            "lookback_id",
            "pooling_id",
            "activation_id",
            "batch_id",
            "population_fingerprint",
        ):
            _field_matches(issues, winner, reference, field)

        if winner_run_id is not None:
            source_config_path = Path("artifacts/runs") / winner_run_id / "config.json"
            source_config_payload, source_config_error = _load_object(root, source_config_path)
            if source_config_error is not None:
                _add_issue(issues, source_config_path, source_config_error)
            elif source_config_payload is not None:
                source_config_fingerprint = _string(source_config_payload, "config_fingerprint")
                expected_fingerprint = _string(reference, "winner_config_fingerprint")
                optimizer_fingerprint = _string(reference, "winner_optimizer_config_fingerprint")
                if source_config_payload.get("run_id") != winner_run_id:
                    _add_issue(issues, source_config_path, "RUN_ID_MISMATCH")
                if source_config_fingerprint is None or expected_fingerprint is None:
                    _add_issue(issues, source_config_path, "CONFIG_FINGERPRINT_MISSING")
                elif source_config_fingerprint != expected_fingerprint:
                    _add_issue(issues, source_config_path, "CONFIG_FINGERPRINT_MISMATCH")
                if optimizer_fingerprint is None:
                    _add_issue(issues, PHASE_30_REFERENCE_PATH, "OPTIMIZER_CONFIG_FINGERPRINT_MISSING")
                config = source_config_payload.get("config")
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
                            (training.get("batch_size"), expected_batch, "BATCH_DRIFT"),
                            (training.get("optimizer_name"), "AdamW", "OPTIMIZER_IDENTITY_DRIFT"),
                            (training.get("weight_decay"), REFERENCE_WEIGHT_DECAY, "REFERENCE_WEIGHT_DECAY_MISMATCH"),
                            (training.get("loss_name"), "MSE", "LOSS_DRIFT"),
                            (training.get("max_epochs"), 50, "EPOCH_BUDGET_DRIFT"),
                            (training.get("early_stopping_patience"), 10, "PATIENCE_DRIFT"),
                            (training.get("gradient_clipping_enabled"), True, "GRADIENT_CLIP_DRIFT"),
                            (training.get("gradient_clip_max_norm"), 1.0, "GRADIENT_CLIP_DRIFT"),
                            (training.get("scheduler_name"), None, "SCHEDULER_DRIFT"),
                            (model.get("dropout"), 0.1, "DROPOUT_DRIFT"),
                            (reproducibility.get("seed"), SEED, "SEED_DRIFT"),
                            (lineage.get("population_fingerprint"), winner.get("population_fingerprint"), "POPULATION_MISMATCH"),
                        )
                        for actual, expected, reason in checks:
                            if actual != expected:
                                _add_issue(issues, source_config_path, reason)
                        actual_learning_rate = training.get("learning_rate")
                        if selected_learning_rate is None or not isinstance(actual_learning_rate, (int, float)):
                            _add_issue(issues, source_config_path, "SELECTED_LEARNING_RATE_MISSING")
                        elif not math.isclose(float(actual_learning_rate), selected_learning_rate, rel_tol=0.0, abs_tol=1e-12):
                            _add_issue(issues, source_config_path, "SELECTED_LEARNING_RATE_MISMATCH")

    frozen_configuration = None
    if winner is not None and reference is not None and not issues:
        frozen_configuration = {
            "feature_variant_id": winner["feature_variant_id"],
            "target_scaling_id": winner["target_scaling_id"],
            "lookback_id": winner["lookback_id"],
            "pooling_id": winner["pooling_id"],
            "activation_id": winner["activation_id"],
            "batch_id": winner["batch_id"],
            "learning_rate": selected_learning_rate,
            "population_fingerprint": winner["population_fingerprint"],
            "optimizer": "AdamW",
            "dropout": 0.1,
            "max_epochs": 50,
            "patience": 10,
            "gradient_clip_norm": 1.0,
            "seed": SEED,
            "scheduler": "NONE",
            "warmup": "NONE",
            "gradient_accumulation": 1,
            "loss": "MSE",
            "explicit_l2_loss": False,
            "test_access": TEST_ACCESS,
            "source_config_path": str(source_config_path),
            "source_config_fingerprint": source_config_fingerprint,
        }

    processing_log_observation = {
        "path": str(PHASE_30_LOG_PATH),
        "available": processing_log_error is None,
        "status": processing_log.get("status") if processing_log is not None else processing_log_error,
        "authoritative": False,
    }
    return {
        "valid": not issues,
        "issues": issues,
        "source_paths": [
            str(PHASE_30_SIGNOFF_PATH),
            str(PHASE_30_WINNER_PATH),
            str(PHASE_30_REFERENCE_PATH),
        ],
        "selected_learning_rate": selected_learning_rate,
        "winner_run_id": winner_run_id,
        "frozen_configuration": frozen_configuration,
        "processing_log_observation": processing_log_observation,
    }


def build_phase_31_preflight(project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    decision = plan_phase_resume(PHASE_ID, root)
    handoff = inspect_phase_30_handoff(root)
    return {
        "phase_id": PHASE_ID,
        "phase_version": PHASE_VERSION,
        "sweep_id": SWEEP_ID,
        "sweep_version": SWEEP_VERSION,
        "conditions": [asdict(condition) for condition in CONDITIONS],
        "reference_condition_id": REFERENCE_CONDITION_ID,
        "selected_learning_rate": handoff["selected_learning_rate"],
        "test_access": TEST_ACCESS,
        "handoff": handoff,
        "decision": decision,
        "ready": handoff["valid"] and decision["readiness"]["ready"],
    }


def prepare_phase_31_condition(
    condition_id: str,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    condition_by_id = {condition.condition_id: condition for condition in CONDITIONS}
    if condition_id not in condition_by_id:
        raise ValueError(f"Condition {condition_id} is not registered for Phase 31")
    handoff = inspect_phase_30_handoff(root)
    if not handoff["valid"]:
        raise RuntimeError(f"Phase 31 handoff is invalid: {handoff['issues']}")
    condition = condition_by_id[condition_id]
    if condition.execution_mode == "REUSE_REFERENCE":
        evidence = resolve_phase_conditions(PHASE_ID, root)
        verified = {
            item["condition_id"]: item
            for item in evidence["verified_conditions"]
        }
        if condition_id not in verified:
            raise RuntimeError(
                f"Phase 31 reference evidence is invalid: {evidence}"
            )
        return {
            "phase_id": PHASE_ID,
            "sweep_id": SWEEP_ID,
            "condition_id": condition.condition_id,
            "execution_mode": condition.execution_mode,
            "weight_decay": condition.weight_decay,
            "reference_run_id": handoff["winner_run_id"],
            "reference_evidence": verified[condition_id],
            "frozen_configuration": handoff["frozen_configuration"],
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
        "weight_decay": condition.weight_decay,
        "reference_run_id": handoff["winner_run_id"],
        "frozen_configuration": handoff["frozen_configuration"],
        "test_access": TEST_ACCESS,
    }
