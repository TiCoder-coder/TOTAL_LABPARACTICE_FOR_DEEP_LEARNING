from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from course_work.experiments.phase_execution import (
    plan_phase_resume,
    resolve_phase_conditions,
    validate_condition_request,
)
from course_work.models.transformer_regressor import TransformerRegressor, validate_transformer_config
from course_work.utils.artifacts import canonical_json_bytes, get_project_root, read_json, sha256_bytes


PHASE_ID = 32
PHASE_VERSION = "PHASE-32-v1"
SWEEP_ID = "S10_DROPOUT"
SWEEP_VERSION = "SWEEP_S10_DROPOUT-v1"
REFERENCE_CONDITION_ID = "DR01"
REFERENCE_DROPOUT = 0.1
SEED = 42
TEST_ACCESS = "FORBIDDEN"
PHASE_31_ROOT = Path("artifacts/sweeps/S9_weight_decay")
PHASE_31_SIGNOFF_PATH = PHASE_31_ROOT / "phase_31_signoff.json"
PHASE_31_WINNER_PATH = PHASE_31_ROOT / "s9_weight_decay_winner.json"
PHASE_31_REFERENCE_PATH = PHASE_31_ROOT / "s9_reference_update.json"
PHASE_31_LOG_PATH = Path("docs/save_log_in_processing/phase_31_s9_weight_decay_log.json")


@dataclass(frozen=True)
class DropoutCondition:
    condition_id: str
    dropout_probability: float
    execution_mode: str


CONDITIONS = (
    DropoutCondition("DR01", 0.1, "REUSE_REFERENCE"),
    DropoutCondition("DR02", 0.2, "TRAIN_NEW"),
    DropoutCondition("DR03", 0.3, "TRAIN_NEW"),
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
    return str(value).upper() in {
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


def inspect_dropout_scope(model_config: dict[str, Any]) -> dict[str, Any]:
    config = validate_transformer_config(model_config)
    model = TransformerRegressor(config)
    sites: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(model.encoder.layers):
        layer_sites = (
            (
                f"encoder.layers.{layer_index}.self_attn",
                "MultiheadAttention",
                "ATTENTION_WEIGHT",
                float(layer.self_attn.dropout),
            ),
            (
                f"encoder.layers.{layer_index}.dropout_attn_residual",
                "Dropout",
                "ATTENTION_RESIDUAL",
                float(layer.dropout_attn_residual.p),
            ),
            (
                f"encoder.layers.{layer_index}.dropout_ff_hidden",
                "Dropout",
                "FFN_HIDDEN",
                float(layer.dropout_ff_hidden.p),
            ),
            (
                f"encoder.layers.{layer_index}.dropout_ff_residual",
                "Dropout",
                "FFN_RESIDUAL",
                float(layer.dropout_ff_residual.p),
            ),
        )
        for module_path, module_type, semantic_location, probability in layer_sites:
            sites.append(
                {
                    "site_id": f"D{len(sites) + 1}",
                    "module_path": module_path,
                    "module_type": module_type,
                    "semantic_location": semantic_location,
                    "layer_index": layer_index,
                    "controlled_by_global_dropout": True,
                    "dropout_probability": probability,
                    "parameter_count": 0,
                }
            )
    topology = [
        {
            "module_path": site["module_path"],
            "module_type": site["module_type"],
            "semantic_location": site["semantic_location"],
            "layer_index": site["layer_index"],
            "controlled_by_global_dropout": site["controlled_by_global_dropout"],
        }
        for site in sites
    ]
    fingerprint = sha256_bytes(canonical_json_bytes(topology))
    expected_probability = float(config.dropout)
    probability_matches = all(
        math.isclose(site["dropout_probability"], expected_probability, rel_tol=0.0, abs_tol=1e-12)
        for site in sites
    )
    return {
        "scope_version": "DROPOUT-SCOPE-v1",
        "sites": sites,
        "site_count": len(sites),
        "scope_fingerprint": fingerprint,
        "expected_probability": expected_probability,
        "probability_matches": probability_matches,
        "input_dropout_present": any("input" in site["semantic_location"].lower() for site in sites),
        "head_dropout_present": any("head" in site["semantic_location"].lower() for site in sites),
        "pooling_dropout_present": any("pool" in site["semantic_location"].lower() for site in sites),
        "status": "PASS" if probability_matches and len(sites) == config.num_layers * 4 else "FAIL",
    }


def verify_dropout_mode_semantics(probability: float) -> dict[str, Any]:
    if probability not in {condition.dropout_probability for condition in CONDITIONS}:
        raise ValueError("Dropout probability is not registered for Phase 32")
    layer = nn.Dropout(probability)
    sample = torch.ones(4096, dtype=torch.float32)
    layer.train()
    train_first = layer(sample)
    train_second = layer(sample)
    train_stochastic = not torch.equal(train_first, train_second)
    layer.eval()
    eval_first = layer(sample)
    eval_second = layer(sample)
    eval_repeatable = torch.equal(eval_first, eval_second)
    eval_identity = torch.equal(eval_first, sample)
    status = "PASS" if train_stochastic and eval_repeatable and eval_identity else "FAIL"
    return {
        "dropout_probability": probability,
        "train_stochastic": train_stochastic,
        "eval_repeatable": eval_repeatable,
        "eval_identity": eval_identity,
        "mc_dropout": False,
        "status": status,
    }


def inspect_phase_31_handoff(project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    signoff, signoff_error = _load_object(root, PHASE_31_SIGNOFF_PATH)
    winner, winner_error = _load_object(root, PHASE_31_WINNER_PATH)
    reference, reference_error = _load_object(root, PHASE_31_REFERENCE_PATH)
    processing_log, processing_log_error = _load_object(root, PHASE_31_LOG_PATH)
    issues: list[dict[str, str]] = []

    for path, error in (
        (PHASE_31_SIGNOFF_PATH, signoff_error),
        (PHASE_31_WINNER_PATH, winner_error),
        (PHASE_31_REFERENCE_PATH, reference_error),
    ):
        if error is not None:
            _add_issue(issues, path, error)

    if signoff is not None:
        if not _is_pass(signoff):
            _add_issue(issues, PHASE_31_SIGNOFF_PATH, "SIGNOFF_NOT_PASS")
        if signoff.get("approved_for_phase32") is not True:
            _add_issue(issues, PHASE_31_SIGNOFF_PATH, "PHASE_32_NOT_APPROVED")
        if not _test_is_locked(signoff.get("test_status")):
            _add_issue(issues, PHASE_31_SIGNOFF_PATH, "TEST_FIREWALL_NOT_CONFIRMED")

    if winner is not None:
        if not _is_pass(winner):
            _add_issue(issues, PHASE_31_WINNER_PATH, "WINNER_NOT_PASS")
        if not _test_is_locked(winner.get("test_status")):
            _add_issue(issues, PHASE_31_WINNER_PATH, "TEST_FIREWALL_NOT_CONFIRMED")

    if reference is not None:
        if reference.get("approved_for_phase32") is not True:
            _add_issue(issues, PHASE_31_REFERENCE_PATH, "PHASE_32_NOT_APPROVED")
        dropout_state = reference.get("dropout_state")
        selected_dropout = _number(reference, "selected_dropout", "dropout")
        if dropout_state not in {None, "DR01_0P1"}:
            _add_issue(issues, PHASE_31_REFERENCE_PATH, "REFERENCE_DROPOUT_MISMATCH")
        if selected_dropout is not None and not math.isclose(
            selected_dropout,
            REFERENCE_DROPOUT,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            _add_issue(issues, PHASE_31_REFERENCE_PATH, "REFERENCE_DROPOUT_MISMATCH")

    selected_weight_decay = None
    winner_run_id = None
    source_config_path = None
    source_config_fingerprint = None
    dropout_scope = None
    if winner is not None and reference is not None:
        winner_run_id = _string(winner, "winner_run_id")
        reference_run_id = _string(reference, "winner_run_id", "current_reference_run_id")
        if winner_run_id is None or reference_run_id is None:
            _add_issue(issues, "winner_run_id", "REFERENCE_RUN_ID_MISSING")
        elif winner_run_id != reference_run_id:
            _add_issue(issues, "winner_run_id", "REFERENCE_RUN_ID_MISMATCH")

        winner_weight_decay = _number(winner, "winner_weight_decay", "weight_decay")
        reference_weight_decay = _number(reference, "selected_weight_decay", "weight_decay")
        if winner_weight_decay is None or reference_weight_decay is None:
            _add_issue(issues, "weight_decay", "SELECTED_WEIGHT_DECAY_MISSING")
        elif not math.isclose(winner_weight_decay, reference_weight_decay, rel_tol=0.0, abs_tol=1e-12):
            _add_issue(issues, "weight_decay", "SELECTED_WEIGHT_DECAY_MISMATCH")
        else:
            selected_weight_decay = winner_weight_decay

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
            source_payload, source_error = _load_object(root, source_config_path)
            if source_error is not None:
                _add_issue(issues, source_config_path, source_error)
            elif source_payload is not None:
                source_config_fingerprint = _string(source_payload, "config_fingerprint")
                expected_fingerprint = _string(reference, "winner_config_fingerprint")
                if source_payload.get("run_id") != winner_run_id:
                    _add_issue(issues, source_config_path, "RUN_ID_MISMATCH")
                if source_config_fingerprint is None or expected_fingerprint is None:
                    _add_issue(issues, source_config_path, "CONFIG_FINGERPRINT_MISSING")
                elif source_config_fingerprint != expected_fingerprint:
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
                            (model.get("dropout"), REFERENCE_DROPOUT, "DROPOUT_DRIFT"),
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
                        actual_weight_decay = training.get("weight_decay")
                        if selected_weight_decay is None or not isinstance(actual_weight_decay, (int, float)):
                            _add_issue(issues, source_config_path, "SELECTED_WEIGHT_DECAY_MISSING")
                        elif not math.isclose(
                            float(actual_weight_decay),
                            selected_weight_decay,
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        ):
                            _add_issue(issues, source_config_path, "SELECTED_WEIGHT_DECAY_MISMATCH")
                        try:
                            dropout_scope = inspect_dropout_scope(model)
                        except (TypeError, ValueError, RuntimeError):
                            _add_issue(issues, source_config_path, "DROPOUT_SCOPE_INVALID")
                        else:
                            if dropout_scope["status"] != "PASS":
                                _add_issue(issues, source_config_path, "DROPOUT_SCOPE_INVALID")

    frozen_configuration = None
    if winner is not None and reference is not None and not issues:
        frozen_configuration = {
            "feature_variant_id": winner["feature_variant_id"],
            "target_scaling_id": winner["target_scaling_id"],
            "lookback_id": winner["lookback_id"],
            "pooling_id": winner["pooling_id"],
            "activation_id": winner["activation_id"],
            "batch_id": winner["batch_id"],
            "learning_rate": _number(winner, "winner_learning_rate", "learning_rate"),
            "weight_decay": selected_weight_decay,
            "population_fingerprint": winner["population_fingerprint"],
            "optimizer": "AdamW",
            "d_model": 64,
            "num_heads": 4,
            "num_layers": 2,
            "ffn_dim": 128,
            "dropout": REFERENCE_DROPOUT,
            "dropout_scope_fingerprint": dropout_scope["scope_fingerprint"],
            "dropout_site_count": dropout_scope["site_count"],
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
            str(PHASE_31_SIGNOFF_PATH),
            str(PHASE_31_WINNER_PATH),
            str(PHASE_31_REFERENCE_PATH),
        ],
        "selected_weight_decay": selected_weight_decay,
        "winner_run_id": winner_run_id,
        "frozen_configuration": frozen_configuration,
        "dropout_scope": dropout_scope,
        "processing_log_observation": {
            "path": str(PHASE_31_LOG_PATH),
            "available": processing_log_error is None,
            "status": processing_log.get("status") if processing_log is not None else processing_log_error,
            "authoritative": False,
        },
    }


def build_phase_32_preflight(project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    decision = plan_phase_resume(PHASE_ID, root)
    handoff = inspect_phase_31_handoff(root)
    mode_semantics = [
        verify_dropout_mode_semantics(condition.dropout_probability)
        for condition in CONDITIONS
    ]
    return {
        "phase_id": PHASE_ID,
        "phase_version": PHASE_VERSION,
        "sweep_id": SWEEP_ID,
        "sweep_version": SWEEP_VERSION,
        "conditions": [asdict(condition) for condition in CONDITIONS],
        "reference_condition_id": REFERENCE_CONDITION_ID,
        "selected_weight_decay": handoff["selected_weight_decay"],
        "test_access": TEST_ACCESS,
        "handoff": handoff,
        "mode_semantics": mode_semantics,
        "decision": decision,
        "ready": handoff["valid"] and decision["readiness"]["ready"] and all(
            item["status"] == "PASS" for item in mode_semantics
        ),
    }


def prepare_phase_32_condition(
    condition_id: str,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    condition_by_id = {condition.condition_id: condition for condition in CONDITIONS}
    if condition_id not in condition_by_id:
        raise ValueError(f"Condition {condition_id} is not registered for Phase 32")
    handoff = inspect_phase_31_handoff(root)
    if not handoff["valid"]:
        raise RuntimeError(f"Phase 32 handoff is invalid: {handoff['issues']}")
    condition = condition_by_id[condition_id]
    if condition.execution_mode == "REUSE_REFERENCE":
        evidence = resolve_phase_conditions(PHASE_ID, root)
        verified = {item["condition_id"]: item for item in evidence["verified_conditions"]}
        if condition_id not in verified:
            raise RuntimeError(f"Phase 32 reference evidence is invalid: {evidence}")
        return {
            "phase_id": PHASE_ID,
            "sweep_id": SWEEP_ID,
            "condition_id": condition.condition_id,
            "execution_mode": condition.execution_mode,
            "dropout_probability": condition.dropout_probability,
            "reference_run_id": handoff["winner_run_id"],
            "reference_evidence": verified[condition_id],
            "frozen_configuration": handoff["frozen_configuration"],
            "dropout_scope": handoff["dropout_scope"],
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
        "dropout_probability": condition.dropout_probability,
        "reference_run_id": handoff["winner_run_id"],
        "frozen_configuration": handoff["frozen_configuration"],
        "dropout_scope": handoff["dropout_scope"],
        "test_access": TEST_ACCESS,
    }
