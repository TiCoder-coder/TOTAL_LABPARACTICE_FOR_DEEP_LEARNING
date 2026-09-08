"""Phase 43 LT1-LT5 tuning space.

Hard-coded candidate spaces per Phase 43 plan section 14 and 19:

LT1 hidden_size: [32, 64, 128]
LT2 num_layers: [1, 2]
LT3 dropout: [0.0, 0.1, 0.2] applicable only when num_layers >= 2
LT4 learning_rate: [1e-4, 3e-4, 1e-3]
LT5 weight_decay: [0, 1e-4, 1e-3]

Fixed contract: MSE, E50, P10, GC1, scheduler None, warmup None, seed42.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


LT1_FACTOR = "model.hidden_size"
LT2_FACTOR = "model.num_layers"
LT3_FACTOR = "model.dropout"
LT4_FACTOR = "training.learning_rate"
LT5_FACTOR = "training.weight_decay"

LT1_VALUES = (32, 64, 128)
LT2_VALUES = (1, 2)
LT3_VALUES = (0.0, 0.1, 0.2)
LT4_VALUES = (1e-4, 3e-4, 1e-3)
LT5_VALUES = (0.0, 1e-4, 1e-3)

LT3_APPLICABLE_IF_NUM_LAYERS_GTE_2 = True

LT_STAGE_ORDER = ("LT1", "LT2", "LT3", "LT4", "LT5")

REFERENCE_HYPERPARAMETERS = {
    "model.hidden_size": 64,
    "model.num_layers": 2,
    "model.dropout": 0.1,
    "training.learning_rate": 3e-4,
    "training.weight_decay": 1e-4,
}

FIXED_TRAINING_CONTRACT = {
    "training.optimizer_name": "AdamW",
    "training.loss_name": "MSE",
    "training.max_epochs": 50,
    "training.early_stopping_patience": 10,
    "training.early_stopping_enabled": True,
    "training.gradient_clip_max_norm": 1.0,
    "training.gradient_clipping_enabled": True,
    "training.scheduler_name": None,
    "training.scheduler_config": None,
    "training.revin_enabled": False,
    "reproducibility.seed": 42,
}

MAX_FRESH_SCIENTIFIC_RUNS = 10


def _set_path(d: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = d
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _override_path(d: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = d
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _drop_field(d: dict[str, Any], path: str) -> None:
    parts = path.split(".")
    cursor = d
    for part in parts[:-1]:
        if not isinstance(cursor, dict) or part not in cursor:
            return
        cursor = cursor[part]
    if isinstance(cursor, dict):
        cursor.pop(parts[-1], None)


def _candidate_code(stage: str, value: Any) -> str:
    if stage == "LT1":
        return f"LH{int(value)}"
    if stage == "LT2":
        return f"LN{int(value)}"
    if stage == "LT3":
        return f"LD{int(round(float(value) * 10))}"
    if stage == "LT4":
        if abs(float(value) - 1e-4) < 1e-12:
            return "LLR1"
        if abs(float(value) - 3e-4) < 1e-12:
            return "LLR2"
        if abs(float(value) - 1e-3) < 1e-12:
            return "LLR3"
        raise ValueError(f"LT4 LR not in registered space: {value!r}")
    if stage == "LT5":
        if float(value) == 0.0:
            return "LWD0"
        if abs(float(value) - 1e-4) < 1e-12:
            return "LWD1"
        if abs(float(value) - 1e-3) < 1e-12:
            return "LWD2"
        raise ValueError(f"LT5 WD not in registered space: {value!r}")
    raise ValueError(f"Unknown stage {stage!r}")


def build_stage_candidates(stage: str, current_winner_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if stage == "LT1":
        values = LT1_VALUES
        factor = LT1_FACTOR
    elif stage == "LT2":
        values = LT2_VALUES
        factor = LT2_FACTOR
    elif stage == "LT3":
        values = LT3_VALUES
        factor = LT3_FACTOR
    elif stage == "LT4":
        values = LT4_VALUES
        factor = LT4_FACTOR
    elif stage == "LT5":
        values = LT5_VALUES
        factor = LT5_FACTOR
    else:
        raise ValueError(f"Unknown stage {stage!r}")

    candidates: dict[str, dict[str, Any]] = {}
    for value in values:
        cfg = deepcopy(current_winner_config)
        _set_path(cfg, factor, value)
        candidates[_candidate_code(stage, value)] = cfg
    return candidates


def stage_factor(stage: str) -> str:
    if stage == "LT1":
        return LT1_FACTOR
    if stage == "LT2":
        return LT2_FACTOR
    if stage == "LT3":
        return LT3_FACTOR
    if stage == "LT4":
        return LT4_FACTOR
    if stage == "LT5":
        return LT5_FACTOR
    raise ValueError(f"Unknown stage {stage!r}")


def stage_values(stage: str) -> tuple[Any, ...]:
    if stage == "LT1":
        return LT1_VALUES
    if stage == "LT2":
        return LT2_VALUES
    if stage == "LT3":
        return LT3_VALUES
    if stage == "LT4":
        return LT4_VALUES
    if stage == "LT5":
        return LT5_VALUES
    raise ValueError(f"Unknown stage {stage!r}")


def lt3_applicable(num_layers: int) -> bool:
    return bool(LT3_APPLICABLE_IF_NUM_LAYERS_GTE_2 and num_layers >= 2)


def reference_value(stage: str) -> Any:
    return REFERENCE_HYPERPARAMETERS[stage_factor(stage)]


def assert_reference_included(stage: str, candidates: dict[str, dict[str, Any]]) -> None:
    ref_value = reference_value(stage)
    expected_code = _candidate_code(stage, ref_value)
    if expected_code not in candidates:
        raise ValueError(
            f"Stage {stage} missing current reference candidate {expected_code} (value={ref_value})"
        )


def candidate_differs_only_one_factor(stage: str, candidate_config: dict[str, Any], reference_config: dict[str, Any]) -> bool:
    factor = stage_factor(stage)
    candidate_view = deepcopy(candidate_config)
    reference_view = deepcopy(reference_config)
    _override_path(candidate_view, factor, _get_path(reference_view, factor))
    return candidate_view == reference_view


def _get_path(d: dict[str, Any], path: str) -> Any:
    parts = path.split(".")
    cursor = d
    for part in parts:
        cursor = cursor.get(part) if isinstance(cursor, dict) else None
        if cursor is None:
            return None
    return cursor


def candidate_value(stage: str, candidate_config: dict[str, Any]) -> Any:
    return _get_path(candidate_config, stage_factor(stage))


def normalise_dropout_arg(raw_value: Any) -> float:
    return float(raw_value)
