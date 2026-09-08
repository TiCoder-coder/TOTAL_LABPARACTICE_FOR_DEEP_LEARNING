"""LSTM_T0_REF resolution.

Phase 20 LSTM_B0 may be reused as the Phase 43 reference only when its
config and data context exactly match the canonical Phase 42/43 shared
contract. Otherwise a fresh LSTM_T0_REF must be trained.

Exact-match fields required:
    feature_variant_id
    target_scaling_id
    lookback_steps
    boundary_protocol
    window_population_version
    batch_size
    hidden_size == 64
    num_layers == 2
    dropout == 0.1
    optimizer_name == AdamW
    learning_rate == 3e-4
    weight_decay == 1e-4
    loss_name == MSE
    max_epochs == 50
    patience == 10
    gradient_clip_max_norm == 1.0
    seed == 42
    training_engine_version (literal 'TRAINING_ENGINE-v1')
    metric_version
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

from course_work.experiments.registry import (
    ExperimentRegistry,
    load_upstream_context,
)


REFERENCE_FIELDS_TO_COMPARE = (
    "data.feature_variant_id",
    "data.target_scaling_option",
    "data.lookback_steps",
    "data.boundary_protocol",
    "training.batch_size",
    "model.hidden_size",
    "model.num_layers",
    "model.dropout",
    "training.optimizer_name",
    "training.learning_rate",
    "training.weight_decay",
    "training.loss_name",
    "training.max_epochs",
    "training.patience",
    "training.gradient_clip_max_norm",
    "reproducibility.seed",
)


@dataclass(frozen=True)
class ReferenceResolution:
    source: str
    reference_run_id: str | None
    phase20_run_id: str | None
    phase20_exact_match: bool
    mismatch_fields: list[str] = field(default_factory=list)
    planned_status: str = "PLANNED_FRESH_REFERENCE"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _set_path(d: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = d
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _get_path(d: dict[str, Any], path: str) -> Any:
    parts = path.split(".")
    cursor = d
    for part in parts:
        cursor = cursor.get(part) if isinstance(cursor, dict) else None
        if cursor is None:
            return None
    return cursor


def find_phase20_lstm_b0_run(registry: ExperimentRegistry) -> dict[str, Any] | None:
    candidates = []
    for record in registry._load_records():
        if record.get("experiment_family") == "LSTM_BASELINE" and record.get("status") == "COMPLETED":
            config = record.get("config", {})
            model = config.get("model", {})
            if model.get("model_family") == "LSTM":
                candidates.append(record)
    if not candidates:
        return None
    candidates.sort(key=lambda r: r.get("completed_at") or r.get("created_at") or "")
    return candidates[-1]


def _reference_comparable_config(
    registry_config: dict[str, Any],
    boundary_protocol_value: str,
    pop_fingerprint: str,
    metric_version: str,
) -> dict[str, Any]:
    cfg = {
        "data": {},
        "training": {},
        "model": {},
        "reproducibility": {},
        "lineage": {
            "population_fingerprint": pop_fingerprint,
            "metric_version": metric_version,
        },
        "runtime": {
            "training_engine_version": "TRAINING_ENGINE-v1",
        },
    }
    cfg["data"]["boundary_protocol"] = boundary_protocol_value
    src_data = registry_config.get("data", {})
    cfg["data"]["feature_variant_id"] = src_data.get("feature_variant_id")
    cfg["data"]["target_scaling_option"] = src_data.get("target_scaling_option")
    cfg["data"]["lookback_steps"] = src_data.get("lookback_steps")
    src_training = registry_config.get("training", {})
    cfg["training"]["batch_size"] = src_training.get("batch_size")
    cfg["training"]["optimizer_name"] = src_training.get("optimizer_name")
    cfg["training"]["learning_rate"] = src_training.get("learning_rate")
    cfg["training"]["weight_decay"] = src_training.get("weight_decay")
    cfg["training"]["loss_name"] = src_training.get("loss_name")
    cfg["training"]["max_epochs"] = src_training.get("max_epochs")
    cfg["training"]["patience"] = src_training.get("early_stopping_patience", src_training.get("patience"))
    cfg["training"]["gradient_clip_max_norm"] = src_training.get("gradient_clip_max_norm")
    src_model = registry_config.get("model", {})
    cfg["model"]["hidden_size"] = src_model.get("hidden_size")
    cfg["model"]["num_layers"] = src_model.get("num_layers")
    cfg["model"]["dropout"] = src_model.get("dropout")
    src_repro = registry_config.get("reproducibility", {})
    cfg["reproducibility"]["seed"] = src_repro.get("seed")
    return cfg


def _build_expected_reference_config(shared_contract, environment: dict[str, Any]) -> dict[str, Any]:
    cfg = {
        "data": {
            "feature_variant_id": shared_contract.feature_variant_id,
            "target_scaling_option": shared_contract.target_scaling_id,
            "lookback_steps": shared_contract.lookback_steps,
            "boundary_protocol": shared_contract.boundary_protocol,
        },
        "training": {
            "batch_size": shared_contract.batch_size,
            "optimizer_name": "AdamW",
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "loss_name": "MSE",
            "max_epochs": 50,
            "patience": 10,
            "gradient_clip_max_norm": 1.0,
        },
        "model": {
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.1,
        },
        "reproducibility": {
            "seed": environment.get("development_seed", 42),
        },
        "lineage": {
            "population_fingerprint": shared_contract.population_fingerprint,
            "metric_version": shared_contract.metric_version,
        },
        "runtime": {
            "training_engine_version": "TRAINING_ENGINE-v1",
        },
    }
    return cfg


def evaluate_phase20_exact_match(
    registry_config: dict[str, Any],
    expected: dict[str, Any],
) -> tuple[bool, list[str]]:
    expected_comparable = _reference_comparable_config(
        expected,
        expected["data"]["boundary_protocol"],
        expected["lineage"]["population_fingerprint"],
        expected["lineage"]["metric_version"],
    )
    registry_comparable = _reference_comparable_config(
        registry_config,
        expected["data"]["boundary_protocol"],
        expected["lineage"]["population_fingerprint"],
        expected["lineage"]["metric_version"],
    )
    mismatches: list[str] = []
    for field in REFERENCE_FIELDS_TO_COMPARE:
        expected_value = _get_path(expected_comparable, field)
        registry_value = _get_path(registry_comparable, field)
        if expected_value != registry_value:
            mismatches.append(field)
    expected_engine = expected["runtime"]["training_engine_version"]
    registry_engine = registry_comparable.get("runtime", {}).get("training_engine_version", "TRAINING_ENGINE-v1")
    if registry_engine != expected_engine:
        mismatches.append("runtime.training_engine_version")
    return (not mismatches), mismatches


def resolve_lstm_t0_reference(
    registry: ExperimentRegistry,
    shared_contract,
) -> ReferenceResolution:
    context = load_upstream_context(registry.project_root)
    environment = context["environment"]
    expected_reference = _build_expected_reference_config(shared_contract, environment)

    phase20_record = find_phase20_lstm_b0_run(registry)
    if phase20_record is None:
        return ReferenceResolution(
            source="FRESH_PHASE43",
            reference_run_id=None,
            phase20_run_id=None,
            phase20_exact_match=False,
            mismatch_fields=[],
            planned_status="PLANNED_FRESH_REFERENCE",
        )

    registry_config = phase20_record.get("config", {})
    exact_match, mismatches = evaluate_phase20_exact_match(registry_config, expected_reference)
    if not exact_match:
        return ReferenceResolution(
            source="FRESH_PHASE43",
            reference_run_id=None,
            phase20_run_id=phase20_record["run_id"],
            phase20_exact_match=False,
            mismatch_fields=mismatches,
            planned_status="PLANNED_FRESH_REFERENCE",
        )
    return ReferenceResolution(
        source="REUSED_PHASE20",
        reference_run_id=phase20_record["run_id"],
        phase20_run_id=phase20_record["run_id"],
        phase20_exact_match=True,
        mismatch_fields=[],
        planned_status="PLANNED_REUSED_REFERENCE",
    )


def build_expected_reference_config_from_contract(shared_contract, environment: dict[str, Any]) -> dict[str, Any]:
    return _build_expected_reference_config(shared_contract, environment)
