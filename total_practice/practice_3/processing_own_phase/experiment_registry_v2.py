"""Experiment registry and immutable run-config infrastructure for Practice 3 v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from filelock import FileLock

from .experiment_protocol_v2 import (
    LEARNING_RATES,
    PROTOCOL_VERSION,
    RUN_IDS,
    V2_EXPERIMENTS_DIR,
    V2_RUNS_DIR,
    atomic_write_json,
    build_run_config,
    validate_run_configs,
)


REGISTRY_PATH_NAME = "experiment_registry.json"
ALLOWED_TRANSITIONS = {
    "PLANNED": {"RUNNING"},
    "RUNNING": {"COMPLETED", "FAILED", "INVALID"},
    "FAILED": set(),
    "INVALID": set(),
    "COMPLETED": set(),
}
TRAINING_HISTORY_FIELDS = (
    "epoch",
    "global_step",
    "train_loss",
    "eval_loss",
    "eval_accuracy",
    "eval_precision",
    "eval_recall",
    "eval_f1",
    "learning_rate",
)


def _registry_record(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": config["run_id"],
        "protocol_version": PROTOCOL_VERSION,
        "learning_rate": config["learning_rate"],
        "seed": config["seed"],
        "max_epochs": config["max_epochs"],
        "early_stopping_patience": config["early_stopping_patience"],
        "early_stopping_threshold": config["early_stopping_threshold"],
        "stopped_epoch": None,
        "best_epoch": None,
        "best_val_loss": None,
        "best_val_accuracy": None,
        "best_val_precision": None,
        "best_val_recall": None,
        "best_val_f1": None,
        "best_checkpoint": None,
        "runtime_seconds": None,
        "tensorboard_log_dir": config["tensorboard_log_dir"],
        "config_snapshot": str(
            (V2_EXPERIMENTS_DIR / config["run_id"] / "run_config.json")
        ),
        "config_hash": config["config_hash"],
        "dataset_fingerprint": config["dataset_fingerprint"],
        "split_fingerprint": config["split_fingerprint"],
        "status": "PLANNED",
    }


def validate_registry(
    registry: dict[str, Any],
    configs: Iterable[dict[str, Any]],
    *,
    require_planned: bool = False,
) -> None:
    config_list = list(configs)
    validate_run_configs(config_list)
    records = registry.get("runs", [])
    if registry.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Registry protocol version mismatch")
    if len(records) != 3 or len({record.get("run_id") for record in records}) != 3:
        raise ValueError("Registry must contain three unique run IDs")
    config_by_id = {record["run_id"]: record for record in config_list}
    for record in records:
        run_id = record.get("run_id")
        if run_id not in RUN_IDS:
            raise ValueError(f"Unknown registry run ID: {run_id}")
        if float(record.get("learning_rate")) not in LEARNING_RATES:
            raise ValueError(f"Invalid registry learning rate: {record.get('learning_rate')}")
        if record.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("Run protocol version mismatch")
        if record.get("config_hash") != config_by_id[run_id]["config_hash"]:
            raise ValueError("Registry config hash mismatch")
        if record.get("status") not in ALLOWED_TRANSITIONS:
            raise ValueError("Invalid registry status")
        if require_planned and record.get("status") != "PLANNED":
            raise ValueError("Part 2.2 registry records must remain PLANNED")
        if record.get("status") == "COMPLETED":
            required_results = (
                "stopped_epoch", "best_epoch", "best_val_loss",
                "best_val_accuracy", "best_val_precision", "best_val_recall",
                "best_val_f1", "best_checkpoint", "runtime_seconds",
            )
            if any(record.get(field) is None for field in required_results):
                raise ValueError("COMPLETED registry record has incomplete artifacts")
    if [record["run_id"] for record in records] != list(RUN_IDS):
        raise ValueError("Registry order must follow deterministic run IDs")


def initialize_registry(
    result_dir: Path,
    dataset_fingerprint: str,
    split_fingerprint: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    configs = [
        build_run_config(rate, dataset_fingerprint, split_fingerprint)
        for rate in LEARNING_RATES
    ]
    validate_run_configs(configs)
    for config in configs:
        config_path = result_dir / "experiments" / config["run_id"] / "run_config.json"
        atomic_write_json(config_path, config)
        (V2_RUNS_DIR / config["run_id"]).mkdir(parents=True, exist_ok=True)
    registry = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "NOT_STARTED",
        "run_count": 3,
        "allowed_lifecycle": {
            "normal": ["PLANNED", "RUNNING", "COMPLETED"],
            "failure": ["RUNNING", "FAILED"],
            "invalid": ["RUNNING", "INVALID"],
        },
        "training_history_schema": list(TRAINING_HISTORY_FIELDS),
        "runs": [_registry_record(config) for config in configs],
        "real_experiments_executed": False,
        "synthetic_results_in_registry": False,
    }
    # Store result-local config paths rather than depending on the repository absolute path.
    for record in registry["runs"]:
        record["config_snapshot"] = (
            f"experiments/{record['run_id']}/run_config.json"
        )
    validate_registry(registry, configs, require_planned=True)
    atomic_write_json(result_dir / REGISTRY_PATH_NAME, registry)
    return configs, registry


def transition_run(
    registry: dict[str, Any],
    run_id: str,
    new_status: str,
    *,
    config_hash: str,
    result_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply a guarded lifecycle transition to an in-memory/synthetic registry."""
    copied = json.loads(json.dumps(registry))
    matches = [record for record in copied["runs"] if record["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError("Run ID is absent or duplicated")
    record = matches[0]
    if record["config_hash"] != config_hash:
        raise ValueError("Config hash mismatch")
    if new_status not in ALLOWED_TRANSITIONS.get(record["status"], set()):
        raise ValueError(f"Invalid lifecycle transition {record['status']} -> {new_status}")
    if record["status"] == "COMPLETED":
        raise ValueError("A completed run cannot be overwritten")
    record["status"] = new_status
    if result_fields:
        prohibited = [key for key in result_fields if "test" in key.lower() or "holdout" in key.lower()]
        if prohibited:
            raise ValueError(f"Evaluation-only fields are prohibited: {prohibited}")
        record.update(result_fields)
    return copied


def persist_run_transition(
    registry_path: Path,
    configs: Iterable[dict[str, Any]],
    run_id: str,
    new_status: str,
    *,
    config_hash: str,
    expected_status: str,
    result_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Lock, validate, transition, and atomically persist the real registry."""
    config_list = list(configs)
    lock = FileLock(str(registry_path) + ".lock")
    with lock:
        if not registry_path.is_file():
            raise FileNotFoundError(registry_path)
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        validate_registry(registry, config_list)
        records = [item for item in registry["runs"] if item["run_id"] == run_id]
        if len(records) != 1 or records[0]["status"] != expected_status:
            observed = records[0]["status"] if len(records) == 1 else "MISSING_OR_DUPLICATED"
            raise RuntimeError(
                f"Registry state changed concurrently: expected {expected_status}, got {observed}"
            )
        updated = transition_run(
            registry,
            run_id,
            new_status,
            config_hash=config_hash,
            result_fields=result_fields,
        )
        updated["real_experiments_executed"] = any(
            item["status"] in {"RUNNING", "COMPLETED", "FAILED", "INVALID"}
            for item in updated["runs"]
        )
        updated["status"] = (
            "COMPLETED" if all(item["status"] == "COMPLETED" for item in updated["runs"])
            else "IN_PROGRESS"
        )
        validate_registry(updated, config_list)
        atomic_write_json(registry_path, updated, overwrite=True)
        return updated


def completed_run_action(record: dict[str, Any], config_hash: str, artifacts_valid: bool) -> str:
    """Return LOAD for a valid completed run; never silently retrain it."""
    if record.get("status") != "COMPLETED":
        return "RUN"
    if record.get("config_hash") == config_hash and artifacts_valid:
        return "LOAD"
    return "INVALID"
