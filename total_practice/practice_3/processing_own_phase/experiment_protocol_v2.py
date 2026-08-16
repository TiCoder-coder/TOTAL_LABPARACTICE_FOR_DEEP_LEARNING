"""Frozen constants and validation helpers for Practice 3 protocol v2."""

from __future__ import annotations

import hashlib
import json
import platform
from functools import cmp_to_key
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

import torch

from .config import PROJECT_ROOT, RESULT_DIR


PROTOCOL_VERSION = "practice_3_v2.0"
LEARNING_RATES = (2e-5, 3e-5, 5e-5)
RUN_IDS = ("p3v2_lr_2e-5", "p3v2_lr_3e-5", "p3v2_lr_5e-5")
RUN_ID_BY_LR = dict(zip(LEARNING_RATES, RUN_IDS))
MAX_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_THRESHOLD = 1e-6
RANKING_TOLERANCE = 1e-6
SPLIT_SEED = 42
V2_RESULT_DIR = RESULT_DIR / "practice_3_v2"
V2_EXPERIMENTS_DIR = V2_RESULT_DIR / "experiments"
V2_RUNS_DIR = PROJECT_ROOT / "runs" / "practice_3_v2"
PART_2_1_PLAN = (
    PROJECT_ROOT
    / "docs"
    / "plan-doc"
    / "plan_before_process"
    / "part_02_experiment_protocol_v2_plan_2026-08-15.md"
)
PROTOCOL_MANIFEST_PATH = V2_RESULT_DIR / "protocol_manifest.json"

FIXED_VARIABLES = {
    "model": "distilbert-base-uncased",
    "tokenizer": "distilbert-base-uncased",
    "num_labels": 2,
    "label_mapping": {"0": "NEGATIVE", "1": "POSITIVE"},
    "max_length": 80,
    "seed": 42,
    "data_seed": 42,
    "split_seed": SPLIT_SEED,
    "batch_size": 16,
    "eval_batch_size": 32,
    "optimizer": "ADAMW_TORCH_FUSED",
    "scheduler": "linear",
    "warmup_steps": 0,
    "weight_decay": 0.01,
    "gradient_clipping": 1.0,
    "evaluation_strategy": "epoch",
    "logging_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "selection_metric": "validation_loss",
    "greater_is_better": False,
    "max_epochs": MAX_EPOCHS,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "early_stopping_threshold": EARLY_STOPPING_THRESHOLD,
    "ranking_tolerance": RANKING_TOLERANCE,
}


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize stable UTF-8 JSON for hashing and immutable snapshots."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any, *, overwrite: bool = False) -> Path:
    """Write canonical JSON atomically and reject accidental overwrites."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing == payload:
            return path
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing v2 metadata: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)
    if json.loads(path.read_text(encoding="utf-8")) != payload:
        raise RuntimeError(f"JSON read-back failed: {path}")
    return path


def library_versions() -> dict[str, str]:
    names = ("torch", "transformers", "datasets", "numpy", "scikit-learn", "tensorboard")
    result: dict[str, str] = {"python": platform.python_version()}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "NOT_INSTALLED"
    return result


def selected_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def optimizer_compatibility_preflight() -> dict[str, Any]:
    """Check fused AdamW construction only; never call backward or step."""
    device = selected_device()
    try:
        parameter = torch.nn.Parameter(torch.zeros(1, device=device))
        optimizer = torch.optim.AdamW([parameter], lr=LEARNING_RATES[0], fused=True)
        supported = optimizer.defaults.get("fused") is True
        del optimizer, parameter
        error = None
    except Exception as exc:  # environment-specific compatibility evidence
        supported = False
        error = f"{type(exc).__name__}: {exc}"
    return {
        "status": "SUPPORTED" if supported else "BLOCKED",
        "optimizer": "ADAMW_TORCH_FUSED",
        "device": device,
        "torch_version": torch.__version__,
        "construction_only": True,
        "backward_called": False,
        "optimizer_step_called": False,
        "error": error,
    }


def build_run_config(
    learning_rate: float,
    dataset_fingerprint: str,
    split_fingerprint: str,
) -> dict[str, Any]:
    if learning_rate not in RUN_ID_BY_LR:
        raise ValueError(f"Learning rate is not registered: {learning_rate}")
    payload = {
        "run_id": RUN_ID_BY_LR[learning_rate],
        "protocol_version": PROTOCOL_VERSION,
        **FIXED_VARIABLES,
        "learning_rate": learning_rate,
        "dataset_fingerprint": dataset_fingerprint,
        "split_fingerprint": split_fingerprint,
        "tensorboard_log_dir": str(
            (V2_RUNS_DIR / RUN_ID_BY_LR[learning_rate]).relative_to(PROJECT_ROOT)
        ),
        "library_versions": library_versions(),
    }
    payload["config_hash"] = sha256_payload(payload)
    return payload


def validate_run_configs(configs: Iterable[dict[str, Any]]) -> None:
    records = list(configs)
    if len(records) != 3:
        raise ValueError("Exactly three run configurations are required")
    if [record["run_id"] for record in records] != list(RUN_IDS):
        raise ValueError("Run IDs or order do not match the frozen protocol")
    if tuple(float(record["learning_rate"]) for record in records) != LEARNING_RATES:
        raise ValueError("Learning-rate candidates do not match the frozen protocol")
    for record in records:
        supplied_hash = record.get("config_hash")
        body = {key: value for key, value in record.items() if key != "config_hash"}
        if supplied_hash != sha256_payload(body):
            raise ValueError(f"Config hash mismatch for {record.get('run_id')}")
    comparison = [
        {key: value for key, value in record.items() if key not in {"run_id", "learning_rate", "config_hash", "tensorboard_log_dir"}}
        for record in records
    ]
    if not all(item == comparison[0] for item in comparison[1:]):
        raise ValueError("Run configurations differ in fields other than learning rate")
    log_dirs = [record["tensorboard_log_dir"] for record in records]
    if len(set(log_dirs)) != 3:
        raise ValueError("TensorBoard directories must be unique")


def _reject_evaluation_fields(record: dict[str, Any]) -> None:
    prohibited = [
        key for key in record
        if "test" in key.lower() or "holdout" in key.lower()
    ]
    if prohibited:
        raise ValueError(f"Ranking record contains prohibited fields: {prohibited}")


def rank_experiments(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank completed Validation-only records using the frozen tolerance rule."""
    candidates = [dict(record) for record in records]
    required = {
        "run_id", "status", "best_val_loss", "best_val_f1",
        "best_val_accuracy", "best_epoch",
    }
    if not candidates:
        raise ValueError("At least one experiment record is required")
    for record in candidates:
        _reject_evaluation_fields(record)
        if not required.issubset(record):
            raise ValueError(f"Ranking record missing fields: {required - set(record)}")
        if record["status"] != "COMPLETED":
            raise ValueError("Only COMPLETED experiments may be ranked")
        if record["run_id"] not in RUN_IDS:
            raise ValueError("Unknown run_id in ranking record")

    def compare(left: dict[str, Any], right: dict[str, Any]) -> int:
        comparisons = (
            ("best_val_loss", False),
            ("best_val_f1", True),
            ("best_val_accuracy", True),
        )
        for field, higher_is_better in comparisons:
            delta = float(left[field]) - float(right[field])
            if abs(delta) > RANKING_TOLERANCE:
                if higher_is_better:
                    return -1 if delta > 0 else 1
                return -1 if delta < 0 else 1
        epoch_delta = float(left["best_epoch"]) - float(right["best_epoch"])
        if abs(epoch_delta) > RANKING_TOLERANCE:
            return -1 if epoch_delta < 0 else 1
        return -1 if left["run_id"] < right["run_id"] else (1 if left["run_id"] > right["run_id"] else 0)

    return sorted(candidates, key=cmp_to_key(compare))


def build_protocol_manifest(
    optimizer_preflight: dict[str, Any],
    part_2_1_plan_hash: str,
    dataset_split_manifest_hash: str,
) -> dict[str, Any]:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "NOT_STARTED",
        "part_2_1_plan_sha256": part_2_1_plan_hash,
        "dataset_split_manifest_hash": dataset_split_manifest_hash,
        "learning_rates": list(LEARNING_RATES),
        "run_ids": list(RUN_IDS),
        "max_epochs": MAX_EPOCHS,
        "early_stopping": {
            "patience": EARLY_STOPPING_PATIENCE,
            "threshold": EARLY_STOPPING_THRESHOLD,
            "metric": "validation_loss",
            "greater_is_better": False,
        },
        "ranking": {
            "primary": "validation_loss",
            "secondary": "validation_f1",
            "tertiary": "validation_accuracy",
            "fourth": "earlier_best_epoch",
            "final_deterministic_fallback": "lexical_run_id",
            "tolerance": RANKING_TOLERANCE,
        },
        "fixed_variables": FIXED_VARIABLES,
        "tensorboard": {
            "root": str(V2_RUNS_DIR.relative_to(PROJECT_ROOT)),
            "required_scalars": [
                "train_loss", "validation_loss", "validation_accuracy",
                "validation_f1", "learning_rate", "epoch", "global_step",
            ],
            "event_files_created": False,
        },
        "optimizer_preflight": optimizer_preflight,
        "winner_status": "NOT_SELECTED",
        "holdout_status": "SEALED",
        "holdout_evaluation_count": 0,
        "training_performed": False,
        "trainer_created": False,
        "v1_test_accessed": False,
        "v2_holdout_evaluated": False,
    }
    payload["protocol_manifest_hash"] = sha256_payload(payload)
    return payload


def validate_protocol_manifest(manifest: dict[str, Any]) -> None:
    supplied_hash = manifest.get("protocol_manifest_hash")
    body = {key: value for key, value in manifest.items() if key != "protocol_manifest_hash"}
    checks = (
        manifest.get("protocol_version") == PROTOCOL_VERSION,
        manifest.get("status") == "NOT_STARTED",
        tuple(manifest.get("learning_rates", ())) == LEARNING_RATES,
        manifest.get("max_epochs") == 10,
        manifest.get("early_stopping", {}).get("patience") == 2,
        manifest.get("early_stopping", {}).get("threshold") == 1e-6,
        manifest.get("ranking", {}).get("tolerance") == 1e-6,
        manifest.get("winner_status") == "NOT_SELECTED",
        manifest.get("holdout_status") == "SEALED",
        manifest.get("holdout_evaluation_count") == 0,
        supplied_hash == sha256_payload(body),
    )
    if not all(checks):
        raise ValueError("Protocol manifest violates the frozen Part 2.1 contract")
