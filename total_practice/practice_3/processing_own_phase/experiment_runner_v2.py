"""Reusable, non-training helpers for the Practice 3 v2 runners.

The original module was referenced by the v2.3 runner and tests but was absent
from source control.  This repair intentionally contains no Trainer creation,
training loop, optimizer step, or artifact-writing orchestration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .experiment_protocol_v2 import RANKING_TOLERANCE, RUN_IDS, V2_RESULT_DIR


HISTORY_FIELDS = (
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


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def collect_training_history(log_history: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Merge Hugging Face train/eval log rows into one record per evaluation.

    Trainer normally emits a training row followed by an evaluation row at the
    same epoch/global step. Only metrics actually present in those rows are
    copied. Rows are returned in the original evaluation order.
    """
    pending_train: dict[tuple[Any, Any], dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for raw in log_history:
        if not isinstance(raw, Mapping):
            continue
        epoch = raw.get("epoch")
        step = raw.get("step", raw.get("global_step"))
        key = (epoch, step)
        if "loss" in raw and "eval_loss" not in raw:
            train: dict[str, Any] = {}
            if _finite_number(raw.get("loss")):
                train["train_loss"] = raw["loss"]
            if _finite_number(raw.get("learning_rate")):
                train["learning_rate"] = raw["learning_rate"]
            pending_train[key] = train
            continue
        if "eval_loss" not in raw:
            continue
        record: dict[str, Any] = {}
        if _finite_number(epoch):
            record["epoch"] = epoch
        if _finite_number(step):
            record["global_step"] = step
        record.update(pending_train.get(key, {}))
        mapping = {
            "eval_loss": "eval_loss",
            "eval_accuracy": "eval_accuracy",
            "eval_precision": "eval_precision",
            "eval_recall": "eval_recall",
            "eval_f1": "eval_f1",
            "learning_rate": "learning_rate",
        }
        for source, target in mapping.items():
            if _finite_number(raw.get(source)):
                record[target] = raw[source]
        records.append({field: record[field] for field in HISTORY_FIELDS if field in record})
    return records


def select_best_epoch(history: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Select by Validation loss with the frozen v2 tolerance/tie policy."""
    candidates = [dict(row) for row in history if _finite_number(row.get("eval_loss"))]
    if not candidates:
        raise ValueError("Training history contains no Validation-loss records")

    def better(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        loss_delta = float(left["eval_loss"]) - float(right["eval_loss"])
        if abs(loss_delta) > RANKING_TOLERANCE:
            return loss_delta < 0
        for field in ("eval_f1", "eval_accuracy"):
            left_value = float(left[field]) if _finite_number(left.get(field)) else -math.inf
            right_value = float(right[field]) if _finite_number(right.get(field)) else -math.inf
            delta = left_value - right_value
            if abs(delta) > RANKING_TOLERANCE:
                return delta > 0
        left_epoch = float(left["epoch"]) if _finite_number(left.get("epoch")) else math.inf
        right_epoch = float(right["epoch"]) if _finite_number(right.get("epoch")) else math.inf
        if abs(left_epoch - right_epoch) > RANKING_TOLERANCE:
            return left_epoch < right_epoch
        left_step = float(left["global_step"]) if _finite_number(left.get("global_step")) else math.inf
        right_step = float(right["global_step"]) if _finite_number(right.get("global_step")) else math.inf
        return left_step < right_step

    best = candidates[0]
    for candidate in candidates[1:]:
        if better(candidate, best):
            best = candidate
    return best


def checkpoint_fingerprint(checkpoint: str | Path) -> dict[str, Any]:
    """Hash the checkpoint weight file without loading or modifying it."""
    directory = Path(checkpoint).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {directory}")
    weight_path = None
    for filename in ("model.safetensors", "pytorch_model.bin"):
        candidate = directory / filename
        if candidate.is_file():
            weight_path = candidate
            break
    if weight_path is None:
        raise FileNotFoundError(f"Checkpoint weight file is missing: {directory}")
    digest = hashlib.sha256()
    with weight_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "checkpoint_weight_file": weight_path.name,
        "checkpoint_hash": digest.hexdigest(),
    }


def build_parser() -> argparse.ArgumentParser:
    """Return the frozen legacy-v2 CLI parser; parsing never starts training."""
    parser = argparse.ArgumentParser(description="Run one authorized Practice 3 v2 experiment")
    parser.add_argument("--run-id", required=True, choices=RUN_IDS)
    return parser


def preflight_run(run_id: str) -> dict[str, Any]:
    """Read the legacy registry and reject every run that is not PLANNED.

    The executed `p3v2_lr_2e-5` evidence predates the retained v2.3 artifacts.
    If its legacy registry snapshot is unavailable, fail closed and preserve it
    as completed evidence rather than authorizing an accidental rerun.
    """
    if run_id not in RUN_IDS:
        raise ValueError(f"Unknown run ID: {run_id}")
    registry_path = V2_RESULT_DIR / "experiment_registry.json"
    if registry_path.is_file():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        records = registry.get("runs", [])
        matches = [record for record in records if record.get("run_id") == run_id]
        if len(matches) != 1:
            raise RuntimeError(f"Run registry record is missing or duplicated: {run_id}")
        record = dict(matches[0])
    elif run_id == "p3v2_lr_2e-5":
        record = {"run_id": run_id, "status": "COMPLETED"}
    else:
        raise FileNotFoundError(f"Legacy v2 registry is unavailable: {registry_path}")
    if record.get("status") != "PLANNED":
        raise RuntimeError(f"Run must be PLANNED, got {record.get('status')}")
    return record


__all__ = (
    "build_parser",
    "checkpoint_fingerprint",
    "collect_training_history",
    "preflight_run",
    "select_best_epoch",
)
