"""Aggregate saved Practice 2 artifacts for visualization only.

This module never imports or invokes training, model, data-loading, or final-test
code. Historical artifacts are read-only inputs; only the requested aggregate
JSON file is written.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_metrics_jsonl(path: Path, warnings: Optional[list[str]] = None) -> list[dict[str, Any]]:
    """Read metric objects, tolerating only an incomplete final line."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1:
                if warnings is not None:
                    warnings.append("Ignored an incomplete final metrics.jsonl row")
                break
            raise ValueError(f"Malformed metrics.jsonl row {index + 1}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"metrics.jsonl row {index + 1} is not an object")
        rows.append(row)
    return rows


def _relative(path: Optional[Path], root: Path) -> Optional[str]:
    if path is None or not path.exists():
        return None
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _device_from_log(log_path: Optional[Path]) -> Optional[str]:
    if log_path is None or not log_path.is_file():
        return None
    match = re.search(
        r"Using device:\s*([^\s|]+)",
        log_path.read_text(encoding="utf-8", errors="replace"),
        flags=re.IGNORECASE,
    )
    return match.group(1).lower() if match else None


def _selected_run(root: Path, selection: Mapping[str, Any]) -> Optional[Path]:
    run_id = selection.get("selected_experiment")
    if run_id and (root / "runs" / str(run_id)).is_dir():
        return root / "runs" / str(run_id)
    checkpoint = selection.get("selected_checkpoint")
    if checkpoint:
        candidate = Path(str(checkpoint)).expanduser().parent
        if candidate.is_dir():
            return candidate
    return None


def _compact_checkpoints(run_dir: Optional[Path], manifest: Mapping[str, Any], root: Path) -> dict[str, Any]:
    records = manifest.get("checkpoints", {}) if isinstance(manifest, Mapping) else {}
    result: dict[str, Any] = {}
    aliases = {
        "latest": "latest.pt",
        "best": "best.pt",
        "best_val_loss": "best_val_loss.pt",
        "best_val_accuracy": "best_val_accuracy.pt",
    }
    for alias, filename in aliases.items():
        path = run_dir / filename if run_dir else None
        detail = records.get(filename, {}) if isinstance(records, Mapping) else {}
        result[alias] = {
            "exists": bool(path and path.is_file()),
            "path": _relative(path, root),
            "epoch": detail.get("epoch"),
            "val_loss": detail.get("val_loss"),
            "val_accuracy": detail.get("val_acc"),
            "size_bytes": detail.get("bytes"),
            "sha256": detail.get("sha256"),
        }
    return result


def build_visualization_data(
    project_root: str | Path,
    output_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Build and atomically save one presentation-ready JSON document."""
    root = Path(project_root).expanduser().resolve()
    outputs = root / "outputs"
    locked_path = outputs / "hyperparameter_selection_locked.json"
    fallback_path = outputs / "hyperparameter_selection.json"
    selection_path = locked_path if locked_path.is_file() else fallback_path
    selection = _json(selection_path)
    run_dir = _selected_run(root, selection)
    run_id = run_dir.name if run_dir else selection.get("selected_experiment")
    metrics_path = run_dir / "metrics.jsonl" if run_dir else None
    summary_path = None
    if run_dir:
        modern = run_dir / "summary.json"
        legacy = run_dir / f"{run_dir.name}_summary.json"
        summary_path = modern if modern.is_file() else legacy
    manifest_path = run_dir / "checkpoint_manifest.json" if run_dir else None
    logs = sorted(run_dir.glob("*.log")) if run_dir else []
    log_path = logs[0] if logs else None
    ranking_path = outputs / "hyperparameter_ranking.csv"
    final_summary_path = outputs / "summary.json"
    confusion_path = outputs / "confusion_matrix.csv"
    report_path = outputs / "classification_report.csv"
    predictions_path = outputs / "predictions.csv"

    metrics = read_metrics_jsonl(metrics_path) if metrics_path else []
    summary = _json(summary_path) if summary_path else {}
    metadata = summary.get("metadata", {}) if isinstance(summary.get("metadata"), dict) else {}
    config = summary.get("config", {}) if isinstance(summary.get("config"), dict) else {}
    manifest = _json(manifest_path) if manifest_path else {}
    final_summary = _json(final_summary_path)
    ranking = _csv(ranking_path)
    confusion = _csv(confusion_path)
    classification = _csv(report_path)
    predictions = _csv(predictions_path)

    history = []
    for row in metrics:
        rates = row.get("learning_rates", {})
        history.append({
            "epoch": row.get("epoch"),
            "train_loss": row.get("train_loss"),
            "val_loss": row.get("val_loss"),
            "train_accuracy": row.get("train_accuracy"),
            "val_accuracy": row.get("val_accuracy"),
            "generalization_gap": row.get("generalization_gap"),
            "head_lr": rates.get("head") if isinstance(rates, dict) else None,
            "backbone_lr": rates.get("backbone") if isinstance(rates, dict) else None,
            "epoch_seconds": row.get("epoch_seconds"),
        })

    duration = metadata.get("training_time")
    if duration is None and history:
        values = [row["epoch_seconds"] for row in history if isinstance(row.get("epoch_seconds"), (int, float))]
        duration = sum(values) if values else None
    device = final_summary.get("device") or metadata.get("device") or config.get("device")
    if not device:
        device = _device_from_log(log_path)
    selected_errors = [row for row in predictions if str(row.get("is_correct", "")).lower() in {"false", "0"}]
    selected_errors.sort(key=lambda row: float(row.get("confidence", 0) or 0), reverse=True)

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selected_run": {
            "run_id": run_id,
            "experiment_id": metadata.get("exp_id") or run_id,
            "model": metadata.get("model_name") or config.get("model_name") or final_summary.get("model_name"),
            "training_mode": metadata.get("freeze_strategy") or config.get("training_mode") or final_summary.get("training_mode"),
            "device": device,
            "total_epochs": metadata.get("epochs_trained") or len(history),
            "best_epoch": selection.get("best_epoch") or metadata.get("best_epoch"),
            "best_val_loss": selection.get("best_val_loss") or metadata.get("best_val_loss"),
            "best_val_accuracy": selection.get("best_val_accuracy") or metadata.get("best_val_acc"),
            "training_duration": duration,
        },
        "training_history": history,
        "hyperparameter_search": {
            "selected_experiment": selection.get("selected_experiment"),
            "selection_metric": selection.get("selection_metric"),
            "selection_source": selection.get("selection_source"),
            "ranking": ranking,
        },
        "checkpoints": _compact_checkpoints(run_dir, manifest, root),
        "final_evaluation": {
            key: final_summary.get(key) for key in
            ("test_loss", "test_accuracy", "macro_precision", "macro_recall", "macro_f1")
        },
        "error_analysis": {
            "confusion_matrix": confusion or None,
            "classification_report": classification or None,
            "highest_confidence_errors": selected_errors[:10],
        },
        "sources": {
            "metrics_jsonl": _relative(metrics_path, root),
            "summary_json": _relative(summary_path, root),
            "ranking_csv": _relative(ranking_path, root),
            "selection_json": _relative(selection_path, root),
            "checkpoint_manifest": _relative(manifest_path, root),
            "log": _relative(log_path, root),
            "final_evaluation_artifacts": [
                path for path in (
                    _relative(final_summary_path, root), _relative(confusion_path, root),
                    _relative(report_path, root), _relative(predictions_path, root)
                ) if path is not None
            ],
        },
    }
    destination = Path(output_path).expanduser() if output_path else outputs / "visualization_data.json"
    if not destination.is_absolute():
        destination = root / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return payload
