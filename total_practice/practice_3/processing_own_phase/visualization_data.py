"""Read-only aggregation of saved Practice 3 presentation artifacts."""
from __future__ import annotations

import csv, json, math, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUN_IDS = ("E1_lr_1e-5", "E2_lr_2e-5", "E3_lr_3e-5", "E4_weight_decay_0.05", "E5_classifier_dropout_0.40", "E6_staged_finetune")

def _value(path: Path, default: Any = None) -> Any:
    if not path.is_file(): return default
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return default

def _json(path: Path) -> dict[str, Any]:
    value = _value(path, {})
    return value if isinstance(value, dict) else {}

def _csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file(): return []
    try:
        with path.open(newline="", encoding="utf-8") as stream: return list(csv.DictReader(stream))
    except (OSError, csv.Error): return []

def _relative(path: Path, root: Path) -> str:
    try: return path.resolve().relative_to(root).as_posix()
    except ValueError: return str(path.resolve())

def _device(path: Path) -> str | None:
    if not path.is_file(): return None
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Using device:\s*([^\s|]+)", text, re.I)
    if match: return match.group(1).lower()
    return next((name for name in ("mps", "cuda", "cpu") if re.search(rf"\b{name}\b", text, re.I)), None)

def _artifact(path: Path, reader: str = "json") -> dict[str, Any]:
    data: Any = None if reader == "binary" else (_csv(path) if reader == "csv" else _value(path))
    available = path.is_file() if reader == "binary" else path.is_file() and data not in (None, {}, [])
    return {"available": available, "path": str(path), "data": data, "message": None if available else "Artifact not generated yet"}

def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file(): return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try: row = json.loads(line)
        except json.JSONDecodeError: continue
        if isinstance(row, dict): rows.append(row)
    return rows

def _history(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("records", [])
    if not isinstance(records, list): return []
    return [{
        "epoch": r.get("epoch"), "global_step": r.get("global_step"), "train_loss": r.get("train_loss"),
        "validation_loss": r.get("eval_loss"), "train_accuracy": r.get("train_accuracy"),
        "validation_accuracy": r.get("eval_accuracy"), "validation_precision": r.get("eval_precision"),
        "validation_recall": r.get("eval_recall"), "validation_f1": r.get("eval_f1"),
        "learning_rate": r.get("learning_rate"), "runtime_seconds": r.get("runtime_seconds", r.get("epoch_runtime_seconds")),
        # Existing detailed notebook plots use the raw Trainer names.
        "eval_loss": r.get("eval_loss"), "eval_accuracy": r.get("eval_accuracy"),
        "eval_precision": r.get("eval_precision"), "eval_recall": r.get("eval_recall"),
        "eval_f1": r.get("eval_f1"),
    } for r in records if isinstance(r, dict)]

def build_visualization_data(project_root: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    """Aggregate real artifacts and atomically save one presentation JSON."""
    root = Path(project_root).expanduser().resolve()
    result = root / "docs/result/practice_3_v2_3"
    registry_path = result / "experiment_registry.json"
    registry = _json(registry_path)
    registry_rows = {r.get("experiment_id"): r for r in registry.get("experiments", []) if isinstance(r, dict)}
    sources: dict[str, Any] = {
        "experiment_registry": _relative(registry_path, root),
        "training_authorization": _relative(result / "training_authorization.json", root),
        "protocol_manifest": _relative(result / "protocol_manifest.json", root), "experiments": {},
    }
    experiments, histories = [], {}
    for run_id in RUN_IDS:
        run_result = result / "experiments" / run_id
        paths = {"config": run_result / "experiment_config.json", "summary": run_result / "run_summary.json", "training_history": run_result / "training_history.json", "terminal_log": root / "save_logs" / f"{run_id}.log", "tensorboard_dir": root / "runs/practice_3_v2_3" / run_id}
        config, summary, registry_row = _json(paths["config"]), _json(paths["summary"]), registry_rows.get(run_id, {})
        history = _history(_json(paths["training_history"])); histories[run_id] = history
        controlled = config.get("controlled_variable") or registry_row.get("controlled_variable")
        controlled_value = config.get(controlled) if controlled else None
        if controlled == "seq_classif_dropout": controlled_value = config.get("seq_classif_dropout", config.get("classifier_dropout"))
        experiments.append({
            "run_id": run_id, "short_id": run_id.split("_")[0], "experiment_name": run_id.split("_", 1)[1].replace("_", " "),
            "experiment_type": controlled, "controlled_value": controlled_value, "stage": config.get("stage", registry_row.get("stage")),
            "status": summary.get("status") or registry_row.get("status", "MISSING"),
            "learning_rate": config.get("learning_rate", registry_row.get("learning_rate")), "weight_decay": config.get("weight_decay", registry_row.get("weight_decay")),
            "dropout": config.get("seq_classif_dropout", config.get("classifier_dropout")), "fine_tuning_strategy": config.get("fine_tuning_strategy"),
            "best_epoch": summary.get("best_epoch", registry_row.get("best_epoch")), "stopped_epoch": summary.get("stopped_epoch", registry_row.get("stopped_epoch")),
            "validation_loss": summary.get("best_val_loss", registry_row.get("best_val_loss")), "validation_accuracy": summary.get("best_val_accuracy", registry_row.get("best_val_accuracy")),
            "validation_precision": summary.get("best_val_precision", registry_row.get("best_val_precision")), "validation_recall": summary.get("best_val_recall", registry_row.get("best_val_recall")),
            "validation_f1": summary.get("best_val_f1", registry_row.get("best_val_f1")), "runtime_seconds": summary.get("runtime_seconds", registry_row.get("runtime_seconds")),
            # Backward-compatible notebook aliases.
            "best_val_loss": summary.get("best_val_loss", registry_row.get("best_val_loss")),
            "best_val_accuracy": summary.get("best_val_accuracy", registry_row.get("best_val_accuracy")),
            "best_val_f1": summary.get("best_val_f1", registry_row.get("best_val_f1")),
            "device": _device(paths["terminal_log"]), "best_checkpoint": summary.get("best_checkpoint", registry_row.get("best_checkpoint")),
            "checkpoint_fingerprint": summary.get("checkpoint_hash", registry_row.get("checkpoint_hash")),
            "tensorboard_available": paths["tensorboard_dir"].is_dir() and any(paths["tensorboard_dir"].glob("events.out.tfevents.*")), "history": history,
        })
        sources["experiments"][run_id] = {k: _relative(v, root) for k, v in paths.items()}
    by_id = {r["run_id"]: r for r in experiments}
    lock_path, lock = result / "final_validation_winner_lock.json", _json(result / "final_validation_winner_lock.json")
    computed = sorted((r for r in experiments if r["status"] == "COMPLETED" and isinstance(r["validation_loss"], (int, float)) and math.isfinite(r["validation_loss"])), key=lambda r: (r["validation_loss"], -float(r["validation_f1"] or 0), r["run_id"]))
    order = lock.get("validation_ranking") if isinstance(lock.get("validation_ranking"), list) else [r["run_id"] for r in computed]
    ranking = [{"rank": i, **by_id[rid]} for i, rid in enumerate(order, 1) if rid in by_id]
    winner_id, winner_row = lock.get("winner_run_id"), by_id.get(lock.get("winner_run_id"), {})
    holdout_path, holdout = result / "final_holdout_metrics.json", _json(result / "final_holdout_metrics.json")
    classification_path, errors_path, high_path = result / "classification_report.csv", result / "holdout_errors.csv", result / "high_confidence_holdout_errors.csv"
    classification, errors, high_errors = _csv(classification_path), _csv(errors_path), _csv(high_path)
    custom_json, custom_csv = result / "custom_inference_results.json", result / "custom_inference_results.csv"
    custom = _value(custom_json); custom = custom if isinstance(custom, list) else _csv(custom_csv)
    save_path, save_reload = result / "save_reload_verification.json", _json(result / "save_reload_verification.json")
    payload = {
        "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(), "protocol_version": registry.get("protocol_version"), "registry_status": registry.get("status"),
        "holdout_status": registry.get("holdout_status"), "validation_leader": winner_id or (computed[0]["run_id"] if computed else None),
        "experiments": experiments, "training_history": histories,
        "experiment_comparison": {"ranking": ranking, "selected_winner": winner_id, "selection_metric": lock.get("selection_metric"), "winner_reason": lock.get("ranking_policy"), "source": "final_validation_winner_lock.json" if lock else None},
        "winner": {"available": bool(lock), "run_id": winner_id, "best_checkpoint": lock.get("checkpoint_path", winner_row.get("best_checkpoint")), "best_epoch": lock.get("best_epoch", winner_row.get("best_epoch")), "validation_loss": lock.get("best_val_loss", winner_row.get("validation_loss")), "validation_accuracy": lock.get("best_val_accuracy", winner_row.get("validation_accuracy")), "validation_precision": winner_row.get("validation_precision"), "validation_recall": winner_row.get("validation_recall"), "validation_f1": lock.get("best_val_f1", winner_row.get("validation_f1")), "checkpoint_fingerprint": lock.get("checkpoint_hash", winner_row.get("checkpoint_fingerprint"))},
        "final_holdout": {"available": bool(holdout), "values": holdout or None, "message": None if holdout else "Artifact not generated yet"},
        "error_analysis": {"available": bool(holdout and (classification or errors)), "confusion_matrix": holdout.get("confusion_matrix") if holdout else None, "classification_report": classification or None, "error_count": len(errors) if errors else None, "high_confidence_error_count": len(high_errors) if high_errors else None, "highest_confidence_errors": high_errors[:10] if high_errors else None, "message": None if holdout and (classification or errors) else "Artifact not generated yet"},
        "custom_inference": {"available": bool(custom), "records": custom or None, "message": None if custom else "Artifact not generated yet"},
        "save_reload_verification": {"available": bool(save_reload), "values": save_reload or None, "message": None if save_reload else "Artifact not generated yet"},
        "sources": sources | {"winner_lock": _relative(lock_path, root), "final_holdout": _relative(holdout_path, root), "classification_report": _relative(classification_path, root), "holdout_errors": _relative(errors_path, root), "high_confidence_errors": _relative(high_path, root), "custom_inference": _relative(custom_json if custom_json.is_file() else custom_csv, root), "save_reload_verification": _relative(save_path, root)},
    }
    destination = Path(output_path).expanduser() if output_path else result / "visualization_data.json"
    if not destination.is_absolute(): destination = root / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"); temporary.replace(destination)
    return payload

def load_presentation_data(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve(); result = root / "docs/result/practice_3_v2_3"
    return {"exercise_1": _artifact(result / "exercise_1_pretrained_sentiment.json"), "dataset_split": _artifact(result / "dataset_split_reference.json"), "tokenization": _artifact(result / "tokenization_demo.json"), "model_architecture": _artifact(result / "final_saved_model/config.json"), "training": build_visualization_data(root), "final_holdout": {"winner_lock": _artifact(result / "final_validation_winner_lock.json"), "metrics": _artifact(result / "final_holdout_metrics.json")}, "error_analysis": {"errors": _artifact(result / "holdout_errors.csv", "csv"), "high_confidence_errors": _artifact(result / "high_confidence_holdout_errors.csv", "csv"), "confusion_matrix": _artifact(result / "figures/final_holdout_confusion_matrix.png", "binary"), "confidence_figure": _artifact(result / "figures/correct_vs_incorrect_confidence.png", "binary")}, "custom_inference": _artifact(result / "custom_inference_results.csv", "csv"), "save_reload": _artifact(result / "save_reload_verification.json")}

def load_live_training_snapshot(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve(); demo = root / "runs/live_terminal_demo"; status = _json(demo / "status.json"); active = status.get("status") == "RUNNING"
    return {"active": active, "message": "Active training session snapshot." if active else "No active training session.", "status": status, "metrics": _jsonl(demo / "metrics.jsonl"), "summary": _json(demo / "final_summary.json")}
