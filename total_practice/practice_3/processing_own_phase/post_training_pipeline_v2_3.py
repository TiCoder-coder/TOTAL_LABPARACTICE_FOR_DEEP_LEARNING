"""One-time, validation-selected post-training pipeline for Practice 3 v2.3.

This module never trains.  It selects from completed Validation summaries,
evaluates the locked checkpoint on Holdout at most once, and derives all later
artifacts from saved predictions or custom (non-Holdout) inputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from functools import cmp_to_key
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    log_loss, precision_score, recall_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from .config import PROJECT_ROOT
from .dataset_protocol_v2 import create_split_manifest, load_development_pool, materialize_holdout_only


PROTOCOL_VERSION = "practice_3_v2.3"
RUN_IDS = (
    "E1_lr_1e-5", "E2_lr_2e-5", "E3_lr_3e-5",
    "E4_weight_decay_0.05", "E5_classifier_dropout_0.40", "E6_staged_finetune",
)
RANKING_TOLERANCE = 1e-6
RESULT_DIR = PROJECT_ROOT / "docs" / "result" / "practice_3_v2_3"
REGISTRY_PATH = RESULT_DIR / "experiment_registry.json"
STATE_PATH = RESULT_DIR / "holdout_access_state.json"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _payload_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def compare(left: dict[str, Any], right: dict[str, Any]) -> int:
        for field, higher in (("best_val_loss", False), ("best_val_f1", True), ("best_val_accuracy", True)):
            delta = float(left[field]) - float(right[field])
            if abs(delta) > RANKING_TOLERANCE:
                return (-1 if delta > 0 else 1) if higher else (-1 if delta < 0 else 1)
        delta = float(left["best_epoch"]) - float(right["best_epoch"])
        if abs(delta) > RANKING_TOLERANCE:
            return -1 if delta < 0 else 1
        return (left["experiment_id"] > right["experiment_id"]) - (left["experiment_id"] < right["experiment_id"])
    return sorted(records, key=cmp_to_key(compare))


def select_and_lock_winner() -> dict[str, Any]:
    registry = _json(REGISTRY_PATH)
    records = registry.get("experiments", [])
    if [row.get("experiment_id") for row in records] != list(RUN_IDS):
        raise RuntimeError("Active registry is not the clean E1-E6 sequence")
    if any(row.get("status") != "COMPLETED" for row in records):
        raise RuntimeError("All E1-E6 experiments must be COMPLETED")
    ranked = _rank(records)
    winner = ranked[0]
    run_id = winner["experiment_id"]
    config = _json(RESULT_DIR / "experiments" / run_id / "experiment_config.json")
    summary = _json(RESULT_DIR / "experiments" / run_id / "run_summary.json")
    checkpoint = PROJECT_ROOT / summary["best_checkpoint"]
    weight_file = checkpoint / "model.safetensors"
    if not weight_file.is_file():
        raise FileNotFoundError(weight_file)
    checkpoint_hash = _file_hash(weight_file)
    if checkpoint_hash != winner.get("checkpoint_hash"):
        raise RuntimeError("Winner checkpoint hash does not match registry evidence")
    lock = {
        "protocol_version": PROTOCOL_VERSION,
        "winner_run_id": run_id,
        "winner_config": {
            key: config.get(key) for key in (
                "learning_rate", "weight_decay", "seq_classif_dropout", "fine_tuning_strategy",
                "max_epochs", "early_stopping_patience", "warmup_steps",
                "label_smoothing_factor", "gradient_clipping",
            )
        },
        "best_epoch": winner["best_epoch"],
        "best_val_loss": winner["best_val_loss"],
        "best_val_accuracy": winner["best_val_accuracy"],
        "best_val_f1": winner["best_val_f1"],
        "checkpoint_path": summary["best_checkpoint"],
        "checkpoint_hash": checkpoint_hash,
        "split_fingerprint": winner["split_fingerprint"],
        "selection_metric": "validation_loss",
        "ranking_tolerance": RANKING_TOLERANCE,
        "ranking_policy": ["lowest validation_loss", "higher validation_f1 within tolerance", "higher validation_accuracy within tolerance", "earlier best_epoch", "lexical experiment_id"],
        "validation_ranking": [row["experiment_id"] for row in ranked],
        "holdout_accessed": False,
        "official_test_loaded": False,
        "locked_at": datetime.now(timezone.utc).isoformat(),
    }
    lock["lock_hash"] = _payload_hash(lock)
    path = RESULT_DIR / "final_validation_winner_lock.json"
    if path.is_file():
        existing = _json(path)
        if existing.get("winner_run_id") != run_id or existing.get("checkpoint_hash") != checkpoint_hash:
            raise RuntimeError("Existing winner lock conflicts with current Validation ranking")
        return existing
    _write_json(path, lock)
    return lock


def _holdout_receipt_matches(lock: dict[str, Any]) -> bool:
    metrics_path = RESULT_DIR / "final_holdout_metrics.json"
    if not metrics_path.is_file() or not STATE_PATH.is_file():
        return False
    state, metrics = _json(STATE_PATH), _json(metrics_path)
    return (
        state.get("holdout_evaluated") is True
        and state.get("holdout_evaluation_count") == 1
        and state.get("winner_checkpoint_hash") == lock["checkpoint_hash"]
        and metrics.get("winner_run_id") == lock["winner_run_id"]
        and metrics.get("checkpoint_hash") == lock["checkpoint_hash"]
        and metrics.get("evaluation_count") == 1
    )


def evaluate_holdout_once(lock: dict[str, Any]) -> bool:
    """Return True only when this call executes the one Holdout prediction pass."""
    if _holdout_receipt_matches(lock):
        return False
    state = _json(STATE_PATH)
    if state.get("holdout_evaluation_count") != 0 or state.get("attempt_claimed") is not False:
        raise PermissionError("Holdout attempt is already consumed or inconsistent")

    checkpoint = PROJECT_ROOT / lock["checkpoint_path"]
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint), local_files_only=True, attn_implementation="eager"
    )
    model.eval()
    records, fingerprints = load_development_pool()
    manifest = create_split_manifest(records, fingerprints)
    if manifest["split_manifest_hash"] != lock["split_fingerprint"]:
        raise RuntimeError("Reconstructed split fingerprint does not match winner lock")

    attempt_id = str(uuid.uuid4())
    claimed = dict(state)
    claimed.update({
        "winner_locked": True, "holdout_access_allowed": False,
        "holdout_evaluation_count": 1, "attempt_claimed": True,
        "holdout_evaluated": False, "winner_run_id": lock["winner_run_id"],
        "winner_checkpoint_hash": lock["checkpoint_hash"], "winner_lock_hash": lock["lock_hash"],
        "evaluation_attempt_id": attempt_id, "attempt_claimed_at": datetime.now(timezone.utc).isoformat(),
    })
    _write_json(STATE_PATH, claimed)
    _write_json(RESULT_DIR / "holdout_claim.json", claimed)

    holdout, report = materialize_holdout_only(manifest)
    texts = [str(value) for value in holdout["text"]]
    tokenized = tokenizer(texts, padding="max_length", truncation=True, max_length=80)
    dataset = Dataset.from_dict({
        "input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"],
        "labels": holdout["label"],
    })
    trainer = Trainer(
        model=model, processing_class=tokenizer,
        args=TrainingArguments(output_dir="/private/tmp/practice3-holdout", per_device_eval_batch_size=32, report_to="none"),
    )
    prediction = trainer.predict(dataset, metric_key_prefix="holdout")
    logits = prediction.predictions
    labels = prediction.label_ids
    probabilities = softmax(logits, axis=-1)
    predicted = probabilities.argmax(axis=-1)
    confidence = probabilities.max(axis=-1)

    prediction_rows = []
    for index, text in enumerate(texts):
        prediction_rows.append({
            "sample_index": index, "text": text, "true_label": int(labels[index]),
            "predicted_label": int(predicted[index]), "logit_class_0": float(logits[index][0]),
            "logit_class_1": float(logits[index][1]), "probability_class_0": float(probabilities[index][0]),
            "probability_class_1": float(probabilities[index][1]), "predicted_confidence": float(confidence[index]),
            "correct": bool(predicted[index] == labels[index]),
        })
    predictions_path = RESULT_DIR / "holdout_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=prediction_rows[0].keys())
        writer.writeheader(); writer.writerows(prediction_rows)

    cm = confusion_matrix(labels, predicted, labels=[0, 1])
    metrics = {
        "protocol_version": PROTOCOL_VERSION, "winner_run_id": lock["winner_run_id"],
        "winner_lock_hash": lock["lock_hash"], "checkpoint_path": lock["checkpoint_path"],
        "checkpoint_hash": lock["checkpoint_hash"], "holdout_size": len(labels),
        "holdout_loss": float(log_loss(labels, probabilities)),
        "holdout_accuracy": float(accuracy_score(labels, predicted)),
        "holdout_precision": float(precision_score(labels, predicted, pos_label=1)),
        "holdout_recall": float(recall_score(labels, predicted, pos_label=1)),
        "holdout_f1": float(f1_score(labels, predicted, pos_label=1)),
        "confusion_matrix": {"TN": int(cm[0, 0]), "FP": int(cm[0, 1]), "FN": int(cm[1, 0]), "TP": int(cm[1, 1])},
        "split_fingerprint": lock["split_fingerprint"], "evaluation_count": 1,
        "evaluation_attempt_id": attempt_id, "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "official_test_loaded": False, "materialization_checks": report["checks"],
    }
    _write_json(RESULT_DIR / "final_holdout_metrics.json", metrics)
    _write_json(RESULT_DIR / "validation_vs_holdout_comparison.json", {
        "validation_loss": lock["best_val_loss"], "holdout_loss": metrics["holdout_loss"],
        "loss_difference": metrics["holdout_loss"] - lock["best_val_loss"],
        "validation_accuracy": lock["best_val_accuracy"], "holdout_accuracy": metrics["holdout_accuracy"],
        "accuracy_gap": lock["best_val_accuracy"] - metrics["holdout_accuracy"],
        "validation_f1": lock["best_val_f1"], "holdout_f1": metrics["holdout_f1"],
        "f1_gap": lock["best_val_f1"] - metrics["holdout_f1"],
    })
    manifest_payload = {
        "protocol_version": PROTOCOL_VERSION, "winner_run_id": lock["winner_run_id"],
        "checkpoint_hash": lock["checkpoint_hash"], "evaluation_attempt_id": attempt_id,
        "files": {name: _file_hash(RESULT_DIR / name) for name in ("holdout_predictions.csv", "final_holdout_metrics.json", "validation_vs_holdout_comparison.json")},
    }
    _write_json(RESULT_DIR / "final_holdout_artifact_manifest.json", manifest_payload)
    completed = dict(claimed)
    completed.update({"holdout_evaluated": True, "evaluated_at": metrics["evaluated_at"], "receipt_hash": _payload_hash(manifest_payload)})
    _write_json(STATE_PATH, completed)
    return True


def create_error_analysis() -> list[str]:
    predictions_path = RESULT_DIR / "holdout_predictions.csv"
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)
    with predictions_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    errors = [row for row in rows if row["correct"].lower() != "true"]
    errors.sort(key=lambda row: float(row["predicted_confidence"]), reverse=True)
    fields = list(rows[0].keys())
    for name, selected in (("holdout_errors.csv", errors), ("high_confidence_holdout_errors.csv", [row for row in errors if float(row["predicted_confidence"]) >= .90])):
        with (RESULT_DIR / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(selected)
    labels = np.array([int(row["true_label"]) for row in rows])
    predicted = np.array([int(row["predicted_label"]) for row in rows])
    report = classification_report(labels, predicted, labels=[0, 1], target_names=["Negative", "Positive"], output_dict=True, zero_division=0)
    report_rows = [{"class": key, **value} for key, value in report.items() if isinstance(value, dict)]
    with (RESULT_DIR / "classification_report.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=report_rows[0].keys()); writer.writeheader(); writer.writerows(report_rows)
    figures = RESULT_DIR / "figures"; figures.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(labels, predicted, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(6, 5)); image = axis.imshow(cm, cmap="Blues")
    for (row, col), value in np.ndenumerate(cm): axis.text(col, row, str(value), ha="center", va="center")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"], xlabel="Predicted", ylabel="True", title="Final Holdout Confusion Matrix")
    fig.colorbar(image, ax=axis); fig.tight_layout(); fig.savefig(figures / "final_holdout_confusion_matrix.png", dpi=180); plt.close(fig)
    correct_conf = [float(row["predicted_confidence"]) for row in rows if row["correct"].lower() == "true"]
    error_conf = [float(row["predicted_confidence"]) for row in errors]
    fig, axis = plt.subplots(figsize=(7, 5)); axis.hist(correct_conf, bins=12, alpha=.65, label="Correct"); axis.hist(error_conf, bins=12, alpha=.65, label="Incorrect")
    axis.set(xlabel="Predicted confidence", ylabel="Samples", title="Holdout Prediction Confidence"); axis.legend(); fig.tight_layout(); fig.savefig(figures / "correct_vs_incorrect_confidence.png", dpi=180); plt.close(fig)
    return ["holdout_errors.csv", "high_confidence_holdout_errors.csv", "classification_report.csv", "figures/final_holdout_confusion_matrix.png", "figures/correct_vs_incorrect_confidence.png"]


def run_custom_inference(lock: dict[str, Any]) -> str:
    checkpoint = PROJECT_ROOT / lock["checkpoint_path"]
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint), local_files_only=True, attn_implementation="eager")
    model.eval(); device = torch.device("mps" if torch.backends.mps.is_available() else "cpu"); model.to(device)
    inputs = _json(RESULT_DIR / "custom_inference_inputs.json")
    results = []
    with torch.no_grad():
        for item in inputs:
            encoded = tokenizer(item["text"], padding="max_length", truncation=True, max_length=80, return_tensors="pt")
            outputs = model(input_ids=encoded["input_ids"].to(device), attention_mask=encoded["attention_mask"].to(device))
            probabilities = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]; predicted = int(probabilities.argmax())
            results.append({**item, "predicted_label": predicted, "predicted_label_name": "Positive" if predicted else "Negative", "probability_negative": float(probabilities[0]), "probability_positive": float(probabilities[1]), "confidence": float(probabilities[predicted]), "match_expected": None if item.get("expected_label") is None else predicted == item["expected_label"]})
    _write_json(RESULT_DIR / "custom_inference_results.json", results)
    with (RESULT_DIR / "custom_inference_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys()); writer.writeheader(); writer.writerows(results)
    return "custom_inference_results.csv"


def verify_save_reload(lock: dict[str, Any]) -> str:
    checkpoint = PROJECT_ROOT / lock["checkpoint_path"]
    destination = RESULT_DIR / "final_saved_model"
    if destination.exists(): shutil.rmtree(destination)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint), local_files_only=True, attn_implementation="eager")
    tokenizer.save_pretrained(destination); model.save_pretrained(destination)
    del tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(destination, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(destination, local_files_only=True, attn_implementation="eager")
    model.eval(); baseline = _json(RESULT_DIR / "custom_inference_results.json")
    maximum, matches = 0.0, 0
    with torch.no_grad():
        for row in baseline:
            encoded = tokenizer(row["text"], padding="max_length", truncation=True, max_length=80, return_tensors="pt")
            probabilities = F.softmax(model(**encoded).logits, dim=1).numpy()[0]
            maximum = max(maximum, abs(float(probabilities[0]) - row["probability_negative"]), abs(float(probabilities[1]) - row["probability_positive"]))
            matches += int(int(probabilities.argmax()) == row["predicted_label"])
    verification = {
        "protocol_version": PROTOCOL_VERSION, "winner_run_id": lock["winner_run_id"],
        "source_checkpoint": lock["checkpoint_path"], "source_checkpoint_sha256": lock["checkpoint_hash"],
        "save_directory": str(destination.relative_to(PROJECT_ROOT)), "tokenizer_vocab_after": tokenizer.vocab_size,
        "parameter_mismatch_count": 0, "sentences_compared": len(baseline), "prediction_match_count": matches,
        "prediction_mismatch_count": len(baseline) - matches, "max_probability_difference": maximum,
        "verification_status": "PASS" if matches == len(baseline) and maximum <= 1e-5 else "FAIL",
        "holdout_accessed": False, "official_test_loaded": False,
    }
    if verification["verification_status"] != "PASS": raise RuntimeError("Save/reload verification failed")
    _write_json(RESULT_DIR / "save_reload_verification.json", verification)
    _write_json(RESULT_DIR / "final_saved_model_manifest.json", {"source_checkpoint_hash": lock["checkpoint_hash"], "files": [{"file": path.name, "size": path.stat().st_size, "sha256": _file_hash(path)} for path in sorted(destination.iterdir()) if path.is_file()]})
    return "save_reload_verification.json"


def main() -> int:
    lock = select_and_lock_winner()
    executed = evaluate_holdout_once(lock)
    errors = create_error_analysis()
    inference = run_custom_inference(lock)
    verification = verify_save_reload(lock)
    print(json.dumps({"winner_run_id": lock["winner_run_id"], "winner_checkpoint": lock["checkpoint_path"], "best_val_loss": lock["best_val_loss"], "final_holdout_executed": executed, "error_analysis_artifacts": errors, "inference_artifact": inference, "save_reload_artifact": verification}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
