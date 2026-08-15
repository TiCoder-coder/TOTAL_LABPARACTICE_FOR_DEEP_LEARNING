"""Phase 11: guarded Validation verification and one-time final Test evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import tempfile
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

from .config import RANDOM_SEED, RESULT_DIR
from .phase_08_metrics_training_configuration import compute_metrics


PHASE_09_DIR = RESULT_DIR / "phase_09_training"
PHASE_11_DIR = RESULT_DIR / "phase_11_evaluation"
SELECTED_PATH = PHASE_09_DIR / "selected_checkpoint.json"
PHASE_09_MANIFEST_PATH = PHASE_09_DIR / "phase_09_training_manifest.json"
CHECKPOINT_RECORDS_PATH = PHASE_09_DIR / "checkpoint_records.json"
TRAINER_STATE_PATH = PHASE_09_DIR / "trainer_state.json"
VALIDATION_HISTORY_PATH = PHASE_09_DIR / "validation_metrics_by_epoch.json"
PHASE_07_VERIFICATION_PATH = RESULT_DIR / "phase_07_model_verification.json"

CHECKPOINT_VERIFICATION_PATH = PHASE_11_DIR / "checkpoint_verification.json"
VALIDATION_EVALUATION_PATH = PHASE_11_DIR / "validation_evaluation.json"
TEST_EVALUATION_PATH = PHASE_11_DIR / "test_evaluation.json"
TEST_PREDICTIONS_PATH = PHASE_11_DIR / "test_predictions.json"
ATTEMPT_RECEIPT_PATH = PHASE_11_DIR / "final_test_attempt_receipt.json"
MANIFEST_PATH = PHASE_11_DIR / "phase_11_evaluation_manifest.json"

LOSS_TOLERANCE = 1e-5
METRIC_TOLERANCE = 1e-6
EXPECTED_COUNTS = {"validation": 1066, "test": 1066}
EXPECTED_ID2LABEL = {0: "NEGATIVE", 1: "POSITIVE"}
EXPECTED_LABEL2ID = {"NEGATIVE": 0, "POSITIVE": 1}
METRIC_NAMES = ("loss", "accuracy", "precision", "recall", "f1")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    if _load_json(path) != json.loads(serialized):
        raise RuntimeError(f"JSON read-back failed: {path}")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalize_id2label(mapping: Dict[Any, Any]) -> Dict[int, str]:
    return {int(key): str(value) for key, value in mapping.items()}


def _validate_split(split: Any, expected_count: int, split_name: str) -> None:
    if len(split) != expected_count:
        raise ValueError(f"{split_name} count {len(split)} != {expected_count}")
    columns = set(split.column_names)
    required = {"input_ids", "attention_mask", "labels"}
    if not required.issubset(columns):
        raise ValueError(f"{split_name} missing fields: {required - columns}")
    labels = {int(value) for value in split["labels"]}
    if not labels.issubset({0, 1}):
        raise ValueError(f"{split_name} labels are invalid: {sorted(labels)}")


def verify_authoritative_checkpoint() -> Tuple[Any, Dict[str, Any]]:
    """Verify and reload the fixed Phase 9 checkpoint without reranking."""
    selected = _load_json(SELECTED_PATH)
    phase_09_manifest = _load_json(PHASE_09_MANIFEST_PATH)
    checkpoint_records = _load_json(CHECKPOINT_RECORDS_PATH)
    trainer_state = _load_json(TRAINER_STATE_PATH)
    phase_07 = _load_json(PHASE_07_VERIFICATION_PATH)

    if phase_09_manifest.get("status") != "PASS":
        raise RuntimeError("Phase 9 manifest is not PASS")
    if phase_09_manifest.get("test_accessed") is not False:
        raise RuntimeError("Phase 9 Test isolation is not verified")
    if phase_07.get("status") != "PASS":
        raise RuntimeError("Phase 7 model verification is not PASS")

    checkpoint = Path(selected["checkpoint"]).resolve()
    selected_record = next(
        (
            record for record in checkpoint_records
            if float(record["epoch"]) == 2.0 and int(record["step"]) == 1068
        ),
        None,
    )
    path_sources = {
        "selected_checkpoint": str(checkpoint),
        "phase_09_manifest": str(Path(phase_09_manifest["selected_checkpoint"]).resolve()),
        "trainer_state": str(Path(trainer_state["best_model_checkpoint"]).resolve()),
        "checkpoint_record": (
            str(Path(selected_record["checkpoint"]).resolve()) if selected_record else None
        ),
    }
    paths_agree = len(set(path_sources.values())) == 1
    identity_pass = (
        float(selected["epoch"]) == 2.0
        and int(selected["step"]) == 1068
        and checkpoint.name == "checkpoint-1068"
        and selected_record is not None
        and int(trainer_state["best_global_step"]) == 1068
        and paths_agree
        and selected.get("test_used_for_selection") is False
    )
    if not identity_pass:
        raise RuntimeError(f"Authoritative checkpoint identity mismatch: {path_sources}")
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Selected checkpoint directory is missing: {checkpoint}")
    weight_candidates = [checkpoint / "model.safetensors", checkpoint / "pytorch_model.bin"]
    weight_path = next((path for path in weight_candidates if path.is_file()), None)
    if weight_path is None or weight_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Checkpoint weights are missing or empty: {checkpoint}")

    weight_sha256 = _sha256_file(weight_path)
    selection_sha256 = _sha256_json(selected)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint), local_files_only=True
    )
    model.eval()
    id2label = _normalize_id2label(model.config.id2label)
    label2id = {str(key): int(value) for key, value in model.config.label2id.items()}
    config_pass = (
        model.__class__.__name__ == "DistilBertForSequenceClassification"
        and model.config.model_type == "distilbert"
        and int(model.config.num_labels) == 2
        and id2label == EXPECTED_ID2LABEL
        and label2id == EXPECTED_LABEL2ID
        and model.training is False
    )
    if not config_pass:
        raise RuntimeError("Reloaded checkpoint model/config verification failed")

    verification = {
        "status": "PASS",
        "verified_at_utc": _utc_now(),
        "checkpoint": str(checkpoint),
        "epoch": float(selected["epoch"]),
        "step": int(selected["step"]),
        "path_sources": path_sources,
        "paths_agree": paths_agree,
        "checkpoint_reranked": False,
        "weight_file": str(weight_path.resolve()),
        "weight_bytes": weight_path.stat().st_size,
        "weight_sha256": weight_sha256,
        "selection_record_sha256": selection_sha256,
        "model_class": model.__class__.__name__,
        "model_type": model.config.model_type,
        "num_labels": int(model.config.num_labels),
        "id2label": id2label,
        "label2id": label2id,
        "model_eval_mode": model.training is False,
        "test_used_for_selection": False,
        "test_accessed": False,
    }
    _atomic_json(CHECKPOINT_VERIFICATION_PATH, verification)
    return model, verification


def _evaluation_trainer(model: Any, data_collator: Any, tokenizer: Any) -> Trainer:
    arguments = TrainingArguments(
        output_dir=str(PHASE_11_DIR / "evaluation_runtime"),
        per_device_eval_batch_size=32,
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,
        eval_strategy="no",
        logging_strategy="no",
        save_strategy="no",
        report_to="none",
    )
    return Trainer(
        model=model,
        args=arguments,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )


def _normalized_metrics(metrics: Dict[str, Any], prefix: str) -> Dict[str, float]:
    normalized = {
        name: float(metrics[f"{prefix}_{name}"])
        for name in METRIC_NAMES
    }
    if not math.isfinite(normalized["loss"]) or normalized["loss"] < 0:
        raise RuntimeError(f"{prefix} loss is invalid: {normalized['loss']}")
    if not all(
        math.isfinite(normalized[name]) and 0.0 <= normalized[name] <= 1.0
        for name in ("accuracy", "precision", "recall", "f1")
    ):
        raise RuntimeError(f"{prefix} metrics are not finite/in [0,1]")
    return normalized


def verify_reloaded_validation(
    trainer: Trainer, validation_dataset: Any, checkpoint_verification: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate Validation and gate Test on Phase 9 reproducibility."""
    _validate_split(validation_dataset, EXPECTED_COUNTS["validation"], "validation")
    output = trainer.predict(validation_dataset, metric_key_prefix="validation")
    actual = _normalized_metrics(output.metrics, "validation")
    history = _load_json(VALIDATION_HISTORY_PATH)
    reference_record = next(
        record for record in history
        if float(record["epoch"]) == 2.0 and int(record["step"]) == 1068
    )
    expected = {
        "loss": float(reference_record["eval_loss"]),
        "accuracy": float(reference_record["eval_accuracy"]),
        "precision": float(reference_record["eval_precision"]),
        "recall": float(reference_record["eval_recall"]),
        "f1": float(reference_record["eval_f1"]),
    }
    comparisons = {}
    for name in METRIC_NAMES:
        tolerance = LOSS_TOLERANCE if name == "loss" else METRIC_TOLERANCE
        delta = abs(actual[name] - expected[name])
        comparisons[name] = {
            "phase_09_expected": expected[name],
            "reloaded_value": actual[name],
            "absolute_delta": delta,
            "absolute_tolerance": tolerance,
            "relative_tolerance": 0.0,
            "pass": delta <= tolerance,
        }
    status = "PASS" if all(row["pass"] for row in comparisons.values()) else "FAIL"
    result = {
        "status": status,
        "evaluated_at_utc": _utc_now(),
        "checkpoint": checkpoint_verification["checkpoint"],
        "checkpoint_weight_sha256": checkpoint_verification["weight_sha256"],
        "sample_count": len(validation_dataset),
        "label_set": sorted({int(value) for value in validation_dataset["labels"]}),
        "metrics": actual,
        "comparisons": comparisons,
        "test_accessed_before_validation_pass": False,
        "training_performed": False,
        "backward_called": False,
        "optimizer_step_performed": False,
        "scheduler_step_performed": False,
    }
    _atomic_json(VALIDATION_EVALUATION_PATH, result)
    if status != "PASS":
        raise RuntimeError(f"Validation reload reproducibility failed: {comparisons}")
    return result


def _prediction_records(logits: np.ndarray, labels: np.ndarray, raw_test: Any) -> list[Dict[str, Any]]:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    predictions = np.argmax(logits, axis=1)
    records = []
    for index in range(len(labels)):
        true_label = int(labels[index])
        predicted_label = int(predictions[index])
        negative_probability = float(probabilities[index, 0])
        positive_probability = float(probabilities[index, 1])
        confidence = float(probabilities[index, predicted_label])
        record = {
            "sample_index": index,
            "sample_id": f"rotten_tomatoes:test:{index}",
            "text": str(raw_test[index]["text"]),
            "true_label": true_label,
            "true_label_name": EXPECTED_ID2LABEL[true_label],
            "predicted_label": predicted_label,
            "predicted_label_name": EXPECTED_ID2LABEL[predicted_label],
            "negative_probability": negative_probability,
            "positive_probability": positive_probability,
            "confidence": confidence,
            "correct": predicted_label == true_label,
        }
        probability_pass = (
            all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (
                negative_probability, positive_probability, confidence
            ))
            and abs(negative_probability + positive_probability - 1.0) <= 1e-6
            and abs(confidence - probabilities[index, predicted_label]) <= 1e-12
        )
        if not probability_pass:
            raise RuntimeError(f"Invalid probability record at Test index {index}")
        records.append(record)
    return records


def _evaluate_test_once(
    trainer: Trainer,
    test_data_provider: Callable[[], Tuple[Any, Any]],
    checkpoint_verification: Dict[str, Any],
    validation_verification: Dict[str, Any],
) -> Tuple[Dict[str, Any], list[Dict[str, Any]]]:
    if validation_verification.get("status") != "PASS":
        raise RuntimeError("Test is forbidden before Validation verification PASS")
    attempt_id = str(uuid.uuid4())
    receipt = {
        "status": "STARTED",
        "attempt_id": attempt_id,
        "started_at_utc": _utc_now(),
        "checkpoint": checkpoint_verification["checkpoint"],
        "checkpoint_weight_sha256": checkpoint_verification["weight_sha256"],
        "selection_record_sha256": checkpoint_verification["selection_record_sha256"],
        "validation_status": "PASS",
        "intended_test_samples": EXPECTED_COUNTS["test"],
        "test_evaluation_count": 0,
    }
    _atomic_json(ATTEMPT_RECEIPT_PATH, receipt)

    tokenized_test, raw_test = test_data_provider()
    _validate_split(tokenized_test, EXPECTED_COUNTS["test"], "test")
    if len(raw_test) != EXPECTED_COUNTS["test"]:
        raise ValueError(f"raw Test count {len(raw_test)} != {EXPECTED_COUNTS['test']}")
    if not {"text", "label"}.issubset(set(raw_test.column_names)):
        raise ValueError("Raw Test split must contain text and label")
    tokenized_labels = [int(value) for value in tokenized_test["labels"]]
    raw_labels = [int(value) for value in raw_test["label"]]
    if tokenized_labels != raw_labels:
        raise RuntimeError("Raw and tokenized Test label alignment failed")

    output = trainer.predict(tokenized_test, metric_key_prefix="test")
    logits = np.asarray(output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions)
    labels = np.asarray(output.label_ids).reshape(-1)
    if logits.shape != (EXPECTED_COUNTS["test"], 2):
        raise RuntimeError(f"Unexpected Test logits shape: {logits.shape}")
    if len(labels) != EXPECTED_COUNTS["test"] or not np.isfinite(logits).all():
        raise RuntimeError("Test logits/label verification failed")
    metrics = _normalized_metrics(output.metrics, "test")
    predictions = _prediction_records(logits, labels, raw_test)
    indices = [record["sample_index"] for record in predictions]
    sample_ids = [record["sample_id"] for record in predictions]
    if indices != list(range(EXPECTED_COUNTS["test"])) or len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError("Test prediction indices/IDs are missing or duplicated")

    current_weight_sha256 = _sha256_file(Path(checkpoint_verification["weight_file"]))
    checkpoint_unchanged = current_weight_sha256 == checkpoint_verification["weight_sha256"]
    if not checkpoint_unchanged:
        raise RuntimeError("Authoritative checkpoint changed during final evaluation")
    test_result = {
        "status": "PASS",
        "evaluated_at_utc": _utc_now(),
        "attempt_id": attempt_id,
        "checkpoint": checkpoint_verification["checkpoint"],
        "checkpoint_weight_sha256": current_weight_sha256,
        "sample_count": len(labels),
        "prediction_count": len(predictions),
        "label_set": sorted(set(labels.astype(int).tolist())),
        "logits_shape": list(logits.shape),
        "logits_finite": bool(np.isfinite(logits).all()),
        "metrics": metrics,
        "test_evaluation_count": 1,
        "validation_verified_before_test": True,
        "test_used_for_selection": False,
        "checkpoint_changed_after_test": False,
        "training_performed": False,
        "backward_called": False,
        "optimizer_step_performed": False,
        "scheduler_step_performed": False,
    }
    _atomic_json(TEST_PREDICTIONS_PATH, predictions)
    _atomic_json(TEST_EVALUATION_PATH, test_result)
    receipt.update({
        "status": "FINAL_TEST_COMPLETE",
        "completed_at_utc": _utc_now(),
        "test_evaluation_count": 1,
        "test_evaluation_path": str(TEST_EVALUATION_PATH.resolve()),
        "test_predictions_path": str(TEST_PREDICTIONS_PATH.resolve()),
    })
    _atomic_json(ATTEMPT_RECEIPT_PATH, receipt)
    return test_result, predictions


def _artifact_hashes() -> Dict[str, Dict[str, Any]]:
    paths = (
        CHECKPOINT_VERIFICATION_PATH,
        VALIDATION_EVALUATION_PATH,
        TEST_EVALUATION_PATH,
        TEST_PREDICTIONS_PATH,
        ATTEMPT_RECEIPT_PATH,
    )
    return {
        path.name: {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in paths
    }


def _complete_manifest(
    checkpoint: Dict[str, Any], validation: Dict[str, Any], test: Dict[str, Any]
) -> Dict[str, Any]:
    manifest = {
        "status": "FINAL_TEST_COMPLETE",
        "created_at_utc": _utc_now(),
        "python_version": platform.python_version(),
        "transformers_version": version("transformers"),
        "torch_version": torch.__version__,
        "checkpoint": checkpoint["checkpoint"],
        "checkpoint_weight_sha256": checkpoint["weight_sha256"],
        "selection_record_sha256": checkpoint["selection_record_sha256"],
        "selected_epoch": checkpoint["epoch"],
        "selected_step": checkpoint["step"],
        "validation_samples": validation["sample_count"],
        "test_samples": test["sample_count"],
        "checkpoint_selected_before_test": True,
        "validation_verified_before_test": True,
        "test_used_for_selection": False,
        "test_evaluation_count": 1,
        "training_performed": False,
        "backward_called": False,
        "optimizer_step_performed": False,
        "scheduler_step_performed": False,
        "checkpoint_changed_after_test": False,
        "phase_12_started": False,
        "artifact_files": _artifact_hashes(),
    }
    _atomic_json(MANIFEST_PATH, manifest)
    return manifest


def _valid_existing_evaluation() -> Dict[str, Any] | None:
    if not MANIFEST_PATH.is_file():
        return None
    manifest = _load_json(MANIFEST_PATH)
    if not (
        manifest.get("status") == "FINAL_TEST_COMPLETE"
        and manifest.get("test_evaluation_count") == 1
        and manifest.get("test_used_for_selection") is False
        and manifest.get("validation_verified_before_test") is True
        and manifest.get("training_performed") is False
        and manifest.get("checkpoint_changed_after_test") is False
        and manifest.get("phase_12_started") is False
    ):
        return None
    for metadata in manifest.get("artifact_files", {}).values():
        path = Path(metadata["path"])
        if not path.is_file() or path.stat().st_size != int(metadata["bytes"]):
            return None
        if _sha256_file(path) != metadata["sha256"]:
            return None
    receipt = _load_json(ATTEMPT_RECEIPT_PATH)
    test = _load_json(TEST_EVALUATION_PATH)
    predictions = _load_json(TEST_PREDICTIONS_PATH)
    validation = _load_json(VALIDATION_EVALUATION_PATH)
    checkpoint = _load_json(CHECKPOINT_VERIFICATION_PATH)
    weight_path = Path(checkpoint.get("weight_file", ""))
    checkpoint_fingerprint_valid = (
        weight_path.is_file()
        and _sha256_file(weight_path) == manifest.get("checkpoint_weight_sha256")
    )
    selection_fingerprint_valid = (
        SELECTED_PATH.is_file()
        and _sha256_json(_load_json(SELECTED_PATH))
        == manifest.get("selection_record_sha256")
    )
    if not (
        receipt.get("status") == "FINAL_TEST_COMPLETE"
        and receipt.get("test_evaluation_count") == 1
        and test.get("status") == "PASS"
        and test.get("test_evaluation_count") == 1
        and validation.get("status") == "PASS"
        and checkpoint.get("status") == "PASS"
        and len(predictions) == EXPECTED_COUNTS["test"]
        and checkpoint.get("weight_sha256") == manifest.get("checkpoint_weight_sha256")
        and checkpoint_fingerprint_valid
        and selection_fingerprint_valid
    ):
        return None
    return manifest


def load_phase_11_report() -> Dict[str, Any]:
    """Load verified Phase 11 artifacts without model/Test evaluation."""
    manifest = _valid_existing_evaluation()
    if manifest is None:
        raise RuntimeError("No complete verified Phase 11 evaluation is available")
    return {
        "manifest": manifest,
        "checkpoint": _load_json(CHECKPOINT_VERIFICATION_PATH),
        "validation": _load_json(VALIDATION_EVALUATION_PATH),
        "test": _load_json(TEST_EVALUATION_PATH),
        "receipt": _load_json(ATTEMPT_RECEIPT_PATH),
        "prediction_count": len(_load_json(TEST_PREDICTIONS_PATH)),
        "prediction_path": str(TEST_PREDICTIONS_PATH.resolve()),
    }


def run_or_load_phase_11(
    validation_dataset: Any,
    data_collator: Any,
    tokenizer: Any,
    test_data_provider: Callable[[], Tuple[Any, Any]],
) -> Dict[str, Any]:
    """Run final evaluation once; later calls load artifacts without Test access."""
    existing = _valid_existing_evaluation()
    if existing is not None:
        return {
            **load_phase_11_report(),
            "guard_action": "loaded_verified_phase_11_artifacts_no_test_reevaluation",
        }
    if PHASE_11_DIR.exists() and any(PHASE_11_DIR.iterdir()):
        raise RuntimeError(
            "Phase 11 contains partial/unverified artifacts; refusing automatic Test retry"
        )

    model, checkpoint = verify_authoritative_checkpoint()
    trainer = _evaluation_trainer(model, data_collator, tokenizer)
    validation = verify_reloaded_validation(trainer, validation_dataset, checkpoint)
    test, _ = _evaluate_test_once(
        trainer, test_data_provider, checkpoint, validation
    )
    manifest = _complete_manifest(checkpoint, validation, test)
    report = load_phase_11_report()
    return {
        **report,
        "manifest": manifest,
        "guard_action": "executed_validation_then_one_final_test_evaluation",
    }
