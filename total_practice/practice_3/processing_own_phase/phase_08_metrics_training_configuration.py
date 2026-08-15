"""Phase 8: metrics, baseline TrainingArguments, and checkpoint policy."""

import json
import math
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import EvalPrediction, TrainingArguments

from .config import RANDOM_SEED, RESULT_DIR


METRIC_NAMES = ("accuracy", "precision", "recall", "f1")
POSITIVE_CLASS = 1
BASELINE = {
    "num_train_epochs": 3.0,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "seed": RANDOM_SEED,
    "data_seed": RANDOM_SEED,
}


def compute_metrics(eval_prediction: EvalPrediction) -> Dict[str, float]:
    """Compute binary metrics with label 1 as positive and zero_division=0."""
    predictions = eval_prediction.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    logits = np.asarray(predictions)
    labels = np.asarray(eval_prediction.label_ids).reshape(-1)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"Expected binary logits [N, 2], received {logits.shape}")
    predicted_labels = np.argmax(logits, axis=-1).reshape(-1)
    if len(predicted_labels) != len(labels):
        raise ValueError("Prediction and label counts do not match")
    if not set(labels.tolist()).issubset({0, 1}):
        raise ValueError("Labels must be a subset of {0, 1}")
    if not set(predicted_labels.tolist()).issubset({0, 1}):
        raise ValueError("Predictions must be a subset of {0, 1}")

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predicted_labels,
        average="binary",
        pos_label=POSITIVE_CLASS,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predicted_labels)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def create_training_arguments(output_dir: str | Path | None = None) -> TrainingArguments:
    """Construct the approved Transformers 5.14.1 baseline configuration."""
    checkpoint_dir = output_dir or RESULT_DIR / "phase_09_training" / "checkpoints"
    return TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=BASELINE["num_train_epochs"],
        per_device_train_batch_size=BASELINE["per_device_train_batch_size"],
        per_device_eval_batch_size=BASELINE["per_device_eval_batch_size"],
        learning_rate=BASELINE["learning_rate"],
        weight_decay=BASELINE["weight_decay"],
        seed=BASELINE["seed"],
        data_seed=BASELINE["data_seed"],
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )


def select_best_checkpoint(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Rank validation records by loss, F1, then earlier epoch/checkpoint."""
    candidates: List[Dict[str, Any]] = [dict(record) for record in records]
    if not candidates:
        raise ValueError("At least one validation checkpoint record is required")
    required = {"checkpoint", "epoch", "eval_loss", "eval_f1"}
    for record in candidates:
        if any("test" in str(key).lower() for key in record):
            raise ValueError("Test fields are prohibited from checkpoint selection")
        if not required.issubset(record):
            raise ValueError(f"Checkpoint record is missing {required - set(record)}")
        if not math.isfinite(float(record["eval_loss"])):
            raise ValueError("Validation loss must be finite")
        if not math.isfinite(float(record["eval_f1"])):
            raise ValueError("Validation F1 must be finite")
    return min(
        candidates,
        key=lambda item: (
            float(item["eval_loss"]),
            -float(item["eval_f1"]),
            float(item["epoch"]),
            str(item["checkpoint"]),
        ),
    )


def _strategy_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def verify_phase_08(training_args: TrainingArguments) -> Dict[str, Any]:
    """Verify metrics, effective arguments and validation-only selection policy."""
    fixture_logits = np.array([[3.0, 1.0], [0.1, 0.9], [0.2, 0.8], [0.8, 0.2]])
    fixture_labels = np.array([0, 0, 1, 1])
    fixture_predictions = np.argmax(fixture_logits, axis=-1).tolist()
    metrics = compute_metrics(EvalPrediction(fixture_logits, fixture_labels))
    expected_metrics = {name: 0.5 for name in METRIC_NAMES}
    known_answer_pass = all(
        math.isclose(metrics[name], expected_metrics[name], abs_tol=1e-12)
        for name in METRIC_NAMES
    )
    zero_division_metrics = compute_metrics(EvalPrediction(
        np.array([[2.0, 0.0], [1.0, 0.0]]), np.array([0, 0])
    ))
    zero_division_pass = all(
        zero_division_metrics[name] == 0.0 for name in ("precision", "recall", "f1")
    )
    metrics_finite_and_in_range = all(
        math.isfinite(value) and 0.0 <= value <= 1.0 for value in metrics.values()
    )

    tie_break_records = [
        {"checkpoint": "checkpoint-1", "epoch": 1, "eval_loss": 0.50, "eval_f1": 0.70},
        {"checkpoint": "checkpoint-2", "epoch": 2, "eval_loss": 0.40, "eval_f1": 0.75},
        {"checkpoint": "checkpoint-3", "epoch": 3, "eval_loss": 0.40, "eval_f1": 0.82},
    ]
    tie_break_winner = select_best_checkpoint(tie_break_records)
    exact_tie_records = [
        {"checkpoint": "checkpoint-2", "epoch": 2, "eval_loss": 0.40, "eval_f1": 0.82},
        {"checkpoint": "checkpoint-3", "epoch": 3, "eval_loss": 0.40, "eval_f1": 0.82},
    ]
    exact_tie_winner = select_best_checkpoint(exact_tie_records)
    policy_pass = (
        tie_break_winner["checkpoint"] == "checkpoint-3"
        and exact_tie_winner["checkpoint"] == "checkpoint-2"
    )

    effective_arguments = {
        "output_dir": training_args.output_dir,
        "num_train_epochs": training_args.num_train_epochs,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
        "learning_rate": training_args.learning_rate,
        "weight_decay": training_args.weight_decay,
        "seed": training_args.seed,
        "data_seed": training_args.data_seed,
        "eval_strategy": _strategy_value(training_args.eval_strategy),
        "logging_strategy": _strategy_value(training_args.logging_strategy),
        "save_strategy": _strategy_value(training_args.save_strategy),
        "load_best_model_at_end": training_args.load_best_model_at_end,
        "metric_for_best_model": training_args.metric_for_best_model,
        "greater_is_better": training_args.greater_is_better,
        "report_to": training_args.report_to,
    }
    arguments_pass = (
        effective_arguments["num_train_epochs"] == 3.0
        and effective_arguments["per_device_train_batch_size"] == 16
        and effective_arguments["per_device_eval_batch_size"] == 32
        and math.isclose(effective_arguments["learning_rate"], 2e-5)
        and math.isclose(effective_arguments["weight_decay"], 0.01)
        and effective_arguments["seed"] == 42
        and effective_arguments["data_seed"] == 42
        and effective_arguments["eval_strategy"] == "epoch"
        and effective_arguments["logging_strategy"] == "epoch"
        and effective_arguments["save_strategy"] == "epoch"
        and effective_arguments["load_best_model_at_end"] is True
        and effective_arguments["metric_for_best_model"] == "eval_loss"
        and effective_arguments["greater_is_better"] is False
        and effective_arguments["report_to"] == []
    )
    all_pass = all([
        set(metrics) == set(METRIC_NAMES), known_answer_pass,
        zero_division_pass, metrics_finite_and_in_range,
        policy_pass, arguments_pass,
    ])
    return {
        "status": "PASS" if all_pass else "FAIL",
        "transformers_version": version("transformers"),
        "metrics_verification": {
            "metric_names": list(METRIC_NAMES),
            "positive_class": POSITIVE_CLASS,
            "zero_division": 0,
            "fixture_labels": fixture_labels.tolist(),
            "fixture_predictions": fixture_predictions,
            "expected_metrics": expected_metrics,
            "executed_metrics": metrics,
            "known_answer_pass": known_answer_pass,
            "finite_and_in_range": metrics_finite_and_in_range,
            "zero_division_fixture_metrics": zero_division_metrics,
            "zero_division_pass": zero_division_pass,
        },
        "effective_training_arguments": effective_arguments,
        "training_arguments_verification_pass": arguments_pass,
        "checkpoint_policy": {
            "primary": "minimum validation loss",
            "tie_breaker": "maximum validation F1",
            "remaining_tie": "earlier epoch/checkpoint",
            "records_source": "synthetic validation records",
            "tie_break_records": tie_break_records,
            "tie_break_winner": tie_break_winner,
            "exact_tie_records": exact_tie_records,
            "exact_tie_winner": exact_tie_winner,
            "validation_only": True,
            "test_fields_used": False,
            "verification_pass": policy_pass,
        },
        "trainer_created": False,
        "training_performed": False,
        "backward_called": False,
        "optimizer_step_performed": False,
        "scheduler_step_performed": False,
        "test_accessed": False,
        "checkpoint_saved": False,
    }


def save_phase_08_verification(
    verification: Dict[str, Any], path: Path | None = None
) -> Path:
    """Save executed Phase 8 evidence and verify exact JSON read-back."""
    output_path = path or RESULT_DIR / "phase_08_metrics_training_configuration_verification.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    readback = json.loads(output_path.read_text(encoding="utf-8"))
    if readback != json.loads(json.dumps(verification)):
        raise RuntimeError("Phase 8 artifact read-back verification failed")
    return output_path
