"""Validation metric contract used by the Practice 3 Hugging Face runners.

This repaired module is intentionally evaluation-only: it creates no Trainer,
TrainingArguments, optimizer, scheduler, checkpoint, or result artifact.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(eval_prediction: Any) -> dict[str, float]:
    """Return the established binary metrics for a Trainer evaluation batch.

    Hugging Face passes either an ``EvalPrediction`` object or a two-item
    ``(predictions, label_ids)`` tuple. Some model outputs wrap logits in a
    tuple, so the first prediction item is used in that case. Class ``1`` is
    the positive class and undefined positive-class metrics resolve to 0.0,
    matching the historical Phase 08 contract.
    """
    if hasattr(eval_prediction, "predictions") and hasattr(eval_prediction, "label_ids"):
        logits = eval_prediction.predictions
        labels = eval_prediction.label_ids
    else:
        try:
            logits, labels = eval_prediction
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "eval_prediction must provide predictions and label_ids"
            ) from exc

    if isinstance(logits, (tuple, list)):
        if not logits:
            raise ValueError("Predictions tuple is empty")
        logits = logits[0]
    logits_array = np.asarray(logits)
    labels_array = np.asarray(labels).reshape(-1)
    if logits_array.ndim != 2 or logits_array.shape[1] != 2:
        raise ValueError(
            f"Expected binary logits with shape (n, 2), got {logits_array.shape}"
        )
    if logits_array.shape[0] != labels_array.shape[0]:
        raise ValueError("Predictions and labels have different sample counts")
    if labels_array.size == 0:
        raise ValueError("Cannot compute metrics for an empty evaluation batch")

    predictions = np.argmax(logits_array, axis=1)
    return {
        "accuracy": float(accuracy_score(labels_array, predictions)),
        "precision": float(
            precision_score(labels_array, predictions, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(labels_array, predictions, pos_label=1, zero_division=0)
        ),
        "f1": float(f1_score(labels_array, predictions, pos_label=1, zero_division=0)),
    }


__all__ = ("compute_metrics",)
