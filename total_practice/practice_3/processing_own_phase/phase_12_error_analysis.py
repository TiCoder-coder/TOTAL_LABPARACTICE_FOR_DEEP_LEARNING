"""Phase 12: artifact-only confusion matrix and deterministic error analysis."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib
import numpy as np

from .config import RESULT_DIR


# Phase 12 writes a PNG artifact and never needs an interactive GUI backend.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PHASE_11_DIR = RESULT_DIR / "phase_11_evaluation"
PREDICTIONS_PATH = PHASE_11_DIR / "test_predictions.json"
TEST_EVALUATION_PATH = PHASE_11_DIR / "test_evaluation.json"
PHASE_11_MANIFEST_PATH = PHASE_11_DIR / "phase_11_evaluation_manifest.json"

CONFUSION_FIGURE_PATH = RESULT_DIR / "phase_12_confusion_matrix.png"
CONFUSION_JSON_PATH = RESULT_DIR / "phase_12_confusion_matrix.json"
ERROR_SAMPLES_PATH = RESULT_DIR / "phase_12_error_samples.json"
ERROR_ANALYSIS_PATH = RESULT_DIR / "phase_12_error_analysis.json"

EXPECTED_COUNT = 1066
METRIC_TOLERANCE = 1e-12
LABEL_NAMES = {0: "NEGATIVE", 1: "POSITIVE"}
REQUIRED_FIELDS = {
    "sample_index", "sample_id", "text", "true_label", "true_label_name",
    "predicted_label", "predicted_label_name", "negative_probability",
    "positive_probability", "confidence", "correct",
}
ERROR_DIRECTIONS = ("NEGATIVE_TO_POSITIVE", "POSITIVE_TO_NEGATIVE")
NEGATION_PATTERN = re.compile(r"\b(?:not|no|never)\b|n['’]t\b", re.IGNORECASE)
CONTRAST_PATTERN = re.compile(
    r"\b(?:but|however|although|though|yet|while)\b", re.IGNORECASE
)


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen Phase 11 artifact is missing: {path}")
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_file_check(manifest: Dict[str, Any], filename: str, path: Path) -> bool:
    metadata = manifest.get("artifact_files", {}).get(filename, {})
    return (
        path.is_file()
        and path.stat().st_size == int(metadata.get("bytes", -1))
        and _sha256(path) == metadata.get("sha256")
    )


def _validate_frozen_inputs(
    predictions: list[Dict[str, Any]],
    test_evaluation: Dict[str, Any],
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    manifest_pass = (
        manifest.get("status") == "FINAL_TEST_COMPLETE"
        and manifest.get("test_evaluation_count") == 1
        and manifest.get("test_used_for_selection") is False
        and manifest.get("training_performed") is False
        and manifest.get("checkpoint_changed_after_test") is False
        and manifest.get("phase_12_started") is False
    )
    frozen_hashes_pass = (
        _manifest_file_check(manifest, "test_predictions.json", PREDICTIONS_PATH)
        and _manifest_file_check(manifest, "test_evaluation.json", TEST_EVALUATION_PATH)
    )
    test_evaluation_pass = (
        test_evaluation.get("status") == "PASS"
        and test_evaluation.get("test_evaluation_count") == 1
        and test_evaluation.get("test_used_for_selection") is False
        and test_evaluation.get("sample_count") == EXPECTED_COUNT
        and test_evaluation.get("prediction_count") == EXPECTED_COUNT
    )
    count_pass = len(predictions) == EXPECTED_COUNT
    indices = [int(record.get("sample_index", -1)) for record in predictions]
    ids = [record.get("sample_id") for record in predictions]
    identity_pass = (
        indices == list(range(EXPECTED_COUNT))
        and len(set(indices)) == EXPECTED_COUNT
        and len(set(ids)) == EXPECTED_COUNT
    )
    schema_pass = all(REQUIRED_FIELDS.issubset(record) for record in predictions)
    record_pass = True
    if schema_pass:
        for record in predictions:
            true_label = int(record["true_label"])
            predicted_label = int(record["predicted_label"])
            probabilities = (
                float(record["negative_probability"]),
                float(record["positive_probability"]),
            )
            confidence = float(record["confidence"])
            expected_confidence = probabilities[predicted_label] if predicted_label in (0, 1) else math.nan
            valid = (
                true_label in (0, 1)
                and predicted_label in (0, 1)
                and record["true_label_name"] == LABEL_NAMES.get(true_label)
                and record["predicted_label_name"] == LABEL_NAMES.get(predicted_label)
                and bool(record["correct"]) == (true_label == predicted_label)
                and all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (*probabilities, confidence))
                and abs(sum(probabilities) - 1.0) <= 1e-6
                and abs(confidence - expected_confidence) <= 1e-12
                and isinstance(record["text"], str)
            )
            if not valid:
                record_pass = False
                break
    else:
        record_pass = False
    checks = {
        "phase_11_manifest_pass": manifest_pass,
        "frozen_artifact_hashes_pass": frozen_hashes_pass,
        "phase_11_test_evaluation_pass": test_evaluation_pass,
        "prediction_count_1066": count_pass,
        "sample_identity_complete_ordered_unique": identity_pass,
        "prediction_schema_pass": schema_pass,
        "all_prediction_records_valid": record_pass,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Frozen Phase 11 input verification failed: {checks}")
    return checks


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _confusion_result(
    predictions: list[Dict[str, Any]], test_evaluation: Dict[str, Any]
) -> Dict[str, Any]:
    tn = sum(record["true_label"] == 0 and record["predicted_label"] == 0 for record in predictions)
    fp = sum(record["true_label"] == 0 and record["predicted_label"] == 1 for record in predictions)
    fn = sum(record["true_label"] == 1 and record["predicted_label"] == 0 for record in predictions)
    tp = sum(record["true_label"] == 1 and record["predicted_label"] == 1 for record in predictions)
    counts = {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    total = sum(counts.values())
    actual_negative = tn + fp
    actual_positive = fn + tp
    correct = tn + tp
    incorrect = fp + fn
    cell_definitions = (
        ("TN", "NEGATIVE", "NEGATIVE", tn, actual_negative),
        ("FP", "NEGATIVE", "POSITIVE", fp, actual_negative),
        ("FN", "POSITIVE", "NEGATIVE", fn, actual_positive),
        ("TP", "POSITIVE", "POSITIVE", tp, actual_positive),
    )
    cells = [
        {
            "cell": name,
            "actual": actual,
            "predicted": predicted,
            "count": int(count),
            "overall_rate": _safe_divide(count, total),
            "row_rate": _safe_divide(count, row_total),
        }
        for name, actual, predicted, count, row_total in cell_definitions
    ]
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    derived = {
        "accuracy": _safe_divide(correct, total),
        "precision": precision,
        "recall": recall,
        "f1": _safe_divide(2 * precision * recall, precision + recall),
    }
    comparisons = []
    for metric, derived_value in derived.items():
        phase_11_value = float(test_evaluation["metrics"][metric])
        delta = abs(derived_value - phase_11_value)
        comparisons.append({
            "metric": metric,
            "phase_11_value": phase_11_value,
            "derived_value": derived_value,
            "absolute_delta": delta,
            "tolerance": METRIC_TOLERANCE,
            "pass": delta <= METRIC_TOLERANCE,
        })
    class_level = [
        {
            "actual_class": "NEGATIVE",
            "total": int(actual_negative),
            "correct": int(tn),
            "errors": int(fp),
            "conditional_accuracy": _safe_divide(tn, actual_negative),
            "error_rate": _safe_divide(fp, actual_negative),
            "error_direction": "NEGATIVE_TO_POSITIVE",
        },
        {
            "actual_class": "POSITIVE",
            "total": int(actual_positive),
            "correct": int(tp),
            "errors": int(fn),
            "conditional_accuracy": _safe_divide(tp, actual_positive),
            "error_rate": _safe_divide(fn, actual_positive),
            "error_direction": "POSITIVE_TO_NEGATIVE",
        },
    ]
    matrix = [[int(tn), int(fp)], [int(fn), int(tp)]]
    row_rates = [
        [_safe_divide(tn, actual_negative), _safe_divide(fp, actual_negative)],
        [_safe_divide(fn, actual_positive), _safe_divide(tp, actual_positive)],
    ]
    checks = {
        "confusion_sum_1066": total == EXPECTED_COUNT,
        "correct_plus_incorrect_1066": correct + incorrect == EXPECTED_COUNT,
        "diagonal_equals_correct": correct == sum(bool(record["correct"]) for record in predictions),
        "off_diagonal_equals_incorrect": incorrect == sum(not bool(record["correct"]) for record in predictions),
        "row_totals_match_true_labels": (
            actual_negative == sum(record["true_label"] == 0 for record in predictions)
            and actual_positive == sum(record["true_label"] == 1 for record in predictions)
        ),
        "column_totals_match_predictions": (
            tn + fn == sum(record["predicted_label"] == 0 for record in predictions)
            and fp + tp == sum(record["predicted_label"] == 1 for record in predictions)
        ),
        "counts_non_negative_integers": all(isinstance(value, int) and value >= 0 for value in counts.values()),
        "rates_finite_in_range": all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for cell in cells for value in (cell["overall_rate"], cell["row_rate"])
        ),
        "row_rates_sum_one": all(abs(sum(row) - 1.0) <= METRIC_TOLERANCE for row in row_rates),
        "derived_metrics_match_phase_11": all(row["pass"] for row in comparisons),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "source": "frozen_phase_11_predictions_only",
        "total": total,
        "correct": int(correct),
        "incorrect": int(incorrect),
        "counts": counts,
        "matrix_actual_rows_predicted_columns": matrix,
        "cells": cells,
        "row_percentages": [[100 * value for value in row] for row in row_rates],
        "class_level": class_level,
        "derived_metrics": derived,
        "phase_11_metric_comparisons": comparisons,
        "validation_checks": checks,
    }


def _text_flags(text: str) -> Dict[str, bool]:
    return {
        "negation_cue": bool(NEGATION_PATTERN.search(text)),
        "contrast_cue": bool(CONTRAST_PATTERN.search(text)),
        "question_mark": "?" in text,
        "exclamation_mark": "!" in text,
    }


def _enrich(record: Dict[str, Any]) -> Dict[str, Any]:
    true_label = int(record["true_label"])
    predicted_label = int(record["predicted_label"])
    direction = None
    if true_label == 0 and predicted_label == 1:
        direction = "NEGATIVE_TO_POSITIVE"
    elif true_label == 1 and predicted_label == 0:
        direction = "POSITIVE_TO_NEGATIVE"
    return {
        **record,
        "character_length": len(record["text"]),
        "word_length": len(record["text"].split()),
        "lexical_flags": _text_flags(record["text"]),
        "error_direction": direction,
    }


def _describe(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    result: Dict[str, Any] = {"count": len(rows)}
    for field in ("confidence", "character_length", "word_length"):
        values = np.asarray([float(row[field]) for row in rows], dtype=float)
        if len(values) == 0:
            result[field] = {
                "mean": None, "median": None, "minimum": None,
                "maximum": None, "p25": None, "p75": None, "p90": None,
            }
            continue
        result[field] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75)),
            "p90": float(np.percentile(values, 90)),
        }
    return result


def _flag_summary(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    flags = ("negation_cue", "contrast_cue", "question_mark", "exclamation_mark")
    return {
        flag: {
            "count": sum(bool(row["lexical_flags"][flag]) for row in rows),
            "rate": _safe_divide(
                sum(bool(row["lexical_flags"][flag]) for row in rows), len(rows)
            ),
        }
        for flag in flags
    }


def _error_results(predictions: list[Dict[str, Any]], confusion: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    enriched = [_enrich(record) for record in predictions]
    errors = [record for record in enriched if not record["correct"]]
    errors = sorted(errors, key=lambda row: (-float(row["confidence"]), int(row["sample_index"])))
    for rank, record in enumerate(errors, start=1):
        record["global_error_rank"] = rank
    by_direction: Dict[str, list[Dict[str, Any]]] = {}
    representatives = []
    for direction in ERROR_DIRECTIONS:
        direction_rows = [row for row in errors if row["error_direction"] == direction]
        direction_rows = sorted(
            direction_rows,
            key=lambda row: (-float(row["confidence"]), int(row["sample_index"])),
        )
        for direction_rank, record in enumerate(direction_rows, start=1):
            record["direction_rank"] = direction_rank
        by_direction[direction] = direction_rows
        for record in direction_rows[:5]:
            representatives.append({
                **record,
                "selection_reason": "top_confidence_within_error_direction",
            })
    error_samples = {
        "status": "PASS",
        "source": "frozen_phase_11_predictions_only",
        "selection_policy": {
            "primary_sort": "confidence_descending",
            "tie_breaker": "sample_index_ascending",
            "representative_rule": "first_five_per_error_direction",
            "manual_cherry_pick": False,
        },
        "total_errors": len(errors),
        "direction_counts": {key: len(value) for key, value in by_direction.items()},
        "representative_count": len(representatives),
        "representative_errors": representatives,
        "all_ranked_errors": errors,
    }
    groups = {
        "correct": [row for row in enriched if row["correct"]],
        "incorrect": errors,
        "negative_to_positive": by_direction["NEGATIVE_TO_POSITIVE"],
        "positive_to_negative": by_direction["POSITIVE_TO_NEGATIVE"],
        "actual_negative": [row for row in enriched if row["true_label"] == 0],
        "actual_positive": [row for row in enriched if row["true_label"] == 1],
    }
    direction_order_pass = all(
        rows == sorted(rows, key=lambda row: (-float(row["confidence"]), int(row["sample_index"])))
        for rows in by_direction.values()
    )
    representative_ids = [row["sample_id"] for row in representatives]
    checks = {
        "all_and_only_incorrect_preserved": (
            len(errors) == confusion["incorrect"]
            and {row["sample_id"] for row in errors}
            == {row["sample_id"] for row in predictions if not row["correct"]}
        ),
        "direction_counts_match_fp_fn": (
            len(by_direction["NEGATIVE_TO_POSITIVE"]) == confusion["counts"]["FP"]
            and len(by_direction["POSITIVE_TO_NEGATIVE"]) == confusion["counts"]["FN"]
        ),
        "global_sort_pass": errors == sorted(
            errors, key=lambda row: (-float(row["confidence"]), int(row["sample_index"]))
        ),
        "direction_sort_pass": direction_order_pass,
        "representative_first_five_per_direction": all(
            [row["sample_id"] for row in representatives if row["error_direction"] == direction]
            == [row["sample_id"] for row in by_direction[direction][:5]]
            for direction in ERROR_DIRECTIONS
        ),
        "representative_ids_unique": len(representative_ids) == len(set(representative_ids)),
        "representative_count_expected": len(representatives) == sum(min(5, len(by_direction[key])) for key in ERROR_DIRECTIONS),
    }
    interpretation = (
        f"The frozen artifact contains {len(errors)} errors: "
        f"{len(by_direction['NEGATIVE_TO_POSITIVE'])} NEGATIVE→POSITIVE and "
        f"{len(by_direction['POSITIVE_TO_NEGATIVE'])} POSITIVE→NEGATIVE. "
        f"The highest-confidence error is {errors[0]['confidence']:.4f}. "
        "Length, confidence and lexical flags below are descriptive co-occurrences "
        "only; they do not establish that a text feature caused an error."
    )
    analysis = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "source": "frozen_phase_11_predictions_only",
        "group_summaries": {name: _describe(rows) for name, rows in groups.items()},
        "lexical_flag_summaries": {name: _flag_summary(rows) for name, rows in groups.items()},
        "error_direction_counts": error_samples["direction_counts"],
        "representative_count": len(representatives),
        "interpretation": interpretation,
        "validation_checks": checks,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "dataset_loaded": False,
        "test_provider_called": False,
        "test_evaluated": False,
        "training_performed": False,
        "prediction_performed": False,
        "checkpoint_changed": False,
        "phase_13_started": False,
    }
    return error_samples, analysis


def _plot_confusion(confusion: Dict[str, Any]) -> None:
    matrix = np.asarray(confusion["matrix_actual_rows_predicted_columns"], dtype=int)
    percentages = np.asarray(confusion["row_percentages"], dtype=float)
    figure, axis = plt.subplots(figsize=(7.5, 6.2))
    image = axis.imshow(matrix, cmap="Blues")
    labels = ("NEGATIVE", "POSITIVE")
    axis.set_xticks([0, 1], labels=labels)
    axis.set_yticks([0, 1], labels=labels)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Actual label")
    axis.set_title("Confusion Matrix — Frozen Final-Test Predictions")
    threshold = matrix.max() / 2
    for row in range(2):
        for column in range(2):
            color = "white" if matrix[row, column] > threshold else "black"
            axis.text(
                column,
                row,
                f"{matrix[row, column]}\n{percentages[row, column]:.2f}% of actual class",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
                fontweight="bold",
            )
    figure.colorbar(image, ax=axis, label="Prediction count")
    figure.tight_layout()
    figure.savefig(CONFUSION_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def analyze_phase_12() -> Dict[str, Any]:
    """Create Phase 12 results solely from frozen Phase 11 JSON artifacts."""
    predictions = _load_json(PREDICTIONS_PATH)
    test_evaluation = _load_json(TEST_EVALUATION_PATH)
    manifest = _load_json(PHASE_11_MANIFEST_PATH)
    source_hashes_before = {
        "test_predictions.json": _sha256(PREDICTIONS_PATH),
        "test_evaluation.json": _sha256(TEST_EVALUATION_PATH),
        "phase_11_evaluation_manifest.json": _sha256(PHASE_11_MANIFEST_PATH),
    }
    input_checks = _validate_frozen_inputs(predictions, test_evaluation, manifest)
    confusion = _confusion_result(predictions, test_evaluation)
    error_samples, analysis = _error_results(predictions, confusion)
    if confusion["status"] != "PASS" or analysis["status"] != "PASS":
        raise RuntimeError("Phase 12 derived verification failed before artifact save")

    _plot_confusion(confusion)
    image = plt.imread(CONFUSION_FIGURE_PATH)
    figure_check = {
        "path": str(CONFUSION_FIGURE_PATH.resolve()),
        "exists": CONFUSION_FIGURE_PATH.is_file(),
        "non_empty": CONFUSION_FIGURE_PATH.stat().st_size > 0,
        "decodable": image.ndim in (2, 3),
        "pixel_shape": list(image.shape),
    }
    source_hashes_after = {
        "test_predictions.json": _sha256(PREDICTIONS_PATH),
        "test_evaluation.json": _sha256(TEST_EVALUATION_PATH),
        "phase_11_evaluation_manifest.json": _sha256(PHASE_11_MANIFEST_PATH),
    }
    frozen_sources_unchanged = source_hashes_before == source_hashes_after
    confusion["source_artifacts"] = {
        "test_predictions": str(PREDICTIONS_PATH.resolve()),
        "test_evaluation": str(TEST_EVALUATION_PATH.resolve()),
        "phase_11_manifest": str(PHASE_11_MANIFEST_PATH.resolve()),
    }
    confusion["input_validation_checks"] = input_checks
    confusion["figure"] = figure_check
    confusion["frozen_source_hashes"] = source_hashes_before
    confusion["frozen_sources_unchanged"] = frozen_sources_unchanged
    analysis["confusion_counts"] = confusion["counts"]
    analysis["confusion_figure"] = figure_check
    analysis["frozen_source_hashes"] = source_hashes_before
    analysis["frozen_sources_unchanged"] = frozen_sources_unchanged
    analysis["test_evaluation_count"] = manifest["test_evaluation_count"]
    analysis["artifacts"] = {
        "confusion_figure": str(CONFUSION_FIGURE_PATH.resolve()),
        "confusion_json": str(CONFUSION_JSON_PATH.resolve()),
        "error_samples": str(ERROR_SAMPLES_PATH.resolve()),
        "error_analysis": str(ERROR_ANALYSIS_PATH.resolve()),
    }
    analysis["status"] = "PASS" if (
        analysis["status"] == "PASS"
        and frozen_sources_unchanged
        and all(input_checks.values())
        and all(figure_check[key] for key in ("exists", "non_empty", "decodable"))
    ) else "FAIL"

    _atomic_json(CONFUSION_JSON_PATH, confusion)
    _atomic_json(ERROR_SAMPLES_PATH, error_samples)
    _atomic_json(ERROR_ANALYSIS_PATH, analysis)
    if analysis["status"] != "PASS":
        raise RuntimeError("Phase 12 final verification failed")
    return {
        "status": analysis["status"],
        "confusion": confusion,
        "error_samples": error_samples,
        "analysis": analysis,
    }
