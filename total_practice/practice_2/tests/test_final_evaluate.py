"""Tests for final checkpoint verification and artifact consistency."""

import json
import numpy as np
import pandas as pd
import pytest

from configs import CLASS_NAMES
from processing_own_phase.final_evaluate import (
    _assert_validation_match,
    _load_selection_record,
    _top_prediction_indices,
    _validate_confusion_matrix,
    validate_final_artifacts,
)


def test_top_prediction_indices_selects_correct_and_incorrect():
    predictions = np.array([0, 1, 1, 0, 1])
    labels = np.array([0, 0, 1, 1, 1])
    probabilities = np.array(
        [
            [0.90, 0.10],
            [0.20, 0.80],
            [0.30, 0.70],
            [0.65, 0.35],
            [0.40, 0.60],
        ]
    )

    selected = _top_prediction_indices(
        predictions,
        labels,
        probabilities,
        top_k=2,
    )

    assert selected.tolist() == [0, 2, 1, 3]


def test_selection_record_requires_validation_only_checkpoint(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps({
        "selection_metric": "val_accuracy",
        "selection_source": "validation_only",
        "selected_experiment": "E2_resnet18_partial",
        "selected_checkpoint": str(checkpoint),
        "best_val_accuracy": 0.89,
        "test_data_used": False,
    }))

    selection = _load_selection_record(str(selection_path))

    assert selection["selected_checkpoint"] == str(checkpoint.resolve())
    assert selection["best_val_accuracy"] == pytest.approx(0.89)


def test_selection_record_accepts_validation_loss_selection(tmp_path):
    checkpoint = tmp_path / "best_val_loss.pt"
    checkpoint.touch()
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps({
        "selection_metric": "val_loss",
        "selection_source": "validation_only",
        "selected_experiment": "winner",
        "selected_checkpoint": str(checkpoint),
        "best_val_accuracy": 0.89,
        "best_val_loss": 0.4,
        "test_data_used": False,
    }))
    assert _load_selection_record(str(selection_path))["best_val_loss"] == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("selection_metric", "test_accuracy"),
        ("selection_source", "test_assisted"),
        ("test_data_used", True),
    ],
)
def test_selection_record_rejects_non_validation_selection(
    tmp_path,
    field,
    value,
):
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    record = {
        "selection_metric": "val_accuracy",
        "selection_source": "validation_only",
        "selected_experiment": "E2_resnet18_partial",
        "selected_checkpoint": str(checkpoint),
        "best_val_accuracy": 0.89,
        "test_data_used": False,
    }
    record[field] = value
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(record))

    with pytest.raises(ValueError):
        _load_selection_record(str(selection_path))


def test_validation_mismatch_stops_before_final_test():
    with pytest.raises(RuntimeError, match="Final Test was not run"):
        _assert_validation_match(0.89, 0.88, tolerance=1e-6)


def test_confusion_matrix_consistency():
    matrix = np.zeros((10, 10), dtype=np.int64)
    np.fill_diagonal(matrix, 900)
    matrix[0, 1] = 1000

    accuracy = _validate_confusion_matrix(
        matrix,
        test_samples=10_000,
        expected_accuracy=0.9,
    )

    assert accuracy == pytest.approx(0.9)


def test_confusion_matrix_rejects_sample_or_accuracy_mismatch():
    matrix = np.eye(10, dtype=np.int64)
    with pytest.raises(RuntimeError, match="sample count mismatch"):
        _validate_confusion_matrix(matrix, test_samples=11, expected_accuracy=1.0)
    with pytest.raises(RuntimeError, match="accuracy mismatch"):
        _validate_confusion_matrix(matrix, test_samples=10, expected_accuracy=0.9)


def test_validate_final_artifacts_checks_summary_predictions_and_matrix(
    tmp_path,
):
    summary = {"test_samples": 10, "test_accuracy": 0.9}
    (tmp_path / "summary.json").write_text(json.dumps(summary))

    matrix = np.zeros((10, 10), dtype=np.int64)
    np.fill_diagonal(matrix, 1)
    matrix[0, 0] = 0
    matrix[0, 1] = 1
    pd.DataFrame(matrix).to_csv(tmp_path / "confusion_matrix.csv")
    prediction_data = {
        "is_correct": [True] * 9 + [False],
        "predicted_label_id": list(range(1, 10)) + [1],
        "confidence": [1.0] * 10,
    }
    for class_index, class_name in enumerate(CLASS_NAMES):
        prediction_data[f"probability_{class_name}"] = [
            float(predicted_class == class_index)
            for predicted_class in prediction_data["predicted_label_id"]
        ]
    pd.DataFrame(prediction_data).to_csv(
        tmp_path / "predictions.csv",
        index=False,
    )

    result = validate_final_artifacts(output_dir=str(tmp_path))

    assert result["status"] == "PASS"
    assert result["test_samples"] == 10
    assert result["confusion_matrix_accuracy"] == pytest.approx(0.9)
