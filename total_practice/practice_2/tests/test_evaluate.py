"""Tests for the evaluate module."""

import os
import tempfile
import numpy as np
import pandas as pd

from processing_own_phase.evaluate import (
    compute_classification_metrics,
    generate_classification_report,
    export_predictions
)


def test_compute_classification_metrics():
    """Test metric calculations with fixed inputs."""
    predictions = np.array([0, 1, 2, 0, 1, 2, 0, 2])
    labels =      np.array([0, 1, 2, 1, 1, 0, 0, 2])
    class_names = ["A", "B", "C"]
    num_classes = 3
    
    # Class 0 (A): 
    # True: 0, 0, 0 (3 times)
    # Pred: 0, 0, 0 (3 times) -> TP=2, FP=1 (idx 3 is pred 0, true 1), FN=1 (idx 5 is true 0, pred 2)
    # P = 2/3, R = 2/3, F1 = 2/3
    
    metrics = compute_classification_metrics(predictions, labels, num_classes, class_names)
    
    assert np.isclose(metrics["per_class"]["A"]["precision"], 2/3)
    assert np.isclose(metrics["per_class"]["A"]["recall"], 2/3)
    
    # Check overall accuracy: correct are 0, 1, 2, 4, 6, 7 (6 out of 8)
    assert np.isclose(metrics["accuracy"], 6/8)


def test_generate_classification_report():
    """Test exporting report to txt and csv."""
    predictions = np.array([0, 1, 0, 1])
    labels =      np.array([0, 1, 1, 1])
    class_names = ["A", "B"]
    metrics = compute_classification_metrics(predictions, labels, 2, class_names)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        txt_path = os.path.join(tmpdir, "report.txt")
        csv_path = os.path.join(tmpdir, "report.csv")
        
        report_str = generate_classification_report(metrics, class_names, txt_path, csv_path)
        
        assert os.path.exists(txt_path)
        assert os.path.exists(csv_path)
        
        assert "accuracy" in report_str
        assert "macro avg" in report_str
        
        df = pd.read_csv(csv_path)
        assert len(df) == 5  # 2 classes + accuracy + macro avg + weighted avg
        assert "class" in df.columns


def test_export_predictions():
    """Test exporting predictions with confidence to CSV."""
    predictions = np.array([0, 1])
    labels =      np.array([0, 0])
    probabilities = np.array([[0.9, 0.1], [0.4, 0.6]])
    class_names = ["A", "B"]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "preds.csv")
        
        df = export_predictions(predictions, labels, probabilities, class_names, csv_path)
        
        assert os.path.exists(csv_path)
        assert len(df) == 2
        assert df["confidence"].iloc[0] == 0.9
        assert df["confidence"].iloc[1] == 0.6
        assert df["is_correct"].iloc[0] == True
        assert df["is_correct"].iloc[1] == False
        assert df["probability_A"].tolist() == [0.9, 0.4]
        assert df["probability_B"].tolist() == [0.1, 0.6]
