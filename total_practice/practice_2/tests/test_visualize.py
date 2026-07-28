import os
import shutil
import numpy as np
import pytest
import torch
from pathlib import Path
from torch.utils.data import Dataset

from configs import REPORTS_DIR, CLASS_NAMES
from processing_own_phase.visualize import (
    plot_class_distribution,
    plot_class_examples,
    plot_training_curves,
    plot_learning_rate,
    plot_metrics_bar,
    plot_experiment_comparison,
    plot_prediction_gallery,
    plot_confusion_matrix,
    plot_confidence_distribution,
    plot_data_samples
)


class DummyDataset(Dataset):
    def __init__(self, size=10):
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Return a normalized dummy image (C, H, W)
        return torch.randn(3, 32, 32), np.random.randint(0, 10)


@pytest.fixture(autouse=True)
def setup_teardown():
    # Setup
    os.makedirs(REPORTS_DIR, exist_ok=True)
    yield
    # We do not want to remove REPORTS_DIR after because we actually want to see the charts.
    pass


def test_plot_class_distribution():
    distributions = {
        "Train": {c: np.random.randint(400, 500) for c in CLASS_NAMES},
        "Val": {c: np.random.randint(40, 50) for c in CLASS_NAMES},
        "Test": {c: np.random.randint(90, 100) for c in CLASS_NAMES},
    }
    path = str(REPORTS_DIR / "test_class_distribution.png")
    plot_class_distribution(distributions, save_path=path)
    assert os.path.exists(path)


def test_plot_class_examples():
    dataset = DummyDataset(size=50)
    path = str(REPORTS_DIR / "test_class_examples.png")
    plot_class_examples(dataset, CLASS_NAMES, num_examples=2, save_path=path)
    assert os.path.exists(path)


def test_plot_training_curves():
    history = {
        "train_loss": [2.0, 1.5, 1.0, 0.8],
        "val_loss": [2.1, 1.6, 1.1, 0.9],
        "train_acc": [20, 40, 60, 80],
        "val_acc": [18, 35, 55, 75]
    }
    path = str(REPORTS_DIR / "test_training_curves.png")
    plot_training_curves(history, save_path=path)
    assert os.path.exists(path)


def test_plot_learning_rate():
    history = {
        "lr": [0.001, 0.001, 0.0005, 0.0001]
    }
    path = str(REPORTS_DIR / "test_learning_rate.png")
    plot_learning_rate(history, save_path=path)
    assert os.path.exists(path)


def test_plot_metrics_bar():
    metrics = {
        "per_class": {
            c: {"precision": np.random.rand(), "recall": np.random.rand(), "f1-score": np.random.rand()}
            for c in CLASS_NAMES
        }
    }
    path = str(REPORTS_DIR / "test_metrics_bar.png")
    plot_metrics_bar(metrics, CLASS_NAMES, save_path=path)
    assert os.path.exists(path)


def test_plot_experiment_comparison():
    results = [
        {"exp_id": "E1", "best_val_acc": 80.5, "f1_score": 0.78, "training_time": 120, "inference_time": 0.5},
        {"exp_id": "E2", "metadata": {"accuracy": 85.0, "f1_score": 0.82, "training_time": 150}, "inference_time": 0.6}
    ]
    path = str(REPORTS_DIR / "test_exp_comparison.png")
    plot_experiment_comparison(results, save_path=path)
    assert os.path.exists(path)


def test_plot_prediction_gallery():
    size = 20
    images = torch.randn(size, 3, 32, 32)
    predictions = np.random.randint(0, 10, size)
    labels = np.random.randint(0, 10, size)
    probabilities = np.random.rand(size, 10)
    
    plot_prediction_gallery(images, predictions, labels, probabilities, CLASS_NAMES, save_dir=str(REPORTS_DIR))
    
    assert os.path.exists(REPORTS_DIR / "prediction_gallery_correct.png")
    assert os.path.exists(REPORTS_DIR / "prediction_gallery_incorrect.png")


def test_plot_confusion_matrix():
    size = 50
    predictions = np.random.randint(0, 10, size)
    labels = np.random.randint(0, 10, size)
    plot_confusion_matrix(predictions, labels, CLASS_NAMES, save_dir=str(REPORTS_DIR))
    
    assert os.path.exists(REPORTS_DIR / "confusion_matrix_raw.png")
    assert os.path.exists(REPORTS_DIR / "confusion_matrix_normalized.png")


def test_plot_confidence_distribution():
    size = 50
    predictions = np.random.randint(0, 10, size)
    labels = np.random.randint(0, 10, size)
    probabilities = np.random.rand(size, 10)
    
    path = str(REPORTS_DIR / "test_confidence_distribution.png")
    plot_confidence_distribution(probabilities, labels, predictions, save_path=path)
    assert os.path.exists(path)
