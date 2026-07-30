import os
import numpy as np
import torch
from torch.utils.data import Dataset

from configs import CLASS_NAMES
from processing_own_phase.visualize import (
    plot_class_distribution,
    plot_class_examples,
    plot_experiment_configuration,
    plot_training_curves,
    plot_learning_rate,
    plot_metrics_bar,
    plot_experiment_comparison,
    plot_validation_test_comparison,
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


def test_plot_class_distribution(tmp_path):
    distributions = {
        "Train": {c: np.random.randint(400, 500) for c in CLASS_NAMES},
        "Val": {c: np.random.randint(40, 50) for c in CLASS_NAMES},
        "Test": {c: np.random.randint(90, 100) for c in CLASS_NAMES},
    }
    path = str(tmp_path / "class_distribution.png")
    plot_class_distribution(distributions, save_path=path)
    assert os.path.exists(path)


def test_plot_class_examples(tmp_path):
    dataset = DummyDataset(size=50)
    path = str(tmp_path / "class_examples.png")
    plot_class_examples(dataset, CLASS_NAMES, num_examples=2, save_path=path)
    assert os.path.exists(path)


def test_plot_experiment_configuration(tmp_path):
    path = str(tmp_path / "experiment_configuration.png")
    experiments = {
        "E1": {"learning_rate": 0.001, "batch_size": 64, "epochs": 10},
        "E2": {"learning_rate": 0.0005, "batch_size": 64, "epochs": 10},
    }
    plot_experiment_configuration(experiments, save_path=path)
    assert os.path.exists(path)


def test_plot_training_curves(tmp_path):
    history = {
        "train_loss": [2.0, 1.5, 1.0, 0.8],
        "val_loss": [2.1, 1.6, 1.1, 0.9],
        "train_acc": [20, 40, 60, 80],
        "val_acc": [18, 35, 55, 75]
    }
    path = str(tmp_path / "training_curves.png")
    plot_training_curves(history, save_path=path)
    assert os.path.exists(path)


def test_plot_learning_rate(tmp_path):
    history = {
        "lr": [0.001, 0.001, 0.0005, 0.0001]
    }
    path = str(tmp_path / "learning_rate.png")
    plot_learning_rate(history, save_path=path)
    assert os.path.exists(path)


def test_plot_metrics_bar(tmp_path):
    metrics = {
        "per_class": {
            c: {"precision": np.random.rand(), "recall": np.random.rand(), "f1-score": np.random.rand()}
            for c in CLASS_NAMES
        }
    }
    path = str(tmp_path / "metrics_bar.png")
    plot_metrics_bar(metrics, CLASS_NAMES, save_path=path)
    assert os.path.exists(path)


def test_plot_experiment_comparison(tmp_path):
    results = [
        {"exp_id": "E1", "best_val_acc": 80.5, "f1_score": 0.78, "training_time": 120, "inference_time": 0.5},
        {"exp_id": "E2", "metadata": {"accuracy": 85.0, "f1_score": 0.82, "training_time": 150}, "inference_time": 0.6}
    ]
    path = str(tmp_path / "experiment_comparison.png")
    plot_experiment_comparison(results, save_path=path)
    assert os.path.exists(path)


def test_plot_validation_test_comparison(tmp_path):
    path = str(tmp_path / "validation_test_comparison.png")
    plot_validation_test_comparison(
        validation_accuracy=0.8688,
        test_accuracy=0.8623,
        validation_loss=0.3742,
        test_loss=0.3935,
        save_path=path,
    )
    assert os.path.exists(path)


def test_plot_prediction_gallery(tmp_path):
    size = 20
    images = torch.randn(size, 3, 32, 32)
    labels = np.random.randint(0, 10, size)
    predictions = labels.copy()
    predictions[size // 2:] = (labels[size // 2:] + 1) % len(CLASS_NAMES)
    probabilities = np.random.rand(size, 10)
    
    plot_prediction_gallery(images, predictions, labels, probabilities, CLASS_NAMES, save_dir=str(tmp_path))
    
    assert os.path.exists(tmp_path / "prediction_gallery_correct.png")
    assert os.path.exists(tmp_path / "prediction_gallery_incorrect.png")


def test_plot_confusion_matrix(tmp_path):
    size = 50
    predictions = np.random.randint(0, 10, size)
    labels = np.random.randint(0, 10, size)
    plot_confusion_matrix(predictions, labels, CLASS_NAMES, save_dir=str(tmp_path))
    
    assert os.path.exists(tmp_path / "confusion_matrix_raw.png")
    assert os.path.exists(tmp_path / "confusion_matrix_normalized.png")


def test_plot_confidence_distribution(tmp_path):
    size = 50
    predictions = np.random.randint(0, 10, size)
    labels = np.random.randint(0, 10, size)
    probabilities = np.random.rand(size, 10)
    
    path = str(tmp_path / "confidence_distribution.png")
    plot_confidence_distribution(probabilities, labels, predictions, save_path=path)
    assert os.path.exists(path)
