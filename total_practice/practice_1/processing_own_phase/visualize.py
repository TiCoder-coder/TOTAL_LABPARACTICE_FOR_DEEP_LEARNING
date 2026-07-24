"""Visualization utilities for FashionMNIST classification."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for script use
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from .config import CLASS_NAMES, OUTPUT_DIR


def _compute_confusion_matrix(predictions: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute a confusion matrix without sklearn."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(labels, predictions):
        cm[int(true), int(pred)] += 1
    return cm


def _save_or_show(fig, save_path: Optional[str], show: bool):
    """Save figure to file and optionally show it."""
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_data_samples(
    images: torch.Tensor,
    labels: np.ndarray,
    class_names=CLASS_NAMES,
    n_rows: int = 4,
    n_cols: int = 5,
    save_path: Optional[str] = str(OUTPUT_DIR / "data_samples.png"),
    show: bool = False,
) -> None:
    """Plot a grid of sample images with labels."""
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    for i, ax in enumerate(np.array(axes).flat):
        if i >= len(images):
            ax.axis("off")
            continue
        img = images[i]
        if img.ndim == 3:
            img = img.squeeze(0)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{class_names[labels[i]]}", fontsize=9)
        ax.axis("off")
    fig.suptitle("FashionMNIST Sample Images", fontsize=14)
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_class_distribution(
    distributions: Dict[str, Dict[str, int]],
    save_path: Optional[str] = str(OUTPUT_DIR / "class_distribution.png"),
    show: bool = False,
) -> None:
    """Plot grouped bar chart of class distributions across splits."""
    fig, ax = plt.subplots(figsize=(12, 5))
    class_names = CLASS_NAMES
    x = np.arange(len(class_names))
    width = 0.27
    for i, (split_name, counts) in enumerate(distributions.items()):
        counts_arr = [counts[name] for name in class_names]
        ax.bar(x + i * width - width, counts_arr, width, label=split_name)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_loss_curves(
    history: List[Dict],
    save_path: Optional[str] = str(OUTPUT_DIR / "loss_curve.png"),
    show: bool = False,
) -> None:
    """Plot training and validation loss curves."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="Train Loss", marker="o")
    ax.plot(epochs, val_loss, label="Validation Loss", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_accuracy_curves(
    history: List[Dict],
    save_path: Optional[str] = str(OUTPUT_DIR / "accuracy_curve.png"),
    show: bool = False,
) -> None:
    """Plot training and validation accuracy curves."""
    epochs = [h["epoch"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_acc, label="Train Acc", marker="o")
    ax.plot(epochs, val_acc, label="Validation Acc", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_experiment_comparison(
    results: List[Dict],
    save_path: Optional[str] = str(OUTPUT_DIR / "experiment_comparison.png"),
    show: bool = False,
) -> None:
    """Plot a bar chart comparing best validation accuracy across experiments."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ids = [r["exp_id"] for r in results]
    accs = [r["best_val_acc"] for r in results]
    bars = ax.bar(ids, accs, color="steelblue", edgecolor="black")
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{acc:.3f}", ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Best Validation Accuracy")
    ax.set_title("Experiment Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_predictions_grid(
    images: torch.Tensor,
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names=CLASS_NAMES,
    n_rows: int = 4,
    n_cols: int = 4,
    save_path: Optional[str] = str(OUTPUT_DIR / "predictions_grid.png"),
    show: bool = False,
) -> None:
    """Plot a grid of test images with predicted vs actual labels."""
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.0, n_rows * 2.2))
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    idx = np.arange(len(images))
    for i, ax in enumerate(np.array(axes).flat):
        if i >= len(images):
            ax.axis("off")
            continue
        img = images[idx[i]]
        if img.ndim == 3:
            img = img.squeeze(0)
        ax.imshow(img, cmap="gray")
        pred_class = predictions[idx[i]]
        true_class = labels[idx[i]]
        confidence = probabilities[idx[i], pred_class] * 100
        correct = pred_class == true_class
        color = "green" if correct else "red"
        ax.set_title(
            f"True: {class_names[true_class]}\nPred: {class_names[pred_class]} ({confidence:.1f}%)",
            fontsize=8, color=color,
        )
        ax.axis("off")
    fig.suptitle("Predictions on Test Set (green=correct, red=wrong)", fontsize=12)
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names=CLASS_NAMES,
    save_path: Optional[str] = str(OUTPUT_DIR / "confusion_matrix.png"),
    show: bool = False,
) -> None:
    """Plot a confusion matrix heatmap."""
    cm = _compute_confusion_matrix(predictions, labels, len(class_names))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    _save_or_show(fig, save_path, show)
