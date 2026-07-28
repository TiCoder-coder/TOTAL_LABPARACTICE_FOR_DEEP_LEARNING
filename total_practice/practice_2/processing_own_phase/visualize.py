"""Visualization utilities for Practice 2."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for script use
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import Dataset

from configs import CLASS_NAMES, REPORTS_DIR

logger = logging.getLogger(__name__)


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
        logger.info(f"Saved plot to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def _denormalize_image(img_tensor: np.ndarray) -> np.ndarray:
    """De-normalize image using ImageNet params."""
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    
    img = img_tensor.transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img


def plot_class_distribution(
    distributions: Dict[str, Dict[str, int]],
    save_path: Optional[str] = str(REPORTS_DIR / "class_distribution.png"),
    show: bool = False
) -> None:
    """Plot the class distribution for Train, Val, and Test datasets.
    
    Args:
        distributions: Dict like {"Train": {"airplane": 500, ...}, "Val": {...}}
        save_path: Where to save the plot.
        show: Whether to display the plot.
    """
    splits = list(distributions.keys())
    if not splits:
        return
        
    classes = list(distributions[splits[0]].keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(classes))
    width = 0.8 / len(splits)
    
    for i, split in enumerate(splits):
        counts = [distributions[split].get(c, 0) for c in classes]
        ax.bar(x + i * width - 0.4 + width/2, counts, width, label=split)
        
    ax.set_ylabel('Number of Samples')
    ax.set_title('Class Distribution across Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_class_examples(
    dataset: Dataset,
    class_names: List[str] = CLASS_NAMES,
    num_examples: int = 5,
    save_path: Optional[str] = str(REPORTS_DIR / "class_examples.png"),
    show: bool = False
) -> None:
    """Plot random examples for each class in the dataset."""
    num_classes = len(class_names)
    fig, axes = plt.subplots(num_classes, num_examples, figsize=(num_examples * 2, num_classes * 2))
    
    # Pre-collect indices for each class
    class_indices = {i: [] for i in range(num_classes)}
    
    # We sample up to a certain point to avoid full dataset iteration if it's huge
    max_search = min(len(dataset), 5000)
    for idx in range(max_search):
        _, label = dataset[idx]
        if len(class_indices[label]) < num_examples:
            class_indices[label].append(idx)
            
        # Break early if we have enough examples for all classes
        if all(len(indices) >= num_examples for indices in class_indices.values()):
            break

    for class_idx in range(num_classes):
        indices = class_indices[class_idx]
        for ex_idx in range(num_examples):
            ax = axes[class_idx, ex_idx]
            if ex_idx < len(indices):
                img_tensor, _ = dataset[indices[ex_idx]]
                img = _denormalize_image(img_tensor.numpy())
                ax.imshow(img)
                if ex_idx == 0:
                    ax.set_ylabel(class_names[class_idx], rotation=0, size='large', ha='right')
            ax.set_xticks([])
            ax.set_yticks([])
            
    fig.suptitle("Dataset Examples per Class", fontsize=16)
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_training_curves(
    history: Dict[str, list],
    save_path: Optional[str] = str(REPORTS_DIR / "training_curves.png"),
    show: bool = False
) -> None:
    """Plot Training and Validation Loss & Accuracy curves."""
    if not history or "train_loss" not in history:
        logger.warning("Empty or invalid history provided for training curves.")
        return
        
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, history["train_loss"], 'bo-', label='Training Loss')
    ax1.plot(epochs, history["val_loss"], 'ro-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], 'bo-', label='Training Accuracy')
    ax2.plot(epochs, history["val_acc"], 'ro-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_learning_rate(
    history: Dict[str, list],
    save_path: Optional[str] = str(REPORTS_DIR / "learning_rate.png"),
    show: bool = False
) -> None:
    """Plot Learning Rate schedule over epochs."""
    if not history or "lr" not in history:
        return
        
    epochs = range(1, len(history["lr"]) + 1)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["lr"], 'go-', linewidth=2)
    ax.set_title('Learning Rate over Epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_metrics_bar(
    metrics_dict: dict,
    class_names: List[str] = CLASS_NAMES,
    save_path: Optional[str] = str(REPORTS_DIR / "metrics_bar.png"),
    show: bool = False
) -> None:
    """Plot Precision, Recall, and F1-Score for each class."""
    if "per_class" not in metrics_dict:
        logger.warning("No per_class metrics found in dict.")
        return
        
    per_class = metrics_dict["per_class"]
    classes = [c for c in class_names if c in per_class]
    
    precisions = [per_class[c]["precision"] for c in classes]
    recalls = [per_class[c]["recall"] for c in classes]
    f1s = [per_class[c]["f1-score"] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.bar(x - width, precisions, width, label='Precision', color='lightblue')
    ax.bar(x, recalls, width, label='Recall', color='lightgreen')
    ax.bar(x + width, f1s, width, label='F1-Score', color='salmon')
    
    ax.set_ylabel('Score')
    ax.set_title('Metrics per Class')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_data_samples(
    images: torch.Tensor,
    labels: np.ndarray,
    class_names=CLASS_NAMES,
    n_rows: int = 4,
    n_cols: int = 5,
    save_path: Optional[str] = str(REPORTS_DIR / "data_samples.png"),
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
        
        img = _denormalize_image(images[i])
        ax.imshow(img)
        ax.set_title(f"{class_names[labels[i]]}", fontsize=9)
        ax.axis("off")
        
    fig.suptitle("CIFAR-10 Sample Images", fontsize=14)
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_experiment_comparison(
    results: List[Dict],
    save_path: Optional[str] = str(REPORTS_DIR / "experiment_comparison.png"),
    show: bool = False,
) -> None:
    """Plot a bar chart comparing multiple metrics across experiments if available."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    ids = [r.get("exp_id", f"Exp {i}") for i, r in enumerate(results)]
    
    metrics_to_plot = [
        ("best_val_acc", "Best Val Accuracy (%)", 0, 100.0),
        ("f1_score", "Test F1-Score", 0, 1.0),
        ("training_time", "Training Time (s)", None, None),
        ("inference_time", "Inference Time (s)", None, None)
    ]
    
    for idx, (metric_key, title, y_min, y_max) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        vals = [r.get(metric_key, r.get("metadata", {}).get(metric_key, 0)) for r in results]
        # In results dict from Phase 5, some might be nested in 'metadata'
        if all(v == 0 for v in vals) and metric_key not in results[0].get("metadata", {}):
            ax.set_title(f"{title} (Data Unavailable)")
            ax.axis('off')
            continue
            
        bars = ax.bar(ids, vals, color="steelblue", edgecolor="black")
        for bar, val in zip(bars, vals):
            if isinstance(val, float):
                text = f"{val:.3f}"
            else:
                text = str(val)
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                text, ha="center", va="bottom", fontsize=9,
            )
        if y_min is not None and y_max is not None:
            # Check if it's accuracy in percentage
            if metric_key == "best_val_acc" and any(v > 1.0 for v in vals):
                ax.set_ylim(0, 105)
            else:
                ax.set_ylim(y_min, y_max)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis='x', rotation=20)
        
    fig.suptitle("Experiment Comparison", fontsize=16)
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_prediction_gallery(
    images: torch.Tensor,
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names=CLASS_NAMES,
    top_k: int = 10,
    save_dir: Optional[str] = str(REPORTS_DIR),
    show: bool = False
) -> None:
    """Plot galleries for correct and incorrect predictions sorted by confidence."""
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
        
    confidences = probabilities[np.arange(len(predictions)), predictions]
    is_correct = predictions == labels
    
    _plot_gallery_subset(
        images, predictions, labels, confidences, is_correct, 
        True, class_names, top_k, 
        save_path=f"{save_dir}/prediction_gallery_correct.png" if save_dir else None, show=show
    )
    
    _plot_gallery_subset(
        images, predictions, labels, confidences, is_correct, 
        False, class_names, top_k, 
        save_path=f"{save_dir}/prediction_gallery_incorrect.png" if save_dir else None, show=show
    )


def _plot_gallery_subset(
    images: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    confidences: np.ndarray,
    is_correct: np.ndarray,
    correct_mode: bool,
    class_names: List[str],
    top_k: int,
    save_path: Optional[str],
    show: bool
):
    """Helper to plot either correct or incorrect prediction galleries."""
    mask = is_correct if correct_mode else ~is_correct
    indices = np.where(mask)[0]
    
    if len(indices) == 0:
        logger.info(f"No {'correct' if correct_mode else 'incorrect'} samples found!")
        return
        
    subset_conf = confidences[indices]
    # Sort by highest confidence
    sorted_idx = np.argsort(subset_conf)[::-1]
    top_indices = indices[sorted_idx[:top_k]]
    
    n_cols = 5
    n_rows = (min(top_k, len(top_indices)) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 3.0))
    axes = np.array(axes).flatten()
    
    for i, ax in enumerate(axes):
        if i >= len(top_indices):
            ax.axis("off")
            continue
            
        idx = top_indices[i]
        img = _denormalize_image(images[idx])
        ax.imshow(img)
        
        pred_class = predictions[idx]
        true_class = labels[idx]
        conf = confidences[idx] * 100
        
        color = "green" if correct_mode else "red"
        ax.set_title(
            f"True: {class_names[true_class]}\nPred: {class_names[pred_class]}\nConf: {conf:.1f}%",
            fontsize=9, color=color,
        )
        ax.axis("off")
        
    title = "Top Confident Correct Predictions" if correct_mode else "Top Confident Misclassified Predictions"
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names=CLASS_NAMES,
    save_dir: Optional[str] = str(REPORTS_DIR),
    show: bool = False,
) -> None:
    """Plot and save both raw and normalized confusion matrix heatmaps."""
    cm = _compute_confusion_matrix(predictions, labels, len(class_names))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1e-6)
    
    # 1. Normalized
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
    _save_or_show(fig, f"{save_dir}/confusion_matrix_normalized.png" if save_dir else None, show)
    
    # 2. Raw Count
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Raw Count Confusion Matrix")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    _save_or_show(fig, f"{save_dir}/confusion_matrix_raw.png" if save_dir else None, show)


def plot_confidence_distribution(
    probabilities: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    save_path: Optional[str] = str(REPORTS_DIR / "confidence_distribution.png"),
    show: bool = False
):
    """Plot the distribution of confidences for correct and incorrect predictions."""
    confidences = probabilities[np.arange(len(predictions)), predictions]
    correct = (predictions == labels)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.histplot(confidences[correct], bins=20, color='green', alpha=0.5, label='Correct', ax=ax, kde=True)
    sns.histplot(confidences[~correct], bins=20, color='red', alpha=0.5, label='Incorrect', ax=ax, kde=True)
    
    ax.set_xlabel("Confidence Probability")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution")
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, save_path, show)
