"""Visualization utilities for FashionMNIST classification."""

from pathlib import Path
import sys
from typing import Dict, List, Optional

import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
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


def plot_eda_class_distribution(
    analysis: Dict,
    class_names=CLASS_NAMES,
    save_path: Optional[str] = str(
        OUTPUT_DIR / "eda_class_distribution.png"
    ),
    show: bool = False,
) -> None:
    distribution = analysis["report"]["class_distribution"]
    counts = np.array([
        distribution[name]["count"]
        for name in class_names
    ])
    percentages = np.array([
        distribution[name]["percentage"]
        for name in class_names
    ])
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(
        class_names,
        counts,
        color="#287271",
        edgecolor="#183A3A",
    )
    for bar, count, percentage in zip(
        bars,
        counts,
        percentages,
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.015,
            f"{count:,}\n{percentage:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0, counts.max() * 1.14)
    ax.set_xlabel("Class")
    ax.set_ylabel("Image count")
    ax.set_title("Official Training Pool Class Distribution")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_pixel_intensity_distribution(
    analysis: Dict,
    save_path: Optional[str] = str(
        OUTPUT_DIR / "pixel_intensity_distribution.png"
    ),
    show: bool = False,
) -> None:
    histogram = analysis["pixel_histogram"].cpu().numpy()
    percentages = histogram / histogram.sum() * 100.0
    intensities = np.arange(256) / 255.0
    statistics = analysis["report"]["pixel_statistics"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(
        intensities,
        percentages,
        color="#005F73",
        linewidth=1.8,
    )
    axes[0].fill_between(
        intensities,
        percentages,
        color="#94D2BD",
        alpha=0.45,
    )
    axes[0].axvline(
        statistics["mean"],
        color="#AE2012",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean {statistics['mean']:.3f}",
    )
    axes[0].axvline(
        statistics["median"],
        color="#CA6702",
        linestyle=":",
        linewidth=1.8,
        label=f"Median {statistics['median']:.3f}",
    )
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_xlabel("Scaled pixel intensity")
    axes[0].set_ylabel("Percentage of all pixels")
    axes[0].set_title("Full Distribution")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        intensities[1:],
        percentages[1:],
        color="#005F73",
        linewidth=1.5,
    )
    axes[1].fill_between(
        intensities[1:],
        percentages[1:],
        color="#94D2BD",
        alpha=0.45,
    )
    axes[1].axvline(
        statistics["mean"],
        color="#AE2012",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean {statistics['mean']:.3f}",
    )
    axes[1].set_xlim(1.0 / 255.0, 1.0)
    axes[1].set_xlabel("Scaled non-zero pixel intensity")
    axes[1].set_ylabel("Percentage of all pixels")
    axes[1].set_title("Non-Zero Distribution")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.suptitle("Pixel Intensity Distribution", fontsize=14)
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_image_brightness_contrast(
    analysis: Dict,
    save_path: Optional[str] = str(
        OUTPUT_DIR / "image_brightness_contrast.png"
    ),
    show: bool = False,
) -> None:
    image_means = analysis["image_means"].cpu().numpy()
    image_stds = analysis["image_stds"].cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(
        image_means,
        bins=50,
        color="#287271",
        edgecolor="white",
    )
    axes[0].axvline(
        image_means.mean(),
        color="#AE2012",
        linestyle="--",
        label=f"Mean {image_means.mean():.3f}",
    )
    axes[0].set_xlabel("Per-image mean intensity")
    axes[0].set_ylabel("Image count")
    axes[0].set_title("Image Brightness")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    axes[1].hist(
        image_stds,
        bins=50,
        color="#E9C46A",
        edgecolor="white",
    )
    axes[1].axvline(
        image_stds.mean(),
        color="#AE2012",
        linestyle="--",
        label=f"Mean {image_stds.mean():.3f}",
    )
    axes[1].set_xlabel("Per-image pixel standard deviation")
    axes[1].set_ylabel("Image count")
    axes[1].set_title("Image Contrast")
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_per_class_intensity_boxplots(
    analysis: Dict,
    class_names=CLASS_NAMES,
    save_path: Optional[str] = str(
        OUTPUT_DIR / "per_class_intensity_boxplot.png"
    ),
    show: bool = False,
) -> None:
    targets = analysis["targets"].cpu().numpy()
    image_means = analysis["image_means"].cpu().numpy()
    image_stds = analysis["image_stds"].cpu().numpy()
    brightness = [
        image_means[targets == class_index]
        for class_index in range(len(class_names))
    ]
    contrast = [
        image_stds[targets == class_index]
        for class_index in range(len(class_names))
    ]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    brightness_plot = axes[0].boxplot(
        brightness,
        tick_labels=class_names,
        patch_artist=True,
        showfliers=False,
    )
    contrast_plot = axes[1].boxplot(
        contrast,
        tick_labels=class_names,
        patch_artist=True,
        showfliers=False,
    )
    for patch in brightness_plot["boxes"]:
        patch.set_facecolor("#94D2BD")
    for patch in contrast_plot["boxes"]:
        patch.set_facecolor("#E9C46A")
    axes[0].set_ylabel("Mean intensity")
    axes[0].set_title("Brightness by Class")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].set_ylabel("Pixel standard deviation")
    axes[1].set_title("Contrast by Class")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_class_samples(
    image_data: torch.Tensor,
    analysis: Dict,
    class_names=CLASS_NAMES,
    save_path: Optional[str] = str(
        OUTPUT_DIR / "class_samples.png"
    ),
    show: bool = False,
) -> None:
    samples_per_class = analysis["report"]["samples_per_class"]
    sample_indices = analysis["sample_indices"]
    fig, axes = plt.subplots(
        samples_per_class,
        len(class_names),
        figsize=(16, 2.2 * samples_per_class),
        squeeze=False,
    )
    for class_index, class_name in enumerate(class_names):
        for sample_index in range(samples_per_class):
            position = (
                class_index * samples_per_class
                + sample_index
            )
            dataset_index = sample_indices[position]
            axis = axes[sample_index, class_index]
            axis.imshow(
                image_data[dataset_index].cpu().numpy(),
                cmap="gray",
                vmin=0,
                vmax=255,
            )
            if sample_index == 0:
                axis.set_title(class_name, fontsize=9)
            axis.axis("off")
    fig.suptitle(
        "Class-Stratified FashionMNIST Samples",
        fontsize=14,
    )
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_class_mean_images(
    analysis: Dict,
    class_names=CLASS_NAMES,
    save_path: Optional[str] = str(
        OUTPUT_DIR / "class_mean_images.png"
    ),
    show: bool = False,
) -> None:
    mean_images = analysis["class_mean_images"].cpu().numpy()
    fig, axes = plt.subplots(2, 5, figsize=(12, 6.2))
    for class_index, axis in enumerate(axes.flat):
        axis.imshow(
            mean_images[class_index],
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        axis.set_title(class_names[class_index], fontsize=10)
        axis.axis("off")
    fig.suptitle("Mean Image by Class", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.5)
    _save_or_show(fig, save_path, show)


def plot_eda_outliers(
    image_data: torch.Tensor,
    targets: torch.Tensor,
    analysis: Dict,
    class_names=CLASS_NAMES,
    save_path: Optional[str] = str(
        OUTPUT_DIR / "eda_outliers.png"
    ),
    show: bool = False,
) -> None:
    groups = list(analysis["outlier_indices"].items())
    column_count = max(
        len(indices)
        for _, indices in groups
    )
    fig, axes = plt.subplots(
        len(groups),
        column_count,
        figsize=(10, 9),
        squeeze=False,
    )
    image_means = analysis["image_means"]
    image_stds = analysis["image_stds"]
    for row_index, (group_name, indices) in enumerate(groups):
        for column_index, dataset_index in enumerate(indices):
            axis = axes[row_index, column_index]
            axis.imshow(
                image_data[dataset_index].cpu().numpy(),
                cmap="gray",
                vmin=0,
                vmax=255,
            )
            label = class_names[int(targets[dataset_index])]
            if "contrast" in group_name:
                value = image_stds[dataset_index]
            else:
                value = image_means[dataset_index]
            axis.set_title(
                f"{label}\n{float(value):.3f}",
                fontsize=8,
            )
            axis.axis("off")
        axes[row_index, 0].text(
            -0.35,
            0.5,
            group_name.replace("_", " ").title(),
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=9,
        )
    fig.suptitle(
        "Brightness and Contrast Extremes",
        fontsize=14,
    )
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
    val_loss = [h["validation_loss"] for h in history]
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
    train_acc = [h["train_accuracy"] for h in history]
    val_acc = [h["validation_accuracy"] for h in history]
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
    ids = [r["experiment_id"] for r in results]
    accs = [r["best_validation_accuracy"] for r in results]
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
