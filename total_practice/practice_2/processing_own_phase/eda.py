"""Training-only CIFAR-10 EDA adapted from the Practice 1 presentation flow.

This module is descriptive only. It never reads Validation/Test pixels, changes
transforms, trains a model, or selects a checkpoint.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Sequence

import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _duplicate_audit(images: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    flat = np.ascontiguousarray(images.reshape(len(images), -1))
    row_type = np.dtype((np.void, flat.dtype.itemsize * flat.shape[1]))
    rows = flat.view(row_type).ravel()
    _, inverse, counts = np.unique(rows, return_inverse=True, return_counts=True)
    duplicate_ids = np.flatnonzero(counts > 1)
    conflicts = sum(np.unique(labels[inverse == group]).size > 1 for group in duplicate_ids)
    return {
        "duplicate_group_count": int(len(duplicate_ids)),
        "additional_duplicate_count": int(np.sum(counts[counts > 1] - 1)),
        "conflicting_label_group_count": int(conflicts),
    }


def _histogram_quantile(histogram: np.ndarray, quantile: float) -> float:
    cumulative = np.cumsum(histogram)
    rank = int(quantile * (int(cumulative[-1]) - 1)) + 1
    return float(np.searchsorted(cumulative, rank) / 255.0)


def analyze_cifar10_training_subset(
    raw_pool: np.ndarray,
    pool_labels: np.ndarray,
    train_indices: Sequence[int] | np.ndarray,
    class_names: Sequence[str],
    seed: int = 42,
    samples_per_class: int = 2,
    projection_samples_per_class: int = 100,
) -> dict[str, Any]:
    """Analyze exactly the indexed Train subset of a raw CIFAR-10 pool."""
    raw_pool = np.asarray(raw_pool)
    pool_labels = np.asarray(pool_labels, dtype=np.int64)
    indices = np.asarray(train_indices, dtype=np.int64)
    if raw_pool.ndim != 4 or raw_pool.shape[1:] != (32, 32, 3):
        raise ValueError("raw_pool must have shape [N, 32, 32, 3]")
    if raw_pool.dtype != np.uint8:
        raise ValueError("raw_pool must use uint8")
    if pool_labels.ndim != 1 or len(pool_labels) != len(raw_pool):
        raise ValueError("pool_labels must have shape [N]")
    if indices.ndim != 1 or not len(indices):
        raise ValueError("train_indices must be a non-empty one-dimensional sequence")
    if np.unique(indices).size != len(indices) or indices.min() < 0 or indices.max() >= len(raw_pool):
        raise ValueError("train_indices must be unique and within the raw pool")

    images = raw_pool[indices]
    labels = pool_labels[indices]
    class_count = len(class_names)
    invalid = (labels < 0) | (labels >= class_count)
    counts = np.bincount(labels[~invalid], minlength=class_count)
    scaled = images.astype(np.float32) / 255.0
    brightness = scaled.mean(axis=(1, 2, 3))
    contrast = scaled.std(axis=(1, 2, 3))
    channel_mean = scaled.mean(axis=(0, 1, 2))
    channel_std = scaled.std(axis=(0, 1, 2))
    channel_histograms = np.stack([
        np.bincount(images[..., channel].ravel(), minlength=256)
        for channel in range(3)
    ])
    flat = images.reshape(len(images), -1)
    duplicates = _duplicate_audit(images, labels)
    quality = {
        "non_finite_pixel_count": 0,
        "invalid_label_count": int(invalid.sum()),
        "missing_class_count": int((counts == 0).sum()),
        "constant_image_count": int(np.sum(np.ptp(flat, axis=1) == 0)),
        **duplicates,
    }

    rng = np.random.default_rng(seed)
    sample_indices: list[int] = []
    projection_indices: list[int] = []
    class_means = []
    per_class = {}
    for class_id, name in enumerate(class_names):
        local = np.flatnonzero(labels == class_id)
        if len(local) < samples_per_class:
            raise ValueError(f"class {name} has too few Train samples")
        sample_indices.extend(rng.choice(local, samples_per_class, replace=False).tolist())
        projection_count = min(projection_samples_per_class, len(local))
        projection_indices.extend(rng.choice(local, projection_count, replace=False).tolist())
        class_means.append(images[local].mean(axis=0).astype(np.uint8))
        per_class[name] = {
            "count": int(len(local)),
            "percentage": float(len(local) / len(images) * 100),
            "mean_brightness": float(brightness[local].mean()),
            "mean_contrast": float(contrast[local].mean()),
        }

    return {
        "report": {
            "scope": "training_subset_only",
            "schema": {"sample_count": len(images), "shape": list(images.shape), "dtype": str(images.dtype), "raw_min": int(images.min()), "raw_max": int(images.max()), "label_min": int(labels.min()), "label_max": int(labels.max()), "class_count": class_count},
            "channel_statistics": {name: {
                "mean": float(channel_mean[i]), "std": float(channel_std[i]),
                "minimum": float(scaled[..., i].min()), "q05": _histogram_quantile(channel_histograms[i], .05),
                "median": _histogram_quantile(channel_histograms[i], .50), "q95": _histogram_quantile(channel_histograms[i], .95),
                "maximum": float(scaled[..., i].max()),
            } for i, name in enumerate(("Red", "Green", "Blue"))},
            "pixel_statistics": {
                "pixel_count": int(images.size), "mean": float(scaled.mean()), "std": float(scaled.std()),
                "minimum": float(scaled.min()), "q05": _histogram_quantile(channel_histograms.sum(axis=0), .05),
                "median": _histogram_quantile(channel_histograms.sum(axis=0), .50),
                "q95": _histogram_quantile(channel_histograms.sum(axis=0), .95), "maximum": float(scaled.max()),
            },
            "class_statistics": per_class,
            "quality": quality,
            "seed": seed,
        },
        "images": images,
        "labels": labels,
        "scaled_images": scaled,
        "brightness": brightness,
        "contrast": contrast,
        "channel_histograms": channel_histograms,
        "class_mean_images": np.stack(class_means),
        "sample_indices": sample_indices,
        "projection_indices": projection_indices,
        "extreme_indices": np.r_[np.argsort(brightness)[:5], np.argsort(brightness)[-5:]].tolist(),
    }


def render_cifar10_eda(
    analysis: dict[str, Any],
    class_names: Sequence[str],
    output_dir: str | Path,
    include_projection: bool = True,
) -> dict[str, Path]:
    """Render deterministic EDA figures from one training-only analysis."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    images, labels = analysis["images"], analysis["labels"]
    scaled, brightness, contrast = analysis["scaled_images"], analysis["brightness"], analysis["contrast"]
    seed = analysis["report"]["seed"]
    rng = np.random.default_rng(seed)
    paths: dict[str, Path] = {}

    histogram_indices = rng.choice(len(images), min(5_000, len(images)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for channel, (name, color) in enumerate(zip(("Red", "Green", "Blue"), ("red", "green", "blue"))):
        axes[0].hist(scaled[histogram_indices, :, :, channel].ravel(), bins=64, range=(0, 1), density=True, alpha=.35, color=color, label=name)
    axes[0].set(title="Training RGB intensity distribution", xlabel="Scaled intensity", ylabel="Density"); axes[0].legend(); axes[0].grid(alpha=.2)
    scatter_indices = histogram_indices[:min(2_000, len(histogram_indices))]
    scatter = axes[1].scatter(brightness[scatter_indices], contrast[scatter_indices], c=labels[scatter_indices], cmap="tab10", s=12, alpha=.55)
    axes[1].set(title="Image brightness versus contrast", xlabel="Mean RGB intensity", ylabel="RGB standard deviation"); axes[1].grid(alpha=.2)
    handles, _ = scatter.legend_elements(num=len(class_names)); axes[1].legend(handles, class_names, fontsize=8, ncol=2)
    fig.tight_layout(); paths["rgb_brightness_contrast"] = output / "eda_rgb_brightness_contrast.png"; fig.savefig(paths["rgb_brightness_contrast"], dpi=140, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for class_id, ax in enumerate(axes.flat):
        ax.imshow(analysis["class_mean_images"][class_id]); ax.set_title(class_names[class_id]); ax.axis("off")
    fig.suptitle("Training-subset class-average images", fontsize=16); fig.tight_layout(); paths["class_means"] = output / "eda_class_mean_images.png"; fig.savefig(paths["class_means"], dpi=140, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for position, (ax, index) in enumerate(zip(axes.flat, analysis["extreme_indices"])):
        group = "Darkest" if position < 5 else "Brightest"; ax.imshow(images[index]); ax.set_title(f"{group}: {class_names[labels[index]]}\nmean={brightness[index]:.3f}", fontsize=9); ax.axis("off")
    fig.suptitle("Training-subset brightness extremes", fontsize=16); fig.tight_layout(); paths["brightness_extremes"] = output / "eda_brightness_extremes.png"; fig.savefig(paths["brightness_extremes"], dpi=140, bbox_inches="tight"); plt.close(fig)

    if include_projection:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        projection_indices = np.asarray(analysis["projection_indices"])
        features = scaled[projection_indices, ::2, ::2, :].reshape(len(projection_indices), -1)
        features = StandardScaler().fit_transform(features)
        components = min(50, len(features) - 1, features.shape[1])
        pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
        pca_features = pca.fit_transform(features)
        perplexity = min(30, max(2, (len(features) - 1) // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto", max_iter=750, init="pca", random_state=seed).fit_transform(pca_features)
        projection_labels = labels[projection_indices]
        fig, axes = plt.subplots(1, 2, figsize=(17, 7))
        for class_id, name in enumerate(class_names):
            mask = projection_labels == class_id
            axes[0].scatter(pca_features[mask, 0], pca_features[mask, 1], s=14, alpha=.65, label=name)
            axes[1].scatter(tsne[mask, 0], tsne[mask, 1], s=14, alpha=.65, label=name)
        axes[0].set(title=f"PCA (50 PCs explain {pca.explained_variance_ratio_.sum():.1%})", xlabel="PC1", ylabel="PC2")
        axes[1].set(title="t-SNE after PCA — Train only", xlabel="t-SNE 1", ylabel="t-SNE 2")
        for ax in axes: ax.grid(alpha=.2); ax.legend(fontsize=8, ncol=2)
        fig.tight_layout(); paths["projection"] = output / "eda_pca_tsne.png"; fig.savefig(paths["projection"], dpi=140, bbox_inches="tight"); plt.close(fig)
    return paths


def render_practice1_style_cifar10_eda(
    analysis: dict[str, Any],
    class_names: Sequence[str],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Render the complete Practice 1 EDA method adapted to RGB CIFAR-10."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    paths = render_cifar10_eda(analysis, class_names, output_dir, include_projection=False)
    output = Path(output_dir).expanduser().resolve()
    images, labels = analysis["images"], analysis["labels"]
    scaled = analysis["scaled_images"]
    brightness, contrast = analysis["brightness"], analysis["contrast"]
    seed = analysis["report"]["seed"]

    # Practice 1: exact class-distribution bar chart.
    counts = np.bincount(labels, minlength=len(class_names))
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(class_names, counts, color="#287271", edgecolor="#183A3A")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + counts.max() * .01, f"{count:,}\n{count/len(labels):.1%}", ha="center", fontsize=8)
    ax.set(title="CIFAR-10 Train-Subset Class Distribution", xlabel="Class", ylabel="Images"); ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); paths["class_distribution"] = output / "eda_train_class_distribution.png"; fig.savefig(paths["class_distribution"], dpi=140, bbox_inches="tight"); plt.close(fig)

    # Practice 1: per-class brightness and contrast boxplots.
    grouped_brightness = [brightness[labels == i] for i in range(len(class_names))]
    grouped_contrast = [contrast[labels == i] for i in range(len(class_names))]
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    for ax, values, title, ylabel, color in (
        (axes[0], grouped_brightness, "Brightness by Class", "Mean RGB intensity", "#94D2BD"),
        (axes[1], grouped_contrast, "Contrast by Class", "RGB standard deviation", "#E9C46A"),
    ):
        boxes = ax.boxplot(values, tick_labels=class_names, patch_artist=True, showfliers=False)
        for box in boxes["boxes"]: box.set_facecolor(color)
        ax.set(title=title, ylabel=ylabel); ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); paths["per_class_boxplots"] = output / "eda_per_class_brightness_contrast.png"; fig.savefig(paths["per_class_boxplots"], dpi=140, bbox_inches="tight"); plt.close(fig)

    # Practice 1: deterministic representative samples from every class.
    fig, axes = plt.subplots(2, len(class_names), figsize=(18, 4.5))
    for ax, sample_index in zip(axes.flat, analysis["sample_indices"]):
        ax.imshow(images[sample_index]); ax.set_title(class_names[labels[sample_index]], fontsize=9); ax.axis("off")
    fig.suptitle("Deterministic Train Samples — Two per Class", fontsize=15); fig.tight_layout(); paths["representative_samples"] = output / "eda_representative_samples.png"; fig.savefig(paths["representative_samples"], dpi=140, bbox_inches="tight"); plt.close(fig)

    # Practice 1: both brightness and contrast extremes.
    extreme_groups = [
        ("Darkest", np.argsort(brightness)[:5], brightness),
        ("Brightest", np.argsort(brightness)[-5:], brightness),
        ("Lowest contrast", np.argsort(contrast)[:5], contrast),
        ("Highest contrast", np.argsort(contrast)[-5:], contrast),
    ]
    fig, axes = plt.subplots(4, 5, figsize=(15, 11))
    for row, (group, indices, metric) in enumerate(extreme_groups):
        for ax, index in zip(axes[row], indices):
            ax.imshow(images[index]); ax.set_title(f"{class_names[labels[index]]}\n{metric[index]:.3f}", fontsize=8); ax.axis("off")
        axes[row, 0].set_ylabel(group, fontsize=10)
    fig.suptitle("Training-Subset Statistical Extremes", fontsize=16); fig.tight_layout(); paths["all_extremes"] = output / "eda_brightness_contrast_extremes.png"; fig.savefig(paths["all_extremes"], dpi=140, bbox_inches="tight"); plt.close(fig)

    # Practice 1: pixel correlation, adapted to sampled RGB spatial features.
    rng = np.random.default_rng(seed)
    correlation_rows = rng.choice(len(images), min(5_000, len(images)), replace=False)
    feature_indices = np.sort(rng.choice(32 * 32 * 3, 100, replace=False))
    correlation_features = scaled[correlation_rows].reshape(len(correlation_rows), -1)[:, feature_indices]
    correlation = np.corrcoef(correlation_features, rowvar=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set(title="Correlation of 100 Deterministic RGB Spatial Features", xlabel="Sampled RGB feature", ylabel="Sampled RGB feature")
    fig.colorbar(image, ax=ax, label="Pearson correlation"); fig.tight_layout(); paths["correlation"] = output / "eda_rgb_feature_correlation.png"; fig.savefig(paths["correlation"], dpi=140, bbox_inches="tight"); plt.close(fig)

    # Practice 1: raw PCA, 3-D view, variance curve and component images.
    projection_indices = np.asarray(analysis["projection_indices"], dtype=np.int64)
    projection_labels = labels[projection_indices]
    raw_features = scaled[projection_indices, ::2, ::2, :].reshape(len(projection_indices), -1)
    component_count = min(50, len(raw_features) - 1, raw_features.shape[1])
    raw_pca = PCA(n_components=component_count, svd_solver="randomized", random_state=seed)
    raw_scores = raw_pca.fit_transform(raw_features)

    fig = plt.figure(figsize=(11, 8)); ax = fig.add_subplot(111, projection="3d")
    for class_id, name in enumerate(class_names):
        mask = projection_labels == class_id; ax.scatter(raw_scores[mask, 0], raw_scores[mask, 1], raw_scores[mask, 2], s=16, alpha=.65, label=name)
    ax.set(title="Raw RGB PCA — First Three Components", xlabel="PC1", ylabel="PC2", zlabel="PC3"); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); paths["pca_3d"] = output / "eda_pca_3d.png"; fig.savefig(paths["pca_3d"], dpi=140, bbox_inches="tight"); plt.close(fig)

    cumulative = np.cumsum(raw_pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(10, 5)); ax.plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o", markersize=3)
    ax.axhline(.80, color="#AE2012", linestyle="--", label="80% variance"); ax.set(title="PCA Cumulative Explained Variance", xlabel="Principal components", ylabel="Cumulative variance"); ax.set_ylim(0, 1); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); paths["explained_variance"] = output / "eda_pca_explained_variance.png"; fig.savefig(paths["explained_variance"], dpi=140, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for index, ax in enumerate(axes):
        component = raw_pca.components_[index].reshape(16, 16, 3)
        component = (component - component.min()) / (component.max() - component.min() + 1e-12)
        ax.imshow(component); ax.set_title(f"RGB Principal Component {index + 1}"); ax.axis("off")
    fig.tight_layout(); paths["component_images"] = output / "eda_rgb_principal_components.png"; fig.savefig(paths["component_images"], dpi=140, bbox_inches="tight"); plt.close(fig)

    # Practice 1: standardized PCA followed by t-SNE.
    standardized = StandardScaler().fit_transform(raw_features)
    standardized_pca = PCA(n_components=component_count, svd_solver="randomized", random_state=seed)
    standardized_scores = standardized_pca.fit_transform(standardized)
    perplexity = min(30, max(2, (len(standardized_scores) - 1) // 3))
    tsne_scores = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto", max_iter=750, init="pca", random_state=seed).fit_transform(standardized_scores)
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    for class_id, name in enumerate(class_names):
        mask = projection_labels == class_id
        axes[0].scatter(standardized_scores[mask, 0], standardized_scores[mask, 1], s=14, alpha=.65, label=name)
        axes[1].scatter(tsne_scores[mask, 0], tsne_scores[mask, 1], s=14, alpha=.65, label=name)
    axes[0].set(title=f"Standardized PCA ({component_count} PCs: {standardized_pca.explained_variance_ratio_.sum():.1%})", xlabel="PC1", ylabel="PC2")
    axes[1].set(title="t-SNE after Standardized PCA — Train Only", xlabel="t-SNE 1", ylabel="t-SNE 2")
    for ax in axes: ax.grid(alpha=.2); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); paths["standardized_pca_tsne"] = output / "eda_standardized_pca_tsne.png"; fig.savefig(paths["standardized_pca_tsne"], dpi=140, bbox_inches="tight"); plt.close(fig)
    return paths
