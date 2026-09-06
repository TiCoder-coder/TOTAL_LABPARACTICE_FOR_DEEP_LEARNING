"""Phase 55 - core figures (HEAD_55_01..HEAD_55_14)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .sources import (
    SEEDS,
    NUM_HEADS,
    NUM_LAYERS,
    LAG_MINUTES,
)


HEAD_LABELS = [f"H{i + 1}" for i in range(NUM_HEADS)]


def _save_fig(fig: plt.Figure, fp: Path) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fp, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _annotate_matrix(ax, m: np.ndarray, title: str, vmin: float, vmax: float, cmap: str) -> None:
    im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(NUM_HEADS))
    ax.set_yticks(range(NUM_HEADS))
    ax.set_xticklabels(HEAD_LABELS)
    ax.set_yticklabels(HEAD_LABELS)
    ax.set_title(title)
    for i in range(NUM_HEADS):
        for j in range(NUM_HEADS):
            v = m[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                        color=("white" if v < (vmin + vmax) / 2 else "black"))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def fig_jsd_matrices(matrices: dict[tuple[int, int], np.ndarray], fp: Path) -> None:
    n = len(matrices)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax in axes:
        ax.axis("off")
    for idx, ((seed, layer), m) in enumerate(sorted(matrices.items())):
        ax = axes[idx]
        ax.axis("on")
        _annotate_matrix(ax, m, f"seed={seed} layer={layer}", 0.0, float(np.log(2.0)), "magma")
    fig.suptitle("HEAD_55_01 — Pairwise JSD matrices (0..ln2)")
    _save_fig(fig, fp)


def fig_cosine_matrices(matrices: dict[tuple[int, int], np.ndarray], fp: Path) -> None:
    n = len(matrices)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax in axes:
        ax.axis("off")
    for idx, ((seed, layer), m) in enumerate(sorted(matrices.items())):
        ax = axes[idx]
        ax.axis("on")
        _annotate_matrix(ax, m, f"seed={seed} layer={layer}", 0.0, 1.0, "viridis")
    fig.suptitle("HEAD_55_02 — Pairwise cosine similarity matrices")
    _save_fig(fig, fp)


def fig_wasserstein_matrices(matrices: dict[tuple[int, int], np.ndarray], fp: Path) -> None:
    n = len(matrices)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    # Common max across all matrices for shared color scale
    all_vals = [v for m in matrices.values() for v in m.flat if np.isfinite(v)]
    common_max = max(all_vals) if all_vals else 1.0
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax in axes:
        ax.axis("off")
    for idx, ((seed, layer), m) in enumerate(sorted(matrices.items())):
        ax = axes[idx]
        ax.axis("on")
        _annotate_matrix(ax, m, f"seed={seed} layer={layer}", 0.0, common_max, "cividis")
    fig.suptitle(f"HEAD_55_03 — Pairwise Wasserstein distance matrices (minutes; common max={common_max:.1f})")
    _save_fig(fig, fp)


def fig_expected_lag_diff_matrices(matrices: dict[tuple[int, int], np.ndarray], fp: Path) -> None:
    n = len(matrices)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    all_vals = [v for m in matrices.values() for v in m.flat if np.isfinite(v)]
    common_max = max(all_vals) if all_vals else 1.0
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax in axes:
        ax.axis("off")
    for idx, ((seed, layer), m) in enumerate(sorted(matrices.items())):
        ax = axes[idx]
        ax.axis("on")
        _annotate_matrix(ax, m, f"seed={seed} layer={layer}", 0.0, common_max, "plasma")
    fig.suptitle(f"HEAD_55_04 — Pairwise |Δ expected lag| matrices (min; common max={common_max:.1f})")
    _save_fig(fig, fp)


def fig_top1_tvd_matrices(matrices: dict[tuple[int, int], np.ndarray], fp: Path) -> None:
    n = len(matrices)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax in axes:
        ax.axis("off")
    for idx, ((seed, layer), m) in enumerate(sorted(matrices.items())):
        ax = axes[idx]
        ax.axis("on")
        _annotate_matrix(ax, m, f"seed={seed} layer={layer}", 0.0, 1.0, "Reds")
    fig.suptitle("HEAD_55_05 — Pairwise top1 lag TVD matrices (0..1)")
    _save_fig(fig, fp)


def fig_metric_by_head(
    behavior_rows: list[Any],
    metric_attr: str,
    title: str,
    ylabel: str,
    fp: Path,
) -> None:
    n = len(SEEDS) * NUM_LAYERS
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax in axes:
        ax.axis("off")
    idx = 0
    for seed in SEEDS:
        for layer in range(NUM_LAYERS):
            ax = axes[idx]
            ax.axis("on")
            vals = []
            for head in range(NUM_HEADS):
                for r in behavior_rows:
                    if r.seed == seed and r.layer_idx0 == layer and r.head_idx0 == head:
                        vals.append(getattr(r, metric_attr))
                        break
                else:
                    vals.append(float("nan"))
            ax.bar(HEAD_LABELS, vals, color="steelblue")
            ax.set_title(f"seed={seed} layer={layer}")
            ax.set_ylabel(ylabel)
            idx += 1
    fig.suptitle(title)
    _save_fig(fig, fp)


def fig_head_to_layer_mean_jsd(rows: list[dict[str, Any]], fp: Path) -> None:
    n = len(SEEDS) * NUM_LAYERS
    cols = min(3, n)
    rows_count = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_count, cols, figsize=(4 * cols, 4 * rows_count))
    axes = np.atleast_1d(axes).flatten()
    for ax in axes:
        ax.axis("off")
    idx = 0
    # Cast seed/layer to int for comparison
    rows_norm = []
    for r in rows:
        rr = dict(r)
        rr["seed"] = int(rr["seed"]) if rr.get("seed") is not None else -1
        rr["layer_idx0"] = int(rr["layer_idx0"]) if rr.get("layer_idx0") is not None else -1
        rows_norm.append(rr)
    for seed in SEEDS:
        for layer in range(NUM_LAYERS):
            ax = axes[idx]
            ax.axis("on")
            vals = [r["jsd_to_layer_head_mean_profile"] for r in rows_norm
                    if r["seed"] == seed and r["layer_idx0"] == layer]
            if not vals:
                vals = [0.0] * NUM_HEADS
            ax.bar(HEAD_LABELS, vals, color="darkorange")
            ax.set_title(f"seed={seed} layer={layer}")
            ax.set_ylabel("JSD to layer mean")
            idx += 1
    fig.suptitle("HEAD_55_11 — JSD to layer head-mean profile (no outlier threshold)")
    _save_fig(fig, fp)


def fig_layer_diversity(layer_div_rows: list[dict[str, Any]], fp: Path) -> None:
    """Transparent separate panels per metric; no weighted composite."""
    metrics = [
        ("mean_pairwise_jsd", "Mean pairwise JSD"),
        ("mean_pairwise_wasserstein_minutes", "Mean pairwise Wasserstein (min)"),
        ("mean_pairwise_l1", "Mean pairwise L1"),
        ("mean_pairwise_abs_expected_lag_diff_minutes", "Mean pairwise |Δ expected lag| (min)"),
        ("mean_pairwise_top1_tvd", "Mean pairwise top1 TVD"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 3 * len(metrics)))
    axes = np.atleast_1d(axes).flatten()
    for ax, (key, label) in zip(axes, metrics):
        labels = []
        vals = []
        for r in layer_div_rows:
            labels.append(f"seed{r['seed']}L{r['layer_idx0']}")
            v = r.get(key, float("nan"))
            vals.append(float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else 0.0)
        ax.bar(labels, vals, color="seagreen")
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("HEAD_55_12 — Layer head-diversity summary (separate panels)")
    fig.tight_layout()
    _save_fig(fig, fp)


def fig_mean_temporal_profiles(
    by_head: dict[tuple[int, int, int], np.ndarray],
    fp: Path,
) -> None:
    """Per (seed, layer) overlay of mean temporal profiles for each head."""
    n = len(SEEDS) * NUM_LAYERS
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax in axes:
        ax.axis("off")
    idx = 0
    for seed in SEEDS:
        for layer in range(NUM_LAYERS):
            ax = axes[idx]
            ax.axis("on")
            for head in range(NUM_HEADS):
                p = by_head.get((seed, layer, head))
                if p is None:
                    continue
                ax.plot(LAG_MINUTES, p, label=f"H{head + 1}")
            ax.set_title(f"seed={seed} layer={layer}")
            ax.set_xlabel("lag (minutes)")
            ax.set_ylabel("mean weight")
            ax.legend(fontsize=7, loc="upper right")
            idx += 1
    fig.suptitle("HEAD_55_13 — Mean temporal profiles by head (architectural order)")
    fig.tight_layout()
    _save_fig(fig, fp)


def fig_paired_difference_distributions(
    paired_rows: list[Any],
    fp: Path,
) -> None:
    """Show all head pairs' paired-difference medians for selected metrics."""
    metrics = ["normalized_entropy", "expected_lag_minutes", "recent_1h_mass", "lag80_minutes"]
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n))
    axes = np.atleast_1d(axes).flatten()
    for ax, metric in zip(axes, metrics):
        # Group by (seed, layer) and draw all pairs in architectural order
        # For each pair, take median_difference
        labels = []
        medians = []
        for seed in SEEDS:
            for layer in range(NUM_LAYERS):
                for ha in range(NUM_HEADS):
                    for hb in range(ha + 1, NUM_HEADS):
                        # find the row
                        for r in paired_rows:
                            if (r.seed == seed and r.layer_idx0 == layer
                                    and r.head_a_idx0 == ha and r.head_b_idx0 == hb
                                    and r.metric == metric):
                                labels.append(f"s{seed}L{layer}H{ha+1}-H{hb+1}")
                                medians.append(r.median_difference if np.isfinite(r.median_difference) else 0.0)
                                break
        ax.bar(labels, medians, color="mediumpurple")
        ax.set_title(f"Paired median difference: {metric}")
        ax.set_ylabel(f"Δ {metric} (A - B)")
        ax.tick_params(axis="x", rotation=70, labelsize=6)
    fig.suptitle("HEAD_55_14 — Paired-difference distributions (architectural pair order)")
    fig.tight_layout()
    _save_fig(fig, fp)
