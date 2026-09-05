"""Phase 54 — figure renderers.

Core figures (Phase 54 detail):
* LASTQ_54_01..06  mean profile per seed/layer (all heads overlaid)
* LASTQ_54_07       layer head-mean profiles (overlay across seeds per layer)
* LASTQ_54_08       normalized entropy by head (per seed/layer)
* LASTQ_54_09       expected lag by head (per seed/layer)
* LASTQ_54_10       top5 mass by head
* LASTQ_54_11       recent mass by head (1h/6h/12h/24h)
* LASTQ_54_12       non-overlap lag bins stacked bars by head
* LASTQ_54_13       Lag50/Lag80/Lag90 by head
* LASTQ_54_14       top1 lag frequency heatmap per seed/layer
* LASTQ_54_15       cumulative recency profiles (mean) by head

Report-case line plots (deterministic, Phase 51 shared ranks 1-5):
* SHARED_R{01..05}_SEED{42,123,2026}_LAST_QUERY.png — one grid per case×seed
  with rows=layers, cols=heads; x=lag minutes, y=raw last-query weight.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .sources import (  # noqa: E402
    CADENCE_MINUTES,
    LOOKBACK,
    N_TEST,
    NUM_HEADS,
    NUM_LAYERS,
    SEEDS,
)

LAYER_HEAD_MEAN_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                          "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def _save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_mean_profiles_per_seed_layer(
    raw: dict[int, np.ndarray],
    out_dir: Path,
) -> list[Path]:
    """LASTQ_54_01..06: per seed/layer, plot all heads as separate lines
    on common y-axis. x = lag minutes, recency order (lag1 left, lagL right)."""
    paths: list[Path] = []
    # Use cumulative mass plot style: x = lag radius, y = mean cumulative
    # mass. Mean cumulative curve is monotonically non-decreasing.
    for seed in SEEDS:
        arr = raw[seed]  # [N, L, H, L]
        mean_block = arr.mean(axis=0)  # [L, H, L]
        for li in range(NUM_LAYERS):
            fig, ax = plt.subplots(figsize=(8.0, 4.5))
            lag_minutes = np.arange(1, LOOKBACK + 1) * CADENCE_MINUTES
            for hi in range(NUM_HEADS):
                # Recency order: newest->oldest
                vec_recency = mean_block[li, hi, ::-1]
                cum = np.cumsum(vec_recency)
                ax.plot(
                    lag_minutes,
                    cum,
                    label=f"H{hi+1}",
                    color=LAYER_HEAD_MEAN_COLORS[hi % len(LAYER_HEAD_MEAN_COLORS)],
                    linewidth=1.6,
                )
            ax.set_xlabel("Lag from forecast target (minutes, recency order: newest left)")
            ax.set_ylabel("Cumulative attention mass (mean across Test)")
            ax.set_title(f"Seed {seed} — Layer L{li+1} — Mean cumulative last-query recency profile")
            ax.set_xlim(CADENCE_MINUTES, LOOKBACK * CADENCE_MINUTES)
            ax.set_ylim(0.0, 1.02)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="lower right", title="Head", ncol=NUM_HEADS, fontsize=9)
            # Lag markers
            for m_lag, m_lab in [(1, "10m"), (6, "1h"), (36, "6h"), (72, "12h")]:
                ax.axvline(m_lag * CADENCE_MINUTES, color="#999", linestyle="--", alpha=0.4)
            _save_fig(fig, out_dir / f"LASTQ_54_mean_cumulative_seed{seed}_layer{li+1:02d}.png")
            paths.append(out_dir / f"LASTQ_54_mean_cumulative_seed{seed}_layer{li+1:02d}.png")
    return paths


def render_layer_head_mean_profiles(
    raw: dict[int, np.ndarray],
    out_dir: Path,
) -> list[Path]:
    """LASTQ_54_07: layer head-mean profile, overlay across seeds per layer."""
    paths: list[Path] = []
    seed_colors = {42: "#1f77b4", 123: "#2ca02c", 2026: "#d62728"}
    fig, axes = plt.subplots(1, NUM_LAYERS, figsize=(13.0, 4.5), sharey=True)
    if NUM_LAYERS == 1:
        axes = [axes]
    for li in range(NUM_LAYERS):
        ax = axes[li]
        lag_minutes = np.arange(1, LOOKBACK + 1) * CADENCE_MINUTES
        for seed in SEEDS:
            arr = raw[seed]
            mean_block = arr.mean(axis=0)  # [L, H, L]
            head_mean = mean_block[li, :, :].mean(axis=0)  # [L]
            # Recency order
            vec_recency = head_mean[::-1]
            cum = np.cumsum(vec_recency)
            ax.plot(
                lag_minutes, cum,
                label=f"seed {seed}",
                color=seed_colors.get(int(seed), "#333"),
                linewidth=2.0,
            )
        ax.set_xlabel("Lag from forecast target (minutes, recency order)")
        if li == 0:
            ax.set_ylabel("Cumulative attention mass (layer head-mean)")
        ax.set_title(f"Layer L{li+1}")
        ax.set_xlim(CADENCE_MINUTES, LOOKBACK * CADENCE_MINUTES)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, linestyle=":", alpha=0.5)
        for m_lag in [1, 6, 36, 72]:
            ax.axvline(m_lag * CADENCE_MINUTES, color="#999", linestyle="--", alpha=0.4)
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Layer head-mean cumulative last-query recency profile (descriptive overlay, NOT stability claim)")
    _save_fig(fig, out_dir / "LASTQ_54_07_layer_head_mean_profiles.png")
    paths.append(out_dir / "LASTQ_54_07_layer_head_mean_profiles.png")
    return paths


def render_distributions_by_head(
    metrics_long: list[dict],
    out_dir: Path,
) -> list[Path]:
    """LASTQ_54_08..11: per-seed/layer, boxplots by head for normalized entropy,
    expected lag (minutes), top5 mass, recent mass."""
    paths: list[Path] = []
    # Group by seed/layer
    grouped: dict[tuple, list[dict]] = {}
    for r in metrics_long:
        try:
            k = (int(r["seed"]), int(r["layer_idx0"]))
        except (KeyError, ValueError):
            continue
        grouped.setdefault(k, []).append(r)

    metric_specs = [
        ("normalized_entropy", "Normalized entropy (H/log L)", "08"),
        ("expected_lag_minutes", "Expected lag (minutes)", "09"),
        ("top5_mass", "Top5 mass", "10"),
        ("recent_1h_mass", "Recent 1h mass", "11"),
    ]

    for (seed, layer), rows in sorted(grouped.items()):
        # Build [head][values] matrix
        for metric, ylabel, fig_no in metric_specs:
            fig, ax = plt.subplots(figsize=(7.0, 4.0))
            data = []
            for hi in range(NUM_HEADS):
                vals = [float(r[metric]) for r in rows if int(r["head_idx0"]) == hi]
                data.append(vals)
            bp = ax.boxplot(
                data,
                positions=range(NUM_HEADS),
                widths=0.55,
                patch_artist=True,
                medianprops={"color": "black", "linewidth": 1.2},
            )
            for patch, color in zip(bp["boxes"], LAYER_HEAD_MEAN_COLORS):
                patch.set_facecolor(color)
                patch.set_alpha(0.55)
            ax.set_xticks(range(NUM_HEADS))
            ax.set_xticklabels([f"H{hi+1}" for hi in range(NUM_HEADS)])
            ax.set_xlabel("Head (architectural order)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Seed {seed} — Layer L{layer+1}")
            ax.grid(True, linestyle=":", alpha=0.5)
            _save_fig(fig, out_dir / f"LASTQ_54_{fig_no}_{metric}_seed{seed}_layer{layer+1:02d}.png")
            paths.append(out_dir / f"LASTQ_54_{fig_no}_{metric}_seed{seed}_layer{layer+1:02d}.png")
    return paths


def render_nonoverlap_lag_bins(
    lag_bin_rows: list[dict],
    out_dir: Path,
) -> list[Path]:
    """LASTQ_54_12: per seed/layer/head, stacked bar of non-overlap lag bins."""
    paths: list[Path] = []
    from collections import defaultdict
    by_seed_layer_head: dict[tuple, list[dict]] = defaultdict(list)
    for r in lag_bin_rows:
        try:
            k = (int(r["seed"]), int(r["layer_idx0"]), int(r["head_idx0"]))
        except (KeyError, ValueError):
            continue
        by_seed_layer_head[k].append(r)

    for seed in SEEDS:
        for layer in range(NUM_LAYERS):
            for head in range(NUM_HEADS):
                rows = by_seed_layer_head.get((seed, layer, head), [])
                if not rows:
                    continue
                labels = []
                masses = []
                applicable = []
                for r in rows:
                    if r.get("applicable") == "True":
                        labels.append(r["bin_label"])
                        masses.append(float(r["mass"]))
                        applicable.append(True)
                if not masses:
                    continue
                fig, ax = plt.subplots(figsize=(6.5, 3.5))
                ax.bar(labels, masses, color="#1f77b4", edgecolor="black", linewidth=0.4)
                ax.set_ylim(0.0, max(1.05, max(masses) * 1.1))
                ax.set_ylabel("Mean mass")
                ax.set_title(f"Seed {seed} L{layer+1} H{head+1} — non-overlap lag bins (sum = {sum(masses):.3f})")
                ax.grid(True, linestyle=":", alpha=0.5, axis="y")
                plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
                _save_fig(fig, out_dir / f"LASTQ_54_12_lag_bins_seed{seed}_layer{layer+1:02d}_head{head+1}.png")
                paths.append(out_dir / f"LASTQ_54_12_lag_bins_seed{seed}_layer{layer+1:02d}_head{head+1}.png")
    return paths


def render_coverage_radii(
    coverage_rows: list[dict],
    out_dir: Path,
) -> list[Path]:
    """LASTQ_54_13: per seed/layer, grouped bar of Lag50/Lag80/Lag90 mean (minutes) per head."""
    paths: list[Path] = []
    from collections import defaultdict
    by_seed_layer_head: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in coverage_rows:
        try:
            k = (int(r["seed"]), int(r["layer_idx0"]), int(r["head_idx0"]))
            lvl = r["coverage_level"]
            mean_min = float(r["mean_minutes"])
        except (KeyError, ValueError):
            continue
        by_seed_layer_head[k][lvl] = mean_min

    for seed in SEEDS:
        for layer in range(NUM_LAYERS):
            fig, ax = plt.subplots(figsize=(7.0, 4.0))
            x = np.arange(NUM_HEADS)
            width = 0.27
            for i, (lvl, lab) in enumerate([("0.50", "Lag50"), ("0.80", "Lag80"), ("0.90", "Lag90")]):
                vals = [by_seed_layer_head.get((seed, layer, hi), {}).get(lvl, 0.0) for hi in range(NUM_HEADS)]
                ax.bar(x + (i - 1) * width, vals, width, label=lab, color=["#1f77b4", "#ff7f0e", "#2ca02c"][i])
            ax.set_xticks(x)
            ax.set_xticklabels([f"H{hi+1}" for hi in range(NUM_HEADS)])
            ax.set_xlabel("Head (architectural order)")
            ax.set_ylabel("Coverage radius (minutes)")
            ax.set_title(f"Seed {seed} — Layer L{layer+1} — coverage radii (descriptive)")
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.5, axis="y")
            _save_fig(fig, out_dir / f"LASTQ_54_13_coverage_radii_seed{seed}_layer{layer+1:02d}.png")
            paths.append(out_dir / f"LASTQ_54_13_coverage_radii_seed{seed}_layer{layer+1:02d}.png")
    return paths


def render_top1_lag_frequency(
    top1_freq_rows: list[dict],
    out_dir: Path,
) -> list[Path]:
    """LASTQ_54_14: per seed/layer, heatmap of top1 lag frequency (rows=heads, cols=lag)."""
    paths: list[Path] = []
    from collections import defaultdict
    by_seed_layer_head: dict[tuple, dict[int, float]] = defaultdict(dict)
    for r in top1_freq_rows:
        try:
            k = (int(r["seed"]), int(r["layer_idx0"]), int(r["head_idx0"]))
            lag = int(r["lag_steps"])
            frac = float(r["fraction"])
        except (KeyError, ValueError):
            continue
        by_seed_layer_head[k][lag] = frac

    for seed in SEEDS:
        for layer in range(NUM_LAYERS):
            matrix = np.zeros((NUM_HEADS, LOOKBACK), dtype=np.float64)
            for hi in range(NUM_HEADS):
                row_dict = by_seed_layer_head.get((seed, layer, hi), {})
                for lag in range(1, LOOKBACK + 1):
                    matrix[hi, lag - 1] = row_dict.get(lag, 0.0)
            fig, ax = plt.subplots(figsize=(10.0, 3.0))
            im = ax.imshow(matrix, aspect="auto", cmap="viridis")
            ax.set_yticks(range(NUM_HEADS))
            ax.set_yticklabels([f"H{hi+1}" for hi in range(NUM_HEADS)])
            ax.set_xticks(np.arange(0, LOOKBACK, max(1, LOOKBACK // 12)))
            ax.set_xticklabels([str(int(lag)) for lag in np.arange(1, LOOKBACK + 1, max(1, LOOKBACK // 12))])
            ax.set_xlabel("Lag (steps)")
            ax.set_ylabel("Head (architectural order)")
            ax.set_title(f"Seed {seed} L{layer+1} — Top1 lag frequency (NEWEST_SOURCE tie rule)")
            fig.colorbar(im, ax=ax, label="Fraction of Test targets")
            _save_fig(fig, out_dir / f"LASTQ_54_14_top1_freq_seed{seed}_layer{layer+1:02d}.png")
            paths.append(out_dir / f"LASTQ_54_14_top1_freq_seed{seed}_layer{layer+1:02d}.png")
    return paths


def render_cumulative_recency_profiles(
    raw: dict[int, np.ndarray],
    out_dir: Path,
) -> list[Path]:
    """LASTQ_54_15: per seed/layer, mean cumulative recency curves by head."""
    paths: list[Path] = []
    lag_minutes = np.arange(1, LOOKBACK + 1) * CADENCE_MINUTES
    for seed in SEEDS:
        arr = raw[seed]
        mean_block = arr.mean(axis=0)  # [L, H, L]
        for li in range(NUM_LAYERS):
            fig, ax = plt.subplots(figsize=(8.0, 4.5))
            for hi in range(NUM_HEADS):
                vec_recency = mean_block[li, hi, ::-1]
                cum = np.cumsum(vec_recency)
                ax.plot(
                    lag_minutes, cum,
                    label=f"H{hi+1}",
                    color=LAYER_HEAD_MEAN_COLORS[hi % len(LAYER_HEAD_MEAN_COLORS)],
                    linewidth=1.6,
                )
            for threshold, label in [(0.50, "Lag50"), (0.80, "Lag80"), (0.90, "Lag90")]:
                ax.axhline(threshold, color="#888", linestyle="--", alpha=0.5)
                ax.text(lag_minutes[-1] * 0.97, threshold + 0.01, label,
                        ha="right", fontsize=8, color="#555")
            ax.set_xlabel("Lag from forecast target (minutes, recency order)")
            ax.set_ylabel("Mean cumulative attention mass")
            ax.set_title(f"Seed {seed} L{li+1} — mean cumulative last-query recency profile")
            ax.set_xlim(CADENCE_MINUTES, LOOKBACK * CADENCE_MINUTES)
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="lower right", fontsize=9)
            _save_fig(fig, out_dir / f"LASTQ_54_15_cumulative_recency_seed{seed}_layer{li+1:02d}.png")
            paths.append(out_dir / f"LASTQ_54_15_cumulative_recency_seed{seed}_layer{li+1:02d}.png")
    return paths


def render_report_case_figure(
    raw: dict[int, np.ndarray],
    case_row_idx0: int,
    target_id: str,
    target_timestamp: str,
    shared_worst_rank: int,
    target_to_row: dict[str, int],
    out_dir: Path,
) -> list[Path]:
    """One grid per case × seed: rows=layers, cols=heads. Each panel:
    x = lag (minutes, recency order), y = raw last-query weight.
    Common y-axis within figure. No smoothing."""
    paths: list[Path] = []
    if target_id not in target_to_row:
        return paths
    target_row = target_to_row[target_id]

    for seed in SEEDS:
        arr = raw[seed]
        fig, axes = plt.subplots(
            NUM_LAYERS, NUM_HEADS, figsize=(11.0, 5.5), sharex=True, sharey=True
        )
        lag_minutes = np.arange(1, LOOKBACK + 1) * CADENCE_MINUTES
        # Determine common y-limits
        max_y = float(arr[target_row, :, :, :].max())
        for li in range(NUM_LAYERS):
            for hi in range(NUM_HEADS):
                ax = axes[li, hi] if NUM_LAYERS > 1 else axes[hi]
                vec = arr[target_row, li, hi, :]
                # Recency order
                vec_recency = vec[::-1]
                ax.plot(lag_minutes, vec_recency, color="#1f77b4", linewidth=1.0)
                ax.set_ylim(0.0, max(max_y * 1.1, 0.05))
                ax.set_xlim(CADENCE_MINUTES, LOOKBACK * CADENCE_MINUTES)
                ax.grid(True, linestyle=":", alpha=0.5)
                if li == NUM_LAYERS - 1:
                    ax.set_xlabel("Lag (min, recency order)")
                if hi == 0:
                    ax.set_ylabel(f"L{li+1}")
                ax.set_title(f"H{hi+1}", fontsize=10)
        fig.suptitle(
            f"Phase 54 — SHARED_R{shared_worst_rank:02d} — "
            f"target {target_id} — ts {target_timestamp} — seed {seed}"
        )
        _save_fig(fig, out_dir / f"SHARED_R{shared_worst_rank:02d}_SEED{seed}_LAST_QUERY.png")
        paths.append(out_dir / f"SHARED_R{shared_worst_rank:02d}_SEED{seed}_LAST_QUERY.png")
    return paths


def render_all_core_figures(
    raw: dict[int, np.ndarray],
    metrics_long: list[dict],
    lag_bin_rows: list[dict],
    coverage_rows: list[dict],
    top1_freq_rows: list[dict],
    out_dir: Path,
) -> list[Path]:
    """Render all Phase 54 core figures."""
    paths: list[Path] = []
    paths += render_mean_profiles_per_seed_layer(raw, out_dir)
    paths += render_layer_head_mean_profiles(raw, out_dir)
    paths += render_distributions_by_head(metrics_long, out_dir)
    paths += render_nonoverlap_lag_bins(lag_bin_rows, out_dir)
    paths += render_coverage_radii(coverage_rows, out_dir)
    paths += render_top1_lag_frequency(top1_freq_rows, out_dir)
    paths += render_cumulative_recency_profiles(raw, out_dir)
    return paths


def render_report_case_figures(
    raw: dict[int, np.ndarray],
    cases: list,
    target_to_row: dict[str, int],
    out_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    for c in cases:
        paths += render_report_case_figure(
            raw,
            c.case_row_idx0,
            c.target_id,
            c.target_timestamp,
            c.shared_worst_rank,
            target_to_row,
            out_dir,
        )
    return paths
