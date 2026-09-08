"""Phase 56 - figures (ERRORATTN_56_01..14)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .core_metrics import compute_core_metrics_vector
from .core_metrics import LAG_MINUTES
from .sources import CORE_ATTENTION_METRICS_V1, NUM_HEADS, NUM_LAYERS, SEEDS


HEAD_LABELS = [f"H{h+1}" for h in range(NUM_HEADS)]
SEED_LABELS = [f"Seed {s}" for s in SEEDS]


def _heatmap(data: np.ndarray, title: str, path: Path, vmin: float = -1, vmax: float = 1, cmap: str = "RdBu_r"):
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(CORE_ATTENTION_METRICS_V1)))
    ax.set_xticklabels(CORE_ATTENTION_METRICS_V1, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(NUM_HEADS))
    ax.set_yticklabels(HEAD_LABELS, fontsize=9)
    for i in range(NUM_HEADS):
        for j in range(len(CORE_ATTENTION_METRICS_V1)):
            ax.text(j, i, f"{data[i, j]:+.2f}", ha="center", va="center", fontsize=7, color="black")
    ax.set_title(title, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_01_abs_error_associations(continuous_matrix_ae: list[dict], outdir: Path):
    """For each seed/layer, heatmap rows=heads cols=metrics value=Spearman rho(AE)."""
    for seed in SEEDS:
        for layer in range(NUM_LAYERS):
            data = np.zeros((NUM_HEADS, len(CORE_ATTENTION_METRICS_V1)))
            for r in continuous_matrix_ae:
                if r["seed"] == seed and r["layer_idx0"] == layer:
                    i = int(r["head_idx0"])
                    j = CORE_ATTENTION_METRICS_V1.index(r["attention_metric"])
                    data[i, j] = float(r["spearman_rho"])
            title = f"AE Spearman rho — Seed {seed}, Layer {layer}"
            _heatmap(data, title, outdir / f"ERRORATTN_56_01_abs_error_seed{seed}_layer{layer}.png")


def fig_02_signed_residual_associations(continuous_matrix_signed: list[dict], outdir: Path):
    for seed in SEEDS:
        for layer in range(NUM_LAYERS):
            data = np.zeros((NUM_HEADS, len(CORE_ATTENTION_METRICS_V1)))
            for r in continuous_matrix_signed:
                if r["seed"] == seed and r["layer_idx0"] == layer:
                    i = int(r["head_idx0"])
                    j = CORE_ATTENTION_METRICS_V1.index(r["attention_metric"])
                    data[i, j] = float(r["spearman_rho"])
            title = f"Signed residual Spearman rho — Seed {seed}, Layer {layer}"
            _heatmap(data, title, outdir / f"ERRORATTN_56_02_signed_seed{seed}_layer{layer}.png", cmap="RdBu_r")


def fig_03_high_low_cliffs_delta(cliffs_matrix: list[dict], outdir: Path):
    for seed in SEEDS:
        for layer in range(NUM_LAYERS):
            data = np.zeros((NUM_HEADS, len(CORE_ATTENTION_METRICS_V1)))
            for r in cliffs_matrix:
                if r["seed"] == seed and r["layer_idx0"] == layer:
                    i = int(r["head_idx0"])
                    j = CORE_ATTENTION_METRICS_V1.index(r["attention_metric"])
                    data[i, j] = float(r["cliffs_delta"])
            title = f"Cliff's delta HIGH vs LOW — Seed {seed}, Layer {layer}"
            _heatmap(data, title, outdir / f"ERRORATTN_56_03_cliffs_delta_seed{seed}_layer{layer}.png")


def fig_04_layer_head_mean_associations(layer_association: list[dict], outdir: Path):
    """Compact panel per seed/layer with 6 core metric bars."""
    n_metrics = len(CORE_ATTENTION_METRICS_V1)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for idx, metric in enumerate(CORE_ATTENTION_METRICS_V1):
        ax = axes[idx // 3, idx % 3]
        bar_x = []
        bar_y = []
        colors = []
        for seed_idx, seed in enumerate(SEEDS):
            for layer in range(NUM_LAYERS):
                val = 0.0
                for r in layer_association:
                    if (r["seed"] == seed and r["layer_idx0"] == layer
                            and r["conditioning_variable"] == "ABS_ERROR"
                            and r["attention_metric"] == metric):
                        val = float(r["spearman_rho"])
                bar_x.append(f"S{seed}\nL{layer}")
                bar_y.append(val)
                colors.append(["tab:blue", "tab:orange", "tab:green"][seed_idx])
        ax.bar(range(len(bar_y)), bar_y, color=colors)
        ax.set_xticks(range(len(bar_x)))
        ax.set_xticklabels(bar_x, fontsize=7, rotation=0)
        ax.set_ylim(-1, 1)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(metric, fontsize=9)
    fig.suptitle("Layer head-mean AE Spearman rho (per seed/layer)", fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "ERRORATTN_56_04_layer_head_mean_associations.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_05_high_vs_low_layer_head_mean_profiles(layer_high_low_profiles: dict, outdir: Path):
    """Per seed, layer: layer head-mean LOW/HIGH profiles."""
    if not layer_high_low_profiles:
        return
    for seed in SEEDS:
        for layer in range(NUM_LAYERS):
            fig, ax = plt.subplots(figsize=(8, 4))
            for cohort_label, cohort_color in [("LOW", "tab:blue"), ("HIGH", "tab:red")]:
                prof = layer_high_low_profiles.get((seed, layer, cohort_label))
                if prof is None:
                    continue
                ax.plot(LAG_MINUTES, prof, label=f"{cohort_label}_ERROR (L{layer})", color=cohort_color, lw=1.5)
            ax.set_xlabel("Lag (minutes, newest -> oldest)")
            ax.set_ylabel("Attention weight")
            ax.set_title(f"Layer head-mean temporal profile — Seed {seed}, Layer {layer}")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(outdir / f"ERRORATTN_56_05_high_vs_low_layer_profile_seed{seed}_layer{layer}.png", dpi=140, bbox_inches="tight")
            plt.close(fig)


def fig_06_high_minus_low_layer_diff(layer_high_low_profiles: dict, outdir: Path):
    for seed in SEEDS:
        for layer in range(NUM_LAYERS):
            p_low = layer_high_low_profiles.get((seed, layer, "LOW"))
            p_high = layer_high_low_profiles.get((seed, layer, "HIGH"))
            if p_low is None or p_high is None:
                continue
            d = p_high - p_low
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.fill_between(LAG_MINUTES, d, 0, alpha=0.5, color="tab:purple")
            ax.axhline(0, color="gray", lw=0.8)
            ax.plot(LAG_MINUTES, d, color="tab:purple", lw=1.2)
            ax.set_xlabel("Lag (minutes, newest -> oldest)")
            ax.set_ylabel("HIGH - LOW")
            ax.set_title(f"Layer head-mean D_HL — Seed {seed}, Layer {layer}")
            ax.grid(alpha=0.3)
            ymax = max(abs(d.min()), abs(d.max())) * 1.1
            ax.set_ylim(-ymax, ymax)
            fig.tight_layout()
            fig.savefig(outdir / f"ERRORATTN_56_06_high_minus_low_diff_seed{seed}_layer{layer}.png", dpi=140, bbox_inches="tight")
            plt.close(fig)


def fig_07_error_decile_metric_trends(decile_metric_summary: list[dict], outdir: Path):
    """For each seed/layer: x=decile 1..10, y=median attention metric, one line per head."""
    import collections
    grouped = collections.defaultdict(list)
    for r in decile_metric_summary:
        key = (r["seed"], r["layer_idx0"], r["attention_metric"])
        grouped[key].append(r)
    for (seed, layer, metric), rows in grouped.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        pivot = {}
        for r in rows:
            pivot.setdefault(int(r["head_idx0"]), {})[int(r["error_decile"])] = float(r["median"])
        for h in range(NUM_HEADS):
            xs = sorted(pivot.get(h, {}).keys())
            ys = [pivot[h][x] for x in xs]
            ax.plot(xs, ys, marker="o", lw=1.2, label=HEAD_LABELS[h])
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Error decile (1=low, 10=high)")
        ax.set_ylabel(f"Median {metric}")
        ax.set_title(f"Decile trend — Seed {seed}, Layer {layer}, {metric}")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / f"ERRORATTN_56_07_decile_trend_seed{seed}_layer{layer}_{metric}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


def fig_08_error_decile_lag_bin_allocation(layer_decile_profiles: dict, outdir: Path):
    """For each seed/layer: line per decile of layer head-mean profile."""
    import collections
    if not layer_decile_profiles:
        return
    for seed in SEEDS:
        for layer in range(NUM_LAYERS):
            fig, ax = plt.subplots(figsize=(8, 4))
            for d in range(1, 11):
                prof = layer_decile_profiles.get((seed, layer, d))
                if prof is None:
                    continue
                ax.plot(LAG_MINUTES, prof, label=f"D{d}", lw=1.0, alpha=0.8)
            ax.set_xlabel("Lag (minutes, newest -> oldest)")
            ax.set_ylabel("Layer head-mean attention weight")
            ax.set_title(f"Layer head-mean profile per decile — Seed {seed}, Layer {layer}")
            ax.legend(fontsize=7, ncol=5, loc="upper right")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(outdir / f"ERRORATTN_56_08_decile_profile_seed{seed}_layer{layer}.png", dpi=140, bbox_inches="tight")
            plt.close(fig)


def fig_09_under_vs_over_metrics(signed_metric: list[dict], outdir: Path):
    """Per seed/layer: scatter UNDER vs OVER median per metric per head."""
    import collections
    grouped = collections.defaultdict(list)
    for r in signed_metric:
        key = (r["seed"], r["layer_idx0"])
        grouped[key].append(r)
    for (seed, layer), rows in grouped.items():
        n_metrics = len(CORE_ATTENTION_METRICS_V1)
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        for idx, metric in enumerate(CORE_ATTENTION_METRICS_V1):
            ax = axes[idx // 3, idx % 3]
            sub = [r for r in rows if r["attention_metric"] == metric]
            xs = [f"{HEAD_LABELS[int(r['head_idx0'])]}" for r in sub]
            under = [float(r["median_under"]) for r in sub]
            over = [float(r["median_over"]) for r in sub]
            ax.scatter(range(len(xs)), under, color="tab:blue", label="UNDER", s=20)
            ax.scatter(range(len(xs)), over, color="tab:red", label="OVER", s=20)
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels(xs, fontsize=8)
            ax.set_title(metric, fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        fig.suptitle(f"UNDER vs OVER median — Seed {seed}, Layer {layer}", fontsize=11)
        fig.tight_layout()
        fig.savefig(outdir / f"ERRORATTN_56_09_under_vs_over_seed{seed}_layer{layer}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


def fig_10_under_vs_over_layer_profile(layer_under_over_profiles: dict, outdir: Path):
    if not layer_under_over_profiles:
        return
    for seed in SEEDS:
        for layer in range(NUM_LAYERS):
            fig, ax = plt.subplots(figsize=(8, 4))
            for cohort_label, cohort_color in [("UNDER", "tab:blue"), ("OVER", "tab:red")]:
                prof = layer_under_over_profiles.get((seed, layer, cohort_label))
                if prof is None:
                    continue
                ax.plot(LAG_MINUTES, prof, label=f"{cohort_label} (L{layer})", color=cohort_color, lw=1.5)
            ax.set_xlabel("Lag (minutes, newest -> oldest)")
            ax.set_ylabel("Layer head-mean attention weight")
            ax.set_title(f"Layer head-mean UNDER vs OVER — Seed {seed}, Layer {layer}")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(outdir / f"ERRORATTN_56_10_under_vs_over_profile_seed{seed}_layer{layer}.png", dpi=140, bbox_inches="tight")
            plt.close(fig)


def fig_11_shared_low_vs_shared_high_layer_profile(shared_layer_profiles: dict, outdir: Path):
    if not shared_layer_profiles:
        return
    for seed in SEEDS:
        for layer in range(NUM_LAYERS):
            fig, ax = plt.subplots(figsize=(8, 4))
            for cohort_label, cohort_color in [("SHARED_LOW", "tab:blue"), ("SHARED_HIGH", "tab:red")]:
                prof = shared_layer_profiles.get((seed, layer, cohort_label))
                if prof is None:
                    continue
                ax.plot(LAG_MINUTES, prof, label=f"{cohort_label}_ERROR (L{layer})", color=cohort_color, lw=1.5)
            ax.set_xlabel("Lag (minutes, newest -> oldest)")
            ax.set_ylabel("Layer head-mean attention weight")
            ax.set_title(f"Layer head-mean SHARED LOW vs HIGH — Seed {seed}, Layer {layer}")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(outdir / f"ERRORATTN_56_11_shared_low_vs_high_seed{seed}_layer{layer}.png", dpi=140, bbox_inches="tight")
            plt.close(fig)


def fig_12_full_matrix_association(full_matrix_assoc: list[dict], outdir: Path):
    """For each seed/layer: heatmap rows=heads cols=full-matrix-metrics value=rho AE."""
    for seed in SEEDS:
        for layer in range(NUM_LAYERS):
            data = np.zeros((NUM_HEADS, len(["mean_query_entropy", "mean_self_attention_weight",
                                              "mean_absolute_query_source_distance_steps",
                                              "forward_within_input_mass"])))
            for r in full_matrix_assoc:
                if r["seed"] == seed and r["layer_idx0"] == layer and r["conditioning_variable"] == "ABS_ERROR":
                    i = int(r["head_idx0"])
                    j = ["mean_query_entropy", "mean_self_attention_weight",
                         "mean_absolute_query_source_distance_steps",
                         "forward_within_input_mass"].index(r["full_matrix_metric"])
                    data[i, j] = float(r["spearman_rho"])
            fig, ax = plt.subplots(figsize=(6, 3.6))
            im = ax.imshow(data, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
            ax.set_xticks(range(data.shape[1]))
            ax.set_xticklabels(["query_ent", "self_attn", "src_dist", "fwd_win"], fontsize=8, rotation=20, ha="right")
            ax.set_yticks(range(NUM_HEADS))
            ax.set_yticklabels(HEAD_LABELS, fontsize=9)
            for i in range(NUM_HEADS):
                for j in range(data.shape[1]):
                    ax.text(j, i, f"{data[i, j]:+.2f}", ha="center", va="center", fontsize=7)
            ax.set_title(f"Full-matrix AE Spearman rho — Seed {seed}, Layer {layer}", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(outdir / f"ERRORATTN_56_12_full_matrix_seed{seed}_layer{layer}.png", dpi=140, bbox_inches="tight")
            plt.close(fig)


def fig_13_error_cohort_regime_composition(regime_composition: list[dict], outdir: Path):
    """For each (seed, cohort), show top-3 regime labels per family.

    Uses horizontal bars per cohort showing top regime labels per family
    (one subplot per cohort, three subplots side-by-side per seed)."""
    if not regime_composition:
        return
    import collections
    grouped = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in regime_composition:
        grouped[(r["seed"], r["error_cohort"])][r["regime_family"]][r["regime_label"]] = float(r["cohort_share"])
    families = ["TARGET_LEVEL", "EXTREME_HIGH", "CHANGE_MAGNITUDE", "CHANGE_DIRECTION", "TIME_OF_DAY", "DAY_TYPE"]
    for seed in SEEDS:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for idx, cohort in enumerate(["LOW_ERROR", "MID_ERROR", "HIGH_ERROR"]):
            ax = axes[idx]
            data = grouped.get((seed, cohort), {})
            label_shares = []
            for fam in families:
                if fam in data:
                    for label, sh in data[fam].items():
                        label_shares.append((f"{fam}={label}", sh))
            label_shares.sort(key=lambda x: -x[1])
            top = label_shares[:8]  
            if top:
                labels = [t[0] for t in top][::-1]
                shares = [t[1] for t in top][::-1]
                ax.barh(labels, shares, color=["tab:blue", "tab:orange", "tab:green", "tab:red",
                                                "tab:purple", "tab:brown", "tab:pink", "tab:gray"][:len(top)],
                        alpha=0.8)
                ax.set_xlabel("Cohort share")
            ax.set_title(f"{cohort} (Seed {seed})", fontsize=10)
            ax.grid(axis="x", alpha=0.3)
        fig.suptitle(f"Error cohort regime composition — Seed {seed}", fontsize=11)
        fig.tight_layout()
        fig.savefig(outdir / f"ERRORATTN_56_13_regime_composition_seed{seed}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


def fig_14_worst_case_context(worst_case_context: list[dict], outdir: Path):
    """Bar of normalized_entropy for shared ranks 1-5 per seed."""
    import collections
    if not worst_case_context:
        return
    grouped = collections.defaultdict(list)
    for r in worst_case_context:
        grouped[(r["seed"], r["shared_rank"], r["layer_idx0"])] = r
    rows_to_plot = []
    for (seed, rank, layer), r in grouped.items():
        rows_to_plot.append(r)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, seed in enumerate(SEEDS):
        ax = axes[idx]
        ranks = sorted(set(r["shared_rank"] for r in rows_to_plot if r["seed"] == seed))
        for layer in range(NUM_LAYERS):
            ys = []
            for rk in ranks:
                vals = [r["normalized_entropy"] for r in rows_to_plot if r["seed"] == seed and r["shared_rank"] == rk and r["layer_idx0"] == layer]
                ys.append(np.mean(vals) if vals else 0.0)
            ax.plot(ranks, ys, marker="o", lw=1.5, label=f"Layer {layer}")
        ax.set_xticks(ranks)
        ax.set_xlabel("Shared W2 rank")
        ax.set_ylabel("Layer head-mean normalized_entropy")
        ax.set_title(f"Seed {seed}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Shared worst cases — layer head-mean normalized_entropy", fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "ERRORATTN_56_14_worst_case_context.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
