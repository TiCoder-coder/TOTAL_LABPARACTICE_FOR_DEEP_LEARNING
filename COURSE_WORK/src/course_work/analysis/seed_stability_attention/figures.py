"""Phase 57 - canonical figure generation (14 figure families)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics_utils import jsd_natural_log, wasserstein_minutes
from .sources import LAG_MINUTES, NUM_LAYERS, NUM_HEADS, SEEDS


# Common styling
PALETTE = {"42": "#3b46c4", "123": "#16a34a", "2026": "#dc2626"}
SEED_LABELS = {42: "seed42", 123: "seed123", 2026: "seed2026"}


def _save(fig, fig_dir: Path, name: str) -> Path:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fp = fig_dir / name
    fig.savefig(fp, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fp


# ===========================================================================
# SEEDATTN_57_01 — Layer head-mean mean profile per seed (per layer)
# ===========================================================================

def fig_01_layer_head_mean_profiles(
    fig_dir: Path,
    mean_profiles: Dict[tuple, np.ndarray],
) -> List[Path]:
    out: List[Path] = []
    for layer in range(NUM_LAYERS):
        fig, ax = plt.subplots(figsize=(8, 4.6))
        for s in SEEDS:
            arr = mean_profiles.get(("layer_seed", s, layer))
            if arr is None:
                # Try building from (seed, layer) -> mean profile
                arr = mean_profiles.get((s, layer))
            if arr is None:
                continue
            ax.plot(LAG_MINUTES, arr, label=SEED_LABELS[s], color=PALETTE[str(s)], linewidth=1.6)
        ax.set_xlabel("Lag (minutes, newest -> oldest)")
        ax.set_ylabel("Mean layer head-mean attention")
        ax.set_title(f"Layer {layer} head-mean mean profile across seeds")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        out.append(_save(fig, fig_dir, f"SEEDATTN_57_01_layer{layer}_layer_head_mean_profiles.png"))
    return out


# ===========================================================================
# SEEDATTN_57_02 — Layer pairwise stability metrics (JSD/Wasserstein/Cosine)
# ===========================================================================

def fig_02_layer_pairwise_stability(
    fig_dir: Path,
    stability_rows: List[dict],
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    pair_labels = ["42-123", "42-2026", "123-2026"]
    jsd_by_layer = {0: [], 1: []}
    wass_by_layer = {0: [], 1: []}
    cos_by_layer = {0: [], 1: []}
    for r in stability_rows:
        idx = pair_labels.index(f"{r['seed_a']}-{r['seed_b']}")
        jsd_by_layer[int(r["layer_idx0"])].append((idx, float(r["jsd"])))
        wass_by_layer[int(r["layer_idx0"])].append((idx, float(r["wasserstein_minutes"])))
        cos_by_layer[int(r["layer_idx0"])].append((idx, float(r["cosine"])))
    for panel_idx, (data_by_layer, title, ylabel) in enumerate([
        (jsd_by_layer, "Pairwise JSD (lower = more similar)", "JSD (nat log)"),
        (wass_by_layer, "Pairwise Wasserstein (minutes)", "W1 (min)"),
        (cos_by_layer, "Pairwise cosine similarity", "cosine"),
    ]):
        ax = axes[panel_idx]
        width = 0.35
        x = np.arange(len(pair_labels))
        for li, layer in enumerate([0, 1]):
            vals = [v for _, v in sorted(data_by_layer[layer], key=lambda x: x[0])]
            ax.bar(x + (li - 0.5) * width, vals, width, label=f"Layer {layer}", color="#3b46c4" if li == 0 else "#16a34a", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(pair_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Layer pairwise stability across seed pairs", fontsize=12)
    return _save(fig, fig_dir, "SEEDATTN_57_02_layer_pairwise_stability.png")


# ===========================================================================
# SEEDATTN_57_03 — Per-target layer stability (ECDF of JSD)
# ===========================================================================

def fig_03_per_target_layer_stability(
    fig_dir: Path,
    per_target_rows: List[dict],
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {0: "#3b46c4", 1: "#16a34a"}
    for layer in [0, 1]:
        sel = [float(r["jsd"]) for r in per_target_rows if int(r["layer_idx0"]) == layer]
        if not sel:
            continue
        arr = np.sort(np.asarray(sel, dtype=np.float64))
        y = np.linspace(0, 1, len(arr))
        ax.plot(arr, y, label=f"Layer {layer}", color=colors[layer], linewidth=1.5)
    ax.set_xlabel("Per-target layer head-mean JSD (all 3 seed pairs)")
    ax.set_ylabel("ECDF")
    ax.set_title("Per-target layer head-mean attention disagreement (ECDF)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, fig_dir, "SEEDATTN_57_03_layer_per_target_stability.png")


# ===========================================================================
# SEEDATTN_57_04 — Head matching cost matrices (JSD, per layer × seed-pair)
# ===========================================================================

def fig_04_head_matching_cost_matrices(
    fig_dir: Path,
    matching_results: Dict[tuple, dict],
) -> List[Path]:
    out: List[Path] = []
    pair_labels = ["42-123", "42-2026", "123-2026"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for li, layer in enumerate([0, 1]):
        for pi, (sa, sb) in enumerate([(42, 123), (42, 2026), (123, 2026)]):
            mr = matching_results.get((layer, sa, sb))
            ax = axes[li, pi]
            if mr is None:
                ax.set_title(f"L{layer} {pair_labels[pi]} (no data)")
                continue
            mat = mr["cost_matrix_jsd"]
            H = mat.shape[0]
            im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=float(np.log(2.0)), aspect="equal")
            ax.set_xticks(range(H))
            ax.set_yticks(range(H))
            ax.set_xticklabels([f"H{h+1}" for h in range(H)], fontsize=8)
            ax.set_yticklabels([f"H{h+1}" for h in range(H)], fontsize=8)
            # Overlay canonical assignment
            for ha, hb in mr["canonical_assignment"].items():
                ax.scatter([hb], [ha], s=160, facecolors="none", edgecolors="red", linewidths=2)
            ax.set_title(f"L{layer} {pair_labels[pi]} (JSD)", fontsize=10)
            ax.set_xlabel("target seed head")
            ax.set_ylabel("source seed head")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Head matching cost matrices (JSD nat-log, [0, ln(2)])", fontsize=12)
    out.append(_save(fig, fig_dir, "SEEDATTN_57_04_head_matching_cost_matrices.png"))
    return out


# ===========================================================================
# SEEDATTN_57_05 — Canonical head mapping diagram
# ===========================================================================

def fig_05_canonical_head_mapping(
    fig_dir: Path,
    canonical_groups: List[dict],
) -> Path:
    fig, axes = plt.subplots(1, NUM_LAYERS, figsize=(12, 5))
    if NUM_LAYERS == 1:
        axes = [axes]
    for li, layer in enumerate([0, 1]):
        ax = axes[li]
        groups_layer = [g for g in canonical_groups if int(g["layer_idx0"]) == layer]
        n = len(groups_layer)
        # Three columns: seed42, seed123, seed2026
        seed42_heads = [g["seed42_head_idx0"] for g in groups_layer]
        seed123_heads = [g["seed123_head_idx0"] for g in groups_layer]
        seed2026_heads = [g["seed2026_head_idx0"] for g in groups_layer]
        # Vertical positions: 1..n
        for g in groups_layer:
            row = int(g["canonical_group"]) - 1
            ax.scatter([0], [row], s=160, color="#3b46c4")
            ax.scatter([1], [row], s=160, color="#16a34a")
            ax.scatter([2], [row], s=160, color="#dc2626")
            ax.text(0, row, f"H{g['seed42_head_idx0']+1}", ha="center", va="center", color="white", fontsize=8)
            ax.text(1, row, f"H{g['seed123_head_idx0']+1}", ha="center", va="center", color="white", fontsize=8)
            ax.text(2, row, f"H{g['seed2026_head_idx0']+1}", ha="center", va="center", color="white", fontsize=8)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["seed42", "seed123", "seed2026"])
        ax.set_yticks(range(n))
        ax.set_yticklabels([f"G{g+1}" for g in range(n)])
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, n - 0.5)
        ax.set_title(f"Layer {layer} canonical mapping")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Canonical head mapping (seed42 anchor)", fontsize=12)
    return _save(fig, fig_dir, "SEEDATTN_57_05_canonical_head_mapping.png")


# ===========================================================================
# SEEDATTN_57_06 — Matching sensitivity agreement (JSD vs Wasserstein)
# ===========================================================================

def fig_06_matching_sensitivity_agreement(
    fig_dir: Path,
    wasserstein_rows: List[dict],
) -> Path:
    pair_labels = ["42-123", "42-2026", "123-2026"]
    fractions = []
    labels = []
    for layer in [0, 1]:
        for sa, sb in [(42, 123), (42, 2026), (123, 2026)]:
            sel = [r for r in wasserstein_rows if int(r["layer_idx0"]) == layer and int(r["seed_a"]) == sa and int(r["seed_b"]) == sb]
            if not sel:
                continue
            agree = sum(1 for r in sel if r["pair_agrees"])
            fractions.append(agree / max(len(sel), 1))
            labels.append(f"L{layer} {pair_labels[pair_labels.index(f'{sa}-{sb}')]}")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(labels))
    ax.bar(x, fractions, color="#3b46c4", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("JSD vs Wasserstein pair-agreement fraction")
    ax.set_title("Matching sensitivity: JSD vs Wasserstein one-to-one mapping")
    ax.grid(True, alpha=0.3, axis="y")
    return _save(fig, fig_dir, "SEEDATTN_57_06_matching_sensitivity_agreement.png")


# ===========================================================================
# SEEDATTN_57_07 — Matched-head mean profiles (overlay)
# ===========================================================================

def fig_07_matched_head_mean_profiles(
    fig_dir: Path,
    canonical_groups: List[dict],
    sources,
) -> List[Path]:
    out: List[Path] = []
    arrs = {s: np.load(sources.raw_last_query_files[s])["last_query_attention"] for s in SEEDS}
    for layer in [0, 1]:
        fig, axes = plt.subplots(1, NUM_HEADS, figsize=(16, 3.6), sharey=True)
        for gi, g in enumerate([gd for gd in canonical_groups if int(gd["layer_idx0"]) == layer]):
            ax = axes[gi]
            h42 = g["seed42_head_idx0"]
            h123 = g["seed123_head_idx0"]
            h2026 = g["seed2026_head_idx0"]
            p42 = arrs[42][:, layer, h42, :].mean(axis=0)
            p123 = arrs[123][:, layer, h123, :].mean(axis=0)
            p2026 = arrs[2026][:, layer, h2026, :].mean(axis=0)
            for s, p, color in (
                (42, p42, "#3b46c4"),
                (123, p123, "#16a34a"),
                (2026, p2026, "#dc2626"),
            ):
                ax.plot(LAG_MINUTES, p, label=f"seed{s}", color=color, linewidth=1.4)
            ax.set_title(f"Layer {layer} Group {g['canonical_group']}", fontsize=9)
            ax.set_xlabel("Lag (min, newest -> oldest)")
            if gi == 0:
                ax.set_ylabel("mean profile weight")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        fig.suptitle(f"Matched-head mean profiles (Layer {layer}, canonical groups)", fontsize=11)
        out.append(_save(fig, fig_dir, f"SEEDATTN_57_07_layer{layer}_matched_head_mean_profiles.png"))
    return out


# ===========================================================================
# SEEDATTN_57_08 — Matched-head pairwise distances
# ===========================================================================

def fig_08_matched_head_pairwise_distances(
    fig_dir: Path,
    matched_profile_rows: List[dict],
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    layer_group_keys = []
    jsd_vals = []
    wass_vals = []
    for r in matched_profile_rows:
        layer_group_keys.append(f"L{r['layer_idx0']}-G{r['canonical_group']}-{r['seed_a']}-{r['seed_b']}")
        jsd_vals.append(float(r["jsd"]))
        wass_vals.append(float(r["wasserstein_minutes"]))
    x = np.arange(len(layer_group_keys))
    ax.bar(x - 0.2, jsd_vals, width=0.4, label="JSD", color="#3b46c4", alpha=0.85)
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, wass_vals, width=0.4, label="Wasserstein (min)", color="#dc2626", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_group_keys, rotation=80, fontsize=6)
    ax.set_ylabel("JSD")
    ax2.set_ylabel("Wasserstein (min)")
    ax.set_title("Matched-head pairwise JSD / Wasserstein across all groups")
    ax.grid(True, alpha=0.3, axis="y")
    return _save(fig, fig_dir, "SEEDATTN_57_08_matched_head_pairwise_distances.png")


# ===========================================================================
# SEEDATTN_57_09 — Matched-head per-target stability (ECDF)
# ===========================================================================

def fig_09_matched_head_per_target_stability(
    fig_dir: Path,
    matched_per_target_rows: List[dict],
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    by_layer = {0: [], 1: []}
    for r in matched_per_target_rows:
        by_layer[int(r["layer_idx0"])].append(float(r["jsd"]))
    for layer in [0, 1]:
        if not by_layer[layer]:
            continue
        arr = np.sort(np.asarray(by_layer[layer], dtype=np.float64))
        y = np.linspace(0, 1, len(arr))
        ax.plot(arr, y, label=f"Layer {layer}", linewidth=1.4)
    ax.set_xlabel("Per-target matched-head JSD (all groups, all seed pairs)")
    ax.set_ylabel("ECDF")
    ax.set_title("Per-target matched-head attention disagreement (ECDF)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, fig_dir, "SEEDATTN_57_09_matched_head_per_target_stability.png")


# ===========================================================================
# SEEDATTN_57_10 — Dense-case full-map stability
# ===========================================================================

def fig_10_dense_case_full_map_stability(
    fig_dir: Path,
    dense_rows: List[dict],
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    if not dense_rows:
        ax.set_title("No dense-case data")
        return _save(fig, fig_dir, "SEEDATTN_57_10_dense_case_full_map_stability.png")
    arr = np.asarray([float(r["mean_query_jsd"]) for r in dense_rows], dtype=np.float64)
    arr = np.sort(arr)
    y = np.linspace(0, 1, arr.size)
    ax.plot(arr, y, color="#3b46c4", linewidth=1.6)
    ax.set_xlabel("Per-case mean query JSD (all layers/groups/pairs)")
    ax.set_ylabel("ECDF")
    ax.set_title("Dense-case full-map attention stability (ECDF)")
    ax.grid(True, alpha=0.3)
    return _save(fig, fig_dir, "SEEDATTN_57_10_dense_case_full_map_stability.png")


# ===========================================================================
# SEEDATTN_57_11 — Error-conditioned layer stability
# ===========================================================================

def fig_11_error_conditioned_layer_stability(
    fig_dir: Path,
    error_layer_rows: List[dict],
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = {"SPEARMAN": 0, "HIGH_LOW_CLIFFS_DELTA": 0, "HIGH_LOW_MEDIAN_DELTA": 0,
              "SHARED_HIGH_LOW_CLIFFS_DELTA": 0, "PROFILE_JSD": 0, "PROFILE_WASSERSTEIN": 0}
    total = {k: 0 for k in counts}
    for r in error_layer_rows:
        t = str(r.get("analysis_type", ""))
        if t not in counts:
            continue
        total[t] += 1
        if r.get("all_defined_same_sign"):
            counts[t] += 1
    types = list(counts.keys())
    fractions = [counts[t] / max(total[t], 1) for t in types]
    x = np.arange(len(types))
    ax.bar(x, fractions, color="#3b46c4", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=20, fontsize=8, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("All-defined-same-sign fraction")
    ax.set_title("Layer error-conditioned sign agreement across seeds (Phase 56 effects)")
    ax.grid(True, alpha=0.3, axis="y")
    return _save(fig, fig_dir, "SEEDATTN_57_11_error_conditioned_layer_stability.png")


# ===========================================================================
# SEEDATTN_57_12 — Shared-cohort layer stability
# ===========================================================================

def fig_12_shared_cohort_layer_stability(
    fig_dir: Path,
    error_layer_rows: List[dict],
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    sh_rows = [r for r in error_layer_rows if str(r.get("analysis_type")) == "SHARED_HIGH_LOW_CLIFFS_DELTA"]
    if not sh_rows:
        ax.set_title("No shared-cohort rows")
        return _save(fig, fig_dir, "SEEDATTN_57_12_shared_cohort_layer_stability.png")
    # Bar chart of seed42/123/2026 values for SHARED_HIGH_LOW_CLIFFS_DELTA per layer
    layers = sorted({int(r["layer_idx0"]) for r in sh_rows})
    metrics = sorted({str(r["attention_metric"]) for r in sh_rows})
    bar_w = 0.25
    x = np.arange(len(metrics))
    for li, layer in enumerate(layers):
        vals = []
        for metric in metrics:
            sel = [r for r in sh_rows if int(r["layer_idx0"]) == layer and str(r["attention_metric"]) == metric]
            if sel:
                vals.append(float(sel[0].get("seed42_value", 0.0)))
            else:
                vals.append(0.0)
        ax.bar(x + (li - 1) * bar_w, vals, bar_w, label=f"Layer {layer} seed42", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Cliff's delta (SHARED_HIGH vs SHARED_LOW) — seed42 reference")
    ax.set_title("Shared-cohort layer stability (seed42 reference)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    return _save(fig, fig_dir, "SEEDATTN_57_12_shared_cohort_layer_stability.png")


# ===========================================================================
# SEEDATTN_57_13 — Prediction spread vs attention disagreement
# ===========================================================================

def fig_13_prediction_spread_vs_attention_disagreement(
    fig_dir: Path,
    pred_assoc_rows: List[dict],
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    if not pred_assoc_rows:
        ax.set_title("No prediction-attention rows")
        return _save(fig, fig_dir, "SEEDATTN_57_13_prediction_spread_vs_attention_disagreement.png")
    labels = [f"L{r['layer_idx0']}-{r['prediction_spread_metric']}-{r['attention_disagreement_metric']}"
              for r in pred_assoc_rows]
    rhos = [float(r["spearman_rho"]) for r in pred_assoc_rows]
    x = np.arange(len(labels))
    ax.bar(x, rhos, color="#3b46c4", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=80, fontsize=7)
    ax.set_ylabel("Spearman rho (prediction spread vs layer attention disagreement)")
    ax.set_title("Prediction spread vs attention disagreement (secondary, descriptive)")
    ax.grid(True, alpha=0.3, axis="y")
    return _save(fig, fig_dir, "SEEDATTN_57_13_prediction_spread_vs_attention_disagreement.png")


# ===========================================================================
# SEEDATTN_57_14 — Stability evidence summary (no composite score)
# ===========================================================================

def fig_14_stability_evidence_summary(
    fig_dir: Path,
    evidence_rows: List[dict],
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    rows_text = []
    for r in evidence_rows:
        rows_text.append(f"{r['evidence_row']:35s} | scope={r['scope'][:35]:35s}")
    ax.text(0.01, 0.99, "\n".join(rows_text), va="top", ha="left", fontsize=8, family="monospace")
    ax.set_axis_off()
    ax.set_title("Phase 57 stability evidence summary (no composite score)", fontsize=11)
    return _save(fig, fig_dir, "SEEDATTN_57_14_stability_evidence_summary.png")
