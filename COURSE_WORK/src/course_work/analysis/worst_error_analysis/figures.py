"""Phase 51-G — deterministic, descriptive figure generation.

All figures are derived from frozen Phase 51-B/C/D/E/F artifacts.
No new inference, no ranking changes, no cherry-picked cases.
Every figure has a deterministic filename (no timestamps) and is
registered in the figure manifest.

Figure inventory (per plan §158 + descriptive augmentations):

  WORST_51_01_top20_abs_error_seed42.png
  WORST_51_02_top20_abs_error_seed123.png
  WORST_51_03_top20_abs_error_seed2026.png
  WORST_51_04_shared_top20_hardness.png
  WORST_51_05_worst_underprediction.png
  WORST_51_06_worst_overprediction.png
  WORST_51_07_seed_top20_overlap.png
  WORST_51_08_hardness_vs_seed_disagreement.png
  WORST_51_09_shared_worst_regime_composition.png
  WORST_51_10_sample_share_vs_worst_case_share.png
  WORST_51_11_error_concentration_sse.png
  WORST_51_12_baseline_context_shared_worst.png
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from course_work.utils.artifacts import atomic_write_bytes, canonical_json_bytes, get_project_root


PHASE51_DIR_REL = "artifacts/worst_error_analysis"
FIGURES_DIR_REL = "artifacts/worst_error_analysis/figures"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load_csv(p: Path) -> list[dict[str, str]]:
    with p.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _save_figure(
    fig: plt.Figure,
    out_dir: Path,
    name: str,
    title: str,
    source: str,
    selection_rule: str,
    descriptive_only: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / name
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(fp, dpi=120, bbox_inches="tight")
    plt.close(fig)
    try:
        os.chmod(fp, 0o444)
    except (OSError, PermissionError):
        pass
    return {
        "filename": name,
        "title": title,
        "source": source,
        "selection_rule": selection_rule,
        "descriptive_only": descriptive_only,
        "sha256": _sha(fp),
        "size_bytes": fp.stat().st_size,
    }


def build_top20_abs_error_per_seed(
    root: Path,
    out_dir: Path,
) -> list[dict[str, Any]]:
    fp = root / PHASE51_DIR_REL / "worst_per_seed_top20.csv"
    rows = _load_csv(fp)
    by_seed: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append(r)
    metas: list[dict[str, Any]] = []
    for seed, srows in by_seed.items():
        srows.sort(key=lambda r: int(r["rank"]))
        ranks = [int(r["rank"]) for r in srows]
        aes = [float(r["absolute_error_wh"]) for r in srows]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(ranks, aes, color="#3b6fb6")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Absolute Error (Wh)")
        ax.set_title(f"W1 Top20 Absolute Error — seed={seed}")
        ax.set_xticks(ranks)
        metas.append(_save_figure(
            fig, out_dir,
            f"WORST_51_0{ {42:1, 123:2, 2026:3}[int(seed)] }_top20_abs_error_seed{seed}.png",
            f"Phase 51 W1 — seed {seed}",
            "artifacts/worst_error_analysis/worst_per_seed_top20.csv",
            "rank position 1..20 from frozen Phase 51-C selection",
        ))
    return metas


def build_shared_top20_hardness(
    root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    fp = root / PHASE51_DIR_REL / "worst_shared_top20.csv"
    rows = sorted(_load_csv(fp), key=lambda r: int(r["rank"]))
    ranks = [int(r["rank"]) for r in rows]
    mean_ae = [float(r["mean_abs_error_wh"]) for r in rows]
    seed42_ae = [float(r["seed42_abs_error_wh"]) for r in rows]
    seed123_ae = [float(r["seed123_abs_error_wh"]) for r in rows]
    seed2026_ae = [float(r["seed2026_abs_error_wh"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ranks, mean_ae, "k-", label="Mean across seeds", linewidth=2)
    ax.scatter(ranks, seed42_ae, color="#3b6fb6", label="seed42", marker="o")
    ax.scatter(ranks, seed123_ae, color="#d97757", label="seed123", marker="s")
    ax.scatter(ranks, seed2026_ae, color="#3aa66f", label="seed2026", marker="^")
    ax.set_xlabel("Shared rank")
    ax.set_ylabel("Absolute Error (Wh)")
    ax.set_title("W2 Shared Top20 Hardness")
    ax.set_xticks(ranks)
    ax.legend()
    return _save_figure(
        fig, out_dir,
        "WORST_51_04_shared_top20_hardness.png",
        "Phase 51 W2 — Shared Top20",
        "artifacts/worst_error_analysis/worst_shared_top20.csv",
        "rank position 1..20 from frozen Phase 51-C shared selection",
    )


def build_signed_top10(
    root: Path,
    out_dir: Path,
) -> list[dict[str, Any]]:
    metas: list[dict[str, Any]] = []
    for fam, fname, figid, ylabel, color in [
        ("W3_UNDERPREDICTION_WORST", "worst_underprediction_top10.csv", 5, "Underprediction (residual_wh)", "#d97757"),
        ("W4_OVERPREDICTION_WORST", "worst_overprediction_top10.csv", 6, "Overprediction magnitude (Wh)", "#3b6fb6"),
    ]:
        rows = _load_csv(root / PHASE51_DIR_REL / fname)
        # Per seed, build a small panel.
        by_seed: dict[str, list[dict[str, str]]] = {}
        for r in rows:
            by_seed.setdefault(r["seed"], []).append(r)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for ax, (seed, srows) in zip(axes, sorted(by_seed.items())):
            srows.sort(key=lambda r: int(r["rank"]))
            ranks = [int(r["rank"]) for r in srows]
            vals = [
                float(r["absolute_error_wh"]) for r in srows
            ]
            ax.bar(ranks, vals, color=color)
            ax.set_title(f"seed={seed}")
            ax.set_xlabel("signed_rank")
            ax.set_xticks(ranks)
            ax.set_ylabel(ylabel)
        fig.suptitle(f"Phase 51 {fam}")
        metas.append(_save_figure(
            fig, out_dir,
            f"WORST_51_0{figid}_{fam.lower().split('_')[0]}prediction.png",
            f"Phase 51 {fam}",
            f"artifacts/worst_error_analysis/{fname}",
            "signed_rank 1..10 from frozen Phase 51-C signed selection",
        ))
    return metas


def build_seed_top20_overlap(
    root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    fp = root / PHASE51_DIR_REL / "seed_overlap_table.csv"
    rows = _load_csv(fp)
    pair_rows = [r for r in rows if r["section"] == "W1_PAIRWISE"]
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [f"{r['set_a']} vs {r['set_b']}" for r in pair_rows]
    jaccards = [float(r["jaccard"]) for r in pair_rows]
    ax.barh(labels, jaccards, color="#3b6fb6")
    ax.set_xlabel("Jaccard similarity")
    ax.set_title("W1 Top20 Pairwise Overlap")
    return _save_figure(
        fig, out_dir,
        "WORST_51_07_seed_top20_overlap.png",
        "Phase 51 W5 — Cross-seed Top20 overlap",
        "artifacts/worst_error_analysis/seed_overlap_table.csv",
        "frozen Phase 51-D overlap computation (W1_PAIRWISE section)",
    )


def build_hardness_vs_seed_spread(
    root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    fp = root / PHASE51_DIR_REL / "hardness_vs_seed_disagreement.csv"
    rows = _load_csv(fp)
    xs = [float(r["mean_abs_error_wh"]) for r in rows]
    ys = [float(r["seed_range_prediction"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, s=6, alpha=0.5, color="#3b6fb6")
    ax.set_xlabel("Mean |e| across seeds (Wh)")
    ax.set_ylabel("seed_range_prediction (Wh)")
    ax.set_title("Hardness vs Phase48 Seed Disagreement (all Test, N=2961)")
    return _save_figure(
        fig, out_dir,
        "WORST_51_08_hardness_vs_seed_disagreement.png",
        "Phase 51 W6 — Hardness vs Seed Spread",
        "artifacts/worst_error_analysis/hardness_vs_seed_disagreement.csv",
        "full Test population descriptive scatter (no threshold)",
    )


def build_shared_worst_regime_composition(
    root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    fp = root / PHASE51_DIR_REL / "regime_overrepresentation.csv"
    rows = [r for r in _load_csv(fp) if r["regime_family"] == "R2_EXTREME_HIGH"]
    # Each row is (family, seed, k=20, label). Keep seed42 as default view.
    rows_s42 = [r for r in rows if r["seed"] == "42"]
    if not rows_s42:
        # fall back: aggregate across seeds (mean)
        rows_s42 = rows
    labels = [r["regime_label"] for r in rows_s42]
    selected_share = [float(r["selected_prevalence"]) for r in rows_s42]
    global_share = [float(r["global_prevalence"]) for r in rows_s42]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    width = 0.4
    ax.bar(x - width / 2, selected_share, width, label="W2 Shared Top20", color="#3b6fb6")
    ax.bar(x + width / 2, global_share, width, label="Full Test", color="#d97757")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Share")
    ax.set_title("R2_EXTREME_HIGH composition: W2 vs Test population")
    ax.legend()
    return _save_figure(
        fig, out_dir,
        "WORST_51_09_shared_worst_regime_composition.png",
        "Phase 51 W2 regime composition (R2)",
        "artifacts/worst_error_analysis/regime_overrepresentation.csv",
        "frozen Phase 50 regime labels reused exactly",
    )


def build_sample_share_vs_worst_case_share(
    root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    fp = root / PHASE51_DIR_REL / "error_concentration_table.csv"
    rows = _load_csv(fp)
    by_seed: dict[str, dict[str, float]] = {}
    n_pop = int(rows[0]["n_population"]) if rows else 2961
    for r in rows:
        if r["k"] == "20":
            sample_share = 20 / n_pop
            by_seed[r["seed"]] = {
                "sample_share": sample_share,
                "SAE_share": float(r["sae_share"]),
                "SSE_share": float(r["sse_share"]),
            }
    seeds = sorted(by_seed.keys())
    sample_share = [by_seed[s]["sample_share"] for s in seeds]
    sae_share = [by_seed[s]["SAE_share"] for s in seeds]
    sse_share = [by_seed[s]["SSE_share"] for s in seeds]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(seeds))
    width = 0.25
    ax.bar(x - width, sample_share, width, label="Sample share", color="#3aa66f")
    ax.bar(x, sae_share, width, label="SAE share", color="#3b6fb6")
    ax.bar(x + width, sse_share, width, label="SSE share", color="#d97757")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed={s}" for s in seeds])
    ax.set_ylabel("Share of Test")
    ax.set_title("Top20 share of Test: sample vs SAE vs SSE")
    ax.legend()
    return _save_figure(
        fig, out_dir,
        "WORST_51_10_sample_share_vs_worst_case_share.png",
        "Phase 51 — Top20 sample vs error share",
        "artifacts/worst_error_analysis/error_concentration_table.csv",
        "frozen Phase 51-D concentration rows",
    )


def build_error_concentration_sse(
    root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    fp = root / PHASE51_DIR_REL / "error_concentration_table.csv"
    rows = _load_csv(fp)
    by_seed: dict[str, list[dict[str, float]]] = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append({
            "k": int(r["k"]),
            "sample_share": float(r["k"]) / int(r["n_population"]),
            "sae_share": float(r["sae_share"]),
            "sse_share": float(r["sse_share"]),
        })
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"42": "#3b6fb6", "123": "#d97757", "2026": "#3aa66f"}
    for seed, srows in sorted(by_seed.items()):
        srows.sort(key=lambda x: x["sample_share"])
        xs = [r["sample_share"] for r in srows]
        ys = [r["sse_share"] for r in srows]
        ax.plot(xs, ys, "o-", color=colors.get(seed, "#333"), label=f"seed {seed}")
    ax.set_xlabel("Sample fraction (descending AE)")
    ax.set_ylabel("Cumulative SSE share")
    ax.set_title("Error concentration (SSE) — k=1..20")
    ax.legend()
    return _save_figure(
        fig, out_dir,
        "WORST_51_11_error_concentration_sse.png",
        "Phase 51 — Error concentration",
        "artifacts/worst_error_analysis/error_concentration_table.csv",
        "k ∈ {1, 2, 5, 10, 20} descriptive curve",
    )


def build_baseline_context_shared_worst(
    root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    fp = root / PHASE51_DIR_REL / "baseline_context.csv"
    rows = _load_csv(fp)
    # Use (selection_family + seed + rank + target_id) as a unique key.
    by_key: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        key = f"{r['selection_family']}|{r['seed']}|r{r['rank']}|{r['target_id']}"
        by_key.setdefault(key, {"t": [], "p": []})
        try:
            by_key[key]["t"].append(float(r["transformer_abs_error_wh"]))
            by_key[key]["p"].append(float(r["persistence_absolute_error_wh"]))
        except (ValueError, KeyError):
            continue
    keys = sorted(by_key.keys())
    t_means = [
        float(np.mean(by_key[k]["t"])) if by_key[k]["t"] else 0.0 for k in keys
    ]
    p_vals = [
        float(np.mean(by_key[k]["p"])) if by_key[k]["p"] else 0.0 for k in keys
    ]
    if not keys:
        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.text(0.5, 0.5, "no baseline context", ha="center", va="center", color="#888")
        ax.set_axis_off()
        return _save_figure(
            fig, out_dir,
            "WORST_51_12_baseline_context_shared_worst.png",
            "Phase 51 — Baseline context (no data)",
            "artifacts/worst_error_analysis/baseline_context.csv",
            "frozen Phase 51-E baseline context",
        )
    # Sample up to 60 keys to keep readable
    if len(keys) > 60:
        keys = keys[:60]
        t_means = t_means[:60]
        p_vals = p_vals[:60]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(keys))
    width = 0.45
    ax.bar(x - width / 2, t_means, width, label="Transformer AE", color="#3b6fb6")
    ax.bar(x + width / 2, p_vals, width, label="Persistence AE", color="#d97757")
    ax.set_xticks(x)
    short = [k.split("|")[0] + "|" + k.split("|")[2] for k in keys]
    ax.set_xticklabels(short, rotation=70, fontsize=5)
    ax.set_ylabel("Absolute Error (Wh)")
    ax.set_title("Selection-conditioned baseline comparison (Phase 51 cases)")
    ax.legend()
    return _save_figure(
        fig, out_dir,
        "WORST_51_12_baseline_context_shared_worst.png",
        "Phase 51 — Baseline context",
        "artifacts/worst_error_analysis/baseline_context.csv",
        "frozen Phase 51-E baseline context (case-level)",
    )


def build_figure_manifest(
    figures: list[dict[str, Any]],
    root: Path,
) -> str:
    out = root / PHASE51_DIR_REL / "figure_manifest.json"
    payload = {
        "phase": 51,
        "subphase": "51-G",
        "version": "PHASE51_FIGURE_MANIFEST-v1",
        "n_figures": len(figures),
        "descriptive_only": True,
        "no_causal_claims": True,
        "figures": figures,
    }
    content = canonical_json_bytes(payload)
    atomic_write_bytes(out, content)
    os.chmod(out, 0o444)
    return _sha(out)


def generate_all_figures(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root if project_root is not None else get_project_root()
    out_dir = root / FIGURES_DIR_REL
    if out_dir.exists():
        # Re-runnable: clear any previous figures deterministically.
        import shutil
        try:
            shutil.rmtree(out_dir)
        except (OSError, PermissionError):
            pass
    out_dir.mkdir(parents=True, exist_ok=True)
    figures: list[dict[str, Any]] = []
    figures += build_top20_abs_error_per_seed(root, out_dir)
    figures.append(build_shared_top20_hardness(root, out_dir))
    figures += build_signed_top10(root, out_dir)
    figures.append(build_seed_top20_overlap(root, out_dir))
    figures.append(build_hardness_vs_seed_spread(root, out_dir))
    figures.append(build_shared_worst_regime_composition(root, out_dir))
    figures.append(build_sample_share_vs_worst_case_share(root, out_dir))
    figures.append(build_error_concentration_sse(root, out_dir))
    figures.append(build_baseline_context_shared_worst(root, out_dir))
    manifest_sha = build_figure_manifest(figures, root)
    return {
        "n_figures": len(figures),
        "figure_manifest_sha256": manifest_sha,
        "figures": figures,
    }
