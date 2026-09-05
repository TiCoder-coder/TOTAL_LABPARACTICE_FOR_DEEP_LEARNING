from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .figure_safety import (
    FIGURES_DIR_REL,
    SEED_COLORS,
    SEED_LABELS,
    SEED_LIST,
    _ensure_figures_dir,
    _save_figure,
)
from ..utils.artifacts import get_project_root


SIGN_POSITIVE = "UNDERPREDICTION"
SIGN_NEGATIVE = "OVERPREDICTION"
SIGN_ZERO = "EXACT"


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _parse_timestamp(s: str) -> datetime:
    if "T" in s:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


# 1. Residual time series by seed
def figure_residual_time_series(project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else get_project_root()
    lt = _read_csv_rows(root / "artifacts/residual_analysis/residual_long_table.csv")
    fig, ax = plt.subplots(figsize=(14, 5))
    for seed in SEED_LIST:
        seed_rows = sorted(
            [r for r in lt if r["seed"] == seed],
            key=lambda r: _parse_timestamp(r["target_timestamp"]),
        )
        ts = [_parse_timestamp(r["target_timestamp"]) for r in seed_rows]
        y = np.asarray([float(r["residual_wh"]) for r in seed_rows], dtype=np.float64)
        ax.plot(ts, y, alpha=0.55, linewidth=0.6, color=SEED_COLORS[seed], label=SEED_LABELS[seed])
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_title("Phase49 — Residual Time Series (y_true - y_pred)\npositive = UNDERPREDICTION, negative = OVERPREDICTION")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Residual (Wh)")
    ax.legend(loc="best")
    fig.autofmt_xdate()
    return _save_figure(fig, project_root, "phase49_fig01_residual_time_series.png")


# 2. Residual distribution / histogram comparison
def figure_residual_distribution(project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else get_project_root()
    lt = _read_csv_rows(root / "artifacts/residual_analysis/residual_long_table.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # density
    for seed in SEED_LIST:
        seed_resid = np.asarray(
            [float(r["residual_wh"]) for r in lt if r["seed"] == seed], dtype=np.float64
        )
        axes[0].hist(
            seed_resid, bins=60, alpha=0.45, density=True,
            color=SEED_COLORS[seed], label=SEED_LABELS[seed],
        )
    axes[0].axvline(0.0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    axes[0].set_title("Residual Density (per seed)")
    axes[0].set_xlabel("Residual (Wh)")
    axes[0].set_ylabel("Density")
    axes[0].legend(loc="best")
    # boxplot
    box_data = [
        np.asarray([float(r["residual_wh"]) for r in lt if r["seed"] == seed], dtype=np.float64)
        for seed in SEED_LIST
    ]
    axes[1].boxplot(
        box_data,
        labels=[SEED_LABELS[s] for s in SEED_LIST],
        showmeans=True,
    )
    axes[1].axhline(0.0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    axes[1].set_title("Residual Box Plot (per seed)")
    axes[1].set_ylabel("Residual (Wh)")
    return _save_figure(fig, project_root, "phase49_fig02_residual_distribution.png")


# 3. Residual ECDF comparison
def figure_residual_ecdf(project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else get_project_root()
    lt = _read_csv_rows(root / "artifacts/residual_analysis/residual_long_table.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for seed in SEED_LIST:
        y = np.sort(
            np.asarray([float(r["residual_wh"]) for r in lt if r["seed"] == seed], dtype=np.float64)
        )
        x = np.arange(1, y.size + 1) / y.size
        ax.plot(y, x, color=SEED_COLORS[seed], label=SEED_LABELS[seed])
    ax.axvline(0.0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_title("Residual ECDF (per seed)")
    ax.set_xlabel("Residual (Wh)")
    ax.set_ylabel("ECDF")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    return _save_figure(fig, project_root, "phase49_fig03_residual_ecdf.png")


# 4. Signed bias comparison
def figure_signed_bias(project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else get_project_root()
    rows = _read_csv_rows(root / "artifacts/residual_analysis/phase49_signed_bias.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [float(r["mean_residual_wh"]) for r in rows]
    medians = [float(r["median_residual_wh"]) for r in rows]
    x = np.arange(len(rows))
    ax.bar(x - 0.2, means, width=0.4, label="mean residual (Wh)")
    ax.bar(x + 0.2, medians, width=0.4, label="median residual (Wh)")
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([SEED_LABELS[r["seed"]] for r in rows])
    ax.set_title("Signed Bias by Seed\n(mean>0 = net underprediction tendency; mean<0 = net overprediction tendency)")
    ax.set_ylabel("Residual (Wh)")
    ax.legend(loc="best")
    return _save_figure(fig, project_root, "phase49_fig04_signed_bias.png")


# 5. Underprediction vs overprediction balance
def figure_sign_balance(project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else get_project_root()
    rows = _read_csv_rows(root / "artifacts/residual_analysis/phase49_sign_balance.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    under = [float(r["underprediction_fraction"]) for r in rows]
    over = [float(r["overprediction_fraction"]) for r in rows]
    exact = [float(r["exact_fraction"]) for r in rows]
    x = np.arange(len(rows))
    ax.bar(x, under, color="#1f77b4", label="UNDERPREDICTION fraction")
    ax.bar(x, over, bottom=under, color="#d62728", label="OVERPREDICTION fraction")
    ax.bar(x, exact, bottom=[u + o for u, o in zip(under, over)], color="#7f7f7f", label="EXACT fraction")
    ax.set_xticks(x)
    ax.set_xticklabels([SEED_LABELS[r["seed"]] for r in rows])
    ax.set_ylabel("Fraction of N=2961")
    ax.set_title("Sign Balance per Seed")
    ax.legend(loc="best")
    ax.set_ylim(0, 1.0)
    return _save_figure(fig, project_root, "phase49_fig05_sign_balance.png")


# 6. Absolute-error tail comparison
def figure_tail_diagnostics(project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else get_project_root()
    rows = _read_csv_rows(root / "artifacts/residual_analysis/phase49_tail_diagnostics.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    p90 = [float(r["abs_error_p90"]) for r in rows]
    p95 = [float(r["abs_error_p95"]) for r in rows]
    p99 = [float(r["abs_error_p99"]) for r in rows]
    mx = [float(r["max_absolute_error"]) for r in rows]
    x = np.arange(len(rows))
    width = 0.2
    ax.bar(x - 1.5 * width, p90, width, label="p90", color="#aec7e8")
    ax.bar(x - 0.5 * width, p95, width, label="p95", color="#1f77b4")
    ax.bar(x + 0.5 * width, p99, width, label="p99", color="#ff7f0e")
    ax.bar(x + 1.5 * width, mx, width, label="max", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels([SEED_LABELS[r["seed"]] for r in rows])
    ax.set_ylabel("|residual| (Wh)")
    ax.set_title("Absolute-Error Tail Diagnostics per Seed")
    ax.legend(loc="best")
    return _save_figure(fig, project_root, "phase49_fig06_tail_diagnostics.png")


# 7. Residual ACF comparison
def figure_residual_acf(project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else get_project_root()
    rows = _read_csv_rows(root / "artifacts/residual_analysis/phase49_residual_acf.csv")
    fig, ax = plt.subplots(figsize=(12, 5))
    for seed in SEED_LIST:
        seed_rows = [r for r in rows if r["seed"] == seed]
        lags = [int(r["lag_steps"]) for r in seed_rows]
        acf = [float(r["acf_value"]) for r in seed_rows]
        ax.plot(lags, acf, color=SEED_COLORS[seed], label=SEED_LABELS[seed], linewidth=1.0)
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_xlabel("Lag steps (1 step = 10 min)")
    ax.set_ylabel("ACF")
    ax.set_title("Residual ACF, lags 1..144, gap-safe (per seed)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    return _save_figure(fig, project_root, "phase49_fig07_residual_acf.png")


# 8. Key residual ACF lags
def figure_key_acf_lags(project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else get_project_root()
    rows = _read_csv_rows(root / "artifacts/residual_analysis/phase49_residual_acf_key_lags.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    key_lags = (1, 6, 12, 36, 72, 144)
    x = np.arange(len(key_lags))
    width = 0.25
    for i, seed in enumerate(SEED_LIST):
        seed_rows = [r for r in rows if r["seed"] == seed]
        acf = [float(r["acf_value"]) for r in seed_rows]
        ax.bar(x + (i - 1) * width, acf, width, color=SEED_COLORS[seed], label=SEED_LABELS[seed])
    ax.set_xticks(x)
    ax.set_xticklabels([f"lag {l}\n({l*10} min)" for l in key_lags])
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_ylabel("ACF value")
    ax.set_title("Residual ACF — Key Lags (1, 6, 12, 36, 72, 144)")
    ax.legend(loc="best")
    return _save_figure(fig, project_root, "phase49_fig08_key_acf_lags.png")


# 9. Sign-run length distribution
def figure_sign_run_distribution(project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else get_project_root()
    fig, ax = plt.subplots(figsize=(10, 5))
    for seed in SEED_LIST:
        rows = _read_csv_rows(
            root / f"artifacts/residual_analysis/phase49_sign_run_table_seed{seed}.csv"
        )
        lengths = [int(r["length"]) for r in rows]
        if not lengths:
            continue
        max_len = max(max(lengths), 30)
        bins = np.arange(0.5, max_len + 1.5, 1)
        ax.hist(lengths, bins=bins, alpha=0.45, color=SEED_COLORS[seed], label=SEED_LABELS[seed])
    ax.set_xlabel("Sign run length")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Sign Run Lengths (per seed)")
    ax.legend(loc="best")
    return _save_figure(fig, project_root, "phase49_fig09_sign_run_distribution.png")


# 10. Sign-transition visualization
def figure_sign_transitions(project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else get_project_root()
    rows = _read_csv_rows(root / "artifacts/residual_analysis/phase49_sign_transitions.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    sign_labels = ["UNDERPREDICTION", "OVERPREDICTION", "EXACT"]
    x = np.arange(len(sign_labels))
    width = 0.25
    seed_data: dict[str, dict[str, float]] = {seed: {label: 0.0 for label in sign_labels} for seed in SEED_LIST}
    for r in rows:
        seed = r["seed"]
        from_sign = r["from_sign"]
        if seed in seed_data and from_sign in sign_labels:
            try:
                v = float(r["probability_given_from_a"])
            except ValueError:
                v = 0.0
            if not (math.isnan(v) or math.isinf(v)):
                seed_data[seed][from_sign] = v
    for i, seed in enumerate(SEED_LIST):
        probs = [seed_data[seed][label] for label in sign_labels]
        ax.bar(x + (i - 1) * width, probs, width, color=SEED_COLORS[seed], label=SEED_LABELS[seed])
    ax.set_xticks(x)
    ax.set_xticklabels(sign_labels)
    ax.set_ylabel("Probability given from-sign")
    ax.set_title(
        "Sign-Transition Probabilities (10-min cadence, exact 0)\n"
        "3 bars per from-sign correspond to seeds 42 / 123 / 2026 (left → right)\n"
        "(NaN probabilities for from-sign with 0 outgoing transitions are plotted as 0)"
    )
    ax.set_ylim(0, 1.0)
    ax.legend(loc="best")
    return _save_figure(fig, project_root, "phase49_fig10_sign_transitions.png")


import math


# 11. Rolling residual mean
def figure_rolling_mean(project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else get_project_root()
    rows = _read_csv_rows(root / "artifacts/residual_analysis/phase49_rolling_residual_diagnostics.csv")
    n_per_seed = len(rows) // 3
    fig, ax = plt.subplots(figsize=(14, 4))
    for i, seed in enumerate(SEED_LIST):
        seed_rows = rows[i * n_per_seed : (i + 1) * n_per_seed]
        ts = [_parse_timestamp(r["window_start_timestamp"]) for r in seed_rows]
        y = [float(r["rolling_residual_mean_wh"]) for r in seed_rows]
        ax.plot(ts, y, alpha=0.7, linewidth=0.7, color=SEED_COLORS[seed], label=SEED_LABELS[seed])
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_xlabel("Window start timestamp")
    ax.set_ylabel("Rolling residual mean (Wh)")
    ax.set_title("Rolling Residual Mean (window = 144 contiguous samples = 24h)")
    ax.legend(loc="best")
    fig.autofmt_xdate()
    return _save_figure(fig, project_root, "phase49_fig11_rolling_mean.png")


# 12. Rolling residual std
def figure_rolling_std(project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else get_project_root()
    rows = _read_csv_rows(root / "artifacts/residual_analysis/phase49_rolling_residual_diagnostics.csv")
    n_per_seed = len(rows) // 3
    fig, ax = plt.subplots(figsize=(14, 4))
    for i, seed in enumerate(SEED_LIST):
        seed_rows = rows[i * n_per_seed : (i + 1) * n_per_seed]
        ts = [_parse_timestamp(r["window_start_timestamp"]) for r in seed_rows]
        y = [float(r["rolling_residual_std_wh"]) for r in seed_rows]
        ax.plot(ts, y, alpha=0.7, linewidth=0.7, color=SEED_COLORS[seed], label=SEED_LABELS[seed])
    ax.set_xlabel("Window start timestamp")
    ax.set_ylabel("Rolling residual std (Wh, ddof=1)")
    ax.set_title("Rolling Residual Std (window = 144 contiguous samples = 24h)")
    ax.legend(loc="best")
    fig.autofmt_xdate()
    return _save_figure(fig, project_root, "phase49_fig12_rolling_std.png")


# 13. Rolling MAE
def figure_rolling_mae(project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else get_project_root()
    rows = _read_csv_rows(root / "artifacts/residual_analysis/phase49_rolling_residual_diagnostics.csv")
    n_per_seed = len(rows) // 3
    fig, ax = plt.subplots(figsize=(14, 4))
    for i, seed in enumerate(SEED_LIST):
        seed_rows = rows[i * n_per_seed : (i + 1) * n_per_seed]
        ts = [_parse_timestamp(r["window_start_timestamp"]) for r in seed_rows]
        y = [float(r["rolling_mae_wh"]) for r in seed_rows]
        ax.plot(ts, y, alpha=0.7, linewidth=0.7, color=SEED_COLORS[seed], label=SEED_LABELS[seed])
    ax.set_xlabel("Window start timestamp")
    ax.set_ylabel("Rolling MAE (Wh)")
    ax.set_title("Rolling MAE (window = 144 contiguous samples = 24h)")
    ax.legend(loc="best")
    fig.autofmt_xdate()
    return _save_figure(fig, project_root, "phase49_fig13_rolling_mae.png")


# 14. Rolling RMSE
def figure_rolling_rmse(project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else get_project_root()
    rows = _read_csv_rows(root / "artifacts/residual_analysis/phase49_rolling_residual_diagnostics.csv")
    n_per_seed = len(rows) // 3
    fig, ax = plt.subplots(figsize=(14, 4))
    for i, seed in enumerate(SEED_LIST):
        seed_rows = rows[i * n_per_seed : (i + 1) * n_per_seed]
        ts = [_parse_timestamp(r["window_start_timestamp"]) for r in seed_rows]
        y = [float(r["rolling_rmse_wh"]) for r in seed_rows]
        ax.plot(ts, y, alpha=0.7, linewidth=0.7, color=SEED_COLORS[seed], label=SEED_LABELS[seed])
    ax.set_xlabel("Window start timestamp")
    ax.set_ylabel("Rolling RMSE (Wh)")
    ax.set_title("Rolling RMSE (window = 144 contiguous samples = 24h)")
    ax.legend(loc="best")
    fig.autofmt_xdate()
    return _save_figure(fig, project_root, "phase49_fig14_rolling_rmse.png")


# 15-18. |residual| and residual vs y_true and y_pred
def _scatter_xy(ax, x, y, title, xlabel, ylabel, color):
    ax.scatter(x, y, alpha=0.20, s=8, color=color, edgecolors="none")
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def figure_magnitude_associations(project_root: Path | None = None) -> list[Path]:
    root = project_root if project_root is not None else get_project_root()
    lt = _read_csv_rows(root / "artifacts/residual_analysis/residual_long_table.csv")
    paths: list[Path] = []
    titles = [
        ("phase49_fig15_abs_residual_vs_y_true.png", "|residual| vs y_true", "y_true (Wh)", "|residual| (Wh)"),
        ("phase49_fig16_abs_residual_vs_y_pred.png", "|residual| vs y_pred", "y_pred (Wh)", "|residual| (Wh)"),
        ("phase49_fig17_residual_vs_y_true.png", "residual vs y_true", "y_true (Wh)", "residual (Wh)"),
        ("phase49_fig18_residual_vs_y_pred.png", "residual vs y_pred", "y_pred (Wh)", "residual (Wh)"),
    ]
    field_pairs = [
        ("abs_residual_y_true", "y_true_wh", lambda r: float(r["absolute_error_wh"]), lambda r: float(r["y_true_wh"])),
        ("abs_residual_y_pred", "y_pred_wh", lambda r: float(r["absolute_error_wh"]), lambda r: float(r["y_pred_wh"])),
        ("residual_y_true", "y_true_wh", lambda r: float(r["residual_wh"]), lambda r: float(r["y_true_wh"])),
        ("residual_y_pred", "y_pred_wh", lambda r: float(r["residual_wh"]), lambda r: float(r["y_pred_wh"])),
    ]
    for (fname, title, xlabel, ylabel), (_id, _y_label, xf, yf) in zip(titles, field_pairs):
        fig, ax = plt.subplots(figsize=(7, 5))
        for seed in SEED_LIST:
            x = np.asarray([xf(r) for r in lt if r["seed"] == seed], dtype=np.float64)
            y = np.asarray([yf(r) for r in lt if r["seed"] == seed], dtype=np.float64)
            ax.scatter(x, y, alpha=0.20, s=8, color=SEED_COLORS[seed], edgecolors="none", label=SEED_LABELS[seed])
        ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
        ax.set_title(f"{title}\n(per seed, DESCRIPTIVE — no causal interpretation)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        paths.append(_save_figure(fig, project_root, fname))
    return paths


# 19. Prediction-decile MAE/RMSE
def figure_prediction_deciles(project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else get_project_root()
    rows = _read_csv_rows(root / "artifacts/residual_analysis/phase49_prediction_deciles.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for seed in SEED_LIST:
        seed_rows = sorted([r for r in rows if r["seed"] == seed], key=lambda r: int(r["decile"]))
        deciles = [int(r["decile"]) for r in seed_rows]
        mae = [float(r["mae"]) for r in seed_rows]
        rmse = [float(r["rmse"]) for r in seed_rows]
        axes[0].plot(deciles, mae, marker="o", color=SEED_COLORS[seed], label=SEED_LABELS[seed])
        axes[1].plot(deciles, rmse, marker="o", color=SEED_COLORS[seed], label=SEED_LABELS[seed])
    axes[0].set_title("Prediction-Decile MAE (DESCRIPTIVE diagnostic only)")
    axes[0].set_xlabel("Prediction decile (y_pred-based)")
    axes[0].set_ylabel("MAE (Wh)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].set_title("Prediction-Decile RMSE (DESCRIPTIVE diagnostic only)")
    axes[1].set_xlabel("Prediction decile (y_pred-based)")
    axes[1].set_ylabel("RMSE (Wh)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")
    return _save_figure(fig, project_root, "phase49_fig19_prediction_deciles.png")


# 20. Cross-seed residual agreement (correlation bar chart)
def figure_cross_seed_agreement(project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else get_project_root()
    rows = _read_csv_rows(root / "artifacts/residual_analysis/phase49_cross_seed_residual_agreement.csv")
    pairs = [f"{r['seed_a']}-{r['seed_b']}" for r in rows]
    pearson = [float(r["pearson_residual_correlation"]) for r in rows]
    spearman = [float(r["spearman_residual_correlation"]) for r in rows]
    x = np.arange(len(pairs))
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    ax.bar(x - width / 2, pearson, width, label="Pearson")
    ax.bar(x + width / 2, spearman, width, label="Spearman")
    ax.set_xticks(x)
    ax.set_xticklabels(pairs)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Residual correlation")
    ax.set_title("Cross-Seed Residual Agreement (DESCRIPTIVE, NOT model-performance comparison)")
    ax.legend(loc="best")
    return _save_figure(fig, project_root, "phase49_fig20_cross_seed_agreement.png")


# 21. Cross-seed sign consensus
def figure_cross_seed_sign_consensus(project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else get_project_root()
    rows = _read_csv_rows(root / "artifacts/residual_analysis/phase49_cross_seed_sign_consensus.csv")
    classes = [r["consensus_class"] for r in rows]
    counts = [int(r["count"]) for r in rows]
    fractions = [float(r["fraction"]) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2ca02c", "#d62728", "#7f7f7f", "#1f77b4", "#ff7f0e", "#9467bd"]
    ax.bar(classes, fractions, color=colors[: len(classes)])
    ax.set_ylabel("Fraction of N=2961")
    ax.set_title("Cross-Seed Sign Consensus (exact-zero policy, no epsilon)")
    ax.set_ylim(0, max(fractions) * 1.2 if fractions else 1.0)
    for i, (cls, c) in enumerate(zip(classes, counts)):
        ax.text(i, fractions[i] + 0.005, f"{c}", ha="center", fontsize=9)
    return _save_figure(fig, project_root, "phase49_fig21_cross_seed_sign_consensus.png")


# 22. Persistence residual context
def figure_persistence_context(project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else get_project_root()
    rows = _read_csv_rows(root / "artifacts/residual_analysis/phase49_persistence_context.csv")
    r = rows[0]
    labels = ["MAE (Wh)", "RMSE (Wh)", "Mean residual (Wh)"]
    values = [float(r["mae"]), float(r["rmse"]), float(r["mean_residual"])]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom")
    ax.set_ylabel("Value (Wh)")
    ax.set_title(
        "Persistence Baseline Residual Context (DESCRIPTIVE baseline)\n"
        f"N={r['N']} sha256 first16={Path('artifacts/final_test/predictions/final_test_predictions_persistence.csv').stat().st_size} bytes verified"
    )
    return _save_figure(fig, project_root, "phase49_fig22_persistence_context.png")


def generate_all_figures(project_root: Path | None = None) -> list[Path]:
    out: list[Path] = []
    out.append(figure_residual_time_series(project_root))
    out.append(figure_residual_distribution(project_root))
    out.append(figure_residual_ecdf(project_root))
    out.append(figure_signed_bias(project_root))
    out.append(figure_sign_balance(project_root))
    out.append(figure_tail_diagnostics(project_root))
    out.append(figure_residual_acf(project_root))
    out.append(figure_key_acf_lags(project_root))
    out.append(figure_sign_run_distribution(project_root))
    out.append(figure_sign_transitions(project_root))
    out.append(figure_rolling_mean(project_root))
    out.append(figure_rolling_std(project_root))
    out.append(figure_rolling_mae(project_root))
    out.append(figure_rolling_rmse(project_root))
    out.extend(figure_magnitude_associations(project_root))
    out.append(figure_prediction_deciles(project_root))
    out.append(figure_cross_seed_agreement(project_root))
    out.append(figure_cross_seed_sign_consensus(project_root))
    out.append(figure_persistence_context(project_root))
    return out
