"""Phase 48-E — Figure generation.

All figures are read-only views of Phase 48 derived artifacts.

Strict rules:
* No inference, training, checkpoint loading.
* Seed mean labelled DESCRIPTIVE ONLY.
* Seed spread labelled CROSS-SEED PREDICTION SPREAD (never confidence interval).
* No "best seed" labels.
* Zoom windows deterministic: FIRST / MIDDLE / LAST.
* Scatter plots: common axes across seeds where comparison intended; y=x reference line included.
* Titles say "Held-Out Test" wherever plan requires.
* No Test-driven fit line.

Required figures per plan §88 + user step 2 (17 views):
  PRED_48_01_full_test_actual_vs_all_seeds.png
  PRED_48_02_full_test_actual_vs_seed_mean_spread.png
  PRED_48_03_first_24h_zoom.png
  PRED_48_04_middle_24h_zoom.png
  PRED_48_05_last_24h_zoom.png
  PRED_48_06_scatter_seed42.png
  PRED_48_07_scatter_seed123.png
  PRED_48_08_scatter_seed2026.png
  PRED_48_09_prediction_ecdf.png
  PRED_48_10_change_magnitude_distribution.png
  PRED_48_11_cross_seed_spread_over_time.png
  PRED_48_12_pairwise_seed_prediction_scatter.png
  PRED_48_13_lag_cross_correlation.png
  PRED_48_14_acf_actual_vs_predictions.png
  PRED_48_15_local_peak_capture.png
  PRED_48_16_rolling_24h_mean_tracking.png
  PRED_48_17_rolling_24h_std_tracking.png

Optional heatmaps (plan §78, §80, §81, §125) — emitted because Test
covers ~20 full days at 10-min cadence:
  PRED_48_18_daily_actual_heatmap.png
  PRED_48_19_daily_seed_mean_heatmap.png
  PRED_48_20_seed_spread_heatmap.png
"""
from __future__ import annotations

import csv
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from course_work.analysis.prediction_analysis.contract import (
    FIGURES_DIR_NAME,
    LAG_RANGE,
    OUTPUT_DIR,
    PEAK_TIMING_WINDOW_STEPS,
    ROLLING_WINDOW,
)


FIGURES_DIR = OUTPUT_DIR / FIGURES_DIR_NAME
ZOOM_HOURS = 24
SAMPLES_PER_HOUR = 6
ZOOM_SAMPLES = ZOOM_HOURS * SAMPLES_PER_HOUR  # 144


def _ensure_dir() -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


def _read_csv(name: str) -> list[dict]:
    p = OUTPUT_DIR / name
    with p.open() as fh:
        return list(csv.DictReader(fh))


def _parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def _set_common_style(ax, title: str, ylabel: str, xlabel: str = "") -> None:
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _load_wide_arrays() -> dict[str, np.ndarray]:
    rows = _read_csv("prediction_wide_table.csv")
    return {
        "target_id": np.array([r["target_id"] for r in rows]),
        "ts": np.array([_parse_ts(r["target_timestamp"]) for r in rows]),
        "y_true": np.array([float(r["y_true_wh"]) for r in rows]),
        "y42": np.array([float(r["y_pred_seed42"]) for r in rows]),
        "y123": np.array([float(r["y_pred_seed123"]) for r in rows]),
        "y2026": np.array([float(r["y_pred_seed2026"]) for r in rows]),
        "y_mean": np.array([float(r["seed_mean_prediction"]) for r in rows]),
        "y_std": np.array([float(r["seed_std_prediction"]) for r in rows]),
        "y_min": np.array([float(r["seed_min_prediction"]) for r in rows]),
        "y_max": np.array([float(r["seed_max_prediction"]) for r in rows]),
    }


def _zoom_window_indices(n: int) -> tuple[slice, slice, slice]:
    """FIRST / MIDDLE / LAST deterministic 24h = 144-sample windows."""
    last_start = max(0, n - ZOOM_SAMPLES)
    mid_start = max(0, (n - ZOOM_SAMPLES) // 2)
    return slice(0, ZOOM_SAMPLES), slice(mid_start, mid_start + ZOOM_SAMPLES), slice(last_start, last_start + ZOOM_SAMPLES)


# ---------------------------------------------------------------------------
# 1. Full test actual vs all seeds
# ---------------------------------------------------------------------------

def fig_01_full_test_actual_vs_all_seeds(arrs: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(arrs["ts"], arrs["y_true"], color="black", lw=0.9, label="Actual", alpha=0.85)
    ax.plot(arrs["ts"], arrs["y42"], color="#2196F3", lw=0.7, label="seed42", alpha=0.7)
    ax.plot(arrs["ts"], arrs["y123"], color="#4CAF50", lw=0.7, label="seed123", alpha=0.7)
    ax.plot(arrs["ts"], arrs["y2026"], color="#FF9800", lw=0.7, label="seed2026", alpha=0.7)
    _set_common_style(ax, "Held-Out Test — Actual vs 3 Transformer seeds", "Appliances (Wh)")
    ax.legend(loc="upper right", fontsize=9, ncol=4)
    plt.tight_layout()
    p = _ensure_dir() / "PRED_48_01_full_test_actual_vs_all_seeds.png"
    plt.savefig(p, dpi=120)
    plt.close()
    return p


# ---------------------------------------------------------------------------
# 2. Full test actual vs seed-mean + cross-seed spread band (min/max)
# ---------------------------------------------------------------------------

def fig_02_full_test_actual_vs_seed_mean_spread(arrs: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(arrs["ts"], arrs["y_true"], color="black", lw=0.9, label="Actual", alpha=0.85)
    ax.plot(arrs["ts"], arrs["y_mean"], color="#7B1FA2", lw=1.0,
            label="Seed mean (DESCRIPTIVE only, NOT ensemble)", alpha=0.9)
    ax.fill_between(arrs["ts"], arrs["y_min"], arrs["y_max"],
                    color="#7B1FA2", alpha=0.18,
                    label="Cross-seed spread (min/max band, NOT confidence interval)")
    _set_common_style(ax, "Held-Out Test — Actual vs Descriptive seed mean + cross-seed spread", "Appliances (Wh)")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    plt.tight_layout()
    p = _ensure_dir() / "PRED_48_02_full_test_actual_vs_seed_mean_spread.png"
    plt.savefig(p, dpi=120)
    plt.close()
    return p


# ---------------------------------------------------------------------------
# 3-5. 24h zoom panels
# ---------------------------------------------------------------------------

def _plot_zoom(arrs: dict, sl: slice, fname: str, label: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(arrs["ts"][sl], arrs["y_true"][sl], color="black", lw=1.0, label="Actual", alpha=0.9)
    ax.plot(arrs["ts"][sl], arrs["y42"][sl], color="#2196F3", lw=0.8, label="seed42", alpha=0.75)
    ax.plot(arrs["ts"][sl], arrs["y123"][sl], color="#4CAF50", lw=0.8, label="seed123", alpha=0.75)
    ax.plot(arrs["ts"][sl], arrs["y2026"][sl], color="#FF9800", lw=0.8, label="seed2026", alpha=0.75)
    ax.fill_between(arrs["ts"][sl], arrs["y_min"][sl], arrs["y_max"][sl],
                    color="#7B1FA2", alpha=0.15, label="Cross-seed spread")
    _set_common_style(ax, f"Held-Out Test — 24h zoom ({label} 144 samples)", "Appliances (Wh)")
    ax.legend(loc="upper right", fontsize=9, ncol=3)
    plt.tight_layout()
    p = _ensure_dir() / fname
    plt.savefig(p, dpi=120)
    plt.close()
    return p


def fig_03_first_24h(arrs: dict) -> Path:
    first, _, _ = _zoom_window_indices(len(arrs["ts"]))
    return _plot_zoom(arrs, first, "PRED_48_03_first_24h_zoom.png", "FIRST")


def fig_04_middle_24h(arrs: dict) -> Path:
    _, mid, _ = _zoom_window_indices(len(arrs["ts"]))
    return _plot_zoom(arrs, mid, "PRED_48_04_middle_24h_zoom.png", "MIDDLE")


def fig_05_last_24h(arrs: dict) -> Path:
    _, _, last = _zoom_window_indices(len(arrs["ts"]))
    return _plot_zoom(arrs, last, "PRED_48_05_last_24h_zoom.png", "LAST")


# ---------------------------------------------------------------------------
# 6-8. Actual vs prediction scatter (common axes + y=x line)
# ---------------------------------------------------------------------------

def _scatter_actual_vs_pred(arrs: dict, y_pred: np.ndarray, fname: str, title_suffix: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo = float(min(arrs["y_true"].min(), y_pred.min()))
    hi = float(max(arrs["y_true"].max(), y_pred.max()))
    pad = 0.05 * (hi - lo)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(arrs["y_true"], y_pred, s=4, alpha=0.35, color="#37474F")
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#D32F2F", lw=1.0, linestyle="--", label="y = x")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    _set_common_style(ax, f"Held-Out Test — Actual vs predicted ({title_suffix})",
                      "y_pred (Wh)", "y_true (Wh)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    p = _ensure_dir() / fname
    plt.savefig(p, dpi=120)
    plt.close()
    return p


def fig_06_scatter_seed42(arrs: dict) -> Path:
    return _scatter_actual_vs_pred(arrs, arrs["y42"], "PRED_48_06_scatter_seed42.png", "seed42")


def fig_07_scatter_seed123(arrs: dict) -> Path:
    return _scatter_actual_vs_pred(arrs, arrs["y123"], "PRED_48_07_scatter_seed123.png", "seed123")


def fig_08_scatter_seed2026(arrs: dict) -> Path:
    return _scatter_actual_vs_pred(arrs, arrs["y2026"], "PRED_48_08_scatter_seed2026.png", "seed2026")


# ---------------------------------------------------------------------------
# 9. Prediction ECDF
# ---------------------------------------------------------------------------

def fig_09_prediction_ecdf(arrs: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, y, color in [
        ("Actual", arrs["y_true"], "black"),
        ("seed42", arrs["y42"], "#2196F3"),
        ("seed123", arrs["y123"], "#4CAF50"),
        ("seed2026", arrs["y2026"], "#FF9800"),
        ("seed mean (descriptive)", arrs["y_mean"], "#7B1FA2"),
    ]:
        sorted_y = np.sort(y)
        ecdf = np.arange(1, sorted_y.size + 1) / sorted_y.size
        ax.plot(sorted_y, ecdf, label=label, color=color, lw=1.2, alpha=0.85)
    _set_common_style(ax, "Held-Out Test — Prediction ECDF (actual + 3 seeds + descriptive mean)", "ECDF", "Appliances (Wh)")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    p = _ensure_dir() / "PRED_48_09_prediction_ecdf.png"
    plt.savefig(p, dpi=120)
    plt.close()
    return p


# ---------------------------------------------------------------------------
# 10. Change magnitude distribution (|delta|)
# ---------------------------------------------------------------------------

def fig_10_change_magnitude_distribution(arrs: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, y, color in [
        ("Actual", arrs["y_true"], "black"),
        ("seed42", arrs["y42"], "#2196F3"),
        ("seed123", arrs["y123"], "#4CAF50"),
        ("seed2026", arrs["y2026"], "#FF9800"),
    ]:
        deltas = np.abs(np.diff(y))
        sorted_d = np.sort(deltas)
        ecdf = np.arange(1, sorted_d.size + 1) / sorted_d.size
        ax.plot(sorted_d, ecdf, label=label, color=color, lw=1.2, alpha=0.85)
    _set_common_style(ax, "Held-Out Test — ECDF of |Δ| (gap-safe 10-min transitions)", "ECDF", "|Δ| (Wh)")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    p = _ensure_dir() / "PRED_48_10_change_magnitude_distribution.png"
    plt.savefig(p, dpi=120)
    plt.close()
    return p


# ---------------------------------------------------------------------------
# 11. Cross-seed spread over time (seed_std + seed_range)
# ---------------------------------------------------------------------------

def fig_11_cross_seed_spread_over_time(arrs: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(arrs["ts"], arrs["y_std"], color="#7B1FA2", lw=0.7,
            label="seed_std_prediction (sample SD, ddof=1)")
    range_pred = arrs["y_max"] - arrs["y_min"]
    ax.plot(arrs["ts"], range_pred, color="#FFA000", lw=0.5, alpha=0.7,
            label="seed_range_prediction (max − min)")
    _set_common_style(ax, "Held-Out Test — Cross-seed prediction spread over time (NOT confidence interval)",
                      "Spread (Wh)")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    p = _ensure_dir() / "PRED_48_11_cross_seed_spread_over_time.png"
    plt.savefig(p, dpi=120)
    plt.close()
    return p


# ---------------------------------------------------------------------------
# 12. Pairwise seed prediction scatter
# ---------------------------------------------------------------------------

def fig_12_pairwise_seed_prediction_scatter(arrs: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    pairs = [("seed42", arrs["y42"], "seed123", arrs["y123"], "#7B1FA2"),
             ("seed42", arrs["y42"], "seed2026", arrs["y2026"], "#00838F"),
             ("seed123", arrs["y123"], "seed2026", arrs["y2026"], "#5D4037")]
    lo = float(min(arrs["y42"].min(), arrs["y123"].min(), arrs["y2026"].min()))
    hi = float(max(arrs["y42"].max(), arrs["y123"].max(), arrs["y2026"].max()))
    pad = 0.05 * (hi - lo)
    for ax, (la, xa, lb, xb, c) in zip(axes, pairs):
        ax.scatter(xa, xb, s=4, alpha=0.35, color=c)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#D32F2F", lw=0.8, linestyle="--")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{la} vs {lb}")
        ax.set_xlabel(f"{la} (Wh)")
        ax.set_ylabel(f"{lb} (Wh)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Held-Out Test — Pairwise seed prediction scatter (SEED_AGREEMENT_DIAGNOSTIC, not model performance)",
                 fontsize=11)
    plt.tight_layout()
    p = _ensure_dir() / "PRED_48_12_pairwise_seed_prediction_scatter.png"
    plt.savefig(p, dpi=120)
    plt.close()
    return p


# ---------------------------------------------------------------------------
# 13. Fixed-lag correlation diagnostic
# ---------------------------------------------------------------------------

def fig_13_lag_cross_correlation(arrs: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_csv("prediction_lag_diagnostics.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for seed_label, color in [("SEED42", "#2196F3"), ("SEED123", "#4CAF50"), ("SEED2026", "#FF9800")]:
        lag_steps = [int(r["lag_steps"]) for r in rows if r["seed"] == seed_label]
        corrs = [float(r["pearson_correlation"]) for r in rows if r["seed"] == seed_label]
        ax.plot(lag_steps, corrs, marker="o", label=seed_label, color=color, lw=1.2)
    ax.axvline(0, color="black", lw=0.5, linestyle="--", alpha=0.5, label="lag 0")
    _set_common_style(ax, "Held-Out Test — Lag cross-correlation diagnostic (diagnostic only, no shift applied)",
                      "Pearson r", "lag (steps; +k means pred compared to truth shifted +k future steps)")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    p = _ensure_dir() / "PRED_48_13_lag_cross_correlation.png"
    plt.savefig(p, dpi=120)
    plt.close()
    return p


# ---------------------------------------------------------------------------
# 14. Prediction ACF
# ---------------------------------------------------------------------------

def fig_14_acf_actual_vs_predictions(arrs: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_csv("prediction_acf_diagnostics.csv")
    series_to_color = {
        "ACTUAL": "black",
        "SEED42": "#2196F3",
        "SEED123": "#4CAF50",
        "SEED2026": "#FF9800",
        "SEED_MEAN_DESCRIPTIVE": "#7B1FA2",
    }
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for sid, color in series_to_color.items():
        lag_steps = []
        acfs = []
        for r in rows:
            if r["series_id"] == sid and r["status"] == "PASS" and r["acf"] not in ("", None):
                lag_steps.append(int(r["lag_steps"]))
                acfs.append(float(r["acf"]))
        if lag_steps:
            ax.plot(lag_steps, acfs, marker="o", label=sid, color=color, lw=1.2)
    _set_common_style(ax, "Held-Out Test — Prediction ACF (NOT residual ACF)", "ACF", "lag (steps)")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    p = _ensure_dir() / "PRED_48_14_acf_actual_vs_predictions.png"
    plt.savefig(p, dpi=120)
    plt.close()
    return p


# ---------------------------------------------------------------------------
# 15. Local peak capture diagnostic
# ---------------------------------------------------------------------------

def fig_15_local_peak_capture(arrs: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_csv("prediction_local_extrema_summary.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in rows:
        ax.scatter(float(r["actual_peak_mean"]), float(r["pred_at_peak_mean"]), s=80, alpha=0.75,
                   label=r["seed"])
    lo = 0.0
    hi = float(max(arrs["y_true"].max(), arrs["y42"].max(), arrs["y123"].max(), arrs["y2026"].max()))
    ax.plot([lo, hi], [lo, hi], color="#D32F2F", lw=1.0, linestyle="--", label="y = x")
    _set_common_style(ax, "Held-Out Test — Local peak capture (mean actual peak vs mean predicted-at-peak)",
                      "Predicted at true local peaks (Wh)", "Actual at true local peaks (Wh)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    p = _ensure_dir() / "PRED_48_15_local_peak_capture.png"
    plt.savefig(p, dpi=120)
    plt.close()
    return p


# ---------------------------------------------------------------------------
# 16. Rolling 24h mean tracking
# ---------------------------------------------------------------------------

def fig_16_rolling_24h_mean_tracking(arrs: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_csv("prediction_rolling_tracking.csv")
    fig, ax = plt.subplots(figsize=(14, 4.5))
    seed_to_color = {"SEED42": "#2196F3", "SEED123": "#4CAF50", "SEED2026": "#FF9800"}
    for seed_label, color in seed_to_color.items():
        ts = []
        rmean = []
        for r in rows:
            if r["seed"] == seed_label:
                ts.append(datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S"))
                rmean.append(float(r["rolling_24h_true_mean"]) - float(r["rolling_24h_pred_mean"]))
        ax.plot(ts, rmean, label=f"{seed_label} (true − pred)", color=color, lw=0.7, alpha=0.85)
    ax.axhline(0, color="black", lw=0.5, linestyle="--", alpha=0.6)
    _set_common_style(ax, "Held-Out Test — Rolling 24h mean gap (true − pred) per seed", "true − pred (Wh)")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    p = _ensure_dir() / "PRED_48_16_rolling_24h_mean_tracking.png"
    plt.savefig(p, dpi=120)
    plt.close()
    return p


# ---------------------------------------------------------------------------
# 17. Rolling 24h std tracking
# ---------------------------------------------------------------------------

def fig_17_rolling_24h_std_tracking(arrs: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_csv("prediction_rolling_tracking.csv")
    fig, ax = plt.subplots(figsize=(14, 4.5))
    seed_to_color = {"SEED42": "#2196F3", "SEED123": "#4CAF50", "SEED2026": "#FF9800"}
    for seed_label, color in seed_to_color.items():
        ts = []
        ratio = []
        for r in rows:
            if r["seed"] == seed_label:
                t_std = float(r["rolling_24h_true_std"])
                p_std = float(r["rolling_24h_pred_std"])
                ts.append(datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S"))
                ratio.append((p_std / t_std) if t_std > 0 else 0.0)
        ax.plot(ts, ratio, label=f"{seed_label} (pred_std / true_std)", color=color, lw=0.7, alpha=0.85)
    ax.axhline(1.0, color="black", lw=0.5, linestyle="--", alpha=0.6, label="ratio = 1 (no compression)")
    _set_common_style(ax, "Held-Out Test — Rolling 24h std ratio (pred / true) per seed", "std ratio")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    p = _ensure_dir() / "PRED_48_17_rolling_24h_std_tracking.png"
    plt.savefig(p, dpi=120)
    plt.close()
    return p


# ---------------------------------------------------------------------------
# 18-20. Optional daily heatmaps (plan §78, §80, §81, §125)
# ---------------------------------------------------------------------------

def _build_day_slot_matrix(arrs: dict) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (dates, slot_indices_0_to_143, day x slot value matrix)."""
    n = len(arrs["ts"])
    date_to_slots: dict[str, dict[int, float]] = {}
    for i in range(n):
        d = arrs["ts"][i].strftime("%Y-%m-%d")
        slot = (arrs["ts"][i].hour * 60 + arrs["ts"][i].minute) // 10
        date_to_slots.setdefault(d, {})[slot] = arrs["y_true"][i]
    dates = sorted(date_to_slots.keys())
    matrix = np.full((len(dates), 144), np.nan)
    for j, d in enumerate(dates):
        for slot, v in date_to_slots[d].items():
            matrix[j, slot] = v
    return dates, np.arange(144), matrix


def _build_day_slot_matrix_for(arrs: dict, series_key: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    n = len(arrs["ts"])
    date_to_slots: dict[str, dict[int, float]] = {}
    for i in range(n):
        d = arrs["ts"][i].strftime("%Y-%m-%d")
        slot = (arrs["ts"][i].hour * 60 + arrs["ts"][i].minute) // 10
        date_to_slots.setdefault(d, {})[slot] = arrs[series_key][i]
    dates = sorted(date_to_slots.keys())
    matrix = np.full((len(dates), 144), np.nan)
    for j, d in enumerate(dates):
        for slot, v in date_to_slots[d].items():
            matrix[j, slot] = v
    return dates, np.arange(144), matrix


def fig_18_daily_actual_heatmap(arrs: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dates, slots, mat = _build_day_slot_matrix(arrs)
    vmin, vmax = float(np.nanmin(mat)), float(np.nanmax(mat))
    fig, ax = plt.subplots(figsize=(14, max(3, 0.35 * len(dates))))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_yticks(range(len(dates)))
    ax.set_yticklabels(dates, fontsize=7)
    ax.set_xlabel("Time of day (10-min slot 0..143)")
    ax.set_title("Held-Out Test — Daily actual Appliances heatmap (Wh)")
    fig.colorbar(im, ax=ax, label="Appliances (Wh)")
    plt.tight_layout()
    p = _ensure_dir() / "PRED_48_18_daily_actual_heatmap.png"
    plt.savefig(p, dpi=120)
    plt.close()
    return p


def fig_19_daily_seed_mean_heatmap(arrs: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dates, slots, mat = _build_day_slot_matrix_for(arrs, "y_mean")
    vmin, vmax = float(np.nanmin(mat)), float(np.nanmax(mat))
    fig, ax = plt.subplots(figsize=(14, max(3, 0.35 * len(dates))))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_yticks(range(len(dates)))
    ax.set_yticklabels(dates, fontsize=7)
    ax.set_xlabel("Time of day (10-min slot 0..143)")
    ax.set_title("Held-Out Test — Daily seed-mean prediction heatmap (Wh, descriptive only)")
    fig.colorbar(im, ax=ax, label="Seed-mean (Wh)")
    plt.tight_layout()
    p = _ensure_dir() / "PRED_48_19_daily_seed_mean_heatmap.png"
    plt.savefig(p, dpi=120)
    plt.close()
    return p


def fig_20_seed_spread_heatmap(arrs: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dates, slots, mat = _build_day_slot_matrix_for(arrs, "y_std")
    vmin, vmax = float(np.nanmin(mat)), float(np.nanmax(mat))
    fig, ax = plt.subplots(figsize=(14, max(3, 0.35 * len(dates))))
    im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
    ax.set_yticks(range(len(dates)))
    ax.set_yticklabels(dates, fontsize=7)
    ax.set_xlabel("Time of day (10-min slot 0..143)")
    ax.set_title("Held-Out Test — Daily cross-seed spread heatmap (sample SD, NOT uncertainty)")
    fig.colorbar(im, ax=ax, label="seed_std_prediction (Wh)")
    plt.tight_layout()
    p = _ensure_dir() / "PRED_48_20_seed_spread_heatmap.png"
    plt.savefig(p, dpi=120)
    plt.close()
    return p


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def render_all_figures() -> list[Path]:
    arrs = _load_wide_arrays()
    paths = [
        fig_01_full_test_actual_vs_all_seeds(arrs),
        fig_02_full_test_actual_vs_seed_mean_spread(arrs),
        fig_03_first_24h(arrs),
        fig_04_middle_24h(arrs),
        fig_05_last_24h(arrs),
        fig_06_scatter_seed42(arrs),
        fig_07_scatter_seed123(arrs),
        fig_08_scatter_seed2026(arrs),
        fig_09_prediction_ecdf(arrs),
        fig_10_change_magnitude_distribution(arrs),
        fig_11_cross_seed_spread_over_time(arrs),
        fig_12_pairwise_seed_prediction_scatter(arrs),
        fig_13_lag_cross_correlation(arrs),
        fig_14_acf_actual_vs_predictions(arrs),
        fig_15_local_peak_capture(arrs),
        fig_16_rolling_24h_mean_tracking(arrs),
        fig_17_rolling_24h_std_tracking(arrs),
        fig_18_daily_actual_heatmap(arrs),
        fig_19_daily_seed_mean_heatmap(arrs),
        fig_20_seed_spread_heatmap(arrs),
    ]
    return paths
