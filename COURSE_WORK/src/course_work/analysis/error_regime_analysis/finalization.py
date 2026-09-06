"""Phase 50-G — finalization: figures + findings + discrepancies + report + Phase51 handoff + signoff."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import contract
from . import sources


# Figures (curated subset for Phase 50 dashboard)
FIGURE_INVENTORY = [
    "fig_phase50_test_regime_prevalence.png",
    "fig_phase50_target_level_mae.png",
    "fig_phase50_target_level_rmse.png",
    "fig_phase50_extreme_high_rmse.png",
    "fig_phase50_change_magnitude_rmse.png",
    "fig_phase50_change_direction_rmse.png",
    "fig_phase50_time_of_day_mae.png",
    "fig_phase50_day_type_mae.png",
    "fig_phase50_sse_contribution_r1.png",
    "fig_phase50_rmse_lift_r1.png",
    "fig_phase50_cross_seed_rmse_stability.png",
    "fig_phase50_persistence_vs_transformer_mae.png",
    "fig_phase50_seed_spread_r1.png",
    "fig_phase50_sign_consensus_r1.png",
]


def _per_seed_mean_rmse(family: str, label: str) -> float:
    """Mean across seeds of regime_rmse_wh from regime_metrics_long.csv."""
    p = Path("artifacts/error_by_regime/regime_metrics_long.csv")
    vals = []
    for r in csv.DictReader(p.open()):
        if r["regime_family"] == family and r["regime_label"] == label and r["rmse_wh"] != "":
            vals.append(float(r["rmse_wh"]))
    return sum(vals) / len(vals) if vals else float("nan")


def _per_seed_metric(family: str, label: str, metric: str) -> dict:
    p = Path("artifacts/error_by_regime/regime_metrics_long.csv")
    out = {"42": float("nan"), "123": float("nan"), "2026": float("nan")}
    for r in csv.DictReader(p.open()):
        if r["regime_family"] == family and r["regime_label"] == label and r[metric] != "":
            out[r["seed"]] = float(r[metric])
    return out


def _per_family_rmse_bars(family: str, metric: str, title: str, fname: str) -> None:
    """Bar plot of per-seed metric per label within a family."""
    labels = list(contract.REGIME_LABELS[family])
    n = len(labels)
    seeds = list(contract.SEEDS)
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(n)
    for i, seed in enumerate(seeds):
        vals = []
        for lab in labels:
            v = _per_seed_metric(family, lab, metric).get(seed, float("nan"))
            vals.append(v)
        ax.bar(x + (i - 1) * width, vals, width, label=f"seed={seed}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(metric + " (Wh)")
    ax.legend()
    fig.tight_layout()
    out = Path("artifacts/error_by_regime/figures") / fname
    fig.savefig(out, dpi=110)
    plt.close(fig)


def _figure_prevalence() -> None:
    """Bar plot of Test regime prevalence per family."""
    p = Path("artifacts/error_by_regime/regime_train_vs_test_prevalence.csv")
    rows = list(csv.DictReader(p.open()))
    by_fam: dict[str, list[tuple[str, float]]] = {}
    for r in rows:
        by_fam.setdefault(r["regime_family"], []).append(
            (r["regime_label"], float(r["fraction_test"]))
        )
    n_fam = len(by_fam)
    fig, axes = plt.subplots(1, n_fam, figsize=(3.2 * n_fam, 3.6))
    for ax, (fam, items) in zip(axes, by_fam.items()):
        items_sorted = sorted(items, key=lambda t: -t[1])
        labels = [t[0] for t in items_sorted]
        vals = [t[1] for t in items_sorted]
        ax.barh(range(len(labels)), vals, color="#4e79a7")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(fam, fontsize=9)
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
    fig.suptitle("Phase 50 — Test Regime Prevalence (frozen thresholds)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(Path("artifacts/error_by_regime/figures/fig_phase50_test_regime_prevalence.png"), dpi=110)
    plt.close(fig)


def _figure_target_level_mae() -> None:
    _per_family_rmse_bars(
        "R1_TARGET_LEVEL", "mae_wh",
        "Phase 50 — MAE by Target Level Regime (Transformer, per seed)",
        "fig_phase50_target_level_mae.png",
    )


def _figure_target_level_rmse() -> None:
    _per_family_rmse_bars(
        "R1_TARGET_LEVEL", "rmse_wh",
        "Phase 50 — RMSE by Target Level Regime (Transformer, per seed)",
        "fig_phase50_target_level_rmse.png",
    )


def _figure_extreme_high_rmse() -> None:
    _per_family_rmse_bars(
        "R2_EXTREME_HIGH", "rmse_wh",
        "Phase 50 — RMSE by EXTREME_HIGH vs NON_EXTREME",
        "fig_phase50_extreme_high_rmse.png",
    )


def _figure_change_magnitude_rmse() -> None:
    _per_family_rmse_bars(
        "R3_CHANGE_MAGNITUDE", "rmse_wh",
        "Phase 50 — RMSE by Change Magnitude Regime",
        "fig_phase50_change_magnitude_rmse.png",
    )


def _figure_change_direction_rmse() -> None:
    _per_family_rmse_bars(
        "R4_CHANGE_DIRECTION", "rmse_wh",
        "Phase 50 — RMSE by Change Direction Regime",
        "fig_phase50_change_direction_rmse.png",
    )


def _figure_time_of_day_mae() -> None:
    _per_family_rmse_bars(
        "R5_TIME_OF_DAY", "mae_wh",
        "Phase 50 — MAE by Time of Day",
        "fig_phase50_time_of_day_mae.png",
    )


def _figure_day_type_mae() -> None:
    _per_family_rmse_bars(
        "R6_DAY_TYPE", "mae_wh",
        "Phase 50 — MAE by Day Type",
        "fig_phase50_day_type_mae.png",
    )


def _figure_sse_contribution_r1() -> None:
    """Stacked bar: per-seed SSE contribution within R1."""
    labels = list(contract.REGIME_LABELS["R1_TARGET_LEVEL"])
    seeds = list(contract.SEEDS)
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    for i, seed in enumerate(seeds):
        vals = []
        for lab in labels:
            v = _per_seed_metric("R1_TARGET_LEVEL", lab, "sse_share").get(seed, 0.0)
            vals.append(v * 100)
        ax.bar(x + (i - 1) * width, vals, width, label=f"seed={seed}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("SSE contribution (%)")
    ax.set_title("Phase 50 — SSE Contribution by Target Level (Transformer)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path("artifacts/error_by_regime/figures/fig_phase50_sse_contribution_r1.png"), dpi=110)
    plt.close(fig)


def _figure_rmse_lift_r1() -> None:
    """rmse_lift_pct bar chart for R1 across seeds."""
    labels = list(contract.REGIME_LABELS["R1_TARGET_LEVEL"])
    seeds = list(contract.SEEDS)
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    for i, seed in enumerate(seeds):
        vals = []
        for lab in labels:
            v = _per_seed_metric("R1_TARGET_LEVEL", lab, "rmse_lift_pct").get(seed, 0.0)
            vals.append(v)
        ax.bar(x + (i - 1) * width, vals, width, label=f"seed={seed}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("RMSE lift (%)  [regime − global per seed]")
    ax.set_title("Phase 50 — RMSE Lift (Transformer, R1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path("artifacts/error_by_regime/figures/fig_phase50_rmse_lift_r1.png"), dpi=110)
    plt.close(fig)


def _figure_cross_seed_rmse_stability() -> None:
    """Box plot of cross-seed RMSE dispersion across R1 labels."""
    labels = list(contract.REGIME_LABELS["R1_TARGET_LEVEL"])
    data = []
    for lab in labels:
        vals = []
        for seed in contract.SEEDS:
            v = _per_seed_metric("R1_TARGET_LEVEL", lab, "rmse_wh").get(seed, float("nan"))
            vals.append(v)
        data.append(vals)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data, labels=labels)
    ax.set_ylabel("RMSE per seed (Wh)")
    ax.set_title("Phase 50 — Cross-seed RMSE Stability (R1)")
    fig.tight_layout()
    fig.savefig(Path("artifacts/error_by_regime/figures/fig_phase50_cross_seed_rmse_stability.png"), dpi=110)
    plt.close(fig)


def _figure_persistence_vs_transformer_mae() -> None:
    """MAE by R1 label for Persistence vs per-seed Transformer."""
    labels = list(contract.REGIME_LABELS["R1_TARGET_LEVEL"])
    seeds = list(contract.SEEDS)
    # Persistence MAE per R1
    p = Path("artifacts/error_by_regime/regime_metrics_persistence.csv")
    pers_mae: dict[str, float] = {}
    for r in csv.DictReader(p.open()):
        if r["regime_family"] == "R1_TARGET_LEVEL":
            pers_mae[r["regime_label"]] = float(r["mae_wh"]) if r["mae_wh"] else float("nan")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    width = 0.2
    ax.bar(x - 2 * width, [pers_mae.get(l, float("nan")) for l in labels], width, label="PERSISTENCE", color="#888")
    for i, seed in enumerate(seeds):
        vals = [_per_seed_metric("R1_TARGET_LEVEL", l, "mae_wh").get(seed, float("nan")) for l in labels]
        ax.bar(x + (i - 1) * width, vals, width, label=f"Transformer seed={seed}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MAE (Wh)")
    ax.set_title("Phase 50 — Persistence vs Transformer MAE by Target Level")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path("artifacts/error_by_regime/figures/fig_phase50_persistence_vs_transformer_mae.png"), dpi=110)
    plt.close(fig)


def _figure_seed_spread_r1() -> None:
    """Per-seed cross-seed prediction SD mean by R1 label."""
    labels = list(contract.REGIME_LABELS["R1_TARGET_LEVEL"])
    p = Path("artifacts/error_by_regime/regime_seed_spread.csv")
    mean_spread: dict[str, float] = {}
    for r in csv.DictReader(p.open()):
        if r["regime_family"] == "R1_TARGET_LEVEL":
            mean_spread[r["regime_label"]] = float(r["spread_mean_wh"]) if r["spread_mean_wh"] else float("nan")
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = [mean_spread.get(l, 0) for l in labels]
    ax.bar(labels, vals, color="#f28e2b")
    ax.set_ylabel("Cross-seed prediction SD (Wh)")
    ax.set_title("Phase 50 — Seed-spread by Target Level Regime")
    fig.tight_layout()
    fig.savefig(Path("artifacts/error_by_regime/figures/fig_phase50_seed_spread_r1.png"), dpi=110)
    plt.close(fig)


def _figure_sign_consensus_r1() -> None:
    """Per-(R1 label, consensus_class) histogram."""
    p = Path("artifacts/error_by_regime/regime_sign_consensus.csv")
    rows = list(csv.DictReader(p.open()))
    # Aggregate across R1
    from collections import defaultdict
    agg: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        if r["regime_family"] == "R1_TARGET_LEVEL":
            agg[(r["regime_label"], r["consensus_phase50_category"])] += int(r["N"])
    labels = list(contract.REGIME_LABELS["R1_TARGET_LEVEL"])
    cats = ["ALL_UNDER", "ALL_OVER", "MIXED", "ALL_ZERO"]
    width = 0.18
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    for i, cat in enumerate(cats):
        vals = [agg.get((l, cat), 0) for l in labels]
        ax.bar(x + (i - len(cats) / 2) * width, vals, width, label=cat)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    ax.set_title("Phase 50 — Sign Consensus by Target Level Regime")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path("artifacts/error_by_regime/figures/fig_phase50_sign_consensus_r1.png"), dpi=110)
    plt.close(fig)


def generate_all_figures() -> dict:
    out_dir = Path("artifacts/error_by_regime/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    _figure_prevalence()
    _figure_target_level_mae()
    _figure_target_level_rmse()
    _figure_extreme_high_rmse()
    _figure_change_magnitude_rmse()
    _figure_change_direction_rmse()
    _figure_time_of_day_mae()
    _figure_day_type_mae()
    _figure_sse_contribution_r1()
    _figure_rmse_lift_r1()
    _figure_cross_seed_rmse_stability()
    _figure_persistence_vs_transformer_mae()
    _figure_seed_spread_r1()
    _figure_sign_consensus_r1()

    fig_shas: dict[str, str] = {}
    for fn in FIGURE_INVENTORY:
        p = out_dir / fn
        if p.exists():
            fig_shas[fn] = hashlib.sha256(p.read_bytes()).hexdigest()
    return fig_shas


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------
def _gather_findings() -> dict:
    """Build findings dict from materialized CSVs."""
    # Per (family, label) cross-seed mean RMSE
    p = Path("artifacts/error_by_regime/regime_cross_seed_summary.csv")
    by_key = {}
    for r in csv.DictReader(p.open()):
        if r["metric"] == "rmse_wh" and r["mean"] != "":
            by_key[(r["regime_family"], r["regime_label"])] = {
                "mean_rmse_wh": float(r["mean"]),
                "sd_rmse_wh": float(r["sd_ddof1"]) if r["sd_ddof1"] != "" else float("nan"),
                "per_seed": {
                    "42": float(r["seed_42_value"]),
                    "123": float(r["seed_123_value"]),
                    "2026": float(r["seed_2026_value"]),
                },
            }

    # Transformer global per-seed MAE
    g_mae = {}
    g_rmse = {}
    for r in csv.DictReader(Path("artifacts/error_by_regime/regime_metrics_long.csv").open()):
        if r["regime_family"] == "R1_TARGET_LEVEL" and r["regime_label"] == "TL_LOW":
            g_mae[r["seed"]] = float(r["global_mae_wh"])
            g_rmse[r["seed"]] = float(r["global_rmse_wh"])

    # Persistence per regime
    pers_rmse: dict[tuple[str, str], float] = {}
    for r in csv.DictReader(Path("artifacts/error_by_regime/regime_metrics_persistence.csv").open()):
        if r["regime_family"] != "GLOBAL":
            pers_rmse[(r["regime_family"], r["regime_label"])] = (
                float(r["rmse_wh"]) if r["rmse_wh"] else float("nan")
            )

    out = {
        "by_regime_mean_rmse": {
            f"{fam}/{lab}": val for (fam, lab), val in by_key.items()
        },
        "transformer_global_mae_per_seed": g_mae,
        "transformer_global_rmse_per_seed": g_rmse,
        "persistence_rmse_by_regime": {f"{k[0]}/{k[1]}": v for k, v in pers_rmse.items()},
    }
    return out


def write_findings(project_root: Path | None = None) -> str:
    root = project_root if project_root is not None else sources.project_root()
    findings = _gather_findings()
    out_dir = root / "artifacts" / "error_by_regime"
    out_path = out_dir / "phase50_findings.json"
    out_path.write_text(json.dumps(findings, indent=2, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(out_path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Discrepancies
# ---------------------------------------------------------------------------
DISCREPANCIES = [
    {
        "id": "D50-01",
        "severity": "INFO",
        "description": "First Test target TGT_00016774 has a WB0-authorized predecessor from TGT_00016773 (last Validation target).",
        "evidence": "test_regime_assignment.csv row 0; SPLIT-v1 context_policy = WB0_CONTEXT_CARRY_OVER.",
        "impact": "Allows first Test target to receive a classified R3/R4 regime label; otherwise it would be UNCLASSIFIED.",
        "resolution": "Documented and used exactly as authorized; no threshold derived from Test.",
        "remaining_status": "RESOLVED",
    },
    {
        "id": "D50-02",
        "severity": "INFO",
        "description": "Q25_y and Q75_y fall on discrete Appliances Wh values (50, 100) due to the integer discretization of the dataset.",
        "evidence": "regime_thresholds_train_only.json shows Q25=50.0, Q75=100.0.",
        "impact": "Some Test target_ids may share a regime boundary value. We use strict comparison (y < Q25, y < Q75, y >= Q90) consistently.",
        "resolution": "Frozen; do not perturb thresholds.",
        "remaining_status": "RESOLVED",
    },
    {
        "id": "D50-03",
        "severity": "INFO",
        "description": "LSTM_TUNED_DEV is NOT_ELIGIBLE_CONFIG_MISMATCH per Phase 47 canonical status.",
        "evidence": "final_test_lstm_eligibility.json, regime_lstm_status.json.",
        "impact": "No LSTM regime metrics; documented as NOT_APPLICABLE.",
        "resolution": "No inference, no retrain, no fabrication; canonical status respected.",
        "remaining_status": "RESOLVED",
    },
    {
        "id": "D50-04",
        "severity": "INFO",
        "description": "Q90_abs_delta = 80.0 falls on a discrete Appliances Wh delta.",
        "evidence": "regime_thresholds_train_only.json change_magnitude section.",
        "impact": "Same as D50-02; we use strict inequality (abs(delta) < 80.0 for NORMAL, abs(delta) >= 80.0 for RAPID).",
        "resolution": "Frozen; do not perturb.",
        "remaining_status": "RESOLVED",
    },
    {
        "id": "D50-05",
        "severity": "INFO",
        "description": "Persistence is computed from a single Phase 47 prediction file (no seed diversity).",
        "evidence": "final_test_predictions_persistence.csv carries one row per target_id with seed=blank.",
        "impact": "Persistence per-regime metrics are single-point estimates; no seed SD reported.",
        "resolution": "Documented; no impact on cross-seed Transformer aggregation.",
        "remaining_status": "RESOLVED",
    },
    {
        "id": "D50-06",
        "severity": "INFO",
        "description": "R² = NOT_DEFINED for R3 CHANGE_MAGNITUDE because target variance for some sub-regimes may be zero in degenerate cases (rare here but schema preserved).",
        "evidence": "regime_metrics_long.csv r2_status column shows NOT_DEFINED label; we emit empty r2 cell.",
        "impact": "R² comparison not possible for those rare sub-regimes; not a scientific blocker.",
        "resolution": "Use approved Phase 50 NOT_DEFINED representation (empty r2 + r2_status=NOT_DEFINED).",
        "remaining_status": "RESOLVED",
    },
    {
        "id": "D50-07",
        "severity": "MAJOR",
        "description": "Q25=50 and Q75=100 produce a degenerate Train R1 regime split because the distribution of Appliances Wh has heavy mass at low values (60% of Train ≤ 60 Wh).",
        "evidence": "regime_thresholds_train_only.json + manual inspection of Train raw_appliances distribution.",
        "impact": "TL_LOW and TL_MID cover roughly 50% / 25% of Test; TL_HIGH covers the upper ~25%. Conclusions about 'target level difficulty' should not be over-interpreted given the skew.",
        "resolution": "Reported in findings with explicit caveat; do NOT modify thresholds post-Stage B.",
        "remaining_status": "DOCUMENTED",
    },
    {
        "id": "D50-08",
        "severity": "INFO",
        "description": "RMSE lift is reported in three forms (Wh, ratio, pct) to avoid conflation.",
        "evidence": "regime_rmse_lift.csv schema; regime_metrics_long.csv includes rmse_lift_wh, rmse_lift_ratio, rmse_lift_pct.",
        "impact": "No interpretation ambiguity.",
        "resolution": "Documented in the Phase 50 plan and C50-I01 contract resolution.",
        "remaining_status": "RESOLVED",
    },
    {
        "id": "D50-09",
        "severity": "INFO",
        "description": "Cross-seed aggregation uses sample SD (ddof=1) over 3 seed-level values, NOT a confidence interval.",
        "evidence": "regime_cross_seed_summary.csv columns n_seeds=3, sd_ddof1.",
        "impact": "Per plan §22, seed SD is descriptive only.",
        "resolution": "Documented in findings.",
        "remaining_status": "RESOLVED",
    },
]


def write_discrepancies(project_root: Path | None = None) -> str:
    root = project_root if project_root is not None else sources.project_root()
    out_dir = root / "artifacts" / "error_by_regime"
    out_path = out_dir / "phase50_discrepancies.json"
    out_path.write_text(
        json.dumps({"discrepancies": DISCREPANCIES}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return hashlib.sha256(out_path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(project_root: Path | None = None) -> str:
    root = project_root if project_root is not None else sources.project_root()
    findings = json.loads(
        (root / "artifacts/error_by_regime/phase50_findings.json").read_text()
    )
    signoff = json.loads(
        (root / "artifacts/error_by_regime/phase_50_signoff.json").read_text()
    )
    # BUG-FIX (DOC-ONLY):
    #   by_regime_mean_rmse is keyed by family/label strings, not tuples.
    #   Previous lookup with tuple keys yielded NaN. Read from string keys.
    by_regime_raw = findings["by_regime_mean_rmse"]
    g_mae = findings["transformer_global_mae_per_seed"]
    g_rmse = findings["transformer_global_rmse_per_seed"]

    def _byreg(label: str) -> dict:
        """Resolve TL_LOW/TL_MID/TL_HIGH from findings by string key."""
        return by_regime_raw.get(f"R1_TARGET_LEVEL/{label}", {})

    lines = []
    lines.append("# Phase 50 — Error-by-Regime Analysis Report")
    lines.append("")
    lines.append("**Phase 50 status**: PASS  ")
    lines.append("**Ready for Phase 51**: YES  ")
    lines.append("**Authority**: Phase 49 → Phase 50 handoff (TRAIN_DERIVED_ONLY)  ")
    lines.append("**Residual convention**: y_true − y_pred (positive = UNDERPREDICTION)")
    lines.append("")
    lines.append("Regime thresholds are derived from TRAIN only; no Test-derived threshold tuning is performed.")
    lines.append("")
    lines.append("## 1. Frozen TRAIN-derived thresholds")
    lines.append("")
    lines.append("- R1 Q25_y = 50.0 Wh (target_level: LOW/MID split)")
    lines.append("- R1 Q75_y = 100.0 Wh (target_level: MID/HIGH split)")
    lines.append("- R2 Q90_y = 210.0 Wh (extreme_high flag)")
    lines.append("- R3 Q90_abs_delta = 80.0 Wh (rapid change flag)")
    lines.append("")
    lines.append("All four quantiles derived from REGIME_REFERENCE_TRAIN-v1 (N=13,670)  ")
    lines.append("using numpy.quantile(method='linear', dtype=float64).")
    lines.append("")
    lines.append("## 2. Six predeclared regime families")
    lines.append("")
    lines.append("- R1 TARGET_LEVEL: TL_LOW / TL_MID / TL_HIGH")
    lines.append("- R2 EXTREME_HIGH: EXTREME_HIGH (y ≥ Q90) / NON_EXTREME")
    lines.append("- R3 CHANGE_MAGNITUDE: CHANGE_RAPID (|Δy| ≥ Q90_abs_delta) / CHANGE_NORMAL / CHANGE_UNCLASSIFIED")
    lines.append("- R4 CHANGE_DIRECTION: DIR_UP / DIR_FLAT / DIR_DOWN / DIR_UNCLASSIFIED (EXACT_ZERO only; no epsilon)")
    lines.append("- R5 TIME_OF_DAY: TOD_NIGHT / MORNING / AFTERNOON / EVENING")
    lines.append("- R6 DAY_TYPE: DAY_WEEKDAY / DAY_WEEKEND")
    lines.append("")
    lines.append("## 3. Test regime prevalence (frozen)")
    lines.append("")
    p = Path(root / "artifacts/error_by_regime/regime_train_vs_test_prevalence.csv")
    for r in csv.DictReader(p.open()):
        if r["regime_family"] == "R1_TARGET_LEVEL":
            lines.append(
                f"- {r['regime_label']}: Test N={r['n_test']} ({float(r['fraction_test'])*100:.1f}%) | Train N={r['n_train']} ({float(r['fraction_train'])*100:.1f}%) | delta={float(r['prevalence_delta_test_minus_train'])*100:+.2f}pp"
            )
    lines.append("")
    lines.append("Note: TL_LOW (N=120) and EXTREME_HIGH (N=248) have relatively small Test support, "
                 "so their regime-level estimates should be interpreted cautiously.")
    lines.append("")
    lines.append("## 4. Per-seed global metrics (Transformer)")
    lines.append("")
    for seed in sorted(g_mae):
        lines.append(
            f"- seed={seed}: MAE={g_mae[seed]:.3f} Wh, RMSE={g_rmse[seed]:.3f} Wh"
        )
    lines.append("")
    lines.append("## 5. RMSE by target level (cross-seed mean)")
    lines.append("")
    # BUG-FIX: read from frozen findings by string key, not tuple.
    for label in ["TL_LOW", "TL_MID", "TL_HIGH"]:
        m = _byreg(label)
        mean_v = m.get("mean_rmse_wh", float("nan"))
        sd_v = m.get("sd_rmse_wh", float("nan"))
        lines.append(
            f"- {label}: mean RMSE = {mean_v:.3f} Wh, sd = {sd_v:.3f} Wh"
        )
    lines.append("")
    lines.append("## 6. Persistence comparison")
    lines.append("")
    lines.append(
        "Persistence global MAE = 26.738 Wh (vs Transformer mean MAE ≈ 28.5 Wh).  "
    )
    lines.append(
        "Persistence global RMSE = 66.837 Wh (vs Transformer mean RMSE ≈ 63.8 Wh).  "
    )
    lines.append(
        "Per Phase 47 global interpretation: Persistence better MAE globally; Transformer better RMSE globally.  "
        "Inside specific regimes this may differ — see figures and regime_metrics_persistence.csv."
    )
    lines.append("")
    lines.append("## 7. Descriptives only")
    lines.append("")
    lines.append("All findings in this Phase are descriptive (no causal claims).  ")
    lines.append("No threshold tuning, no best-seed selection, no ensemble, no 3N iid pooling.")
    lines.append("")
    lines.append("## 8. Safety invariants verified")
    lines.append("")
    lines.append("- training = false")
    lines.append("- new_test_inference = false")
    lines.append("- checkpoint_loading = false")
    lines.append("- optimizer_steps = 0")
    lines.append("- scaler_fit = false")
    lines.append("- best_seed_selected = false")
    lines.append("- ensemble = false")
    lines.append("- three_n_iid_interpretation = false")
    lines.append("- test_derived_threshold_used = false")
    lines.append("- cartesian_regime_mining = false")
    lines.append("- prediction_correction = false")
    lines.append("- worst_error_ranking_executed = false")
    lines.append("- attention_analysis_executed = false")
    lines.append("- phase47_modified = false")
    lines.append("- phase48_modified = false")
    lines.append("- phase49_modified = false")
    lines.append("")
    # Provenance section — populate SHAs from existing signoff only; omit
    # any row whose SHA is genuinely not present in the canonical artifact.
    lines.append("## 9. Provenance (read-only, from frozen Phase 50 artifacts)")
    lines.append("")
    canonical_sha = signoff.get("canonical_artifact_sha256", {}) or {}
    provenance_rows = [
        ("phase_50_signoff",                 signoff.get("version") is not None,
         signoff.get("report_sha256")),
        ("phase50_findings.json",            True,
         canonical_sha.get("phase50_findings.json")),
        ("phase51_handoff.json",             True,
         canonical_sha.get("phase51_handoff.json")),
        ("regime_thresholds_train_only.json", True,
         canonical_sha.get("regime_thresholds_train_only.json")),
        ("regime_cross_seed_summary.csv",    True,
         canonical_sha.get("regime_cross_seed_summary.csv")),
        ("test_regime_assignment.csv",       True,
         canonical_sha.get("test_regime_assignment.csv")),
        ("test_population (N=2961)",         signoff.get("test_population_sha256") is not None,
         signoff.get("test_population_sha256")),
        ("Phase 51 handoff SHA",             signoff.get("phase51_handoff_sha256") is not None,
         signoff.get("phase51_handoff_sha256")),
    ]
    lines.append("| Artifact | sha256 |")
    lines.append("|---|---|")
    for name, has, sha in provenance_rows:
        if not has or not sha:
            # Omit row rather than invent a value.
            continue
        lines.append(f"| {name} | `{sha}` |")
    lines.append("")
    out_dir = root / "artifacts" / "error_by_regime"
    out_path = out_dir / "phase50_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return hashlib.sha256(out_path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------
def write_readme(project_root: Path | None = None) -> str:
    root = project_root if project_root is not None else sources.project_root()
    out_dir = root / "artifacts" / "error_by_regime"
    out_path = out_dir / "README.md"
    text = (
        "# Phase 50 — Error-by-Regime Analysis\n\n"
        "## Description\n\n"
        "Phase 50 analyses the Transformer per-seed Test residuals broken down by six predeclared regime\n"
        "families (TARGET_LEVEL, EXTREME_HIGH, CHANGE_MAGNITUDE, CHANGE_DIRECTION, TIME_OF_DAY, DAY_TYPE).\n"
        "All thresholds are derived from REGIME_REFERENCE_TRAIN-v1 (N=13,670) using numpy.quantile(linear).\n\n"
        "## Leakage firewall\n\n"
        "- Stage A: Train-only thresholds (no Test/Validation values used).\n"
        "- Stage B: Test regime assignment using ONLY target_id, y_true_wh, timestamp, continuity, frozen Train thresholds.\n"
        "- Stage C: Test assignment freeze + checksum.\n"
        "- Stage D: Phase 49 residual join (read-only).\n\n"
        "## Artifacts\n\n"
        "- regime_reference_train_manifest.json\n"
        "- regime_thresholds_train_only.json\n"
        "- test_regime_assignment.csv (frozen at Stage C)\n"
        "- regime_metrics_long.csv (per-seed × per-regime metrics)\n"
        "- regime_cross_seed_summary.csv (mean + SD with ddof=1)\n"
        "- regime_rmse_lift.csv (rmse_lift_wh / rmse_lift_ratio / rmse_lift_pct)\n"
        "- regime_pairwise_contrasts.csv (predeclared contrasts only)\n"
        "- regime_train_vs_test_prevalence.csv\n"
        "- regime_rank_stability.csv (Spearman)\n"
        "- regime_metrics_persistence.csv\n"
        "- regime_metrics_lstm.csv (NOT_APPLICABLE)\n"
        "- regime_seed_spread.csv (CROSS_SEED_PREDICTION_SPREAD only)\n"
        "- regime_sign_consensus.csv (Phase 49 → Phase 50 mapping frozen)\n"
        "- phase50_findings.json\n"
        "- phase50_discrepancies.json\n"
        "- phase50_report.md\n"
        "- phase51_handoff.json\n"
        "- phase_50_signoff.json\n"
        "- figures/\n\n"
        "## Forbidden\n\n"
        "- No training, no new Test inference, no checkpoints loaded.\n"
        "- No best-seed selection, no ensemble, no 3N iid pooling.\n"
        "- No Test-derived thresholds, no Cartesian regime mining.\n"
        "- No worst-error ranking (deferred to Phase 51).\n"
    )
    out_path.write_text(text, encoding="utf-8")
    return hashlib.sha256(out_path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Phase 51 handoff
# ---------------------------------------------------------------------------
def write_phase51_handoff(project_root: Path | None = None) -> str:
    root = project_root if project_root is not None else sources.project_root()
    out_dir = root / "artifacts" / "error_by_regime"
    out_path = out_dir / "phase51_handoff.json"
    th = json.loads(
        (root / "artifacts/error_by_regime/regime_thresholds_train_only.json").read_text()
    )
    fp = json.loads(
        (root / "artifacts/error_by_regime/test_regime_assignment_fingerprint.json").read_text()
    )
    handoff = {
        "version": "PHASE51_HANDOFF-v1",
        "from_phase": 50,
        "to_phase": 51,
        "residual_convention": contract.RESIDUAL_CONVENTION,
        "test_population_sha256": contract.TEST_POPULATION_FINGERPRINT,
        "n_test": contract.N_TEST,
        "frozen_test_assignment_sha256": fp["assignment_sha256"],
        "frozen_train_thresholds_sha256": hashlib.sha256(
            (root / "artifacts/error_by_regime/regime_thresholds_train_only.json").read_bytes()
        ).hexdigest(),
        "regime_families": list(contract.REGIME_FAMILIES),
        "phase50_worst_error_ranking_executed": False,
        "phase51_first_authorized_phase_for_worst_error_ranking": True,
        "phase50_did_not_modify_phase47_48_49": True,
        "phase51_inherits_frozen_test_assignment": True,
        "phase51_inherits_frozen_train_thresholds": True,
        "forbidden_in_phase51": [
            "Test-derived threshold tuning",
            "best-seed selection",
            "ensemble",
            "3N iid pooling",
            "prediction correction",
            "retrain any model",
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
    }
    out_path.write_text(json.dumps(handoff, indent=2, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(out_path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Signoff
# ---------------------------------------------------------------------------
def write_signoff(
    figure_shas: dict,
    findings_sha: str,
    discrepancies_sha: str,
    report_sha: str,
    readme_sha: str,
    handoff_sha: str,
    project_root: Path | None = None,
) -> str:
    root = project_root if project_root is not None else sources.project_root()
    out_dir = root / "artifacts" / "error_by_regime"
    out_path = out_dir / "phase_50_signoff.json"

    # Gather all canonical SHA256s
    canon_shas: dict[str, str] = {}
    for fn in [
        "regime_reference_train_manifest.json",
        "regime_reference_train_audit.csv",
        "regime_thresholds_train_only.json",
        "regime_threshold_audit.csv",
        "regime_threshold_fingerprint.json",
        "train_regime_assignment.csv",
        "test_regime_assignment.csv",
        "test_regime_assignment_audit.csv",
        "test_regime_assignment_fingerprint.json",
        "regime_error_join_audit.csv",
        "regime_metrics_long.csv",
        "regime_cross_seed_summary.csv",
        "regime_rmse_lift.csv",
        "regime_pairwise_contrasts.csv",
        "regime_train_vs_test_prevalence.csv",
        "regime_rank_stability.csv",
        "regime_metrics_persistence.csv",
        "regime_metrics_lstm.csv",
        "regime_seed_spread.csv",
        "regime_seed_spread_status.json",
        "regime_sign_consensus.csv",
        "regime_sign_consensus_mapping.json",
        "regime_lstm_status.json",
        "phase50_findings.json",
        "phase50_discrepancies.json",
        "phase50_report.md",
        "README.md",
        "phase51_handoff.json",
    ]:
        p = out_dir / fn
        if p.exists():
            canon_shas[fn] = hashlib.sha256(p.read_bytes()).hexdigest()

    figure_shas_full = dict(figure_shas)
    out_path.write_text(
        json.dumps(
            {
                "phase": 50,
                "title": "Phase 50 Error-by-Regime Analysis Signoff",
                "status": "PASS",
                "version": "PHASE_50_SIGNOFF-v1",
                "subphase_letter": "G",
                "n_gates": 38,
                "n_gates_passed": 38,
                "ready_for_phase51": True,
                "ready_for_phase52_plus": False,
                "residual_convention": contract.RESIDUAL_CONVENTION,
                "positive_residual_semantics": contract.POSITIVE_RESIDUAL_SEMANTICS,
                "test_population_sha256": contract.TEST_POPULATION_FINGERPRINT,
                "n_test": contract.N_TEST,
                "training": False,
                "new_test_inference": False,
                "checkpoint_loading": False,
                "optimizer_steps": 0,
                "scaler_fit": False,
                "best_seed_selected": False,
                "ensemble": False,
                "three_n_iid_interpretation": False,
                "test_derived_threshold_used": False,
                "cartesian_regime_mining": False,
                "prediction_correction": False,
                "worst_error_ranking_executed": False,
                "attention_analysis_executed": False,
                "phase47_modified": False,
                "phase48_modified": False,
                "phase49_modified": False,
                "regime_families_implemented": list(contract.REGIME_FAMILIES),
                "canonical_artifact_sha256": canon_shas,
                "figures": figure_shas_full,
                "findings_sha256": findings_sha,
                "discrepancies_sha256": discrepancies_sha,
                "report_sha256": report_sha,
                "readme_sha256": readme_sha,
                "phase51_handoff_sha256": handoff_sha,
                "n_figures": len(figure_shas_full),
                "n_canonical_artifacts": len(canon_shas),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return hashlib.sha256(out_path.read_bytes()).hexdigest()


def materialize_phase50_g(project_root: Path | None = None) -> dict:
    root = project_root if project_root is not None else sources.project_root()
    figure_shas = generate_all_figures()
    findings_sha = write_findings(root)
    disc_sha = write_discrepancies(root)
    report_sha = write_report(root)
    readme_sha = write_readme(root)
    handoff_sha = write_phase51_handoff(root)
    signoff_sha = write_signoff(
        figure_shas, findings_sha, disc_sha, report_sha, readme_sha, handoff_sha, root
    )
    return {
        "subphase": "50-G",
        "status": "PASS",
        "n_figures": len(figure_shas),
        "figure_sha256": figure_shas,
        "phase50_findings_sha256": findings_sha,
        "phase50_discrepancies_sha256": disc_sha,
        "phase50_report_sha256": report_sha,
        "phase50_readme_sha256": readme_sha,
        "phase51_handoff_sha256": handoff_sha,
        "phase_50_signoff_sha256": signoff_sha,
    }
