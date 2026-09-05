"""Phase 55 - head behavior cards, head-to-layer-mean distance, layer diversity.

Phase 55 §49: head behavior card per (seed, layer, head) - no rank, no score.
Phase 55 §78: layer-level diversity summary.
Phase 55 §118-§120: schemas.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .sources import (
    FrozenSources55,
    LAG_MINUTES,
    NUM_HEADS,
    NUM_LAYERS,
    SEEDS,
)
from .profile_metrics import (
    jsd_natural_log,
    l1_distance,
    l2_distance,
    cosine_similarity,
    wasserstein_1d_minutes,
)


# ---------------------------------------------------------------------------
# Profile pivot
# ---------------------------------------------------------------------------

def pivot_profiles_by_head(
    sources: FrozenSources55,
) -> dict[tuple[int, int, int], np.ndarray]:
    """Returns (seed, layer, head) -> profile over LAG_MINUTES order (newest=10 -> oldest=720).

    Phase 54 schema: seed,layer_idx0,head_idx0,source_position_idx0,lag_steps,lag_minutes,mean_weight,...
    """
    by_head: dict[tuple[int, int, int], dict[int, float]] = {}
    for r in sources.profile_by_lag:
        try:
            seed = int(r["seed"])
            layer = int(r["layer_idx0"])
            head = int(r["head_idx0"])
            lag = int(r["lag_minutes"])
            w = float(r["mean_weight"])
        except Exception:
            continue
        by_head.setdefault((seed, layer, head), {})[lag] = w
    out: dict[tuple[int, int, int], np.ndarray] = {}
    lag_sorted = sorted(set(LAG_MINUTES.tolist()))
    for key, d in by_head.items():
        vec = np.array([d.get(int(l), 0.0) for l in lag_sorted], dtype=np.float64)
        s = vec.sum()
        if s > 0 and abs(s - 1.0) > 1e-5:
            vec = vec / s
        out[key] = vec
    return out


def pivot_layer_head_mean_profiles(
    sources: FrozenSources55,
) -> dict[tuple[int, int], np.ndarray]:
    """Returns (seed, layer) -> mean profile across heads over LAG_MINUTES order.

    Phase 54 schema: seed,layer_idx0,lag_steps,lag_minutes,head_mean_weight,...
    (column is 'head_mean_weight' — not 'mean_weight').
    """
    by_layer: dict[tuple[int, int], dict[int, float]] = {}
    for r in sources.layer_head_mean_profile:
        try:
            seed = int(r["seed"])
            layer = int(r["layer_idx0"])
            lag = int(r["lag_minutes"])
            w = float(r["head_mean_weight"])
        except Exception:
            continue
        by_layer.setdefault((seed, layer), {})[lag] = w
    out: dict[tuple[int, int], np.ndarray] = {}
    lag_sorted = sorted(set(LAG_MINUTES.tolist()))
    for key, d in by_layer.items():
        vec = np.array([d.get(int(l), 0.0) for l in lag_sorted], dtype=np.float64)
        s = vec.sum()
        if s > 0 and abs(s - 1.0) > 1e-5:
            vec = vec / s
        out[key] = vec
    return out


# ---------------------------------------------------------------------------
# Head behavior summary
# ---------------------------------------------------------------------------

@dataclass
class HeadBehaviorCard:
    seed: int
    layer_idx0: int
    head_idx0: int
    head_display: str
    median_normalized_entropy: float
    mean_normalized_entropy: float
    median_effective_source_count: float
    median_expected_lag_minutes: float
    median_lag_sd_minutes: float
    median_top1_weight: float
    median_top5_mass: float
    median_recent1h_mass: float
    median_recent6h_mass: float
    median_recent12h_mass: float
    median_recent24h_mass: float
    median_lag50_minutes: float
    median_lag80_minutes: float
    median_lag90_minutes: float
    modal_top1_lag_minutes: float
    modal_top1_lag_fraction: float
    jsd_to_layer_mean: float
    l1_to_layer_mean: float
    cosine_to_layer_mean: float
    status: str


def _safe_get(d: dict[str, str], key: str) -> float:
    v = d.get(key, "")
    if v == "" or v is None or v in ("nan", "NaN", "None"):
        return float("nan")
    try:
        return float(v)
    except Exception:
        return float("nan")


def _pivot_metric_summary_by_head(
    summary_rows: list[dict[str, str]],
) -> dict[tuple[int, int, int], dict[str, float]]:
    """Long schema: (seed, layer_idx0, head_idx0, metric, median, mean, ...).
    Returns {(seed,layer,head): {metric_name: median, metric_name_mean: mean}}.
    """
    out: dict[tuple[int, int, int], dict[str, float]] = {}
    for r in summary_rows:
        try:
            key = (int(r["seed"]), int(r["layer_idx0"]), int(r["head_idx0"]))
        except Exception:
            continue
        metric_name = r.get("metric", "")
        median_v = r.get("median", "")
        mean_v = r.get("mean", "")
        if metric_name == "" or (median_v == "" and mean_v == ""):
            continue
        if median_v != "":
            try:
                out.setdefault(key, {})[metric_name] = float(median_v)
            except Exception:
                continue
        if mean_v != "":
            try:
                out.setdefault(key, {})[f"{metric_name}_mean"] = float(mean_v)
            except Exception:
                continue
    return out


def _pivot_recent_mass_by_head(
    recent_rows: list[dict[str, str]],
) -> dict[tuple[int, int, int], dict[str, float]]:
    """Recent mass schema: (seed, layer_idx0, head_idx0, window, median_mass, ...). Returns {(seed,layer,head): {window: median_mass}}."""
    out: dict[tuple[int, int, int], dict[str, float]] = {}
    for r in recent_rows:
        try:
            key = (int(r["seed"]), int(r["layer_idx0"]), int(r["head_idx0"]))
            win = r.get("window", "")
            med = r.get("median_mass", "")
            out.setdefault(key, {})[win] = float(med)
        except Exception:
            continue
    return out


def _pivot_coverage_by_head(
    cov_rows: list[dict[str, str]],
) -> dict[tuple[int, int, int], dict[str, float]]:
    """Coverage schema: (seed, layer_idx0, head_idx0, coverage_level, median_minutes, ...). Returns {(seed,layer,head): {lag_key: median_minutes}}."""
    out: dict[tuple[int, int, int], dict[str, float]] = {}
    for r in cov_rows:
        try:
            key = (int(r["seed"]), int(r["layer_idx0"]), int(r["head_idx0"]))
            level = r.get("coverage_level", "")
            med = r.get("median_minutes", "")
            if level == "" or med == "":
                continue
            tag = f"lag{int(float(level) * 100):02d}_minutes"  # 50 -> lag50_minutes
            out.setdefault(key, {})[tag] = float(med)
        except Exception:
            continue
    return out


def _build_modal_top1(
    sources: FrozenSources55,
    seed: int,
    layer_idx0: int,
    head_idx0: int,
) -> tuple[float, float]:
    """Find modal top1 lag_minutes and its fraction from Phase 54 top1 frequency table.

    Schema: seed,layer_idx0,head_idx0,lag_steps,lag_minutes,count,fraction,total_targets,status
    """
    rows = [r for r in sources.top1_lag_frequency
            if int(r["seed"]) == seed
            and int(r["layer_idx0"]) == layer_idx0
            and int(r["head_idx0"]) == head_idx0]
    if not rows:
        return (float("nan"), float("nan"))
    best = max(rows, key=lambda r: float(r.get("fraction", 0.0)))
    try:
        return (float(best["lag_minutes"]), float(best["fraction"]))
    except Exception:
        return (float("nan"), float("nan"))


def build_head_behavior_cards(
    sources: FrozenSources55,
) -> list[HeadBehaviorCard]:
    cards: list[HeadBehaviorCard] = []
    by_head = pivot_profiles_by_head(sources)
    by_layer = pivot_layer_head_mean_profiles(sources)
    summary_by_head = _pivot_metric_summary_by_head(sources.metric_summary_by_head)
    recent_by_head = _pivot_recent_mass_by_head(sources.recent_mass_summary)
    coverage_by_head = _pivot_coverage_by_head(sources.coverage_radius_summary)

    for seed in SEEDS:
        for layer in range(NUM_LAYERS):
            layer_profile = by_layer.get((seed, layer))
            for head in range(NUM_HEADS):
                key = (seed, layer, head)
                s = summary_by_head.get(key, {})
                rm = recent_by_head.get(key, {})
                cv = coverage_by_head.get(key, {})
                prof_h = by_head.get(key)
                if layer_profile is not None and prof_h is not None:
                    jsd_lm = jsd_natural_log(prof_h, layer_profile)
                    l1_lm = l1_distance(prof_h, layer_profile)
                    cos_lm = cosine_similarity(prof_h, layer_profile)
                else:
                    jsd_lm = l1_lm = cos_lm = float("nan")
                modal_lag, modal_frac = _build_modal_top1(sources, seed, layer, head)
                cards.append(HeadBehaviorCard(
                    seed=seed,
                    layer_idx0=layer,
                    head_idx0=head,
                    head_display=f"H{head + 1}",
                    median_normalized_entropy=s.get("normalized_entropy", float("nan")),
                    mean_normalized_entropy=s.get("normalized_entropy_mean", float("nan")),
                    median_effective_source_count=s.get("effective_source_count", float("nan")),
                    median_expected_lag_minutes=s.get("expected_lag_minutes", float("nan")),
                    median_lag_sd_minutes=s.get("lag_sd_minutes", float("nan")),
                    median_top1_weight=s.get("top1_weight", float("nan")),
                    median_top5_mass=s.get("top5_mass", float("nan")),
                    median_recent1h_mass=rm.get("1h", float("nan")),
                    median_recent6h_mass=rm.get("6h", float("nan")),
                    median_recent12h_mass=rm.get("12h", float("nan")),
                    median_recent24h_mass=rm.get("24h", float("nan")),
                    median_lag50_minutes=cv.get("lag50_minutes", float("nan")),
                    median_lag80_minutes=cv.get("lag80_minutes", float("nan")),
                    median_lag90_minutes=cv.get("lag90_minutes", float("nan")),
                    modal_top1_lag_minutes=modal_lag,
                    modal_top1_lag_fraction=modal_frac,
                    jsd_to_layer_mean=jsd_lm,
                    l1_to_layer_mean=l1_lm,
                    cosine_to_layer_mean=cos_lm,
                    status="OK",
                ))
    return cards


# ---------------------------------------------------------------------------
# Head-to-layer-mean distance
# ---------------------------------------------------------------------------

def build_head_to_layer_mean_distance(
    sources: FrozenSources55,
) -> list[dict[str, Any]]:
    by_head = pivot_profiles_by_head(sources)
    by_layer = pivot_layer_head_mean_profiles(sources)
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for layer in range(NUM_LAYERS):
            layer_profile = by_layer.get((seed, layer))
            if layer_profile is None:
                continue
            for head in range(NUM_HEADS):
                prof = by_head.get((seed, layer, head))
                if prof is None:
                    rows.append({
                        "seed": seed,
                        "layer_idx0": layer,
                        "head_idx0": head,
                        "jsd_to_layer_head_mean_profile": float("nan"),
                        "l1_to_layer_head_mean_profile": float("nan"),
                        "l2_to_layer_head_mean_profile": float("nan"),
                        "cosine_to_layer_head_mean_profile": float("nan"),
                        "wasserstein_to_layer_head_mean_minutes": float("nan"),
                        "status": "MISSING_PROFILE",
                    })
                    continue
                rows.append({
                    "seed": seed,
                    "layer_idx0": layer,
                    "head_idx0": head,
                    "jsd_to_layer_head_mean_profile": jsd_natural_log(prof, layer_profile),
                    "l1_to_layer_head_mean_profile": l1_distance(prof, layer_profile),
                    "l2_to_layer_head_mean_profile": l2_distance(prof, layer_profile),
                    "cosine_to_layer_head_mean_profile": cosine_similarity(prof, layer_profile),
                    "wasserstein_to_layer_head_mean_minutes": wasserstein_1d_minutes(prof, layer_profile, LAG_MINUTES),
                    "status": "OK",
                })
    return rows


# ---------------------------------------------------------------------------
# Layer diversity summary
# ---------------------------------------------------------------------------

def build_layer_diversity_summary(
    pair_metrics_rows: list[Any],
) -> list[dict[str, Any]]:
    """Aggregate pairwise metrics per (seed, layer).

    Inputs: list of PairProfileMetrics.
    """
    from collections import defaultdict
    grouped: dict[tuple[int, int], list[Any]] = defaultdict(list)
    for pm in pair_metrics_rows:
        grouped[(pm.seed, pm.layer_idx0)].append(pm)

    rows: list[dict[str, Any]] = []
    for (seed, layer), pairs in sorted(grouped.items()):
        n = len(pairs)
        jsd_arr = np.array([p.jsd_profile for p in pairs], dtype=np.float64)
        l1_arr = np.array([p.l1_profile for p in pairs], dtype=np.float64)
        l2_arr = np.array([p.l2_profile for p in pairs], dtype=np.float64)
        w_arr = np.array([p.wasserstein_minutes for p in pairs], dtype=np.float64)
        cos_arr = np.array([p.cosine_profile for p in pairs], dtype=np.float64)
        rows.append({
            "seed": seed,
            "layer_idx0": layer,
            "head_count": NUM_HEADS,
            "pair_count": n,
            "mean_pairwise_jsd": float(jsd_arr.mean()),
            "median_pairwise_jsd": float(np.median(jsd_arr)),
            "max_pairwise_jsd": float(jsd_arr.max()),
            "mean_pairwise_l1": float(l1_arr.mean()),
            "mean_pairwise_l2": float(l2_arr.mean()),
            "mean_pairwise_wasserstein_minutes": float(w_arr.mean()),
            "mean_pairwise_abs_expected_lag_diff_minutes": float("nan"),  # filled in from metric diffs separately
            "mean_pairwise_top1_tvd": float("nan"),  # filled in from top1 dist separately
            "min_pairwise_cosine": float(cos_arr.min()),
            "mean_pairwise_cosine": float(cos_arr.mean()),
            "status": "OK",
        })
    return rows


def fill_layer_diversity_extras(
    layer_div_rows: list[dict[str, Any]],
    metric_diff_rows: list[Any],
    top1_dist_rows: list[dict[str, Any]],
) -> None:
    """Augment layer diversity rows with abs expected-lag diff and top1 TVD aggregates."""
    from collections import defaultdict
    eld_per_layer: dict[tuple[int, int], list[float]] = defaultdict(list)
    for r in metric_diff_rows:
        if r.metric == "expected_lag_minutes" and not np.isnan(r.abs_delta):
            eld_per_layer[(r.seed, r.layer_idx0)].append(float(r.abs_delta))
    tvd_per_layer: dict[tuple[int, int], list[float]] = defaultdict(list)
    for r in top1_dist_rows:
        if not (isinstance(r["tvd"], float) and np.isnan(r["tvd"])):
            tvd_per_layer[(r["seed"], r["layer_idx0"])].append(float(r["tvd"]))
    for row in layer_div_rows:
        key = (row["seed"], row["layer_idx0"])
        eld = eld_per_layer.get(key, [])
        row["mean_pairwise_abs_expected_lag_diff_minutes"] = float(np.mean(eld)) if eld else float("nan")
        tvd = tvd_per_layer.get(key, [])
        row["mean_pairwise_top1_tvd"] = float(np.mean(tvd)) if tvd else float("nan")
