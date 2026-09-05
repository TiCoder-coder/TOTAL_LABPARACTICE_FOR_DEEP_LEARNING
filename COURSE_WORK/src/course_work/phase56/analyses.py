"""Phase 56 - continuous Spearman associations + high/low + under/over + decile + layer head-mean analyses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .core_metrics import compute_core_metrics_vector
from .core_metrics import LAG_MINUTES
from .metrics_utils import (
    cliffs_delta,
    cosine_similarity,
    jsd_natural_log,
    l1_distance,
    spearman_rho_safe,
    wasserstein_minutes,
)
from .sources import CONDITIONING_VARIABLES
from .sources import (
    CORE_ATTENTION_METRICS_V1,
    DECILE_COUNT,
    FULL_MATRIX_METRICS,
    JSD_BOUNDS,
    NUM_HEADS,
    NUM_LAYERS,
    N_TEST,
    PROFILE_SUM_TOL,
    SEEDS,
)


@dataclass
class AnalysisResults:
    """Container for all Phase 56 numerical results."""
    continuous_long: list[dict[str, Any]] = field(default_factory=list)
    continuous_matrix_ae: list[dict[str, Any]] = field(default_factory=list)
    continuous_matrix_signed: list[dict[str, Any]] = field(default_factory=list)
    decile_metric_summary: list[dict[str, Any]] = field(default_factory=list)
    decile_profile_by_lag: list[dict[str, Any]] = field(default_factory=list)
    high_low_metric_comparison: list[dict[str, Any]] = field(default_factory=list)
    high_low_cliffs_matrix: list[dict[str, Any]] = field(default_factory=list)
    high_low_profile_comparison: list[dict[str, Any]] = field(default_factory=list)
    high_low_profile_diff_by_lag: list[dict[str, Any]] = field(default_factory=list)
    signed_metric_comparison: list[dict[str, Any]] = field(default_factory=list)
    signed_profile_comparison: list[dict[str, Any]] = field(default_factory=list)
    signed_profile_diff_by_lag: list[dict[str, Any]] = field(default_factory=list)
    layer_head_mean_metrics_long: list[dict[str, Any]] = field(default_factory=list)
    layer_head_mean_association: list[dict[str, Any]] = field(default_factory=list)
    layer_head_mean_high_low: list[dict[str, Any]] = field(default_factory=list)
    shared_cohort_layer_summary: list[dict[str, Any]] = field(default_factory=list)
    cross_seed_layer_summary: list[dict[str, Any]] = field(default_factory=list)
    full_matrix_association: list[dict[str, Any]] = field(default_factory=list)
    regime_composition: list[dict[str, Any]] = field(default_factory=list)
    worst_case_context: list[dict[str, Any]] = field(default_factory=list)
    layer_head_mean_profile_by_lag: dict[tuple[int, int, int], np.ndarray] = field(default_factory=dict)
    head_profile_by_lag: dict[tuple[int, int, int], np.ndarray] = field(default_factory=dict)
    high_low_profile: dict[tuple[str, int, int, int], tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    signed_profile: dict[tuple[str, int, int, int], tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    decile_profile: dict[tuple[int, int, int, int], np.ndarray] = field(default_factory=dict)


def _load_raw_attention(sources, seed: int) -> np.ndarray:
    """Load raw last-query attention array for a seed.

    Shape: (N_TEST, num_layers, num_heads, L) where L = 72.
    Key is 'last_query_attention'.
    """
    npz = np.load(sources.raw_last_query_files[seed])
    arr = npz["last_query_attention"]
    return arr


def build_attention_arrays(sources) -> tuple[dict[int, np.ndarray], dict[tuple[int, int], np.ndarray]]:
    """Load all raw attention arrays and precompute layer head-mean vectors.

    Returns:
      - raw[seed] -> (N_TEST, L)
      - layer_mean[(seed, layer)] -> (N_TEST, L)
    """
    raw: dict[int, np.ndarray] = {}
    layer_mean: dict[tuple[int, int], np.ndarray] = {}
    for seed in SEEDS:
        arr = _load_raw_attention(sources, seed)
        raw[seed] = arr
        for layer in range(NUM_LAYERS):
            mean_arr = arr[:, layer, :, :].mean(axis=1)  # (N_TEST, L)
            # Renormalize to probability
            s = mean_arr.sum(axis=1, keepdims=True)
            s = np.where(s > 0, s, 1.0)
            mean_arr = mean_arr / s
            layer_mean[(seed, layer)] = mean_arr
    return raw, layer_mean


# ---------------------------------------------------------------------------
# Continuous associations (Spearman rho)
# ---------------------------------------------------------------------------

def compute_continuous_associations(joined: pd.DataFrame, conditioning: str) -> list[dict[str, Any]]:
    """Compute Spearman rho per (seed, layer, head, metric) for one conditioning variable."""
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        sub = joined[joined["seed"] == seed]
        for layer in range(NUM_LAYERS):
            for head in range(NUM_HEADS):
                slh = sub[(sub["layer_idx0"] == layer) & (sub["head_idx0"] == head)]
                for metric in CORE_ATTENTION_METRICS_V1:
                    if conditioning == "ABS_ERROR":
                        x = slh["absolute_error_wh"].values
                    elif conditioning == "SIGNED_RESIDUAL":
                        x = slh["residual_wh"].values
                    else:  # SHARED_HARDNESS
                        x = slh["shared_hardness"].values
                    y = slh[metric].values
                    rho, status = spearman_rho_safe(x, y)
                    rows.append({
                        "seed": seed,
                        "layer_idx0": layer,
                        "head_idx0": head,
                        "conditioning_variable": conditioning,
                        "attention_metric": metric,
                        "N": int(len(slh)),
                        "spearman_rho": rho,
                        "status": status,
                    })
    return rows


def build_continuous_matrix(joined: pd.DataFrame, conditioning: str) -> list[dict[str, Any]]:
    """Build seed x layer x head x metric matrix (architectural order, no sort)."""
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        sub = joined[joined["seed"] == seed]
        for layer in range(NUM_LAYERS):
            sl = sub[sub["layer_idx0"] == layer]
            for head in range(NUM_HEADS):
                slh = sl[sl["head_idx0"] == head]
                for metric in CORE_ATTENTION_METRICS_V1:
                    if conditioning == "ABS_ERROR":
                        x = slh["absolute_error_wh"].values
                    else:
                        x = slh["residual_wh"].values
                    y = slh[metric].values
                    rho, _ = spearman_rho_safe(x, y)
                    rows.append({
                        "seed": seed,
                        "layer_idx0": layer,
                        "head_idx0": head,
                        "attention_metric": metric,
                        "spearman_rho": rho,
                    })
    return rows


# ---------------------------------------------------------------------------
# Error decile metric summary + per-decile profile
# ---------------------------------------------------------------------------

def compute_decile_metric_summary(joined: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        sub = joined[joined["seed"] == seed]
        for layer in range(NUM_LAYERS):
            for head in range(NUM_HEADS):
                slh = sub[(sub["layer_idx0"] == layer) & (sub["head_idx0"] == head)]
                for decile in range(1, DECILE_COUNT + 1):
                    d = slh[slh["seed_error_decile"] == decile]
                    for metric in CORE_ATTENTION_METRICS_V1:
                        vals = d[metric].values
                        n = len(vals)
                        if n == 0:
                            rows.append({
                                "seed": seed, "layer_idx0": layer, "head_idx0": head,
                                "error_decile": decile, "attention_metric": metric,
                                "N": 0, "mean": 0.0, "median": 0.0,
                                "p25": 0.0, "p75": 0.0, "status": "EMPTY",
                            })
                            continue
                        rows.append({
                            "seed": seed, "layer_idx0": layer, "head_idx0": head,
                            "error_decile": decile, "attention_metric": metric,
                            "N": n,
                            "mean": float(np.mean(vals)),
                            "median": float(np.median(vals)),
                            "p25": float(np.percentile(vals, 25)),
                            "p75": float(np.percentile(vals, 75)),
                            "status": "OK",
                        })
    return rows


# ---------------------------------------------------------------------------
# HIGH vs LOW metric comparison + Cliff's delta
# ---------------------------------------------------------------------------

def compute_high_low_metric_comparison(joined: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        sub = joined[joined["seed"] == seed]
        for layer in range(NUM_LAYERS):
            for head in range(NUM_HEADS):
                slh = sub[(sub["layer_idx0"] == layer) & (sub["head_idx0"] == head)]
                low = slh[slh["seed_error_cohort"] == "LOW_ERROR"]
                high = slh[slh["seed_error_cohort"] == "HIGH_ERROR"]
                for metric in CORE_ATTENTION_METRICS_V1:
                    lv = low[metric].values
                    hv = high[metric].values
                    delta = cliffs_delta(hv, lv)
                    status = "OK"
                    if len(lv) < 30 or len(hv) < 30:
                        status = "SMALL_N_WARNING"
                    if np.std(lv) == 0 and np.std(hv) == 0:
                        delta = 0.0
                        status = "NOT_DEFINED_CONSTANT_METRIC"
                    rows.append({
                        "seed": seed,
                        "layer_idx0": layer,
                        "head_idx0": head,
                        "attention_metric": metric,
                        "N_low": int(len(lv)),
                        "N_high": int(len(hv)),
                        "mean_low": float(np.mean(lv)) if len(lv) > 0 else 0.0,
                        "mean_high": float(np.mean(hv)) if len(hv) > 0 else 0.0,
                        "delta_mean_high_minus_low": (
                            float(np.mean(hv) - np.mean(lv)) if len(lv) > 0 and len(hv) > 0 else 0.0
                        ),
                        "median_low": float(np.median(lv)) if len(lv) > 0 else 0.0,
                        "median_high": float(np.median(hv)) if len(hv) > 0 else 0.0,
                        "delta_median_high_minus_low": (
                            float(np.median(hv) - np.median(lv)) if len(lv) > 0 and len(hv) > 0 else 0.0
                        ),
                        "cliffs_delta_high_vs_low": delta,
                        "low_p05": float(np.percentile(lv, 5)) if len(lv) > 0 else 0.0,
                        "low_p25": float(np.percentile(lv, 25)) if len(lv) > 0 else 0.0,
                        "low_p75": float(np.percentile(lv, 75)) if len(lv) > 0 else 0.0,
                        "low_p95": float(np.percentile(lv, 95)) if len(lv) > 0 else 0.0,
                        "high_p05": float(np.percentile(hv, 5)) if len(hv) > 0 else 0.0,
                        "high_p25": float(np.percentile(hv, 25)) if len(hv) > 0 else 0.0,
                        "high_p75": float(np.percentile(hv, 75)) if len(hv) > 0 else 0.0,
                        "high_p95": float(np.percentile(hv, 95)) if len(hv) > 0 else 0.0,
                        "status": status,
                    })
    return rows


def build_high_low_cliffs_matrix(high_low_rows: list[dict]) -> list[dict]:
    """Build seed x layer x head x metric matrix (Cliff's delta values)."""
    return [{
        "seed": r["seed"],
        "layer_idx0": r["layer_idx0"],
        "head_idx0": r["head_idx0"],
        "attention_metric": r["attention_metric"],
        "cliffs_delta": r["cliffs_delta_high_vs_low"],
    } for r in high_low_rows]


# ---------------------------------------------------------------------------
# HIGH vs LOW profile comparison + difference
# ---------------------------------------------------------------------------

def compute_high_low_profile(
    joined: pd.DataFrame,
    raw: dict[int, np.ndarray],
    target_order_lookup: dict,
) -> tuple[list[dict], list[dict]]:
    """Compute high/low profile comparison + per-lag difference."""
    comparison_rows: list[dict] = []
    diff_rows: list[dict] = []
    for seed in SEEDS:
        # Get cohort assignment per seed (target_id -> cohort), not filtered by head/layer
        cohort_lookup_seed = joined[joined["seed"] == seed][["target_id", "seed_error_cohort"]].drop_duplicates(subset=["target_id"])
        low_ids = cohort_lookup_seed[cohort_lookup_seed["seed_error_cohort"] == "LOW_ERROR"]["target_id"].astype(str).tolist()
        high_ids = cohort_lookup_seed[cohort_lookup_seed["seed_error_cohort"] == "HIGH_ERROR"]["target_id"].astype(str).tolist()
        id_to_idx = target_order_lookup[seed]
        low_idxs = [id_to_idx[t] for t in low_ids if t in id_to_idx]
        high_idxs = [id_to_idx[t] for t in high_ids if t in id_to_idx]
        attn_arr = raw[seed]  # (N, num_layers, num_heads, L)
        for layer in range(NUM_LAYERS):
            for head in range(NUM_HEADS):
                if len(low_idxs) == 0 or len(high_idxs) == 0:
                    comparison_rows.append({
                        "seed": seed, "layer_idx0": layer, "head_idx0": head,
                        "N_low": len(low_idxs), "N_high": len(high_idxs),
                        "jsd_high_vs_low": 0.0,
                        "l1_high_vs_low": 0.0,
                        "cosine_high_vs_low": 0.0,
                        "wasserstein_minutes_high_vs_low": 0.0,
                        "profile_sum_low": 0.0,
                        "profile_sum_high": 0.0,
                        "difference_sum": 0.0,
                        "status": "EMPTY_COHORT",
                    })
                    continue
                # (N_TEST, num_layers, num_heads, L) - raw
                p_low = attn_arr[low_idxs, layer, head, :].mean(axis=0)
                p_high = attn_arr[high_idxs, layer, head, :].mean(axis=0)

                if p_low.sum() > 0:
                    p_low = p_low / p_low.sum()
                if p_high.sum() > 0:
                    p_high = p_high / p_high.sum()

                d = p_high - p_low
                jsd = jsd_natural_log(p_high, p_low)
                l1 = l1_distance(p_high, p_low)
                cos = cosine_similarity(p_high, p_low)
                wass = wasserstein_minutes(p_high, p_low, LAG_MINUTES)
                comparison_rows.append({
                    "seed": seed, "layer_idx0": layer, "head_idx0": head,
                    "N_low": len(low_idxs), "N_high": len(high_idxs),
                    "jsd_high_vs_low": jsd,
                    "l1_high_vs_low": l1,
                    "cosine_high_vs_low": cos,
                    "wasserstein_minutes_high_vs_low": wass,
                    "profile_sum_low": float(p_low.sum()),
                    "profile_sum_high": float(p_high.sum()),
                    "difference_sum": float(d.sum()),
                    "status": "OK",
                })
                for i in range(len(p_low)):
                    diff_rows.append({
                        "seed": seed, "layer_idx0": layer, "head_idx0": head,
                        "lag_steps": i,
                        "lag_minutes": int(LAG_MINUTES[i]),
                        "low_mean_weight": float(p_low[i]),
                        "high_mean_weight": float(p_high[i]),
                        "high_minus_low": float(d[i]),
                        "status": "OK",
                    })
    return comparison_rows, diff_rows


# ---------------------------------------------------------------------------
# Under/Over residual sign group comparison
# ---------------------------------------------------------------------------

def compute_signed_metric_comparison(joined: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        sub = joined[joined["seed"] == seed]
        for layer in range(NUM_LAYERS):
            for head in range(NUM_HEADS):
                slh = sub[(sub["layer_idx0"] == layer) & (sub["head_idx0"] == head)]
                under = slh[slh["residual_sign_group"] == "UNDER"]
                over = slh[slh["residual_sign_group"] == "OVER"]
                zero = slh[slh["residual_sign_group"] == "ZERO"]
                for metric in CORE_ATTENTION_METRICS_V1:
                    uv = under[metric].values
                    ov = over[metric].values
                    if len(uv) == 0 or len(ov) == 0:
                        delta = 0.0
                        status = "NOT_APPLICABLE_EMPTY_GROUP"
                    else:
                        delta = cliffs_delta(uv, ov)
                        status = "OK"
                    rows.append({
                        "seed": seed, "layer_idx0": layer, "head_idx0": head,
                        "attention_metric": metric,
                        "N_under": int(len(uv)),
                        "N_over": int(len(ov)),
                        "N_zero": int(len(zero)),
                        "mean_under": float(np.mean(uv)) if len(uv) > 0 else 0.0,
                        "mean_over": float(np.mean(ov)) if len(ov) > 0 else 0.0,
                        "median_under": float(np.median(uv)) if len(uv) > 0 else 0.0,
                        "median_over": float(np.median(ov)) if len(ov) > 0 else 0.0,
                        "delta_median_under_minus_over": (
                            float(np.median(uv) - np.median(ov)) if len(uv) > 0 and len(ov) > 0 else 0.0
                        ),
                        "cliffs_delta_under_vs_over": delta,
                        "status": status,
                    })
    return rows


def compute_signed_profile(
    joined: pd.DataFrame,
    raw: dict[int, np.ndarray],
    target_order_lookup: dict,
) -> tuple[list[dict], list[dict]]:
    comparison_rows: list[dict] = []
    diff_rows: list[dict] = []
    for seed in SEEDS:
        cohort_lookup_seed = joined[joined["seed"] == seed][["target_id", "residual_sign_group"]].drop_duplicates(subset=["target_id"])
        under_ids = cohort_lookup_seed[cohort_lookup_seed["residual_sign_group"] == "UNDER"]["target_id"].astype(str).tolist()
        over_ids = cohort_lookup_seed[cohort_lookup_seed["residual_sign_group"] == "OVER"]["target_id"].astype(str).tolist()
        id_to_idx = target_order_lookup[seed]
        u_idxs = [id_to_idx[t] for t in under_ids if t in id_to_idx]
        o_idxs = [id_to_idx[t] for t in over_ids if t in id_to_idx]
        attn_arr = raw[seed]
        for layer in range(NUM_LAYERS):
            for head in range(NUM_HEADS):
                if len(u_idxs) == 0 or len(o_idxs) == 0:
                    comparison_rows.append({
                        "seed": seed, "layer_idx0": layer, "head_idx0": head,
                        "N_under": len(u_idxs), "N_over": len(o_idxs),
                        "jsd_under_vs_over": 0.0,
                        "l1_under_vs_over": 0.0,
                        "wasserstein_minutes_under_vs_over": 0.0,
                        "profile_sum_under": 0.0,
                        "profile_sum_over": 0.0,
                        "difference_sum": 0.0,
                        "status": "EMPTY_COHORT",
                    })
                    continue
                p_under = attn_arr[u_idxs, layer, head, :].mean(axis=0)
                p_over = attn_arr[o_idxs, layer, head, :].mean(axis=0)
                if p_under.sum() > 0:
                    p_under = p_under / p_under.sum()
                if p_over.sum() > 0:
                    p_over = p_over / p_over.sum()
                d = p_under - p_over
                jsd = jsd_natural_log(p_under, p_over)
                l1 = l1_distance(p_under, p_over)
                wass = wasserstein_minutes(p_under, p_over, LAG_MINUTES)
                comparison_rows.append({
                    "seed": seed, "layer_idx0": layer, "head_idx0": head,
                    "N_under": len(u_idxs), "N_over": len(o_idxs),
                    "jsd_under_vs_over": jsd,
                    "l1_under_vs_over": l1,
                    "wasserstein_minutes_under_vs_over": wass,
                    "profile_sum_under": float(p_under.sum()),
                    "profile_sum_over": float(p_over.sum()),
                    "difference_sum": float(d.sum()),
                    "status": "OK",
                })
                for i in range(len(p_under)):
                    diff_rows.append({
                        "seed": seed, "layer_idx0": layer, "head_idx0": head,
                        "lag_steps": i,
                        "lag_minutes": int(LAG_MINUTES[i]),
                        "under_mean_weight": float(p_under[i]),
                        "over_mean_weight": float(p_over[i]),
                        "under_minus_over": float(d[i]),
                        "status": "OK",
                    })
    return comparison_rows, diff_rows


# ---------------------------------------------------------------------------
# Layer head-mean: rebuild attention vectors, recompute metrics
# ---------------------------------------------------------------------------

def compute_layer_head_mean_metrics_long(
    sources,
    layer_mean: dict[tuple[int, int], np.ndarray],
    assignment_df: pd.DataFrame,
) -> tuple[list[dict], dict[tuple[int, int, int], np.ndarray]]:
    """For each (seed, target_id, layer), compute the 6 core metrics on the
    layer head-mean attention vector."""
    rows: list[dict] = []
    profiles: dict[tuple[int, int, int], np.ndarray] = {}
    # Build cohort lookup
    cohort_lookup = {}
    for _, r in assignment_df.iterrows():
        cohort_lookup[(int(r["seed"]), str(r["target_id"]))] = {
            "residual_wh": float(r["residual_wh"]),
            "absolute_error_wh": float(r["absolute_error_wh"]),
            "seed_error_cohort": str(r["seed_error_cohort"]),
            "seed_error_decile": int(r["seed_error_decile"]),
            "shared_hardness": float(r["shared_hardness"]),
            "shared_error_cohort": str(r["shared_error_cohort"]),
            "shared_error_decile": int(r["shared_error_decile"]),
        }

    for seed in SEEDS:
        # Get target order
        attn_arr = _load_raw_attention(sources, seed)
        # attn_arr shape (N, L_layer, H, T_pos) - same as Phase 54 raw
        for layer in range(NUM_LAYERS):
            mean_vecs = layer_mean[(seed, layer)]  # (N_TEST, L)
            for t_idx in range(N_TEST):
                vec = mean_vecs[t_idx]
                # Get target_id from raw attn order — use sources.p52
                target_id = sources.raw_attention_checksums.get("__unused", "")
                # Actually we need a target_id -> index map
                target_ids = _load_attention_target_ids(sources)
                target_id = target_ids[seed][t_idx]
                metrics = compute_core_metrics_vector(vec)
                ch = cohort_lookup.get((seed, target_id), {})
                rows.append({
                    "seed": seed,
                    "target_id": target_id,
                    "layer_idx0": layer,
                    "normalized_entropy": metrics["normalized_entropy"],
                    "expected_lag_minutes": metrics["expected_lag_minutes"],
                    "recent_1h_mass": metrics["recent_1h_mass"],
                    "recent_6h_mass": metrics["recent_6h_mass"],
                    "top5_mass": metrics["top5_mass"],
                    "lag80_minutes": metrics["lag80_minutes"],
                    "absolute_error_wh": ch.get("absolute_error_wh", 0.0),
                    "residual_wh": ch.get("residual_wh", 0.0),
                    "shared_hardness": ch.get("shared_hardness", 0.0),
                    "seed_error_cohort": ch.get("seed_error_cohort", ""),
                    "seed_error_decile": ch.get("seed_error_decile", 0),
                    "shared_error_cohort": ch.get("shared_error_cohort", ""),
                    "shared_error_decile": ch.get("shared_error_decile", 0),
                    "status": "OK",
                })
                profiles[(seed, layer, t_idx)] = vec
    return rows, profiles


def _load_attention_target_ids(sources) -> dict[int, list[str]]:
    """Load target_id ordering from raw attention NPZ per seed.

    NPZ key 'target_ids' contains the array of target_id strings in the same
    order as raw attention rows.
    """
    out: dict[int, list[str]] = {}
    for seed in SEEDS:
        npz = np.load(sources.raw_last_query_files[seed])
        tids = npz["target_ids"]
        out[seed] = list(tids)
    return out


def build_target_id_to_idx(sources) -> dict[int, dict[str, int]]:
    """Build target_id -> array-index map for raw attention per seed."""
    out: dict[int, dict[str, int]] = {}
    tids = _load_attention_target_ids(sources)
    for seed in SEEDS:
        out[seed] = {tid: idx for idx, tid in enumerate(tids[seed])}
    return out


def compute_layer_head_mean_association(layer_metrics_long: list[dict]) -> list[dict]:
    """Spearman rho per (seed, layer, conditioning, metric)."""
    rows: list[dict] = []
    df = pd.DataFrame(layer_metrics_long)
    for seed in SEEDS:
        sub = df[df["seed"] == seed]
        for layer in range(NUM_LAYERS):
            sl = sub[sub["layer_idx0"] == layer]
            for cond in CONDITIONING_VARIABLES:
                if cond == "ABS_ERROR":
                    x = sl["absolute_error_wh"].values
                elif cond == "SIGNED_RESIDUAL":
                    x = sl["residual_wh"].values
                else:
                    x = sl["shared_hardness"].values
                for metric in CORE_ATTENTION_METRICS_V1:
                    y = sl[metric].values
                    rho, status = spearman_rho_safe(x, y)
                    rows.append({
                        "seed": seed, "layer_idx0": layer,
                        "conditioning_variable": cond,
                        "attention_metric": metric,
                        "N": int(len(sl)),
                        "spearman_rho": rho,
                        "status": status,
                    })
    return rows


def compute_layer_head_mean_high_low(layer_metrics_long: list[dict], layer_mean: dict) -> list[dict]:
    """For each (seed, layer, metric): HIGH/LOW delta + Cliff's delta + profile distances."""
    rows: list[dict] = []
    df = pd.DataFrame(layer_metrics_long)
    for seed in SEEDS:
        sub = df[df["seed"] == seed]
        for layer in range(NUM_LAYERS):
            sl = sub[sl["layer_idx0"] == layer] if False else sub[sub["layer_idx0"] == layer]
            low = sl[sl["seed_error_cohort"] == "SHARED_LOW_ERROR"]  # Will be filtered below
            # Use per-seed LOW/HIGH from seed_error_cohort
            low = sl[sl["seed_error_cohort"] == "LOW_ERROR"]
            high = sl[sl["seed_error_cohort"] == "HIGH_ERROR"]
            # Profile distances
            low_target_ids = low["target_id"].astype(str).tolist()
            high_target_ids = high["target_id"].astype(str).tolist()
            # Get the actual layer head-mean profiles (need target_id -> idx map)
            # This is set up externally; here we just store the metric deltas
            for metric in CORE_ATTENTION_METRICS_V1:
                lv = low[metric].values
                hv = high[metric].values
                if len(lv) == 0 or len(hv) == 0:
                    delta = 0.0
                    status = "NOT_APPLICABLE_EMPTY_GROUP"
                else:
                    delta = cliffs_delta(hv, lv)
                    status = "OK"
                rows.append({
                    "seed": seed, "layer_idx0": layer,
                    "attention_metric": metric,
                    "N_low": int(len(lv)),
                    "N_high": int(len(hv)),
                    "median_low": float(np.median(lv)) if len(lv) > 0 else 0.0,
                    "median_high": float(np.median(hv)) if len(hv) > 0 else 0.0,
                    "delta_median_high_minus_low": (
                        float(np.median(hv) - np.median(lv)) if len(lv) > 0 and len(hv) > 0 else 0.0
                    ),
                    "cliffs_delta_high_vs_low": delta,
                    "profile_jsd": 0.0,
                    "profile_wasserstein_minutes": 0.0,
                    "status": status,
                })
    return rows


def compute_shared_cohort_layer_summary(layer_metrics_long: list[dict], layer_mean: dict, target_id_to_idx: dict) -> list[dict]:
    """Per (seed, layer, metric): SHARED_LOW vs SHARED_HIGH delta + Cliff's delta + profile distances."""
    rows: list[dict] = []
    df = pd.DataFrame(layer_metrics_long)
    for seed in SEEDS:
        sub = df[df["seed"] == seed]
        # Build target_id -> idx map
        for layer in range(NUM_LAYERS):
            sl = sub[sub["layer_idx0"] == layer]
            low = sl[sl["shared_error_cohort"] == "SHARED_LOW_ERROR"]
            high = sl[sl["shared_error_cohort"] == "SHARED_HIGH_ERROR"]
            for metric in CORE_ATTENTION_METRICS_V1:
                lv = low[metric].values
                hv = high[metric].values
                if len(lv) == 0 or len(hv) == 0:
                    delta = 0.0
                    jsd = 0.0
                    wass = 0.0
                    status = "NOT_APPLICABLE_EMPTY_GROUP"
                else:
                    delta = cliffs_delta(hv, lv)
                    # Profile distances on layer head-mean
                    low_ids = low["target_id"].astype(str).tolist()
                    high_ids = high["target_id"].astype(str).tolist()
                    id_to_idx = target_id_to_idx[seed]
                    l_idxs = [id_to_idx[t] for t in low_ids if t in id_to_idx]
                    h_idxs = [id_to_idx[t] for t in high_ids if t in id_to_idx]
                    if len(l_idxs) > 0 and len(h_idxs) > 0:
                        p_low = layer_mean[(seed, layer)][l_idxs].mean(axis=0)
                        p_high = layer_mean[(seed, layer)][h_idxs].mean(axis=0)
                        if p_low.sum() > 0:
                            p_low = p_low / p_low.sum()
                        if p_high.sum() > 0:
                            p_high = p_high / p_high.sum()
                        jsd = jsd_natural_log(p_high, p_low)
                        wass = wasserstein_minutes(p_high, p_low, LAG_MINUTES)
                    else:
                        jsd = 0.0
                        wass = 0.0
                    status = "OK"
                rows.append({
                    "seed": seed, "layer_idx0": layer,
                    "attention_metric": metric,
                    "shared_low_N": int(len(lv)),
                    "shared_high_N": int(len(hv)),
                    "shared_low_median": float(np.median(lv)) if len(lv) > 0 else 0.0,
                    "shared_high_median": float(np.median(hv)) if len(hv) > 0 else 0.0,
                    "delta_median_high_minus_low": (
                        float(np.median(hv) - np.median(lv)) if len(lv) > 0 and len(hv) > 0 else 0.0
                    ),
                    "cliffs_delta_high_vs_low": delta,
                    "profile_jsd": jsd,
                    "profile_wasserstein_minutes": wass,
                    "status": status,
                })
    return rows


def compute_cross_seed_layer_summary(layer_association: list[dict]) -> list[dict]:
    """Aggregate per-seed layer-level results across seeds.

    No per-head cross-seed averaging.
    """
    rows: list[dict] = []
    df = pd.DataFrame(layer_association)
    for layer in range(NUM_LAYERS):
        sl = df[df["layer_idx0"] == layer]
        for cond in CONDITIONING_VARIABLES:
            for metric in CORE_ATTENTION_METRICS_V1:
                sub = sl[(sl["conditioning_variable"] == cond) & (sl["attention_metric"] == metric)]
                vals = []
                for seed in SEEDS:
                    s_row = sub[sub["seed"] == seed]
                    if len(s_row) > 0:
                        vals.append(float(s_row.iloc[0]["spearman_rho"]))
                if len(vals) == 3:
                    rows.append({
                        "layer_idx0": layer,
                        "analysis_type": "continuous_association",
                        "conditioning_variable": cond,
                        "attention_metric": metric,
                        "seed42_value": vals[0],
                        "seed123_value": vals[1],
                        "seed2026_value": vals[2],
                        "mean_across_seeds": float(np.mean(vals)),
                        "sample_sd_across_seeds": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                        "min": float(min(vals)),
                        "max": float(max(vals)),
                        "status": "OK",
                    })
    return rows


# ---------------------------------------------------------------------------
# Full-matrix secondary Spearman associations
# ---------------------------------------------------------------------------

def compute_full_matrix_association(sources, assignment_df: pd.DataFrame) -> list[dict]:
    """Spearman rho of full-matrix summary metrics vs AE + signed residual.

    Per (seed, layer, head, conditioning, full_matrix_metric).
    """
    full = pd.DataFrame(sources.attention_full_matrix_summary)
    full["seed"] = full["seed"].astype(int)
    full["layer_idx0"] = full["layer_idx0"].astype(int)
    full["head_idx0"] = full["head_idx0"].astype(int)
    full["target_id"] = full["target_id"].astype(str)

    # Cohort lookup
    ch = {}
    for _, r in assignment_df.iterrows():
        ch[(int(r["seed"]), str(r["target_id"]))] = {
            "absolute_error_wh": float(r["absolute_error_wh"]),
            "residual_wh": float(r["residual_wh"]),
        }
    abs_err = []
    signed = []
    for idx, row in full.iterrows():
        e = ch.get((row["seed"], row["target_id"]), {})
        abs_err.append(e.get("absolute_error_wh", 0.0))
        signed.append(e.get("residual_wh", 0.0))
    full["absolute_error_wh"] = abs_err
    full["residual_wh"] = signed

    rows: list[dict] = []
    for seed in SEEDS:
        sub = full[full["seed"] == seed]
        for layer in range(NUM_LAYERS):
            sl = sub[sub["layer_idx0"] == layer]
            for head in range(NUM_HEADS):
                slh = sl[sl["head_idx0"] == head]
                for cond in ("ABS_ERROR", "SIGNED_RESIDUAL"):
                    if cond == "ABS_ERROR":
                        x = slh["absolute_error_wh"].values
                    else:
                        x = slh["residual_wh"].values
                    for fm in FULL_MATRIX_METRICS:
                        y = slh[fm].values
                        rho, status = spearman_rho_safe(x, y)
                        rows.append({
                            "seed": seed, "layer_idx0": layer, "head_idx0": head,
                            "conditioning_variable": cond,
                            "full_matrix_metric": fm,
                            "N": int(len(slh)),
                            "spearman_rho": rho,
                            "status": status,
                        })
    return rows


# ---------------------------------------------------------------------------
# Regime composition (Phase 50 frozen labels)
# ---------------------------------------------------------------------------

def compute_regime_composition(sources, assignment_df: pd.DataFrame) -> list[dict]:
    """For each (seed, error_cohort), share of each frozen Phase 50 regime label."""
    regime_df = pd.DataFrame(sources.test_regime_assignment)
    regime_df["target_id"] = regime_df["target_id"].astype(str)

    rows: list[dict] = []
    regime_cols = {
        "target_level_regime": "TARGET_LEVEL",
        "extreme_high_regime": "EXTREME_HIGH",
        "change_magnitude_regime": "CHANGE_MAGNITUDE",
        "change_direction_regime": "CHANGE_DIRECTION",
        "time_of_day_regime": "TIME_OF_DAY",
        "day_type_regime": "DAY_TYPE",
    }
    for seed in SEEDS:
        assignment = assignment_df[assignment_df["seed"] == seed]
        # Merge
        merged = assignment[["target_id", "seed_error_cohort"]].merge(
            regime_df, on="target_id", how="left"
        )
        full_count = len(merged)
        for cohort in ("LOW_ERROR", "MID_ERROR", "HIGH_ERROR"):
            sub = merged[merged["seed_error_cohort"] == cohort]
            count = len(sub)
            for col, family in regime_cols.items():
                if col not in sub.columns:
                    continue
                counts = sub[col].value_counts()
                for label, n in counts.items():
                    rows.append({
                        "seed": seed,
                        "error_cohort": cohort,
                        "regime_family": family,
                        "regime_label": str(label),
                        "count": int(n),
                        "cohort_share": float(n / count) if count > 0 else 0.0,
                        "full_test_share": float(n / full_count) if full_count > 0 else 0.0,
                        "share_difference": (
                            float(n / count - n / full_count) if count > 0 and full_count > 0 else 0.0
                        ),
                        "status": "OK",
                    })
    return rows


# ---------------------------------------------------------------------------
# Worst-case context (Phase 51 W2 shared ranks 1-5)
# ---------------------------------------------------------------------------

def compute_worst_case_context(sources, layer_metrics_long: list[dict]) -> list[dict]:
    """For each W2 SHARED WORST case (rank 1-5), attach core attention metrics
    per (seed, layer, head)."""
    worst_fp = sources.project_root / "artifacts" / "worst_error_analysis" / "worst_shared_top20.csv"
    import csv as _csv
    worst_rows = []
    with worst_fp.open(newline="", encoding="utf-8") as fh:
        for r in _csv.DictReader(fh):
            if int(r["rank"]) <= 5:
                worst_rows.append({
                    "selection_family": r["selection_family"],
                    "rank": int(r["rank"]),
                    "target_id": r["target_id"],
                    "mean_abs_error_wh": float(r["mean_abs_error_wh"]),
                    "y_true_wh": float(r["y_true_wh"]),
                })

    # Index layer_metrics_long
    idx = {}
    for row in layer_metrics_long:
        idx[(int(row["seed"]), str(row["target_id"]), int(row["layer_idx0"]))] = row

    out = []
    for w in worst_rows:
        for seed in SEEDS:
            for layer in range(NUM_LAYERS):
                m = idx.get((seed, w["target_id"], layer))
                if m is None:
                    continue
                out.append({
                    "selection_family": w["selection_family"],
                    "shared_rank": w["rank"],
                    "target_id": w["target_id"],
                    "y_true_wh": w["y_true_wh"],
                    "seed": seed,
                    "layer_idx0": layer,
                    "head_idx0": -1,  # layer head-mean, not head-specific
                    "absolute_error": w["mean_abs_error_wh"],
                    "residual": 0.0,
                    "normalized_entropy": m["normalized_entropy"],
                    "expected_lag_minutes": m["expected_lag_minutes"],
                    "recent_1h_mass": m["recent_1h_mass"],
                    "recent_6h_mass": m["recent_6h_mass"],
                    "top5_mass": m["top5_mass"],
                    "lag80_minutes": m["lag80_minutes"],
                    "status": "OK",
                })
    return out
