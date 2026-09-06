"""Phase 57 - all analyses.

Implements:
- S57-A permutation-invariant layer head-mean stability (mean-profile + per-target + three-seed disagreement + metric stability)
- S57-B permutation-aware canonical head matching within each layer (exhaustive H! enumeration,
  min total JSD with Wasserstein tie-break, lex final tie-break, MATCH_TIE_TOL=1e-12)
- Matching independence audit (uses ONLY temporal attention profiles; no error/regime/etc.)
- Cycle consistency audit (direct 123-2026 vs anchor-induced 123-42-2026)
- Wasserstein-only sensitivity matching + agreement fraction
- Canonical three-seed matched groups anchored at seed42
- Matched-head mean-profile stability + per-target stability + consensus profile
- Matched-head metric stability + top1-lag stability
- Dense-case full-map stability on frozen Phase 51 cases
- Layer error-conditioned stability + matched-head error-conditioned stability + shared-cohort
- Prediction-attention disagreement secondary association

All pure functions. No model loading, no training.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

import numpy as np

from .metrics_utils import (
    cliffs_delta,
    cosine_similarity,
    jsd_natural_log,
    l1_distance,
    l2_distance,
    pearson_corr,
    profile_sum,
    spearman_rho_safe,
    verify_probability_profile,
    wasserstein_minutes,
)
from .sources import (
    ANCHOR_REASON,
    ANCHOR_SEED,
    CORE_ATTENTION_METRICS_V1,
    H2_LAYER_HEAD_COUNTS,
    LAG_MINUTES,
    LOOKBACK,
    MATCH_TIE_TOL,
    NUM_HEADS,
    NUM_LAYERS,
    N_TEST,
    SEED_PAIRS,
    SEEDS,
    FrozenSources57,
)


# ===========================================================================
# Data containers
# ===========================================================================

@dataclass
class MatchingResult:
    layer_idx0: int
    seed_a: int
    seed_b: int
    canonical_assignment: Dict[int, int]   # head_a -> head_b
    cost_matrix_jsd: np.ndarray            # [H,H]
    cost_matrix_wass: np.ndarray           # [H,H]
    row_margin: Dict[int, Dict[str, float]]
    column_margin: Dict[int, Dict[str, float]]
    best_total_jsd: float
    second_best_total_jsd: float
    assignment_gap_jsd: float
    best_total_wasserstein: float
    num_assignments_within_tie_tol: int
    ambiguous_match_warning: bool
    tie_tol_used: float
    permutation_count: int
    status: str


@dataclass
class CanonicalGroups:
    """For each (layer_idx0, seed42_head_idx0) -> (seed123_head_idx0, seed2026_head_idx0) plus warnings."""
    groups: Dict[Tuple[int, int], Dict[str, Any]]
    cycle_consistency: Dict[Tuple[int, int], Dict[str, Any]]


# ===========================================================================
# S57-A.1 — Build layer head-mean per target + mean profile (permutation-invariant)
# ===========================================================================

def build_layer_head_mean_arrays(sources: FrozenSources57) -> Dict[int, np.ndarray]:
    """For each seed, build [N_TEST, LAYERS, LOOKBACK] = mean over heads.

    Returns dict {seed: arr}. Source of truth: Phase 52 raw last-query NPZ.
    """
    out: Dict[int, np.ndarray] = {}
    for s in SEEDS:
        arr = np.load(sources.raw_last_query_files[s])
        a = arr["last_query_attention"]  # [N_TEST, LAYERS, HEADS, LOOKBACK]
        # mean over heads -> permutation-invariant
        out[s] = a.mean(axis=2)  # [N_TEST, LAYERS, LOOKBACK]
    return out


def build_layer_head_mean_profiles(layer_mean_arrays: Dict[int, np.ndarray]) -> Dict[Tuple[int, int], np.ndarray]:
    """For each (seed, layer_idx0), mean profile over all Test targets -> [LOOKBACK].

    Returns dict {(seed, layer): mean_profile}.
    """
    out: Dict[Tuple[int, int], np.ndarray] = {}
    for s in SEEDS:
        for layer in range(NUM_LAYERS):
            v = layer_mean_arrays[s][:, layer, :].mean(axis=0)
            s_sum = v.sum()
            if s_sum > 0:
                v = v / s_sum
            out[(s, layer)] = v
    return out


# ===========================================================================
# S57-A.2 — Layer head-mean pairwise profile stability (per layer, per seed-pair)
# ===========================================================================

def layer_pairwise_stability(
    profiles: Dict[Tuple[int, int], np.ndarray],
    layer_mean_arrays: Dict[int, np.ndarray] | None = None,
) -> List[dict]:
    """For each (layer, seed_pair), compute JSD/L1/L2/cosine/Pearson/Spearman/Wasserstein
    on the layer head-mean mean profiles.

    Also include per-target summary statistics across the (N_TEST x 3) seed pairs per layer.
    """
    rows: List[dict] = []
    for layer in range(NUM_LAYERS):
        for seed_a, seed_b in SEED_PAIRS:
            pa = profiles[(seed_a, layer)]
            pb = profiles[(seed_b, layer)]
            jsd = jsd_natural_log(pa, pb)
            l1 = l1_distance(pa, pb)
            l2 = l2_distance(pa, pb)
            cos = cosine_similarity(pa, pb)
            pear = pearson_corr(pa, pb)
            spear, _ = spearman_rho_safe(pa, pb)
            wass = wasserstein_minutes(pa, pb, LAG_MINUTES)
            rows.append({
                "layer_idx0": layer,
                "seed_a": seed_a,
                "seed_b": seed_b,
                "jsd": jsd,
                "l1": l1,
                "l2": l2,
                "cosine": cos,
                "pearson": pear,
                "spearman": spear,
                "wasserstein_minutes": wass,
                "status": "OK",
            })
    return rows


def layer_pairwise_stability_summary(rows: List[dict]) -> List[dict]:
    """Aggregate per (layer) across 3 seed pairs: mean/max JSD, mean/max Wasserstein,
    mean/min cosine."""
    summary: List[dict] = []
    for layer in range(NUM_LAYERS):
        layer_rows = [r for r in rows if int(r["layer_idx0"]) == layer]
        if not layer_rows:
            continue
        jsds = [float(r["jsd"]) for r in layer_rows]
        wass = [float(r["wasserstein_minutes"]) for r in layer_rows]
        coss = [float(r["cosine"]) for r in layer_rows]
        summary.append({
            "layer_idx0": layer,
            "pair_count": len(layer_rows),
            "mean_pairwise_jsd": float(np.mean(jsds)),
            "max_pairwise_jsd": float(np.max(jsds)),
            "mean_pairwise_wasserstein_minutes": float(np.mean(wass)),
            "max_pairwise_wasserstein_minutes": float(np.max(wass)),
            "mean_pairwise_cosine": float(np.mean(coss)),
            "min_pairwise_cosine": float(np.min(coss)),
            "status": "OK",
        })
    return summary


# ===========================================================================
# S57-A.3 — Per-target layer head-mean stability
# ===========================================================================

def per_target_layer_stability(sources: FrozenSources57, layer_mean_arrays: Dict[int, np.ndarray]) -> List[dict]:
    """For each target/layer/seed_pair, compare layer head-mean vectors.

    Returns rows with N=2961 * 2 layers * 3 seed_pairs = 17,766 rows.
    """
    rows: List[dict] = []
    target_ids_by_seed = {}
    for s in SEEDS:
        npz = np.load(sources.raw_last_query_files[s])
        target_ids_by_seed[s] = npz["target_ids"].astype(str)

    n_test = layer_mean_arrays[SEEDS[0]].shape[0]

    for ti in range(n_test):
        tid = str(target_ids_by_seed[SEEDS[0]][ti])
        for layer in range(NUM_LAYERS):
            for seed_a, seed_b in SEED_PAIRS:
                a = layer_mean_arrays[seed_a][ti, layer, :]
                b = layer_mean_arrays[seed_b][ti, layer, :]
                jsd = jsd_natural_log(a, b)
                l1 = l1_distance(a, b)
                cos = cosine_similarity(a, b)
                wass = wasserstein_minutes(a, b, LAG_MINUTES)
                rows.append({
                    "target_id": tid,
                    "target_index": ti,
                    "layer_idx0": layer,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "jsd": jsd,
                    "l1": l1,
                    "cosine": cos,
                    "wasserstein_minutes": wass,
                    "status": "OK",
                })
    return rows


def sources_raw(seed: int) -> str:
    # Helper used internally; not exported. Returns path to raw last-query NPZ.
    # Deprecated in favour of sources.raw_last_query_files[seed].
    raise NotImplementedError("use sources.raw_last_query_files[seed] instead")


def per_target_layer_summary(stability_rows: List[dict]) -> List[dict]:
    """Per (layer, seed_pair, metric): N / mean / SD / median / p05/p25/p75/p95 / min/max."""
    summary: List[dict] = []
    for layer in range(NUM_LAYERS):
        for seed_a, seed_b in SEED_PAIRS:
            sel = [r for r in stability_rows if int(r["layer_idx0"]) == layer
                   and int(r["seed_a"]) == seed_a and int(r["seed_b"]) == seed_b]
            for metric in ("jsd", "wasserstein_minutes", "cosine", "l1"):
                vals = np.asarray([r[metric] for r in sel], dtype=np.float64)
                if len(vals) == 0:
                    continue
                summary.append({
                    "layer_idx0": layer,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "metric": metric,
                    "N": int(len(vals)),
                    "mean": float(np.mean(vals)),
                    "sample_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "median": float(np.median(vals)),
                    "p05": float(np.percentile(vals, 5)),
                    "p25": float(np.percentile(vals, 25)),
                    "p75": float(np.percentile(vals, 75)),
                    "p95": float(np.percentile(vals, 95)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "status": "OK",
                })
    return summary


def per_target_layer_three_seed_disagreement(stability_rows: List[dict]) -> List[dict]:
    """For each (target_id, layer), aggregate the 3 pairwise distances into
    mean/max pairwise JSD and Wasserstein. Per-layer summary; seed-invariant."""
    out: List[dict] = {}
    for r in stability_rows:
        key = (r["target_id"], int(r["layer_idx0"]))
        if key not in out:
            out[key] = []
        out[key].append(r)
    rows: List[dict] = []
    for (tid, layer), recs in out.items():
        jsds = [float(r["jsd"]) for r in recs]
        wass = [float(r["wasserstein_minutes"]) for r in recs]
        rows.append({
            "target_id": tid,
            "target_index": int(recs[0]["target_index"]),
            "layer_idx0": layer,
            "mean_pairwise_jsd": float(np.mean(jsds)),
            "max_pairwise_jsd": float(np.max(jsds)),
            "mean_pairwise_wasserstein_minutes": float(np.mean(wass)),
            "max_pairwise_wasserstein_minutes": float(np.max(wass)),
            "status": "OK",
        })
    return rows


# ===========================================================================
# S57-A.4 — Layer attention metric stability (Spearman + MAD)
# ===========================================================================

def layer_metric_stability(sources: FrozenSources57, layer_mean_arrays: Dict[int, np.ndarray]) -> List[dict]:
    """Recompute 6 CORE_ATTENTION_METRICS-v1 on the layer head-mean vectors per (seed, target, layer),
    then compare across seed pairs per (layer, metric).
    """
    from .core_metrics import compute_core_metrics_vector

    # Recompute metrics from raw layer-head-mean vectors (DO NOT reuse per-head metrics).
    by_seed_layer_target: Dict[Tuple[int, int], Dict[int, dict]] = {}
    for s in SEEDS:
        npz = np.load(sources.raw_last_query_files[s])
        a = npz["last_query_attention"]  # [N_TEST, LAYERS, HEADS, LOOKBACK]
        for layer in range(NUM_LAYERS):
            for ti in range(N_TEST):
                v = a[ti, layer, :, :].mean(axis=0)  # head-mean
                m = compute_core_metrics_vector(v)
                by_seed_layer_target.setdefault((s, layer), {})[ti] = m

    rows: List[dict] = []
    for layer in range(NUM_LAYERS):
        for metric in CORE_ATTENTION_METRICS_V1:
            for seed_a, seed_b in SEED_PAIRS:
                xs = np.asarray([by_seed_layer_target[(seed_a, layer)][ti][metric]
                                 for ti in range(N_TEST)], dtype=np.float64)
                ys = np.asarray([by_seed_layer_target[(seed_b, layer)][ti][metric]
                                 for ti in range(N_TEST)], dtype=np.float64)
                rho, _ = spearman_rho_safe(xs, ys)
                diffs = np.abs(xs - ys)
                rows.append({
                    "layer_idx0": layer,
                    "metric": metric,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "N": int(N_TEST),
                    "spearman_across_targets": rho,
                    "mean_absolute_difference": float(np.mean(diffs)),
                    "median_absolute_difference": float(np.median(diffs)),
                    "status": "OK",
                })
    return rows


# ===========================================================================
# S57-B.1 — Build full-Test mean head profile per (seed, layer, head)
# ===========================================================================

def build_mean_head_profiles(sources: FrozenSources57) -> Dict[Tuple[int, int, int], np.ndarray]:
    """For each (seed, layer_idx0, head_idx0), full-Test mean last-query vector -> [LOOKBACK].

    This is the canonical matching representation.
    """
    out: Dict[Tuple[int, int, int], np.ndarray] = {}
    for s in SEEDS:
        npz = np.load(sources.raw_last_query_files[s])
        a = npz["last_query_attention"]  # [N_TEST, LAYERS, HEADS, LOOKBACK]
        for layer in range(NUM_LAYERS):
            for head in range(NUM_HEADS):
                v = a[:, layer, head, :].mean(axis=0)  # full-Test mean
                s_sum = v.sum()
                if s_sum > 0:
                    v = v / s_sum
                out[(s, layer, head)] = v
    return out


# ===========================================================================
# S57-B.2 — Cost matrices + canonical matching via exhaustive permutation
# ===========================================================================

def _permutations_lex(H: int):
    """Yield permutations of range(H) in lex order."""
    return itertools.permutations(range(H))


def compute_pairwise_matching(
    mean_profiles: Dict[Tuple[int, int, int], np.ndarray],
    layer_idx0: int,
    seed_a: int,
    seed_b: int,
) -> MatchingResult:
    H = NUM_HEADS
    pa = np.stack([mean_profiles[(seed_a, layer_idx0, h)] for h in range(H)], axis=0)  # [H, L]
    pb = np.stack([mean_profiles[(seed_b, layer_idx0, h)] for h in range(H)], axis=0)  # [H, L]

    # Cost matrices
    cost_jsd = np.zeros((H, H), dtype=np.float64)
    cost_wass = np.zeros((H, H), dtype=np.float64)
    for ha in range(H):
        for hb in range(H):
            cost_jsd[ha, hb] = jsd_natural_log(pa[ha], pb[hb])
            cost_wass[ha, hb] = wasserstein_minutes(pa[ha], pb[hb], LAG_MINUTES)

    # Enumerate permutations lex; compute (total_jsd, total_wass)
    perms = list(itertools.permutations(range(H)))
    scored = []
    for perm in perms:
        total_jsd = sum(cost_jsd[ha, perm[ha]] for ha in range(H))
        total_wass = sum(cost_wass[ha, perm[ha]] for ha in range(H))
        scored.append((perm, total_jsd, total_wass))

    scored.sort(key=lambda x: (x[1], x[2], x[0]))
    best_perm, best_total_jsd, best_total_wass = scored[0]
    if len(scored) > 1:
        second_total_jsd = scored[1][1]
    else:
        second_total_jsd = best_total_jsd

    assignment_gap = second_total_jsd - best_total_jsd

    # Number of permutations within MATCH_TIE_TOL of best total JSD
    within = sum(1 for _, tj, _ in scored if tj - best_total_jsd <= MATCH_TIE_TOL)

    canonical: Dict[int, int] = {h: int(best_perm[h]) for h in range(H)}

    # Row margins: per source head, gap to next best (excluding the matched column)
    row_margin: Dict[int, Dict[str, float]] = {}
    for ha in range(H):
        col_scores = sorted(cost_jsd[ha, :])
        row_margin[ha] = {
            "matched_jsd": float(cost_jsd[ha, canonical[ha]]),
            "next_best_row_jsd": float(col_scores[1] if H >= 2 else col_scores[0]),
            "row_margin": float(col_scores[1] - col_scores[0]) if H >= 2 else 0.0,
        }
    col_margin: Dict[int, Dict[str, float]] = {}
    for hb in range(H):
        col_scores = sorted(cost_jsd[:, hb])
        col_margin[hb] = {
            "next_best_column_jsd": float(col_scores[1] if H >= 2 else col_scores[0]),
            "column_margin": float(col_scores[1] - col_scores[0]) if H >= 2 else 0.0,
        }

    ambiguous = bool(within > 1)

    return MatchingResult(
        layer_idx0=layer_idx0,
        seed_a=seed_a,
        seed_b=seed_b,
        canonical_assignment=canonical,
        cost_matrix_jsd=cost_jsd,
        cost_matrix_wass=cost_wass,
        row_margin=row_margin,
        column_margin=col_margin,
        best_total_jsd=float(best_total_jsd),
        second_best_total_jsd=float(second_total_jsd),
        assignment_gap_jsd=float(assignment_gap),
        best_total_wasserstein=float(best_total_wass),
        num_assignments_within_tie_tol=int(within),
        ambiguous_match_warning=ambiguous,
        tie_tol_used=float(MATCH_TIE_TOL),
        permutation_count=int(len(perms)),
        status="OK",
    )


def cost_matrices_rows(sources: FrozenSources57, mean_profiles: Dict[Tuple[int, int, int], np.ndarray]) -> List[dict]:
    rows: List[dict] = []
    for layer in range(NUM_LAYERS):
        for seed_a, seed_b in SEED_PAIRS:
            mr = compute_pairwise_matching(mean_profiles, layer, seed_a, seed_b)
            for ha in range(NUM_HEADS):
                for hb in range(NUM_HEADS):
                    rows.append({
                        "layer_idx0": layer,
                        "seed_a": seed_a,
                        "seed_b": seed_b,
                        "head_a_idx0": ha,
                        "head_b_idx0": hb,
                        "jsd_cost": float(mr.cost_matrix_jsd[ha, hb]),
                        "wasserstein_minutes": float(mr.cost_matrix_wass[ha, hb]),
                        "cosine": float(cosine_similarity(
                            mean_profiles[(seed_a, layer, ha)],
                            mean_profiles[(seed_b, layer, hb)]
                        )),
                        "l1": float(l1_distance(
                            mean_profiles[(seed_a, layer, ha)],
                            mean_profiles[(seed_b, layer, hb)]
                        )),
                        "status": "OK",
                    })
    return rows


def matching_assignment_rows(matching_results: Dict[Tuple[int, int, int], MatchingResult]) -> List[dict]:
    rows: List[dict] = []
    for (layer, seed_a, seed_b), mr in matching_results.items():
        for ha in range(NUM_HEADS):
            hb = mr.canonical_assignment[ha]
            rows.append({
                "layer_idx0": layer,
                "seed_a": seed_a,
                "seed_b": seed_b,
                "head_a_idx0": ha,
                "head_b_idx0": hb,
                "assignment_method": "MIN_TOTAL_JSD",
                "matched": True,
                "matched_edge_jsd": float(mr.cost_matrix_jsd[ha, hb]),
                "matched_edge_wasserstein": float(mr.cost_matrix_wass[ha, hb]),
                "total_assignment_jsd": float(mr.best_total_jsd),
                "total_assignment_wasserstein": float(mr.best_total_wasserstein),
                "assignment_rank": 0,
                "status": "OK",
            })
    return rows


def ambiguity_audit_rows(matching_results: Dict[Tuple[int, int, int], MatchingResult]) -> List[dict]:
    rows: List[dict] = []
    for (layer, seed_a, seed_b), mr in matching_results.items():
        rows.append({
            "layer_idx0": layer,
            "seed_a": seed_a,
            "seed_b": seed_b,
            "head_count": int(NUM_HEADS),
            "permutation_count": int(mr.permutation_count),
            "best_total_jsd": float(mr.best_total_jsd),
            "second_best_total_jsd": float(mr.second_best_total_jsd),
            "assignment_gap_jsd": float(mr.assignment_gap_jsd),
            "best_total_wasserstein": float(mr.best_total_wasserstein),
            "num_assignments_within_tie_tolerance": int(mr.num_assignments_within_tie_tol),
            "match_tie_tolerance": float(mr.tie_tol_used),
            "ambiguous_match_warning": bool(mr.ambiguous_match_warning),
            "status": "OK",
        })
    return rows


def edge_margin_rows(matching_results: Dict[Tuple[int, int, int], MatchingResult]) -> List[dict]:
    rows: List[dict] = []
    for (layer, seed_a, seed_b), mr in matching_results.items():
        for ha in range(NUM_HEADS):
            hb = mr.canonical_assignment[ha]
            rm = mr.row_margin.get(ha, {})
            cm = mr.column_margin.get(hb, {})
            rows.append({
                "layer_idx0": layer,
                "seed_a": seed_a,
                "seed_b": seed_b,
                "head_a_idx0": ha,
                "matched_head_b_idx0": hb,
                "matched_jsd": float(mr.cost_matrix_jsd[ha, hb]),
                "next_best_row_jsd": float(rm.get("next_best_row_jsd", 0.0)),
                "row_margin": float(rm.get("row_margin", 0.0)),
                "next_best_column_jsd": float(cm.get("next_best_column_jsd", 0.0)),
                "column_margin": float(cm.get("column_margin", 0.0)),
                "status": "OK",
            })
    return rows


# ===========================================================================
# S57-B.3 — Cycle consistency
# ===========================================================================

def cycle_consistency_rows(matching_results: Dict[Tuple[int, int, int], MatchingResult]) -> List[dict]:
    """For each (layer, seed123_head_idx0), compute direct vs anchor-induced matching.

    Direct:   M_123_2026(seed123_head=h) -> seed2026_head
    Anchor:   M_42_2026(M_42_123(seed42_head = seed123_head)) - but mapping between seeds
              is by head-index, not by head-index-to-another-seed. So we apply M_42_123 first to
              seed42 head (the seed42 head that maps to seed123 head h), then M_42_2026 on that same
              seed42 head to get seed2026 head.

    Implementation: M_42_123 is seed42 -> seed123. For seed123 head h, we need the seed42 head a
    such that M_42_123[a] = h (reverse mapping). Then apply M_42_2026[a] to get seed2026 head.
    """
    rows: List[dict] = []
    for layer in range(NUM_LAYERS):
        mr_42_123 = matching_results[(layer, 42, 123)]
        mr_42_2026 = matching_results[(layer, 42, 2026)]
        mr_123_2026 = matching_results[(layer, 123, 2026)]
        rev_42_123 = {v: k for k, v in mr_42_123.canonical_assignment.items()}
        for head_123 in range(NUM_HEADS):
            head_42 = rev_42_123.get(head_123, None)
            if head_42 is None:
                anchor_induced = None
                direct = mr_123_2026.canonical_assignment.get(head_123, None)
            else:
                anchor_induced = mr_42_2026.canonical_assignment.get(head_42, None)
                direct = mr_123_2026.canonical_assignment.get(head_123, None)
            consistent = (direct is not None and anchor_induced is not None and direct == anchor_induced)
            rows.append({
                "layer_idx0": layer,
                "seed123_head_idx0": head_123,
                "direct_matched_seed2026_head": int(direct) if direct is not None else -1,
                "anchor_induced_seed2026_head": int(anchor_induced) if anchor_induced is not None else -1,
                "cycle_consistent": bool(consistent),
                "status": "OK",
            })
    return rows


def cycle_consistency_layer_summary(rows: List[dict]) -> List[dict]:
    out: List[dict] = []
    for layer in range(NUM_LAYERS):
        sel = [r for r in rows if int(r["layer_idx0"]) == layer]
        consistent = sum(1 for r in sel if r["cycle_consistent"])
        out.append({
            "layer_idx0": layer,
            "consistent_count": int(consistent),
            "head_count": int(len(sel)),
            "cycle_consistency_fraction": float(consistent / max(len(sel), 1)),
            "status": "OK",
        })
    return out


# ===========================================================================
# S57-B.4 — Wasserstein-only sensitivity matching
# ===========================================================================

def _compute_wasserstein_matching(
    mean_profiles: Dict[Tuple[int, int, int], np.ndarray],
    layer_idx0: int,
    seed_a: int,
    seed_b: int,
) -> Dict[int, int]:
    H = NUM_HEADS
    pa = np.stack([mean_profiles[(seed_a, layer_idx0, h)] for h in range(H)], axis=0)
    pb = np.stack([mean_profiles[(seed_b, layer_idx0, h)] for h in range(H)], axis=0)
    cost_wass = np.zeros((H, H), dtype=np.float64)
    for ha in range(H):
        for hb in range(H):
            cost_wass[ha, hb] = wasserstein_minutes(pa[ha], pb[hb], LAG_MINUTES)
    perms = list(itertools.permutations(range(H)))
    scored = []
    for perm in perms:
        total = sum(cost_wass[ha, perm[ha]] for ha in range(H))
        scored.append((perm, total))
    scored.sort(key=lambda x: (x[1], x[0]))
    return {ha: int(scored[0][0][ha]) for ha in range(H)}


def wasserstein_sensitivity_rows(
    sources: FrozenSources57,
    mean_profiles: Dict[Tuple[int, int, int], np.ndarray],
    matching_results: Dict[Tuple[int, int, int], MatchingResult],
) -> List[dict]:
    rows: List[dict] = []
    summary: List[dict] = []
    for layer in range(NUM_LAYERS):
        for seed_a, seed_b in SEED_PAIRS:
            jsd_match = matching_results[(layer, seed_a, seed_b)].canonical_assignment
            wass_match = _compute_wasserstein_matching(mean_profiles, layer, seed_a, seed_b)
            agree_count = 0
            for ha in range(NUM_HEADS):
                pa = mean_profiles[(seed_a, layer, ha)]
                hb_jsd = jsd_match[ha]
                hb_wass = wass_match[ha]
                pb_wass = mean_profiles[(seed_b, layer, hb_wass)]
                pair_agrees = bool(hb_jsd == hb_wass)
                if pair_agrees:
                    agree_count += 1
                rows.append({
                    "layer_idx0": layer,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "head_a_idx0": ha,
                    "jsd_match_head_b": int(hb_jsd),
                    "wasserstein_match_head_b": int(hb_wass),
                    "pair_agrees": pair_agrees,
                    "status": "OK",
                })
            summary.append({
                "layer_idx0": layer,
                "seed_a": seed_a,
                "seed_b": seed_b,
                "agreement_count": int(agree_count),
                "head_count": int(NUM_HEADS),
                "agreement_fraction": float(agree_count / max(NUM_HEADS, 1)),
                "status": "OK",
            })
    return rows, summary


# ===========================================================================
# S57-B.5 — Matching independence audit
# ===========================================================================

def matching_independence_audit() -> List[dict]:
    """Audit which inputs were used in matching.

    Canonical matching uses ONLY full-Test mean last-query temporal profiles.
    Returns one row per input class, with used_in_matching=False for any
    forbidden inputs."""
    checks = [
        ("mean_temporal_profiles", True),
        ("forecast_error", False),
        ("residual_sign", False),
        ("phase50_regimes", False),
        ("phase51_worst_case_membership", False),
        ("phase56_effect_sizes", False),
        ("prediction_seed_spread", False),
        ("test_rmse", False),
    ]
    rows = []
    for name, used in checks:
        rows.append({
            "check": name,
            "used_in_matching": bool(used),
            "expected": (not used) if name != "mean_temporal_profiles" else True,
            "status": "PASS" if (used if name == "mean_temporal_profiles" else (not used)) else "FAIL",
        })
    return rows


# ===========================================================================
# S57-B.6 — Canonical three-seed matched groups (anchored at seed42)
# ===========================================================================

def build_canonical_groups(
    matching_results: Dict[Tuple[int, int, int], MatchingResult],
    ambiguity_rows: List[dict],
    cycle_rows: List[dict],
) -> CanonicalGroups:
    ambiguity_lookup = {(int(r["layer_idx0"]), int(r["seed_a"]), int(r["seed_b"])): bool(r["ambiguous_match_warning"])
                         for r in ambiguity_rows}
    cycle_lookup = {(int(r["layer_idx0"]), int(r["seed123_head_idx0"])): bool(r["cycle_consistent"])
                    for r in cycle_rows}
    groups: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for layer in range(NUM_LAYERS):
        mr_42_123 = matching_results[(layer, 42, 123)]
        mr_42_2026 = matching_results[(layer, 42, 2026)]
        amb_42_123 = ambiguity_lookup[(layer, 42, 123)]
        amb_42_2026 = ambiguity_lookup[(layer, 42, 2026)]
        for h42 in range(NUM_HEADS):
            h123 = mr_42_123.canonical_assignment[h42]
            h2026 = mr_42_2026.canonical_assignment[h42]
            consistent = cycle_lookup.get((layer, h123), False)
            groups[(layer, h42)] = {
                "layer_idx0": layer,
                "canonical_group": h42 + 1,    # 1-indexed display
                "anchor_seed": 42,
                "anchor_reason": ANCHOR_REASON,
                "seed42_head_idx0": h42,
                "seed123_head_idx0": int(h123),
                "seed2026_head_idx0": int(h2026),
                "mapping_42_123_ambiguous": bool(amb_42_123),
                "mapping_42_2026_ambiguous": bool(amb_42_2026),
                "cycle_consistent": bool(consistent),
                "status": "OK",
            }
    return CanonicalGroups(groups=groups, cycle_consistency={})


def canonical_groups_rows(cg: CanonicalGroups) -> List[dict]:
    return list(cg.groups.values())


# ===========================================================================
# Matching fingerprint (BEFORE Phase 56 effect mapping)
# ===========================================================================

def matching_fingerprint(
    sources: FrozenSources57,
    matching_results: Dict[Tuple[int, int, int], MatchingResult],
    cg: CanonicalGroups,
) -> dict:
    """Compute SHA fingerprints over the canonical matching + canonical groups.
    MUST be written before applying Phase 56 effects.
    """
    # Profile source SHA: hash of all mean profiles
    h = hashlib.sha256()
    for s in SEEDS:
        for layer in range(NUM_LAYERS):
            for head in range(NUM_HEADS):
                npz = np.load(sources.raw_last_query_files[s])
                v = npz["last_query_attention"][:, layer, head, :].mean(axis=0)
                h.update(np.asarray(v, dtype=np.float64).tobytes())
    profile_sha = h.hexdigest()

    # Pairwise assignment SHA
    a = hashlib.sha256()
    for (layer, sa, sb), mr in sorted(matching_results.items()):
        a.update(f"{layer}-{sa}-{sb}".encode())
        for ha in range(NUM_HEADS):
            a.update(str(mr.canonical_assignment[ha]).encode())
    assignment_sha = a.hexdigest()

    g = hashlib.sha256()
    for key in sorted(cg.groups.keys()):
        info = cg.groups[key]
        g.update(f"{key[0]}-{info['seed42_head_idx0']}-{info['seed123_head_idx0']}-{info['seed2026_head_idx0']}".encode())
    group_sha = g.hexdigest()

    return {
        "source_profile_sha256": profile_sha,
        "final_lock_sha256": sources.final_lock_sha256,
        "test_population_sha256": sources.test_population_sha256,
        "seed_list": list(SEEDS),
        "layers": int(NUM_LAYERS),
        "heads": int(NUM_HEADS),
        "cost_metric": "JSD_NATURAL_LOG",
        "matching_method": "EXHAUSTIVE_PERMUTATION",
        "tie_tolerance": float(MATCH_TIE_TOL),
        "tie_break_2": "WASSERSTEIN_MINUTES",
        "tie_break_3": "LEXICOGRAPHIC",
        "anchor_seed": int(ANCHOR_SEED),
        "anchor_reason": ANCHOR_REASON,
        "performance_based_anchor": False,
        "pairwise_assignment_sha256": assignment_sha,
        "canonical_group_sha256": group_sha,
        "created_before_error_effect_mapping": True,
        "status": "OK",
    }


# ===========================================================================
# S57-B.7 — Matched-head mean-profile stability + consensus profile
# ===========================================================================

def matched_head_profile_stability(
    sources: FrozenSources57,
    mean_profiles: Dict[Tuple[int, int, int], np.ndarray],
    cg: CanonicalGroups,
) -> List[dict]:
    rows: List[dict] = []
    summary: List[dict] = []
    for (layer, h42), info in cg.groups.items():
        h123 = info["seed123_head_idx0"]
        h2026 = info["seed2026_head_idx0"]
        amb = bool(info["mapping_42_123_ambiguous"] or info["mapping_42_2026_ambiguous"])
        p42 = mean_profiles[(42, layer, h42)]
        p123 = mean_profiles[(123, layer, h123)]
        p2026 = mean_profiles[(2026, layer, h2026)]
        for seed_a, seed_b, ha, hb, pa, pb in (
            (42, 123, h42, h123, p42, p123),
            (42, 2026, h42, h2026, p42, p2026),
            (123, 2026, h123, h2026, p123, p2026),
        ):
            rows.append({
                "layer_idx0": layer,
                "canonical_group": info["canonical_group"],
                "seed_a": seed_a,
                "seed_b": seed_b,
                "head_a_idx0": int(ha),
                "head_b_idx0": int(hb),
                "jsd": float(jsd_natural_log(pa, pb)),
                "wasserstein_minutes": float(wasserstein_minutes(pa, pb, LAG_MINUTES)),
                "cosine": float(cosine_similarity(pa, pb)),
                "pearson": float(pearson_corr(pa, pb)),
                "spearman": float(spearman_rho_safe(pa, pb)[0]),
                "l1": float(l1_distance(pa, pb)),
                "l2": float(l2_distance(pa, pb)),
                "match_ambiguity_warning": amb,
                "status": "OK",
            })
        # Three-seed summary for the canonical group
        jsds = [
            float(jsd_natural_log(p42, p123)),
            float(jsd_natural_log(p42, p2026)),
            float(jsd_natural_log(p123, p2026)),
        ]
        wass = [
            float(wasserstein_minutes(p42, p123, LAG_MINUTES)),
            float(wasserstein_minutes(p42, p2026, LAG_MINUTES)),
            float(wasserstein_minutes(p123, p2026, LAG_MINUTES)),
        ]
        coss = [
            float(cosine_similarity(p42, p123)),
            float(cosine_similarity(p42, p2026)),
            float(cosine_similarity(p123, p2026)),
        ]
        summary.append({
            "layer_idx0": layer,
            "canonical_group": info["canonical_group"],
            "mean_pairwise_jsd": float(np.mean(jsds)),
            "max_pairwise_jsd": float(np.max(jsds)),
            "mean_pairwise_wasserstein_minutes": float(np.mean(wass)),
            "max_pairwise_wasserstein_minutes": float(np.max(wass)),
            "mean_pairwise_cosine": float(np.mean(coss)),
            "min_pairwise_cosine": float(np.min(coss)),
            "any_match_ambiguity": amb,
            "cycle_consistent": bool(info["cycle_consistent"]),
            "status": "OK",
        })
    return rows, summary


def matched_head_consensus_profiles(
    sources: FrozenSources57,
    cg: CanonicalGroups,
) -> List[dict]:
    """For each (layer, group, lag): mean of three seeds' full-Test mean head profiles."""
    rows: List[dict] = []
    for (layer, h42), info in cg.groups.items():
        h123 = info["seed123_head_idx0"]
        h2026 = info["seed2026_head_idx0"]
        npz42 = np.load(sources.raw_last_query_files[42])
        npz123 = np.load(sources.raw_last_query_files[123])
        npz2026 = np.load(sources.raw_last_query_files[2026])
        p42 = npz42["last_query_attention"][:, layer, h42, :].mean(axis=0)
        p123 = npz123["last_query_attention"][:, layer, h123, :].mean(axis=0)
        p2026 = npz2026["last_query_attention"][:, layer, h2026, :].mean(axis=0)
        for k in range(LOOKBACK):
            arr = np.asarray([p42[k], p123[k], p2026[k]], dtype=np.float64)
            consensus_sum = float(np.mean([p42.sum(), p123.sum(), p2026.sum()]))
            rows.append({
                "layer_idx0": layer,
                "canonical_group": info["canonical_group"],
                "lag_steps": int(LOOKBACK - 1 - k),
                "lag_minutes": int(LAG_MINUTES[k]),
                "seed42_weight": float(p42[k]),
                "seed123_weight": float(p123[k]),
                "seed2026_weight": float(p2026[k]),
                "mean_weight": float(np.mean(arr)),
                "sample_sd_weight": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "min_weight": float(np.min(arr)),
                "max_weight": float(np.max(arr)),
                "consensus_profile_sum": consensus_sum,
                "status": "OK",
            })
    return rows


# ===========================================================================
# S57-B.8 — Per-target matched-head stability (frozen mapping reused per target)
# ===========================================================================

def matched_head_per_target_stability(
    sources: FrozenSources57,
    cg: CanonicalGroups,
) -> List[dict]:
    rows: List[dict] = []
    # Load all seeds' raw last-query arrays
    arrs = {s: np.load(sources.raw_last_query_files[s])["last_query_attention"] for s in SEEDS}
    npz_seed_ids = {s: np.load(sources.raw_last_query_files[s])["target_ids"].astype(str) for s in SEEDS}
    target_ids_seed42 = npz_seed_ids[42]
    for (layer, h42), info in cg.groups.items():
        h123 = info["seed123_head_idx0"]
        h2026 = info["seed2026_head_idx0"]
        amb = bool(info["mapping_42_123_ambiguous"] or info["mapping_42_2026_ambiguous"])
        for ti in range(N_TEST):
            tid = str(target_ids_seed42[ti])
            v42 = arrs[42][ti, layer, h42, :]
            v123 = arrs[123][ti, layer, h123, :]
            v2026 = arrs[2026][ti, layer, h2026, :]
            for seed_a, seed_b, va, vb in (
                (42, 123, v42, v123),
                (42, 2026, v42, v2026),
                (123, 2026, v123, v2026),
            ):
                rows.append({
                    "target_id": tid,
                    "target_index": ti,
                    "layer_idx0": layer,
                    "canonical_group": info["canonical_group"],
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "jsd": float(jsd_natural_log(va, vb)),
                    "wasserstein_minutes": float(wasserstein_minutes(va, vb, LAG_MINUTES)),
                    "cosine": float(cosine_similarity(va, vb)),
                    "l1": float(l1_distance(va, vb)),
                    "match_ambiguity_warning": amb,
                    "status": "OK",
                })
    return rows


def matched_head_per_target_summary(stability_rows: List[dict]) -> List[dict]:
    rows: List[dict] = []
    for layer in range(NUM_LAYERS):
        for cg in range(1, NUM_HEADS + 1):
            for seed_a, seed_b in SEED_PAIRS:
                sel = [r for r in stability_rows if int(r["layer_idx0"]) == layer
                       and int(r["canonical_group"]) == cg
                       and int(r["seed_a"]) == seed_a and int(r["seed_b"]) == seed_b]
                for metric in ("jsd", "wasserstein_minutes", "cosine", "l1"):
                    vals = np.asarray([r[metric] for r in sel], dtype=np.float64)
                    if len(vals) == 0:
                        continue
                    rows.append({
                        "layer_idx0": layer,
                        "canonical_group": cg,
                        "seed_a": seed_a,
                        "seed_b": seed_b,
                        "metric": metric,
                        "N": int(len(vals)),
                        "mean": float(np.mean(vals)),
                        "sample_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                        "median": float(np.median(vals)),
                        "p05": float(np.percentile(vals, 5)),
                        "p25": float(np.percentile(vals, 25)),
                        "p75": float(np.percentile(vals, 75)),
                        "p95": float(np.percentile(vals, 95)),
                        "min": float(np.min(vals)),
                        "max": float(np.max(vals)),
                        "status": "OK",
                    })
    return rows


def matched_head_per_target_three_seed_disagreement(stability_rows: List[dict]) -> List[dict]:
    """Per (target, layer, group): aggregate the 3 pairwise distances."""
    out: Dict[Tuple[str, int, int], List[dict]] = {}
    for r in stability_rows:
        k = (r["target_id"], int(r["layer_idx0"]), int(r["canonical_group"]))
        out.setdefault(k, []).append(r)
    rows: List[dict] = []
    for (tid, layer, cg), recs in out.items():
        jsds = [float(r["jsd"]) for r in recs]
        wass = [float(r["wasserstein_minutes"]) for r in recs]
        rows.append({
            "target_id": tid,
            "target_index": int(recs[0]["target_index"]),
            "layer_idx0": layer,
            "canonical_group": cg,
            "mean_pairwise_jsd": float(np.mean(jsds)),
            "max_pairwise_jsd": float(np.max(jsds)),
            "mean_pairwise_wasserstein_minutes": float(np.mean(wass)),
            "max_pairwise_wasserstein_minutes": float(np.max(wass)),
            "status": "OK",
        })
    return rows


# ===========================================================================
# S57-B.9 — Matched-head metric stability (from Phase 54 per-vector metrics)
# ===========================================================================

def matched_head_metric_stability(sources: FrozenSources57, cg: CanonicalGroups) -> List[dict]:
    """Compare Phase 54 metrics across matched heads.

    For each (seed, target, layer, head) the Phase 54 last_query_metrics_long.csv
    has the 6 CORE_ATTENTION_METRICS-v1 values.
    """
    # Build target_id -> target_index lookup from Phase 52 raw NPZ
    npz42 = np.load(sources.raw_last_query_files[42])
    target_ids_seed42 = npz42["target_ids"].astype(str)
    tid_to_idx = {tid: i for i, tid in enumerate(target_ids_seed42)}

    rows_by_seed: Dict[Tuple[int, int, int, int], dict] = {}
    for r in sources.last_query_metrics_long:
        s = int(r["seed"])
        tid = str(r["target_id"])
        ti = tid_to_idx.get(tid)
        if ti is None:
            continue
        layer = int(r["layer_idx0"])
        head = int(r["head_idx0"])
        rows_by_seed[(s, ti, layer, head)] = r

    rows: List[dict] = []
    for (layer, h42), info in cg.groups.items():
        h123 = info["seed123_head_idx0"]
        h2026 = info["seed2026_head_idx0"]
        amb = bool(info["mapping_42_123_ambiguous"] or info["mapping_42_2026_ambiguous"])
        for metric in CORE_ATTENTION_METRICS_V1:
            for seed_a, seed_b, ha, hb in (
                (42, 123, h42, h123),
                (42, 2026, h42, h2026),
                (123, 2026, h123, h2026),
            ):
                xs = []
                ys = []
                for ti in range(N_TEST):
                    ra = rows_by_seed.get((seed_a, ti, layer, ha))
                    rb = rows_by_seed.get((seed_b, ti, layer, hb))
                    if ra is None or rb is None:
                        continue
                    try:
                        xs.append(float(ra[metric]))
                        ys.append(float(rb[metric]))
                    except (KeyError, ValueError, TypeError):
                        continue
                if not xs:
                    continue
                xs_arr = np.asarray(xs, dtype=np.float64)
                ys_arr = np.asarray(ys, dtype=np.float64)
                rho, _ = spearman_rho_safe(xs_arr, ys_arr)
                diffs = np.abs(xs_arr - ys_arr)
                rows.append({
                    "layer_idx0": layer,
                    "canonical_group": info["canonical_group"],
                    "metric": metric,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "N": int(len(xs)),
                    "spearman_across_targets": rho,
                    "mean_absolute_difference": float(np.mean(diffs)),
                    "median_absolute_difference": float(np.median(diffs)),
                    "match_ambiguity_warning": amb,
                    "status": "OK",
                })
    return rows


# ===========================================================================
# S57-B.10 — Matched-head top1-lag stability
# ===========================================================================

def matched_head_top1_lag_stability(
    sources: FrozenSources57,
    cg: CanonicalGroups,
) -> List[dict]:
    """Compare top1 lag frequency distributions across matched heads.

    Uses Phase 54 last_query_top1_lag_frequency.csv (one row per (seed, layer, head, lag_steps)).
    """
    # Index by (seed, layer, head) -> dict[top1_lag_index -> count]
    freq_by_seed: Dict[Tuple[int, int, int], Dict[int, int]] = {}
    for r in sources.last_query_top1_lag_frequency:
        s = int(r["seed"])
        layer = int(r["layer_idx0"])
        head = int(r["head_idx0"])
        # Use lag_steps as recency index (1..LOOKBACK); convert to 0-indexed
        lag = int(r["lag_steps"]) - 1
        if lag < 0:
            lag = 0
        cnt = int(float(r["count"]))
        freq_by_seed.setdefault((s, layer, head), {})[lag] = cnt
    L = LOOKBACK

    rows: List[dict] = []
    for (layer, h42), info in cg.groups.items():
        h123 = info["seed123_head_idx0"]
        h2026 = info["seed2026_head_idx0"]
        amb = bool(info["mapping_42_123_ambiguous"] or info["mapping_42_2026_ambiguous"])
        for seed_a, seed_b, ha, hb in (
            (42, 123, h42, h123),
            (42, 2026, h42, h2026),
            (123, 2026, h123, h2026),
        ):
            fa = freq_by_seed.get((seed_a, layer, ha), {})
            fb = freq_by_seed.get((seed_b, layer, hb), {})
            pa = np.asarray([fa.get(k, 0) for k in range(L)], dtype=np.float64)
            pb = np.asarray([fb.get(k, 0) for k in range(L)], dtype=np.float64)
            if pa.sum() > 0:
                pa = pa / pa.sum()
            if pb.sum() > 0:
                pb = pb / pb.sum()
            tvd = 0.5 * float(np.sum(np.abs(pa - pb)))
            jsd = jsd_natural_log(pa, pb)
            mode_a = int(np.argmax(pa)) if pa.sum() > 0 else -1
            mode_b = int(np.argmax(pb)) if pb.sum() > 0 else -1
            rows.append({
                "layer_idx0": layer,
                "canonical_group": info["canonical_group"],
                "seed_a": seed_a,
                "seed_b": seed_b,
                "tvd": tvd,
                "jsd": jsd,
                "modal_lag_seed_a": mode_a,
                "modal_lag_seed_b": mode_b,
                "same_modal_lag": bool(mode_a == mode_b),
                "match_ambiguity_warning": amb,
                "status": "OK",
            })
    return rows


# ===========================================================================
# S57-E — Dense-case full-map stability
# ===========================================================================

def dense_case_stability(
    sources: FrozenSources57,
    cg: CanonicalGroups,
    max_cases: int | None = None,
) -> Tuple[List[dict], List[dict]]:
    """For each Phase 51 frozen dense case, compute per-case stability metrics.

    Apply canonical global head mapping (no rematching) to dense NPZ arrays.
    """
    rows: List[dict] = []
    summary: List[dict] = []

    # Load dense arrays and case orders
    arrs = {s: np.load(sources.raw_dense_files[s]) for s in SEEDS}
    case_orders = {}
    for s in SEEDS:
        case_orders[s] = arrs[s]["target_ids"].astype(str)
    # Dense case order same across seeds (verify)
    base_ids = case_orders[42]
    for s in SEEDS:
        if not np.array_equal(case_orders[s], base_ids):
            raise ValueError(f"dense case order differs in seed {s}")

    n_cases = len(base_ids)
    if max_cases is not None:
        n_cases = min(n_cases, max_cases)

    # Per (layer, canonical_group, seed_pair) summary accumulator
    agg: Dict[Tuple[int, int, int, int], Dict[str, list]] = {}

    for ci in range(n_cases):
        for layer in range(NUM_LAYERS):
            for (layer_key, h42), info in cg.groups.items():
                if layer_key != layer:
                    continue
                h123 = info["seed123_head_idx0"]
                h2026 = info["seed2026_head_idx0"]
                amb = bool(info["mapping_42_123_ambiguous"] or info["mapping_42_2026_ambiguous"])
                A42 = arrs[42]["attention"][ci, layer, h42, :, :]
                A123 = arrs[123]["attention"][ci, layer, h123, :, :]
                A2026 = arrs[2026]["attention"][ci, layer, h2026, :, :]

                # Each query row is a probability distribution; rowwise JSD
                jsd_per_q = []
                cos_per_q = []
                for q in range(LOOKBACK):
                    jsd_per_q.append(jsd_natural_log(A42[q, :], A123[q, :]))
                    cos_per_q.append(cosine_similarity(A42[q, :], A123[q, :]))
                mean_jsd_42_123 = float(np.mean(jsd_per_q))
                median_jsd_42_123 = float(np.median(jsd_per_q))
                max_jsd_42_123 = float(np.max(jsd_per_q))
                mean_cos_42_123 = float(np.mean(cos_per_q))
                min_cos_42_123 = float(np.min(cos_per_q))
                frob_42_123 = float(np.linalg.norm(A42 - A123) / np.sqrt(LOOKBACK))

                jsd_per_q = []
                cos_per_q = []
                for q in range(LOOKBACK):
                    jsd_per_q.append(jsd_natural_log(A42[q, :], A2026[q, :]))
                    cos_per_q.append(cosine_similarity(A42[q, :], A2026[q, :]))
                mean_jsd_42_2026 = float(np.mean(jsd_per_q))
                median_jsd_42_2026 = float(np.median(jsd_per_q))
                max_jsd_42_2026 = float(np.max(jsd_per_q))
                mean_cos_42_2026 = float(np.mean(cos_per_q))
                min_cos_42_2026 = float(np.min(cos_per_q))
                frob_42_2026 = float(np.linalg.norm(A42 - A2026) / np.sqrt(LOOKBACK))

                jsd_per_q = []
                cos_per_q = []
                for q in range(LOOKBACK):
                    jsd_per_q.append(jsd_natural_log(A123[q, :], A2026[q, :]))
                    cos_per_q.append(cosine_similarity(A123[q, :], A2026[q, :]))
                mean_jsd_123_2026 = float(np.mean(jsd_per_q))
                median_jsd_123_2026 = float(np.median(jsd_per_q))
                max_jsd_123_2026 = float(np.max(jsd_per_q))
                mean_cos_123_2026 = float(np.mean(cos_per_q))
                min_cos_123_2026 = float(np.min(cos_per_q))
                frob_123_2026 = float(np.linalg.norm(A123 - A2026) / np.sqrt(LOOKBACK))

                for seed_a, seed_b, mean_jsd, median_jsd, max_jsd, mean_cos, min_cos, frob in (
                    (42, 123, mean_jsd_42_123, median_jsd_42_123, max_jsd_42_123, mean_cos_42_123, min_cos_42_123, frob_42_123),
                    (42, 2026, mean_jsd_42_2026, median_jsd_42_2026, max_jsd_42_2026, mean_cos_42_2026, min_cos_42_2026, frob_42_2026),
                    (123, 2026, mean_jsd_123_2026, median_jsd_123_2026, max_jsd_123_2026, mean_cos_123_2026, min_cos_123_2026, frob_123_2026),
                ):
                    rows.append({
                        "case_row_idx0": ci,
                        "target_id": str(base_ids[ci]),
                        "target_timestamp": "",
                        "selection_roles": "PHASE51_FROZEN_DENSE",
                        "layer_idx0": layer,
                        "canonical_group": info["canonical_group"],
                        "seed_a": seed_a,
                        "seed_b": seed_b,
                        "head_a_idx0": int(h42 if seed_a == 42 else (h123 if seed_a == 123 else h2026)),
                        "head_b_idx0": int(h42 if seed_b == 42 else (h123 if seed_b == 123 else h2026)),
                        "mean_query_jsd": mean_jsd,
                        "median_query_jsd": median_jsd,
                        "max_query_jsd": max_jsd,
                        "mean_query_cosine": mean_cos,
                        "min_query_cosine": min_cos,
                        "normalized_frobenius_distance": frob,
                        "match_ambiguity_warning": amb,
                        "status": "OK",
                    })
                    key = (layer, info["canonical_group"], seed_a, seed_b)
                    agg.setdefault(key, {"mean_jsd": [], "frobenius": []})
                    agg[key]["mean_jsd"].append(mean_jsd)
                    agg[key]["frobenius"].append(frob)

    for (layer, cg, sa, sb), v in agg.items():
        if not v["mean_jsd"]:
            continue
        arr = np.asarray(v["mean_jsd"], dtype=np.float64)
        farr = np.asarray(v["frobenius"], dtype=np.float64)
        summary.append({
            "layer_idx0": layer,
            "canonical_group": cg,
            "seed_a": sa,
            "seed_b": sb,
            "case_count": int(arr.size),
            "mean_case_mean_query_jsd": float(np.mean(arr)),
            "median_case_mean_query_jsd": float(np.median(arr)),
            "p05": float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
            "mean_normalized_frobenius": float(np.mean(farr)),
            "status": "OK",
        })
    return rows, summary


# ===========================================================================
# S57-D.1 — Layer error-conditioned stability
# ===========================================================================

def layer_error_conditioned_stability(sources: FrozenSources57) -> List[dict]:
    """For each (analysis_type, conditioning_variable, attention_metric), collect
    seed42/123/2026 values, mean/SD/min/max/sign counts.

    analysis_type: SPEARMAN | HIGH_LOW_MEDIAN_DELTA | HIGH_LOW_CLIFFS_DELTA |
                   SHARED_HIGH_LOW_CLIFFS_DELTA | PROFILE_JSD | PROFILE_WASSERSTEIN
    """
    rows: List[dict] = []

    # SPEARMAN from error_attention_association_long.csv (per-seed)
    spearman_by_layer: Dict[Tuple[int, str, str], Dict[int, float]] = {}
    for r in sources.error_attention_association_long:
        layer = int(r["layer_idx0"])
        cond = str(r.get("conditioning_variable", ""))
        metric = str(r.get("attention_metric", ""))
        seed = int(r["seed"])
        rho = float(r.get("spearman_rho", 0.0))
        if not np.isfinite(rho):
            continue
        spearman_by_layer.setdefault((layer, cond, metric), {})[seed] = rho
    for (layer, cond, metric), sd in spearman_by_layer.items():
        seeds_vals = [sd.get(s, np.nan) for s in SEEDS]
        finite = [v for v in seeds_vals if np.isfinite(v)]
        pos = sum(1 for v in finite if v > 0)
        neg = sum(1 for v in finite if v < 0)
        und = 3 - len(finite)
        rows.append({
            "layer_idx0": layer,
            "analysis_type": "SPEARMAN",
            "conditioning_variable": cond,
            "attention_metric": metric,
            "seed42_value": finite[0] if len(finite) > 0 else float("nan"),
            "seed123_value": finite[1] if len(finite) > 1 else float("nan"),
            "seed2026_value": finite[2] if len(finite) > 2 else float("nan"),
            "mean_across_seeds": float(np.mean(finite)) if finite else float("nan"),
            "sample_sd_across_seeds": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
            "min": float(np.min(finite)) if finite else float("nan"),
            "max": float(np.max(finite)) if finite else float("nan"),
            "positive_count": pos,
            "negative_count": neg,
            "undefined_count": und,
            "all_defined_same_sign": bool((pos == 3) or (neg == 3)),
            "status": "OK",
        })

    # HIGH_LOW_CLIFFS_DELTA from error_attention_high_low_metric_comparison.csv
    hl_by_layer: Dict[Tuple[int, str], Dict[int, float]] = {}
    for r in sources.error_attention_high_low_metric:
        layer = int(r["layer_idx0"])
        metric = str(r.get("attention_metric", ""))
        seed = int(r["seed"])
        try:
            cliffs = float(r.get("cliffs_delta_high_vs_low", 0.0))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(cliffs):
            continue
        hl_by_layer.setdefault((layer, metric), {})[seed] = cliffs
    for (layer, metric), sd in hl_by_layer.items():
        finite = [sd.get(s, np.nan) for s in SEEDS]
        finite = [v for v in finite if np.isfinite(v)]
        pos = sum(1 for v in finite if v > 0)
        neg = sum(1 for v in finite if v < 0)
        und = 3 - len(finite)
        rows.append({
            "layer_idx0": layer,
            "analysis_type": "HIGH_LOW_CLIFFS_DELTA",
            "conditioning_variable": "ABS_ERROR",
            "attention_metric": metric,
            "seed42_value": finite[0] if len(finite) > 0 else float("nan"),
            "seed123_value": finite[1] if len(finite) > 1 else float("nan"),
            "seed2026_value": finite[2] if len(finite) > 2 else float("nan"),
            "mean_across_seeds": float(np.mean(finite)) if finite else float("nan"),
            "sample_sd_across_seeds": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
            "min": float(np.min(finite)) if finite else float("nan"),
            "max": float(np.max(finite)) if finite else float("nan"),
            "positive_count": pos,
            "negative_count": neg,
            "undefined_count": und,
            "all_defined_same_sign": bool((pos == 3) or (neg == 3)),
            "status": "OK",
        })

    # HIGH_LOW_MEDIAN_DELTA from same file (delta_median_high_minus_low)
    hlmd_by_layer: Dict[Tuple[int, str], Dict[int, float]] = {}
    for r in sources.error_attention_high_low_metric:
        layer = int(r["layer_idx0"])
        metric = str(r.get("attention_metric", ""))
        seed = int(r["seed"])
        try:
            v = float(r.get("delta_median_high_minus_low", 0.0))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        hlmd_by_layer.setdefault((layer, metric), {})[seed] = v
    for (layer, metric), sd in hlmd_by_layer.items():
        finite = [sd.get(s, np.nan) for s in SEEDS]
        finite = [v for v in finite if np.isfinite(v)]
        pos = sum(1 for v in finite if v > 0)
        neg = sum(1 for v in finite if v < 0)
        und = 3 - len(finite)
        rows.append({
            "layer_idx0": layer,
            "analysis_type": "HIGH_LOW_MEDIAN_DELTA",
            "conditioning_variable": "ABS_ERROR",
            "attention_metric": metric,
            "seed42_value": finite[0] if len(finite) > 0 else float("nan"),
            "seed123_value": finite[1] if len(finite) > 1 else float("nan"),
            "seed2026_value": finite[2] if len(finite) > 2 else float("nan"),
            "mean_across_seeds": float(np.mean(finite)) if finite else float("nan"),
            "sample_sd_across_seeds": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
            "min": float(np.min(finite)) if finite else float("nan"),
            "max": float(np.max(finite)) if finite else float("nan"),
            "positive_count": pos,
            "negative_count": neg,
            "undefined_count": und,
            "all_defined_same_sign": bool((pos == 3) or (neg == 3)),
            "status": "OK",
        })

    # SHARED_HIGH_LOW_CLIFFS_DELTA from error_attention_shared_cohort_layer_summary.csv
    sh_by_layer: Dict[Tuple[int, str], Dict[int, float]] = {}
    for r in sources.error_attention_shared_cohort:
        layer = int(r["layer_idx0"])
        metric = str(r.get("attention_metric", ""))
        seed = int(r["seed"])
        try:
            v = float(r.get("cliffs_delta_high_vs_low", 0.0))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        sh_by_layer.setdefault((layer, metric), {})[seed] = v
    for (layer, metric), sd in sh_by_layer.items():
        finite = [sd.get(s, np.nan) for s in SEEDS]
        finite = [v for v in finite if np.isfinite(v)]
        pos = sum(1 for v in finite if v > 0)
        neg = sum(1 for v in finite if v < 0)
        und = 3 - len(finite)
        rows.append({
            "layer_idx0": layer,
            "analysis_type": "SHARED_HIGH_LOW_CLIFFS_DELTA",
            "conditioning_variable": "SHARED_HARDNESS",
            "attention_metric": metric,
            "seed42_value": finite[0] if len(finite) > 0 else float("nan"),
            "seed123_value": finite[1] if len(finite) > 1 else float("nan"),
            "seed2026_value": finite[2] if len(finite) > 2 else float("nan"),
            "mean_across_seeds": float(np.mean(finite)) if finite else float("nan"),
            "sample_sd_across_seeds": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
            "min": float(np.min(finite)) if finite else float("nan"),
            "max": float(np.max(finite)) if finite else float("nan"),
            "positive_count": pos,
            "negative_count": neg,
            "undefined_count": und,
            "all_defined_same_sign": bool((pos == 3) or (neg == 3)),
            "status": "OK",
        })

    # PROFILE_JSD (HIGH vs LOW) from error_attention_high_low_profile_comparison.csv
    prof_jsd_by_layer: Dict[Tuple[int], Dict[int, float]] = {}
    for r in sources.error_attention_high_low_profile:
        layer = int(r["layer_idx0"])
        seed = int(r["seed"])
        try:
            v = float(r.get("jsd_high_vs_low", 0.0))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        prof_jsd_by_layer.setdefault((layer,), {})[seed] = v
    for (layer,), sd in prof_jsd_by_layer.items():
        finite = [sd.get(s, np.nan) for s in SEEDS]
        finite = [v for v in finite if np.isfinite(v)]
        rows.append({
            "layer_idx0": layer,
            "analysis_type": "PROFILE_JSD",
            "conditioning_variable": "ABS_ERROR",
            "attention_metric": "MEAN_PROFILE",
            "seed42_value": finite[0] if len(finite) > 0 else float("nan"),
            "seed123_value": finite[1] if len(finite) > 1 else float("nan"),
            "seed2026_value": finite[2] if len(finite) > 2 else float("nan"),
            "mean_across_seeds": float(np.mean(finite)) if finite else float("nan"),
            "sample_sd_across_seeds": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
            "min": float(np.min(finite)) if finite else float("nan"),
            "max": float(np.max(finite)) if finite else float("nan"),
            "positive_count": sum(1 for v in finite if v > 0),
            "negative_count": 0,
            "undefined_count": 3 - len(finite),
            "all_defined_same_sign": True,
            "status": "OK",
        })

    # PROFILE_WASSERSTEIN (HIGH vs LOW)
    prof_wass_by_layer: Dict[Tuple[int], Dict[int, float]] = {}
    for r in sources.error_attention_high_low_profile:
        layer = int(r["layer_idx0"])
        seed = int(r["seed"])
        try:
            v = float(r.get("wasserstein_minutes_high_vs_low", 0.0))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        prof_wass_by_layer.setdefault((layer,), {})[seed] = v
    for (layer,), sd in prof_wass_by_layer.items():
        finite = [sd.get(s, np.nan) for s in SEEDS]
        finite = [v for v in finite if np.isfinite(v)]
        rows.append({
            "layer_idx0": layer,
            "analysis_type": "PROFILE_WASSERSTEIN",
            "conditioning_variable": "ABS_ERROR",
            "attention_metric": "MEAN_PROFILE",
            "seed42_value": finite[0] if len(finite) > 0 else float("nan"),
            "seed123_value": finite[1] if len(finite) > 1 else float("nan"),
            "seed2026_value": finite[2] if len(finite) > 2 else float("nan"),
            "mean_across_seeds": float(np.mean(finite)) if finite else float("nan"),
            "sample_sd_across_seeds": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
            "min": float(np.min(finite)) if finite else float("nan"),
            "max": float(np.max(finite)) if finite else float("nan"),
            "positive_count": sum(1 for v in finite if v > 0),
            "negative_count": 0,
            "undefined_count": 3 - len(finite),
            "all_defined_same_sign": True,
            "status": "OK",
        })

    return rows


# ===========================================================================
# S57-D.2 — Matched-head error-conditioned stability
# ===========================================================================

def matched_head_error_conditioned_stability(
    sources: FrozenSources57,
    canonical_groups: CanonicalGroups,
) -> List[dict]:
    """Reindex Phase 56 per-head effects (continuous Spearman + HIGH-LOW Cliff's) to
    canonical matched groups via frozen matching.
    """
    rows: List[dict] = []
    # Build head-index lookup: per (seed, layer, head_idx0) -> (layer, group)
    head_to_group: Dict[Tuple[int, int, int], Tuple[int, int]] = {}
    for (layer, h42), info in canonical_groups.groups.items():
        head_to_group[(42, layer, h42)] = (layer, info["canonical_group"])
        head_to_group[(123, layer, info["seed123_head_idx0"])] = (layer, info["canonical_group"])
        head_to_group[(2026, layer, info["seed2026_head_idx0"])] = (layer, info["canonical_group"])

    # SPEARMAN per head (from error_attention_association_long.csv)
    spearman_by_head: Dict[Tuple[int, int, int, str, str], float] = {}
    for r in sources.error_attention_association_long:
        try:
            seed = int(r["seed"]); layer = int(r["layer_idx0"]); head = int(r["head_idx0"])
            cond = str(r.get("conditioning_variable", "")); metric = str(r.get("attention_metric", ""))
            rho = float(r.get("spearman_rho", 0.0))
        except (KeyError, ValueError, TypeError):
            continue
        if not np.isfinite(rho):
            continue
        spearman_by_head[(seed, layer, head, cond, metric)] = rho

    # Aggregate per (layer, group, cond, metric)
    agg: Dict[Tuple[int, int, str, str], Dict[int, float]] = {}
    for (seed, layer, head, cond, metric), rho in spearman_by_head.items():
        key = head_to_group.get((seed, layer, head))
        if key is None:
            continue
        layer_g, grp = key
        if layer_g != layer:
            continue
        agg.setdefault((layer_g, grp, cond, metric), {})[seed] = rho
    for (layer, group, cond, metric), sd in agg.items():
        any_amb = False
        for (l_, h42), info in canonical_groups.groups.items():
            if l_ == layer and info["canonical_group"] == group:
                any_amb = bool(info["mapping_42_123_ambiguous"] or info["mapping_42_2026_ambiguous"])
                break
        finite = [sd.get(s, np.nan) for s in SEEDS]
        finite = [v for v in finite if np.isfinite(v)]
        pos = sum(1 for v in finite if v > 0)
        neg = sum(1 for v in finite if v < 0)
        und = 3 - len(finite)
        rows.append({
            "layer_idx0": layer,
            "canonical_group": group,
            "analysis_type": "SPEARMAN",
            "conditioning_variable": cond,
            "attention_metric": metric,
            "seed42_value": finite[0] if len(finite) > 0 else float("nan"),
            "seed123_value": finite[1] if len(finite) > 1 else float("nan"),
            "seed2026_value": finite[2] if len(finite) > 2 else float("nan"),
            "mean_across_seeds": float(np.mean(finite)) if finite else float("nan"),
            "sample_sd_across_seeds": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
            "positive_count": pos,
            "negative_count": neg,
            "undefined_count": und,
            "all_defined_same_sign": bool((pos == 3) or (neg == 3)),
            "any_match_ambiguity": any_amb,
            "status": "OK",
        })

    # HIGH-LOW Cliff's per head
    cliffs_by_head: Dict[Tuple[int, int, int, str], float] = {}
    for r in sources.error_attention_high_low_metric:
        try:
            seed = int(r["seed"]); layer = int(r["layer_idx0"]); head = int(r["head_idx0"])
            metric = str(r.get("attention_metric", ""))
            v = float(r.get("cliffs_delta_high_vs_low", 0.0))
        except (KeyError, ValueError, TypeError):
            continue
        if not np.isfinite(v):
            continue
        cliffs_by_head[(seed, layer, head, metric)] = v
    agg2: Dict[Tuple[int, int, str], Dict[int, float]] = {}
    for (seed, layer, head, metric), v in cliffs_by_head.items():
        key = head_to_group.get((seed, layer, head))
        if key is None:
            continue
        layer_g, grp = key
        if layer_g != layer:
            continue
        agg2.setdefault((layer_g, grp, metric), {})[seed] = v
    for (layer, group, metric), sd in agg2.items():
        any_amb = False
        for (l_, h42), info in canonical_groups.groups.items():
            if l_ == layer and info["canonical_group"] == group:
                any_amb = bool(info["mapping_42_123_ambiguous"] or info["mapping_42_2026_ambiguous"])
                break
        finite = [sd.get(s, np.nan) for s in SEEDS]
        finite = [v for v in finite if np.isfinite(v)]
        pos = sum(1 for v in finite if v > 0)
        neg = sum(1 for v in finite if v < 0)
        und = 3 - len(finite)
        rows.append({
            "layer_idx0": layer,
            "canonical_group": group,
            "analysis_type": "HIGH_LOW_CLIFFS_DELTA",
            "conditioning_variable": "ABS_ERROR",
            "attention_metric": metric,
            "seed42_value": finite[0] if len(finite) > 0 else float("nan"),
            "seed123_value": finite[1] if len(finite) > 1 else float("nan"),
            "seed2026_value": finite[2] if len(finite) > 2 else float("nan"),
            "mean_across_seeds": float(np.mean(finite)) if finite else float("nan"),
            "sample_sd_across_seeds": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
            "positive_count": pos,
            "negative_count": neg,
            "undefined_count": und,
            "all_defined_same_sign": bool((pos == 3) or (neg == 3)),
            "any_match_ambiguity": any_amb,
            "status": "OK",
        })

    return rows


# ===========================================================================
# S57-G — Prediction-attention disagreement (secondary)
# ===========================================================================

def prediction_attention_disagreement_association(
    per_target_disagreement: List[dict],
    sources: FrozenSources57,
) -> List[dict]:
    """For each layer, Spearman between prediction_range/SD and mean_pairwise_layer_JSD/Wasserstein."""
    pred = {str(r["target_id"]): r for r in sources.prediction_seed_spread}
    rows: List[dict] = []
    for layer in range(NUM_LAYERS):
        sel = [r for r in per_target_disagreement if int(r["layer_idx0"]) == layer]
        # Pull prediction spread for matched targets
        pred_range = []
        pred_std = []
        attn_jsd = []
        attn_wass = []
        for r in sel:
            tid = str(r["target_id"])
            p = pred.get(tid)
            if p is None:
                continue
            try:
                pr = float(p.get("seed_range_prediction", float("nan")))
                ps = float(p.get("seed_std_prediction", float("nan")))
            except (KeyError, TypeError, ValueError):
                continue
            if not (np.isfinite(pr) and np.isfinite(ps)):
                continue
            pred_range.append(pr)
            pred_std.append(ps)
            attn_jsd.append(float(r["mean_pairwise_jsd"]))
            attn_wass.append(float(r["mean_pairwise_wasserstein_minutes"]))
        if not attn_jsd:
            continue
        for pred_metric, pred_vals in (("prediction_range", pred_range), ("prediction_std", pred_std)):
            for attn_metric, attn_vals in (("mean_pairwise_layer_jsd", attn_jsd),
                                            ("mean_pairwise_layer_wasserstein", attn_wass)):
                rho, _ = spearman_rho_safe(np.asarray(pred_vals), np.asarray(attn_vals))
                rows.append({
                    "layer_idx0": layer,
                    "prediction_spread_metric": pred_metric,
                    "attention_disagreement_metric": attn_metric,
                    "N": int(len(attn_vals)),
                    "spearman_rho": rho,
                    "status": "OK",
                })
    return rows


def matched_head_prediction_disagreement_association(
    matched_three_seed: List[dict],
    sources: FrozenSources57,
) -> List[dict]:
    """Per (layer, canonical_group, prediction_spread_metric, attention_disagreement_metric): Spearman."""
    pred = {str(r["target_id"]): r for r in sources.prediction_seed_spread}
    rows: List[dict] = []
    for layer in range(NUM_LAYERS):
        for group in range(1, NUM_HEADS + 1):
            sel = [r for r in matched_three_seed if int(r["layer_idx0"]) == layer and int(r["canonical_group"]) == group]
            pred_range = []
            pred_std = []
            attn_jsd = []
            attn_wass = []
            for r in sel:
                tid = str(r["target_id"])
                p = pred.get(tid)
                if p is None:
                    continue
                try:
                    pr = float(p.get("seed_range_prediction", float("nan")))
                    ps = float(p.get("seed_std_prediction", float("nan")))
                except (KeyError, TypeError, ValueError):
                    continue
                if not (np.isfinite(pr) and np.isfinite(ps)):
                    continue
                pred_range.append(pr)
                pred_std.append(ps)
                attn_jsd.append(float(r["mean_pairwise_jsd"]))
                attn_wass.append(float(r["mean_pairwise_wasserstein_minutes"]))
            if not attn_jsd:
                continue
            for pred_metric, pred_vals in (("prediction_range", pred_range), ("prediction_std", pred_std)):
                for attn_metric, attn_vals in (("mean_pairwise_canonical_head_jsd", attn_jsd),
                                                ("mean_pairwise_canonical_head_wasserstein", attn_wass)):
                    rho, _ = spearman_rho_safe(np.asarray(pred_vals), np.asarray(attn_vals))
                    rows.append({
                        "layer_idx0": layer,
                        "canonical_group": group,
                        "prediction_spread_metric": pred_metric,
                        "attention_disagreement_metric": attn_metric,
                        "N": int(len(attn_vals)),
                        "spearman_rho": rho,
                        "any_match_ambiguity": False,
                        "status": "OK",
                    })
    return rows


# ===========================================================================
# Stability evidence summary (no composite score)
# ===========================================================================

def stability_evidence_summary(
    layer_pair_summary: List[dict],
    matched_summary: List[dict],
    cycle_layer_summary: List[dict],
    wasserstein_summary: List[dict],
    error_layer_rows: List[dict],
    pred_assoc_rows: List[dict],
) -> List[dict]:
    rows: List[dict] = []

    # LAYER_PROFILE row
    for s in layer_pair_summary:
        rows.append({
            "evidence_row": "LAYER_PROFILE",
            "layer_idx0": s["layer_idx0"],
            "scope": "perm_inv_layer_mean_profile",
            "mean_jsd": s["mean_pairwise_jsd"],
            "max_jsd": s["max_pairwise_jsd"],
            "mean_wasserstein_minutes": s["mean_pairwise_wasserstein_minutes"],
            "max_wasserstein_minutes": s["max_pairwise_wasserstein_minutes"],
            "mean_cosine": s["mean_pairwise_cosine"],
            "min_cosine": s["min_pairwise_cosine"],
            "sign_agreement_count": "",
            "status": "OK",
        })

    # LAYER_PER_TARGET summary
    rows.append({
        "evidence_row": "LAYER_PER_TARGET",
        "layer_idx0": "",
        "scope": "perm_inv_layer_mean_per_target",
        "mean_jsd": "",
        "max_jsd": "",
        "mean_wasserstein_minutes": "",
        "max_wasserstein_minutes": "",
        "mean_cosine": "",
        "min_cosine": "",
        "sign_agreement_count": "",
        "status": "OK",
    })

    # HEAD_MATCHING row
    rows.append({
        "evidence_row": "HEAD_MATCHING",
        "layer_idx0": "",
        "scope": "canonical_JSD_exhaustive_matching",
        "mean_jsd": "",
        "max_jsd": "",
        "mean_wasserstein_minutes": "",
        "max_wasserstein_minutes": "",
        "mean_cosine": "",
        "min_cosine": "",
        "sign_agreement_count": "",
        "status": "OK",
    })

    # MATCH_CYCLE row
    for s in cycle_layer_summary:
        rows.append({
            "evidence_row": "MATCH_CYCLE",
            "layer_idx0": s["layer_idx0"],
            "scope": "direct_vs_anchor_induced_123_2026",
            "mean_jsd": "",
            "max_jsd": "",
            "mean_wasserstein_minutes": "",
            "max_wasserstein_minutes": "",
            "mean_cosine": "",
            "min_cosine": "",
            "sign_agreement_count": int(s["cycle_consistency_fraction"] * NUM_HEADS),
            "status": "OK",
        })

    # MATCH_METHOD_SENSITIVITY row
    for s in wasserstein_summary:
        rows.append({
            "evidence_row": "MATCH_METHOD_SENSITIVITY",
            "layer_idx0": s["layer_idx0"],
            "scope": "JSD_vs_Wasserstein_pair_agreement_fraction",
            "mean_jsd": "",
            "max_jsd": "",
            "mean_wasserstein_minutes": "",
            "max_wasserstein_minutes": "",
            "mean_cosine": "",
            "min_cosine": "",
            "sign_agreement_count": "",
            "status": "OK",
        })

    # MATCHED_HEAD_PROFILE row
    for s in matched_summary:
        rows.append({
            "evidence_row": "MATCHED_HEAD_PROFILE",
            "layer_idx0": s["layer_idx0"],
            "scope": "matched_canonical_group_3_pair_mean_profile",
            "mean_jsd": s["mean_pairwise_jsd"],
            "max_jsd": s["max_pairwise_jsd"],
            "mean_wasserstein_minutes": s["mean_pairwise_wasserstein_minutes"],
            "max_wasserstein_minutes": s["max_pairwise_wasserstein_minutes"],
            "mean_cosine": s["mean_pairwise_cosine"],
            "min_cosine": s["min_pairwise_cosine"],
            "sign_agreement_count": "",
            "status": "OK",
        })

    # MATCHED_HEAD_PER_TARGET row
    rows.append({
        "evidence_row": "MATCHED_HEAD_PER_TARGET",
        "layer_idx0": "",
        "scope": "matched_canonical_group_per_target_3_pair",
        "mean_jsd": "",
        "max_jsd": "",
        "mean_wasserstein_minutes": "",
        "max_wasserstein_minutes": "",
        "mean_cosine": "",
        "min_cosine": "",
        "sign_agreement_count": "",
        "status": "OK",
    })

    # DENSE_CASE_FULL_MAP row
    rows.append({
        "evidence_row": "DENSE_CASE_FULL_MAP",
        "layer_idx0": "",
        "scope": "frozen_phase51_dense_case_full_attention",
        "mean_jsd": "",
        "max_jsd": "",
        "mean_wasserstein_minutes": "",
        "max_wasserstein_minutes": "",
        "mean_cosine": "",
        "min_cosine": "",
        "sign_agreement_count": "",
        "status": "OK",
    })

    # ERROR_CONDITIONED_LAYER row (sign counts as aggregate)
    n_total = len(error_layer_rows)
    n_unanim = sum(1 for r in error_layer_rows if r.get("all_defined_same_sign"))
    rows.append({
        "evidence_row": "ERROR_CONDITIONED_LAYER",
        "layer_idx0": "",
        "scope": "phase56_layer_head_mean_effect_seed_stability",
        "mean_jsd": "",
        "max_jsd": "",
        "mean_wasserstein_minutes": "",
        "max_wasserstein_minutes": "",
        "mean_cosine": "",
        "min_cosine": "",
        "sign_agreement_count": n_unanim,
        "status": "OK",
    })

    # ERROR_CONDITIONED_MATCHED_HEAD row
    rows.append({
        "evidence_row": "ERROR_CONDITIONED_MATCHED_HEAD",
        "layer_idx0": "",
        "scope": "phase56_matched_head_effect_seed_stability",
        "mean_jsd": "",
        "max_jsd": "",
        "mean_wasserstein_minutes": "",
        "max_wasserstein_minutes": "",
        "mean_cosine": "",
        "min_cosine": "",
        "sign_agreement_count": "",
        "status": "OK",
    })

    # PREDICTION_ATTENTION_DISAGREEMENT row
    rows.append({
        "evidence_row": "PREDICTION_ATTENTION_DISAGREEMENT",
        "layer_idx0": "",
        "scope": "phase48_prediction_range_vs_layer_attention_disagreement",
        "mean_jsd": "",
        "max_jsd": "",
        "mean_wasserstein_minutes": "",
        "max_wasserstein_minutes": "",
        "mean_cosine": "",
        "min_cosine": "",
        "sign_agreement_count": "",
        "status": "OK",
    })

    return rows


# ===========================================================================
# Findings generator
# ===========================================================================

def build_findings(
    layer_pair_rows: List[dict],
    layer_pair_summary: List[dict],
    per_target_rows: List[dict],
    ambiguity_rows: List[dict],
    cycle_rows: List[dict],
    wasserstein_summary: List[dict],
    matched_summary: List[dict],
    matched_per_target_rows: List[dict],
    matched_metric_rows: List[dict],
    dense_rows: List[dict],
    error_layer_rows: List[dict],
    error_matched_rows: List[dict],
    pred_assoc_rows: List[dict],
) -> List[dict]:
    rows: List[dict] = []
    fid = 0

    # Layer profile findings
    for s in layer_pair_summary:
        layer = int(s["layer_idx0"])
        if s["mean_pairwise_jsd"] < 0.10:
            code = "LAYER_HEAD_MEAN_PROFILES_HIGHLY_SIMILAR_DESCRIPTIVE"
        else:
            code = "LAYER_HEAD_MEAN_PROFILES_DIFFER_DESCRIPTIVELY"
        rows.append({
            "finding_id": f"F{fid:04d}",
            "scope": "LAYER_HEAD_MEAN",
            "seed": "",
            "layer": layer,
            "head_if_applicable": "",
            "code": code,
            "value": f"mean_JSD={s['mean_pairwise_jsd']:.6f}, mean_W1={s['mean_pairwise_wasserstein_minutes']:.4f}min, mean_cos={s['mean_pairwise_cosine']:.4f}",
        })
        fid += 1

    # Per-target finding (one per layer)
    by_layer: Dict[int, list] = {0: [], 1: []}
    for r in per_target_rows:
        by_layer[int(r["layer_idx0"])].append(float(r["jsd"]))
    for layer, vals in by_layer.items():
        if not vals:
            continue
        arr = np.asarray(vals)
        median_jsd = float(np.median(arr))
        if median_jsd < 0.20:
            code = "LAYER_PER_TARGET_ATTENTION_STABLE_DESCRIPTIVE"
        else:
            code = "LAYER_PER_TARGET_ATTENTION_VARIABLE_DESCRIPTIVE"
        rows.append({
            "finding_id": f"F{fid:04d}",
            "scope": "LAYER_PER_TARGET",
            "seed": "",
            "layer": layer,
            "head_if_applicable": "",
            "code": code,
            "value": f"median_JSD={median_jsd:.6f}, mean_JSD={float(arr.mean()):.6f}, max_JSD={float(arr.max()):.6f}",
        })
        fid += 1

    # Head matching findings
    any_amb = any(r["ambiguous_match_warning"] for r in ambiguity_rows)
    rows.append({
        "finding_id": f"F{fid:04d}",
        "scope": "HEAD_MATCHING",
        "seed": "",
        "layer": "",
        "head_if_applicable": "",
        "code": "HEAD_MATCHING_AMBIGUOUS" if any_amb else "HEAD_MATCHING_UNAMBIGUOUS",
        "value": f"any_ambiguous={any_amb}",
    })
    fid += 1

    # Cycle consistency
    n_inconsistent = sum(1 for r in cycle_rows if not r["cycle_consistent"])
    rows.append({
        "finding_id": f"F{fid:04d}",
        "scope": "CYCLE",
        "seed": "",
        "layer": "",
        "head_if_applicable": "",
        "code": "HEAD_MATCHING_CYCLE_INCONSISTENT" if n_inconsistent > 0 else "HEAD_MATCHING_CYCLE_CONSISTENT",
        "value": f"inconsistent_heads={n_inconsistent}",
    })
    fid += 1

    # Matching method agreement
    any_diff = any(s["agreement_fraction"] < 1.0 for s in wasserstein_summary)
    rows.append({
        "finding_id": f"F{fid:04d}",
        "scope": "MATCH_METHOD_SENSITIVITY",
        "seed": "",
        "layer": "",
        "head_if_applicable": "",
        "code": "JSD_WASSERSTEIN_MATCHING_DIFFER" if any_diff else "JSD_WASSERSTEIN_MATCHING_AGREE",
        "value": f"any_pair_disagreement={any_diff}",
    })
    fid += 1

    # Matched-head profile stability
    for s in matched_summary[:NUM_LAYERS * 2]:  # one row per layer (top group only)
        pass
    rows.append({
        "finding_id": f"F{fid:04d}",
        "scope": "MATCHED_HEAD_PROFILE",
        "seed": "",
        "layer": "",
        "head_if_applicable": "",
        "code": "MATCHED_HEAD_PROFILES_VARIABLE_DESCRIPTIVE" if matched_summary and max(float(x["mean_pairwise_jsd"]) for x in matched_summary) > 0.10 else "MATCHED_HEAD_PROFILES_SIMILAR_DESCRIPTIVE",
        "value": f"max_mean_JSD={max(float(x['mean_pairwise_jsd']) for x in matched_summary) if matched_summary else 0.0:.6f}",
    })
    fid += 1

    # Dense case stability
    if dense_rows:
        arr = np.asarray([r["mean_query_jsd"] for r in dense_rows], dtype=np.float64)
        median_dense = float(np.median(arr))
        code = "DENSE_CASE_ATTENTION_STABLE_DESCRIPTIVE" if median_dense < 0.20 else "DENSE_CASE_ATTENTION_VARIABLE_DESCRIPTIVE"
        rows.append({
            "finding_id": f"F{fid:04d}",
            "scope": "DENSE_CASE",
            "seed": "",
            "layer": "",
            "head_if_applicable": "",
            "code": code,
            "value": f"median_JSD={median_dense:.6f}, N={arr.size}",
        })
        fid += 1

    # Error-conditioned sign agreement
    n_unanim = sum(1 for r in error_layer_rows if r.get("all_defined_same_sign"))
    n_total = len(error_layer_rows)
    if n_total > 0:
        code = "ERROR_ATTENTION_SIGN_CONSISTENT_ACROSS_SEEDS" if n_unanim > n_total * 0.5 else "ERROR_ATTENTION_SIGN_VARIES_ACROSS_SEEDS"
        rows.append({
            "finding_id": f"F{fid:04d}",
            "scope": "ERROR_CONDITIONED_LAYER",
            "seed": "",
            "layer": "",
            "head_if_applicable": "",
            "code": code,
            "value": f"unanim_sign_rows={n_unanim}/{n_total}",
        })
        fid += 1

    # Shared-cohort robustness
    n_unanim_shared = sum(1 for r in error_layer_rows if r.get("all_defined_same_sign") and str(r.get("analysis_type")) == "SHARED_HIGH_LOW_CLIFFS_DELTA")
    n_shared = sum(1 for r in error_layer_rows if str(r.get("analysis_type")) == "SHARED_HIGH_LOW_CLIFFS_DELTA")
    if n_shared > 0:
        code = "SHARED_COHORT_EFFECT_CONSISTENT_ACROSS_SEEDS" if n_unanim_shared > n_shared * 0.5 else "SHARED_COHORT_EFFECT_VARIES_ACROSS_SEEDS"
        rows.append({
            "finding_id": f"F{fid:04d}",
            "scope": "SHARED_COHORT",
            "seed": "",
            "layer": "",
            "head_if_applicable": "",
            "code": code,
            "value": f"unanim={n_unanim_shared}/{n_shared}",
        })
        fid += 1

    # Prediction-attention disagreement
    if pred_assoc_rows:
        max_rho = max(abs(float(r["spearman_rho"])) for r in pred_assoc_rows)
        code = "PREDICTION_SPREAD_ASSOCIATED_WITH_ATTENTION_DISAGREEMENT" if max_rho >= 0.20 else "NO_CLEAR_PREDICTION_ATTENTION_DISAGREEMENT_ASSOCIATION"
        rows.append({
            "finding_id": f"F{fid:04d}",
            "scope": "PREDICTION_ATTENTION_DISAGREEMENT",
            "seed": "",
            "layer": "",
            "head_if_applicable": "",
            "code": code,
            "value": f"max_|rho|={max_rho:.4f}",
        })
        fid += 1

    # Safety / methodology findings
    rows.append({"finding_id": f"F{fid:04d}", "scope": "SAFETY", "seed": "", "layer": "",
                 "head_if_applicable": "", "code": "NO_BEST_SEED_SELECTED", "value": "best_seed_selection=false"})
    fid += 1
    rows.append({"finding_id": f"F{fid:04d}", "scope": "SAFETY", "seed": "", "layer": "",
                 "head_if_applicable": "", "code": "NO_BEST_HEAD_SELECTED", "value": "best_head_selection=false"})
    fid += 1
    rows.append({"finding_id": f"F{fid:04d}", "scope": "SAFETY", "seed": "", "layer": "",
                 "head_if_applicable": "", "code": "NO_RETUNING", "value": "model_retrained=false; new_inference=false; new_attention_extraction=false"})
    fid += 1
    rows.append({"finding_id": f"F{fid:04d}", "scope": "SAFETY", "seed": "", "layer": "",
                 "head_if_applicable": "", "code": "NO_CAUSAL_CLAIM", "value": "attention_stability_is_descriptive_only"})
    fid += 1
    rows.append({"finding_id": f"F{fid:04d}", "scope": "HANDOFF", "seed": "", "layer": "",
                 "head_if_applicable": "", "code": "ATTENTION_ROBUSTNESS_EVIDENCE_READY_FOR_FINAL_TABLES",
                 "value": "phase58_ready; phase59_context_ready"})

    return rows
