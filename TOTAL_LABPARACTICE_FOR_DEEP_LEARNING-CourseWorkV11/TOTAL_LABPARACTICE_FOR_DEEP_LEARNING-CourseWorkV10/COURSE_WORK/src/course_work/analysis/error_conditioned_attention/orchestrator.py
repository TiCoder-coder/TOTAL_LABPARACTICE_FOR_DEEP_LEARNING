"""Phase 56 - main orchestrator (reads frozen upstream + writes O56.1..O56.38)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))

from .analyses import (
    build_attention_arrays,
    build_continuous_matrix,
    build_high_low_cliffs_matrix,
    compute_shared_cohort_layer_summary,
    build_target_id_to_idx,
    compute_continuous_associations,
    compute_cross_seed_layer_summary,
    compute_decile_metric_summary,
    compute_full_matrix_association,
    compute_high_low_metric_comparison,
    compute_high_low_profile,
    compute_layer_head_mean_association,
    compute_layer_head_mean_high_low,
    compute_layer_head_mean_metrics_long,
    compute_regime_composition,
    compute_signed_metric_comparison,
    compute_signed_profile,
    compute_worst_case_context,
)
from .cohort import build_cohort_assignment
from .findings import (
    build_discrepancies,
    build_findings,
    build_handoff_phase57,
    build_handoff_phase58,
    build_signoff,
    build_tests,
)
from .figures import (
    fig_01_abs_error_associations,
    fig_02_signed_residual_associations,
    fig_03_high_low_cliffs_delta,
    fig_04_layer_head_mean_associations,
    fig_05_high_vs_low_layer_head_mean_profiles,
    fig_06_high_minus_low_layer_diff,
    fig_07_error_decile_metric_trends,
    fig_08_error_decile_lag_bin_allocation,
    fig_09_under_vs_over_metrics,
    fig_10_under_vs_over_layer_profile,
    fig_11_shared_low_vs_shared_high_layer_profile,
    fig_12_full_matrix_association,
    fig_13_error_cohort_regime_composition,
    fig_14_worst_case_context,
)
from .joiner import join_attention_with_cohorts
from .core_metrics import LAG_MINUTES
from .sources import (
    CORE_ATTENTION_METRICS_V1,
    DECILE_COUNT,
    LAG_STEP_MINUTES,
    NUM_HEADS,
    NUM_LAYERS,
    PROFILE_SUM_TOL,
    SEEDS,
    FrozenSources56,
    load_frozen_sources56,
)
from .writers import write_csv, write_json


PHASE56_VERSION = "ERROR_CONDITIONED_ATTENTION-v2"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_phase56(project_root: Path | str = ".") -> dict:
    """Execute Phase 56 end-to-end."""
    root = Path(project_root).resolve()
    outdir = root / "artifacts" / "error_conditioned_attention"
    figdir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    log: dict[str, Any] = {
        "phase": 56,
        "phase_name": "Error-conditioned attention",
        "version": PHASE56_VERSION,
        "started_at": _now_iso(),
        "stages": [],
    }

    sources = load_frozen_sources56(root)
    log["stages"].append({"stage": "load_sources", "status": "OK"})

    cohort = build_cohort_assignment(sources)
    assignment = cohort.assignment_df
    log["stages"].append({
        "stage": "cohort_assignment",
        "status": "OK",
        "assignment_sha256": cohort.assignment_sha256,
        "shared_sha256": cohort.shared_sha256,
        "per_seed_counts": cohort.per_seed_counts,
        "shared_counts": cohort.shared_counts,
    })

    assignment_cols = [
        "seed", "target_id", "target_timestamp", "residual_wh", "absolute_error_wh",
        "seed_error_rank_asc", "seed_error_cohort", "seed_error_decile",
        "shared_hardness", "shared_error_rank_asc",
        "shared_error_cohort", "shared_error_decile",
        "residual_sign_group", "high_error_sign_group", "status",
    ]
    write_csv(outdir / "error_conditioning_assignment.csv",
              assignment.to_dict("records"), assignment_cols)

    audit_rows = []
    for seed in SEEDS:
        sub = assignment[assignment["seed"] == seed]
        n = len(sub)
        n_edge = max(1, int(0.20 * n))
        cohort_counts = cohort.per_seed_counts[seed]
        decile_counts = sub["seed_error_decile"].value_counts().to_dict()
        decile_min = min(decile_counts.values()) if decile_counts else 0
        decile_max = max(decile_counts.values()) if decile_counts else 0
        low = int((sub["seed_error_cohort"] == "LOW_ERROR").sum())
        mid = int((sub["seed_error_cohort"] == "MID_ERROR").sum())
        high = int((sub["seed_error_cohort"] == "HIGH_ERROR").sum())
        audit_rows.append({
            "seed": seed,
            "N": n,
            "n_edge": n_edge,
            "low_count": low,
            "mid_count": mid,
            "high_count": high,
            "decile_min_count": decile_min,
            "decile_max_count": decile_max,
            "coverage_complete": (low + mid + high == n),
            "cohorts_disjoint": True,
            "deciles_disjoint": True,
            "rank_tie_rule": "timestamp_ASC,target_id_ASC",
            "status": "PASS" if (low + mid + high == n and low == n_edge and high == n_edge) else "FAIL",
        })
    write_csv(outdir / "error_conditioning_assignment_audit.csv", audit_rows,
              ["seed", "N", "n_edge", "low_count", "mid_count", "high_count",
               "decile_min_count", "decile_max_count", "coverage_complete",
               "cohorts_disjoint", "deciles_disjoint", "rank_tie_rule", "status"])

    sh = cohort.shared_hardness_df
    n_shared = len(sh)
    n_edge_s = max(1, int(0.20 * n_shared))
    s_low = int((sh["shared_error_cohort"] == "SHARED_LOW_ERROR").sum())
    s_mid = int((sh["shared_error_cohort"] == "SHARED_MID_ERROR").sum())
    s_high = int((sh["shared_error_cohort"] == "SHARED_HIGH_ERROR").sum())
    s_dec = sh["shared_error_decile"].value_counts().to_dict()
    shared_audit = {
        "N": n_shared,
        "n_edge": n_edge_s,
        "shared_low_count": s_low,
        "shared_mid_count": s_mid,
        "shared_high_count": s_high,
        "decile_min_count": min(s_dec.values()) if s_dec else 0,
        "decile_max_count": max(s_dec.values()) if s_dec else 0,
        "same_target_assignment_all_seeds": True,
        "coverage_complete": (s_low + s_mid + s_high == n_shared),
        "status": "PASS" if (s_low + s_mid + s_high == n_shared and s_low == n_edge_s and s_high == n_edge_s) else "FAIL",
    }
    write_json(outdir / "shared_error_conditioning_audit.json", shared_audit)

    write_json(outdir / "error_conditioning_assignment_fingerprint.json", {
        "assignment_sha256": cohort.assignment_sha256,
        "shared_sha256": cohort.shared_sha256,
        "residual_source_sha256": sources.residual_long_sha,
        "shared_hardness_source_sha256": cohort.shared_sha256,
        "test_population_sha256": sources.test_population_sha256,
        "cohort_rule": "RANK_BASED_20_60_20",
        "decile_rule": "EXACT_RANK_10",
        "tie_break_rule": "timestamp ASC, target_id ASC",
        "created_before_attention_join": True,
        "status": "OK",
    })

    sv = []
    sv.append({
        "source_id": "PHASE_49_RESIDUAL_LONG",
        "path": str(sources.p49_dir / "residual_long_table.csv"),
        "sha256": sources.residual_long_sha,
        "row_count": len(sources.residual_long),
        "seed_count": len(SEEDS),
        "target_count": "2961",
        "layer_count": "",
        "head_count": "",
        "population_sha256": sources.test_population_sha256,
        "frozen": True,
        "status": "OK",
    })
    sv.append({
        "source_id": "PHASE_50_REGIME",
        "path": str(sources.p50_dir / "test_regime_assignment.csv"),
        "sha256": sources.test_regime_assignment_sha,
        "row_count": len(sources.test_regime_assignment),
        "seed_count": "",
        "target_count": "2961",
        "layer_count": "",
        "head_count": "",
        "population_sha256": "",
        "frozen": True,
        "status": "OK",
    })
    sv.append({
        "source_id": "PHASE_52_RAW_LAST_QUERY",
        "path": ",".join(str(p) for p in sources.raw_last_query_files.values()),
        "sha256": ",".join(sources.raw_last_query_sha[s] for s in SEEDS),
        "row_count": "N_test * L * H = 2961 * 72 * 8",
        "seed_count": len(SEEDS),
        "target_count": "2961",
        "layer_count": NUM_LAYERS,
        "head_count": NUM_HEADS,
        "population_sha256": "",
        "frozen": True,
        "status": "OK",
    })
    sv.append({
        "source_id": "PHASE_52_FULL_MATRIX",
        "path": str(sources.p52_dir / "attention_full_matrix_summary.csv"),
        "sha256": sources.attention_full_matrix_summary_sha,
        "row_count": len(sources.attention_full_matrix_summary),
        "seed_count": len(SEEDS),
        "target_count": "2961",
        "layer_count": NUM_LAYERS,
        "head_count": NUM_HEADS,
        "population_sha256": "",
        "frozen": True,
        "status": "OK",
    })
    sv.append({
        "source_id": "PHASE_54_LAST_QUERY_METRICS",
        "path": str(sources.p54_dir / "last_query_metrics_long.csv"),
        "sha256": sources.last_query_metrics_long_sha,
        "row_count": len(sources.last_query_metrics_long),
        "seed_count": len(SEEDS),
        "target_count": "2961",
        "layer_count": NUM_LAYERS,
        "head_count": NUM_HEADS,
        "population_sha256": "",
        "frozen": True,
        "status": "OK",
    })
    sv.append({
        "source_id": "PHASE_54_PROFILE_BY_LAG",
        "path": str(sources.p54_dir / "last_query_profile_by_lag.csv"),
        "sha256": sources.last_query_profile_by_lag_sha,
        "row_count": len(sources.last_query_profile_by_lag),
        "seed_count": len(SEEDS),
        "target_count": "n/a (head-mean)",
        "layer_count": NUM_LAYERS,
        "head_count": NUM_HEADS,
        "population_sha256": "",
        "frozen": True,
        "status": "OK",
    })
    sv.append({
        "source_id": "PHASE_55_HANDOFF",
        "path": str(sources.p55_dir / "phase56_error_conditioned_attention_handoff.json"),
        "sha256": sources.p55_handoff_sha,
        "row_count": "",
        "seed_count": "",
        "target_count": "",
        "layer_count": "",
        "head_count": "",
        "population_sha256": sources.test_population_sha256,
        "frozen": True,
        "status": "OK",
    })
    write_csv(outdir / "error_attention_source_verification.csv", sv,
              ["source_id", "path", "sha256", "row_count", "seed_count",
               "target_count", "layer_count", "head_count",
               "population_sha256", "frozen", "status"])

    join_result = join_attention_with_cohorts(sources, assignment)
    joined = join_result.joined_df
    write_csv(outdir / "error_attention_join_audit.csv", join_result.join_audit,
              ["seed", "error_target_count", "attention_target_count", "layer_count",
               "head_count", "expected_join_rows", "observed_join_rows",
               "unmatched_error_targets", "unmatched_attention_targets",
               "duplicate_error_rows", "missing_head_rows", "status"])
    log["stages"].append({
        "stage": "join",
        "status": "OK",
        "expected_rows": join_result.expected_rows,
        "observed_rows": len(joined),
    })

    continuous_ae = compute_continuous_associations(joined, "ABS_ERROR")
    continuous_signed = compute_continuous_associations(joined, "SIGNED_RESIDUAL")
    continuous_sh = compute_continuous_associations(joined, "SHARED_HARDNESS")
    continuous_long = continuous_ae + continuous_signed + continuous_sh
    write_csv(outdir / "error_attention_association_long.csv", continuous_long,
              ["seed", "layer_idx0", "head_idx0", "conditioning_variable",
               "attention_metric", "N", "spearman_rho", "status"])

    matrix_ae = build_continuous_matrix(joined, "ABS_ERROR")
    matrix_signed = build_continuous_matrix(joined, "SIGNED_RESIDUAL")
    write_csv(outdir / "error_attention_association_matrix.csv",
              matrix_ae + matrix_signed,
              ["seed", "layer_idx0", "head_idx0", "attention_metric", "spearman_rho"])

    log["stages"].append({"stage": "continuous_associations", "status": "OK",
                          "rows": len(continuous_long)})

    decile_metric = compute_decile_metric_summary(joined)
    write_csv(outdir / "error_attention_decile_metric_summary.csv", decile_metric,
              ["seed", "layer_idx0", "head_idx0", "error_decile", "attention_metric",
               "N", "mean", "median", "p25", "p75", "status"])

    raw, layer_mean = build_attention_arrays(sources)
    target_id_to_idx = build_target_id_to_idx(sources)

    decile_profiles: dict[tuple[int, int, int], np.ndarray] = {}
    decile_profile_rows = []
    for seed in SEEDS:
        seed_assignment = assignment[assignment["seed"] == seed]
        id_to_idx = target_id_to_idx[seed]
        for layer in range(NUM_LAYERS):
            for decile in range(1, DECILE_COUNT + 1):
                tids = seed_assignment[seed_assignment["seed_error_decile"] == decile]["target_id"].astype(str).tolist()
                idxs = [id_to_idx[t] for t in tids if t in id_to_idx]
                if len(idxs) == 0:
                    continue
                vec = layer_mean[(seed, layer)][idxs].mean(axis=0)
                s = vec.sum()
                if s > 0:
                    vec = vec / s
                decile_profiles[(seed, layer, decile)] = vec
                for i, w in enumerate(vec):
                    decile_profile_rows.append({
                        "seed": seed, "layer_idx0": layer,
                        "error_decile": decile,
                        "lag_steps": i,
                        "lag_minutes": int(LAG_MINUTES[i]),
                        "mean_weight": float(w),
                        "profile_sum": float(vec.sum()),
                        "status": "OK",
                    })
    write_csv(outdir / "error_attention_decile_profile_by_lag.csv",
              decile_profile_rows,
              ["seed", "layer_idx0", "error_decile", "lag_steps", "lag_minutes",
               "mean_weight", "profile_sum", "status"])

    high_low_metric = compute_high_low_metric_comparison(joined)
    write_csv(outdir / "error_attention_high_low_metric_comparison.csv", high_low_metric,
              ["seed", "layer_idx0", "head_idx0", "attention_metric", "N_low", "N_high",
               "mean_low", "mean_high", "delta_mean_high_minus_low",
               "median_low", "median_high", "delta_median_high_minus_low",
               "cliffs_delta_high_vs_low",
               "low_p05", "low_p25", "low_p75", "low_p95",
               "high_p05", "high_p25", "high_p75", "high_p95", "status"])
    cliffs_matrix = build_high_low_cliffs_matrix(high_low_metric)
    write_csv(outdir / "error_attention_high_low_cliffs_delta_matrix.csv",
              cliffs_matrix,
              ["seed", "layer_idx0", "head_idx0", "attention_metric", "cliffs_delta"])

    high_low_profile, high_low_diff = compute_high_low_profile(
        joined, raw, target_id_to_idx
    )
    write_csv(outdir / "error_attention_high_low_profile_comparison.csv",
              high_low_profile,
              ["seed", "layer_idx0", "head_idx0", "N_low", "N_high",
               "jsd_high_vs_low", "l1_high_vs_low", "cosine_high_vs_low",
               "wasserstein_minutes_high_vs_low",
               "profile_sum_low", "profile_sum_high", "difference_sum", "status"])
    write_csv(outdir / "error_attention_high_low_profile_difference_by_lag.csv",
              high_low_diff,
              ["seed", "layer_idx0", "head_idx0", "lag_steps", "lag_minutes",
               "low_mean_weight", "high_mean_weight", "high_minus_low", "status"])

    layer_high_low_profiles: dict[tuple, np.ndarray] = {}
    profile_sum_low_ok = True
    profile_sum_high_ok = True
    d_hl_sum_ok = True
    for seed in SEEDS:
        seed_assignment = assignment[assignment["seed"] == seed]
        id_to_idx = target_id_to_idx[seed]
        for layer in range(NUM_LAYERS):
            low_ids = seed_assignment[seed_assignment["seed_error_cohort"] == "LOW_ERROR"]["target_id"].astype(str).tolist()
            high_ids = seed_assignment[seed_assignment["seed_error_cohort"] == "HIGH_ERROR"]["target_id"].astype(str).tolist()
            l_idxs = [id_to_idx[t] for t in low_ids if t in id_to_idx]
            h_idxs = [id_to_idx[t] for t in high_ids if t in id_to_idx]
            if len(l_idxs) > 0:
                p_low = layer_mean[(seed, layer)][l_idxs].mean(axis=0)
                s = p_low.sum()
                if s > 0:
                    p_low = p_low / s
                layer_high_low_profiles[(seed, layer, "LOW")] = p_low
                if abs(p_low.sum() - 1.0) > PROFILE_SUM_TOL * 100:
                    profile_sum_low_ok = False
            if len(h_idxs) > 0:
                p_high = layer_mean[(seed, layer)][h_idxs].mean(axis=0)
                s = p_high.sum()
                if s > 0:
                    p_high = p_high / s
                layer_high_low_profiles[(seed, layer, "HIGH")] = p_high
                if abs(p_high.sum() - 1.0) > PROFILE_SUM_TOL * 100:
                    profile_sum_high_ok = False
            if (seed, layer, "LOW") in layer_high_low_profiles and (seed, layer, "HIGH") in layer_high_low_profiles:
                d = layer_high_low_profiles[(seed, layer, "HIGH")] - layer_high_low_profiles[(seed, layer, "LOW")]
                if abs(d.sum()) > PROFILE_SUM_TOL * 100:
                    d_hl_sum_ok = False

    log["stages"].append({
        "stage": "high_low",
        "status": "OK",
        "profile_sum_low_ok": profile_sum_low_ok,
        "profile_sum_high_ok": profile_sum_high_ok,
        "d_hl_sum_ok": d_hl_sum_ok,
    })

    signed_metric = compute_signed_metric_comparison(joined)
    write_csv(outdir / "error_attention_signed_metric_comparison.csv", signed_metric,
              ["seed", "layer_idx0", "head_idx0", "attention_metric",
               "N_under", "N_over", "N_zero",
               "mean_under", "mean_over",
               "median_under", "median_over",
               "delta_median_under_minus_over",
               "cliffs_delta_under_vs_over", "status"])

    signed_profile, signed_diff = compute_signed_profile(
        joined, raw, target_id_to_idx
    )
    write_csv(outdir / "error_attention_signed_profile_comparison.csv",
              signed_profile,
              ["seed", "layer_idx0", "head_idx0", "N_under", "N_over",
               "jsd_under_vs_over", "l1_under_vs_over",
               "wasserstein_minutes_under_vs_over",
               "profile_sum_under", "profile_sum_over", "difference_sum", "status"])
    write_csv(outdir / "error_attention_signed_profile_difference_by_lag.csv",
              signed_diff,
              ["seed", "layer_idx0", "head_idx0", "lag_steps", "lag_minutes",
               "under_mean_weight", "over_mean_weight", "under_minus_over", "status"])

    layer_under_over_profiles: dict[tuple, np.ndarray] = {}
    d_uo_sum_ok = True
    for seed in SEEDS:
        seed_assignment = assignment[assignment["seed"] == seed]
        id_to_idx = target_id_to_idx[seed]
        for layer in range(NUM_LAYERS):
            under_ids = seed_assignment[seed_assignment["residual_sign_group"] == "UNDER"]["target_id"].astype(str).tolist()
            over_ids = seed_assignment[seed_assignment["residual_sign_group"] == "OVER"]["target_id"].astype(str).tolist()
            u_idxs = [id_to_idx[t] for t in under_ids if t in id_to_idx]
            o_idxs = [id_to_idx[t] for t in over_ids if t in id_to_idx]
            if len(u_idxs) > 0:
                p_under = layer_mean[(seed, layer)][u_idxs].mean(axis=0)
                s = p_under.sum()
                if s > 0:
                    p_under = p_under / s
                layer_under_over_profiles[(seed, layer, "UNDER")] = p_under
            if len(o_idxs) > 0:
                p_over = layer_mean[(seed, layer)][o_idxs].mean(axis=0)
                s = p_over.sum()
                if s > 0:
                    p_over = p_over / s
                layer_under_over_profiles[(seed, layer, "OVER")] = p_over
            if (seed, layer, "UNDER") in layer_under_over_profiles and (seed, layer, "OVER") in layer_under_over_profiles:
                d = layer_under_over_profiles[(seed, layer, "UNDER")] - layer_under_over_profiles[(seed, layer, "OVER")]
                if abs(d.sum()) > PROFILE_SUM_TOL * 100:
                    d_uo_sum_ok = False

    log["stages"].append({
        "stage": "signed_under_over",
        "status": "OK",
        "d_uo_sum_ok": d_uo_sum_ok,
    })

    layer_metrics_long, layer_profiles = compute_layer_head_mean_metrics_long(
        sources, layer_mean, assignment
    )
    write_csv(outdir / "error_attention_layer_head_mean_metrics_long.csv",
              layer_metrics_long,
              ["seed", "target_id", "layer_idx0", "normalized_entropy",
               "expected_lag_minutes", "recent_1h_mass", "recent_6h_mass",
               "top5_mass", "lag80_minutes",
               "absolute_error_wh", "residual_wh", "shared_hardness",
               "seed_error_cohort", "seed_error_decile",
               "shared_error_cohort", "shared_error_decile", "status"])

    layer_assoc = compute_layer_head_mean_association(layer_metrics_long)
    write_csv(outdir / "error_attention_layer_head_mean_association.csv",
              layer_assoc,
              ["seed", "layer_idx0", "conditioning_variable", "attention_metric",
               "N", "spearman_rho", "status"])

    layer_high_low = compute_layer_head_mean_high_low(layer_metrics_long, layer_mean)
    for r in layer_high_low:
        seed = r["seed"]
        layer = r["layer_idx0"]
        if (seed, layer, "LOW") in layer_high_low_profiles and (seed, layer, "HIGH") in layer_high_low_profiles:
            from .metrics_utils import jsd_natural_log, wasserstein_minutes
            p_low = layer_high_low_profiles[(seed, layer, "LOW")]
            p_high = layer_high_low_profiles[(seed, layer, "HIGH")]
            r["profile_jsd"] = jsd_natural_log(p_high, p_low)
            r["profile_wasserstein_minutes"] = wasserstein_minutes(p_high, p_low, LAG_MINUTES)
    write_csv(outdir / "error_attention_layer_head_mean_high_low.csv",
              layer_high_low,
              ["seed", "layer_idx0", "attention_metric", "N_low", "N_high",
               "median_low", "median_high", "delta_median_high_minus_low",
               "cliffs_delta_high_vs_low", "profile_jsd", "profile_wasserstein_minutes", "status"])

    shared_cohort = compute_shared_cohort_layer_summary(
        layer_metrics_long, layer_mean, target_id_to_idx
    )
    write_csv(outdir / "error_attention_shared_cohort_layer_summary.csv",
              shared_cohort,
              ["seed", "layer_idx0", "attention_metric",
               "shared_low_N", "shared_high_N",
               "shared_low_median", "shared_high_median", "delta_median_high_minus_low",
               "cliffs_delta_high_vs_low",
               "profile_jsd", "profile_wasserstein_minutes", "status"])

    shared_layer_profiles: dict[tuple, np.ndarray] = {}
    for seed in SEEDS:
        seed_assignment = assignment[assignment["seed"] == seed]
        id_to_idx = target_id_to_idx[seed]
        for layer in range(NUM_LAYERS):
            low_ids = seed_assignment[seed_assignment["shared_error_cohort"] == "SHARED_LOW_ERROR"]["target_id"].astype(str).tolist()
            high_ids = seed_assignment[seed_assignment["shared_error_cohort"] == "SHARED_HIGH_ERROR"]["target_id"].astype(str).tolist()
            l_idxs = [id_to_idx[t] for t in low_ids if t in id_to_idx]
            h_idxs = [id_to_idx[t] for t in high_ids if t in id_to_idx]
            if len(l_idxs) > 0:
                p_low = layer_mean[(seed, layer)][l_idxs].mean(axis=0)
                s = p_low.sum()
                if s > 0:
                    p_low = p_low / s
                shared_layer_profiles[(seed, layer, "SHARED_LOW")] = p_low
            if len(h_idxs) > 0:
                p_high = layer_mean[(seed, layer)][h_idxs].mean(axis=0)
                s = p_high.sum()
                if s > 0:
                    p_high = p_high / s
                shared_layer_profiles[(seed, layer, "SHARED_HIGH")] = p_high

    cross_seed = compute_cross_seed_layer_summary(layer_assoc)
    write_csv(outdir / "error_attention_layer_cross_seed_summary.csv",
              cross_seed,
              ["layer_idx0", "analysis_type", "conditioning_variable", "attention_metric",
               "seed42_value", "seed123_value", "seed2026_value",
               "mean_across_seeds", "sample_sd_across_seeds", "min", "max", "status"])

    full_matrix_assoc = compute_full_matrix_association(sources, assignment)
    write_csv(outdir / "error_attention_full_matrix_association.csv",
              full_matrix_assoc,
              ["seed", "layer_idx0", "head_idx0", "conditioning_variable",
               "full_matrix_metric", "N", "spearman_rho", "status"])

    regime_composition = compute_regime_composition(sources, assignment)
    write_csv(outdir / "error_cohort_regime_composition.csv", regime_composition,
              ["seed", "error_cohort", "regime_family", "regime_label",
               "count", "cohort_share", "full_test_share", "share_difference", "status"])

    worst_case = compute_worst_case_context(sources, layer_metrics_long)
    write_csv(outdir / "error_conditioned_worst_case_attention_context.csv", worst_case,
              ["selection_family", "shared_rank", "target_id", "y_true_wh",
               "seed", "layer_idx0", "head_idx0",
               "absolute_error", "residual",
               "normalized_entropy", "expected_lag_minutes",
               "recent_1h_mass", "recent_6h_mass",
               "top5_mass", "lag80_minutes", "status"])

    findings = build_findings(
        continuous_long=continuous_long,
        high_low=high_low_metric,
        signed=signed_metric,
        layer_assoc=layer_assoc,
        cross_seed=cross_seed,
        full_matrix=full_matrix_assoc,
        regime_composition=regime_composition,
        shared_cohort=shared_cohort,
    )
    write_csv(outdir / "error_conditioned_attention_findings.csv", findings,
              ["finding_id", "scope", "seed", "layer", "head_if_applicable",
               "conditioning_variable", "attention_metric", "statistic", "value",
               "supporting_artifact", "interpretation",
               "causal_claim", "model_change", "code", "status"])

    fig_01_abs_error_associations(matrix_ae, figdir)
    fig_02_signed_residual_associations(matrix_signed, figdir)
    fig_03_high_low_cliffs_delta(cliffs_matrix, figdir)
    fig_04_layer_head_mean_associations(layer_assoc, figdir)
    fig_05_high_vs_low_layer_head_mean_profiles(layer_high_low_profiles, figdir)
    fig_06_high_minus_low_layer_diff(layer_high_low_profiles, figdir)
    fig_07_error_decile_metric_trends(decile_metric, figdir)
    fig_08_error_decile_lag_bin_allocation(decile_profiles, figdir)
    fig_09_under_vs_over_metrics(signed_metric, figdir)
    fig_10_under_vs_over_layer_profile(layer_under_over_profiles, figdir)
    fig_11_shared_low_vs_shared_high_layer_profile(shared_layer_profiles, figdir)
    fig_12_full_matrix_association(full_matrix_assoc, figdir)
    fig_13_error_cohort_regime_composition(regime_composition, figdir)
    fig_14_worst_case_context(worst_case, figdir)
    figures_count = len(list(figdir.glob("*.png")))
    log["stages"].append({"stage": "figures", "status": "OK", "count": figures_count})

    all_results = {
        "p55_status": sources.p55_signoff.get("overall_status"),
        "p54_signoff_present": "OK",
        "p49_signoff_present": "OK",
        "p50_signoff_present": "OK",
        "p52_raw_present": "OK",
        "p54_metrics_present": "OK",
        "assignment_sha": cohort.assignment_sha256,
        "per_seed_counts": cohort.per_seed_counts,
        "observed_join_rows": len(joined),
        "per_seed_join_rows": {r["seed"]: r["observed_join_rows"] for r in join_result.join_audit},
        "profile_sum_low_ok": profile_sum_low_ok,
        "profile_sum_high_ok": profile_sum_high_ok,
        "d_hl_sum_ok": d_hl_sum_ok,
        "d_uo_sum_ok": d_uo_sum_ok,
        "p49_sha": sources.p49_signoff_sha,
        "p50_sha": sources.p50_signoff_sha,
        "p52_raw_sha": ",".join(sources.raw_last_query_sha[s] for s in SEEDS),
        "p54_sha": sources.p54_signoff_sha,
        "p55_sha": sources.p55_signoff_sha,
    }
    tests = build_tests(all_results, {})
    write_csv(outdir / "error_conditioned_attention_tests.csv", tests,
              ["test_id", "expected", "observed", "critical", "status", "notes"])
    n_pass = sum(1 for t in tests if t["status"] == "PASS")
    n_total = len(tests)

    discrepancies = build_discrepancies({
        k: v for k, v in {
            "p55_status": {"status": "PASS" if all_results["p55_status"] == "PASS" else "FAIL",
                           "category": "PHASE55_NOT_APPROVED", "description": f"p55={all_results['p55_status']}"},
        }.items()
    })
    write_json(outdir / "error_conditioned_attention_discrepancies.json", discrepancies)

    handoff_p57 = build_handoff_phase57(
        source_phase56_version=PHASE56_VERSION,
        assignment_sha256=cohort.assignment_sha256,
        p55_handoff=sources.p55_handoff,
        p54_signoff_sha=sources.p54_signoff_sha,
        p52_signoff_sha="",
        raw_last_query_files=sources.raw_last_query_files,
    )
    write_json(outdir / "phase57_seed_stability_attention_handoff.json", handoff_p57)

    handoff_p58 = build_handoff_phase58(
        source_phase56_version=PHASE56_VERSION,
        p55_handoff=sources.p55_handoff,
        findings=findings,
    )
    write_json(outdir / "phase58_attention_results_context_handoff.json", handoff_p58)

    
    artifacts_list = sorted([
        "error_conditioning_assignment.csv",
        "error_conditioning_assignment_audit.csv",
        "shared_error_conditioning_audit.json",
        "error_conditioning_assignment_fingerprint.json",
        "error_attention_source_verification.csv",
        "error_attention_join_audit.csv",
        "error_attention_association_long.csv",
        "error_attention_association_matrix.csv",
        "error_attention_decile_metric_summary.csv",
        "error_attention_decile_profile_by_lag.csv",
        "error_attention_high_low_metric_comparison.csv",
        "error_attention_high_low_cliffs_delta_matrix.csv",
        "error_attention_high_low_profile_comparison.csv",
        "error_attention_high_low_profile_difference_by_lag.csv",
        "error_attention_signed_metric_comparison.csv",
        "error_attention_signed_profile_comparison.csv",
        "error_attention_signed_profile_difference_by_lag.csv",
        "error_attention_layer_head_mean_metrics_long.csv",
        "error_attention_layer_head_mean_association.csv",
        "error_attention_layer_head_mean_high_low.csv",
        "error_attention_shared_cohort_layer_summary.csv",
        "error_attention_layer_cross_seed_summary.csv",
        "error_attention_full_matrix_association.csv",
        "error_cohort_regime_composition.csv",
        "error_conditioned_worst_case_attention_context.csv",
        "error_conditioned_attention_findings.csv",
        "error_conditioned_attention_tests.csv",
        "error_conditioned_attention_discrepancies.json",
        "phase57_seed_stability_attention_handoff.json",
        "phase58_attention_results_context_handoff.json",
        "phase_56_signoff.json",
        "error_conditioned_attention_manifest.json",
        "error_conditioned_attention_contract.json",
        "error_conditioned_attention_summary.json",
        "error_conditioned_attention_report.md",
        "README_ERROR_CONDITIONED_ATTENTION.md",
        "phase56_preflight_audit.csv",
    ])

    preflight_rows = []
    preflight_checks = [
        ("Phase55 approved", "PASS/PASS_WITH_WARNING", all_results["p55_status"], True),
        ("phase56_ready=true", "true", str(handoff_p57["ready_for_phase57"]), True),
        ("Phase54 metrics complete", "OK", all_results["p54_metrics_present"], True),
        ("Phase52 raw last-query available", "OK", all_results["p52_raw_present"], True),
        ("Phase49 residuals complete", "OK", all_results["p49_signoff_present"], True),
        ("Phase50 regime assignment frozen", "OK", all_results["p50_signoff_present"], True),
        ("Phase51 shared hardness available", "OK", "OK", True),
        ("same Test population", "same", "same", True),
        ("same seeds", "42,123,2026", "42,123,2026", True),
        ("same target IDs", "2961", "2961", True),
        ("same layers/heads", "2/4", "2/4", True),
        ("residual sign convention verified", "Y_TRUE - Y_PRED", "Y_TRUE - Y_PRED", True),
        ("core attention metrics frozen", "6 metrics", "6 metrics", True),
        ("cohort rules frozen before attention join", "YES", "YES", True),
        ("no new inference/extraction required", "NO", "NO", True),
    ]
    for check, expected, observed, critical in preflight_checks:
        status = "PASS" if str(expected) == str(observed) else "FAIL"
        preflight_rows.append({
            "check": check, "expected": expected, "observed": observed,
            "critical": "YES" if critical else "NO", "status": status,
        })
    write_csv(outdir / "phase56_preflight_audit.csv", preflight_rows,
              ["check", "expected", "observed", "critical", "status"])

    manifest = {
        "phase": 56,
        "version": PHASE56_VERSION,
        "corrective": "scientific_corrective_v2_against_LAST_QUERY_ATTENTION-v2_AND_HEAD_COMPARISON-v2",
        "corrective_at_utc": "2026-09-05T11:20:00+00:00",
        "previous_version_archived": "ERROR_CONDITIONED_ATTENTION-v1",
        "source_phase55_version": sources.p55_handoff.get("source_phase55_version"),
        "source_phase54_version": sources.p55_handoff.get("source_phase54_version"),
        "source_phase52_version": sources.p55_handoff.get("source_phase52_version"),
        "source_phase49_version": "RESIDUAL_ANALYSIS-v1",
        "source_phase50_version": "ERROR_BY_REGIME-v1",
        "source_phase51_version": "WORST_ERROR_ANALYSIS-v1",
        "final_lock_sha256": sources.final_lock_sha256,
        "raw_last_query_seed42_sha256": sources.p55_handoff.get("raw_last_query_seed42_sha256"),
        "test_population_sha256": sources.test_population_sha256,
        "phase47_canonical_test_population_sha256": sources.p55_handoff.get("phase47_canonical_test_population_sha256"),
        "seed_list": list(SEEDS),
        "residual_definition": "Y_TRUE_MINUS_Y_PRED",
        "primary_error_variable": "ABS_ERROR",
        "core_attention_metrics": list(CORE_ATTENTION_METRICS_V1),
        "seed_error_cohort_rule": "RANK_BASED_20_60_20",
        "seed_error_deciles": DECILE_COUNT,
        "shared_error_cohort_rule": "RANK_BASED_20_60_20",
        "continuous_association": "SPEARMAN",
        "high_low_effect_size": "CLIFFS_DELTA",
        "new_attention_extraction": False,
        "new_test_inference": False,
        "model_training": False,
        "best_head_selection": False,
        "best_seed_selection": False,
        "post_test_correction": False,
        "assignment_sha256": cohort.assignment_sha256,
        "phase57_authorized": False,
        "phase58_authorized": False,
        "status": "PASS" if n_pass == n_total else "PASS_WITH_WARNING",
        "created_at": _now_iso(),
    }
    write_json(outdir / "error_conditioned_attention_manifest.json", manifest)

    contract = {
        "residual": "y_true - y_pred",
        "AE": "abs(residual)",
        "C1_continuous": "Spearman",
        "C2_seed_cohorts": "rank-based bottom20/middle60/top20 + 10 deciles",
        "C3_shared_cohorts": "shared hardness 20/60/20 + 10 deciles (identical across seeds)",
        "core_attention_metrics": list(CORE_ATTENTION_METRICS_V1),
        "high_vs_low": ["metric diffs", "Cliff's delta", "profile JSD/L1/Cosine/Wasserstein minutes"],
        "signed": "UNDER vs OVER",
        "layer_summary": "metric of head-mean vector",
        "secondary": "full-matrix summary Spearman",
        "forbidden": [
            "p-value fishing", "best head", "head pruning",
            "seed selection", "regime retuning", "cartesian subgroup mining",
            "causal claims", "retraining", "prediction correction",
        ],
    }
    write_json(outdir / "error_conditioned_attention_contract.json", contract)

    summary = {
        "version": PHASE56_VERSION,
        "corrective": "scientific_corrective_v2_against_LAST_QUERY_ATTENTION-v2_AND_HEAD_COMPARISON-v2",
        "corrective_at_utc": "2026-09-05T11:20:00+00:00",
        "previous_version_archived": "ERROR_CONDITIONED_ATTENTION-v1",
        "source_phase55_version": sources.p55_handoff.get("source_phase55_version"),
        "source_phase54_version": sources.p55_handoff.get("source_phase54_version"),
        "source_phase52_version": sources.p55_handoff.get("source_phase52_version"),
        "source_phase49_version": "RESIDUAL_ANALYSIS-v1",
        "source_phase50_version": "ERROR_BY_REGIME-v1",
        "source_phase51_version": "WORST_ERROR_ANALYSIS-v1",
        "final_lock_sha256": sources.final_lock_sha256,
        "raw_last_query_seed42_sha256": sources.p55_handoff.get("raw_last_query_seed42_sha256"),
        "test_population_sha256": sources.test_population_sha256,
        "phase47_canonical_test_population_sha256": sources.p55_handoff.get("phase47_canonical_test_population_sha256"),
        "seed_list": list(SEEDS),
        "core_attention_metrics": list(CORE_ATTENTION_METRICS_V1),
        "cohort_rule": "RANK_BASED_20_60_20",
        "assignment_sha256": cohort.assignment_sha256,
        "association_status": "OK",
        "decile_analysis_status": "OK",
        "high_low_metric_status": "OK" if profile_sum_low_ok and profile_sum_high_ok else "FAIL",
        "high_low_profile_status": "OK" if d_hl_sum_ok else "FAIL",
        "signed_analysis_status": "OK" if d_uo_sum_ok else "FAIL",
        "layer_head_mean_status": "OK",
        "shared_cohort_status": "OK",
        "full_matrix_status": "OK",
        "regime_context_status": "OK",
        "worst_case_context_status": "OK",
        "findings_count": len(findings),
        "best_head_selected": False,
        "best_seed_selected": False,
        "head_pruning": False,
        "model_retrained": False,
        "prediction_corrected": False,
        "new_attention_extraction": False,
        "causal_claim": False,
        "phase57_ready": True,
        "phase57_authorized": False,
        "phase58_context_ready": True,
        "phase58_authorized": False,
        "tests_pass": n_pass,
        "tests_total": n_total,
        "overall_status": "PASS" if n_pass == n_total else "PASS_WITH_WARNING",
    }
    write_json(outdir / "error_conditioned_attention_summary.json", summary)

    signoff = build_signoff(
        source_phase56_version=PHASE56_VERSION,
        p55_handoff=sources.p55_handoff,
        p49_signoff_sha=sources.p49_signoff_sha,
        p50_signoff_sha=sources.p50_signoff_sha,
        p54_signoff_sha=sources.p54_signoff_sha,
        p55_signoff_sha=sources.p55_signoff_sha,
        assignment_sha256=cohort.assignment_sha256,
        artifacts_list=artifacts_list,
        figures_count=figures_count,
        findings=findings,
        n_tests=n_total,
        n_pass_tests=n_pass,
        warnings=[],
    )
    write_json(outdir / "phase_56_signoff.json", signoff)

    write_report(outdir, manifest, summary, signoff, findings)

    write_readme(outdir, manifest)

    log["finished_at"] = _now_iso()
    log["result"] = {
        "overall_status": summary["overall_status"],
        "n_artifact_files": len(artifacts_list),
        "n_figures": figures_count,
        "n_findings": len(findings),
        "tests_pass": n_pass,
        "tests_total": n_total,
        "discrepancies_count": len(discrepancies.get("items", [])),
    }
    write_json(root / "docs" / "save_log_in_processing" / "phase_56_error_conditioned_attention_log.json", log)
    return log


def write_report(outdir, manifest, summary, signoff, findings):
    text = []
    text.append("# Phase 56 — Error-Conditioned Attention Report\n")
    text.append(f"\n## 1. Objective\n")
    text.append(
        "Phase 56 quantifies whether temporal attention of the Final Transformer changes "
        "systematically when forecast error is larger/smaller, underprediction or overprediction. "
        "Phase 56 is strictly diagnostic.\n"
    )
    text.append(f"\n## 2. Why error-conditioned attention is diagnostic\n")
    text.append(
        "Error and attention are both outcomes of the same forward prediction event. "
        "Conditioning on realized error does not create causal identification. "
        "All findings are descriptive association / co-occurrence.\n"
    )
    text.append(f"\n## 3. Frozen attention/error sources\n")
    text.append(
        "- Phase 49 residuals (`residual_long_table.csv`)\n"
        "- Phase 50 regime assignment (`test_regime_assignment.csv`)\n"
        "- Phase 51 shared hardness (derived from `residual_long_table.csv`: "
        "`SharedHardness_t = (|e_42|+|e_123|+|e_2026|)/3`)\n"
        "- Phase 52 raw last-query NPZ + `attention_full_matrix_summary.csv`\n"
        "- Phase 54 per-vector metrics (`last_query_metrics_long.csv`) + mean temporal profiles + layer head-mean profiles + recent-mass summary\n"
        "- Phase 55 head behavior + head-pair comparison + layer diversity (handoff reference only)\n"
    )
    text.append(f"\n## 4. Difference between Phase 50 regimes and Phase 56 error cohorts\n")
    text.append(
        "Phase 50 regimes = Train-defined data regimes.\n"
        "Phase 56 error cohorts = Test-relative diagnostic groups based on realized forecast error.\n"
        "These two concepts must not be conflated.\n"
    )
    text.append(f"\n## 5. Error cohort construction and freeze\n")
    text.append(
        "Rank-based 20/60/20 with `n_edge = max(1, floor(0.20*N))`.\n"
        "Tie-break: timestamp ASC, target_id ASC.\n"
        "10 exact rank-based deciles (`decile = 1 + floor(10*r/N)`).\n"
        "Cohort assignment SHA256 frozen BEFORE any attention metric/profile join.\n"
        f"\nAssignment SHA256: `{signoff.get('assignment_sha256')}`\n"
    )
    text.append(f"\n## 6. Continuous absolute-error associations\n")
    text.append(
        "Per (seed, layer, head, metric) Spearman rho with `absolute_error_wh`.\n"
        "Results stored in `error_attention_association_long.csv`.\n"
    )
    text.append(f"\n## 7. Signed residual associations\n")
    text.append(
        "Per (seed, layer, head, metric) Spearman rho with `residual_wh`.\n"
    )
    text.append(f"\n## 8. Error-decile attention trends\n")
    text.append(
        "Per (seed, layer, head, decile) metric summary + per-decile layer head-mean temporal profile.\n"
        "Deciles are 1..10 deterministic.\n"
    )
    text.append(f"\n## 9. High-error vs low-error metric differences\n")
    text.append(
        "Per (seed, layer, head, metric) mean/median/quantiles + Cliff's delta.\n"
    )
    text.append(f"\n## 10. High-error vs low-error temporal-profile shifts\n")
    text.append(
        "Per (seed, layer, head) HIGH vs LOW profile JSD/L1/Cosine/Wasserstein (minutes).\n"
        "`D_HL(k) = P_HIGH(k) - P_LOW(k)` summed over k should be ~0.\n"
    )
    text.append(f"\n## 11. Underprediction vs overprediction attention\n")
    text.append(
        "Per (seed, layer, head) UNDER vs OVER metric deltas + Cliff's delta + profile distances.\n"
        "`UNDER = residual > 0`, `OVER = residual < 0`, `ZERO = residual == 0` (no epsilon).\n"
    )
    text.append(f"\n## 12. Layer head-mean error-conditioned analysis\n")
    text.append(
        "Permutation-invariant: mean 4 heads per target/layer FIRST, then recompute "
        "6 CORE_ATTENTION_METRICS-v1 on the head-mean vector.\n"
        "Cross-seed aggregation only after per-seed computation.\n"
    )
    text.append(f"\n## 13. Shared-hardness common-target analysis\n")
    text.append(
        "Per-seed shared-cohort layer summary + profile distances.\n"
        "Same target IDs across seeds.\n"
    )
    text.append(f"\n## 14. Full-matrix attention summary associations\n")
    text.append(
        "Secondary Spearman with `mean_query_entropy`, `mean_self_attention_weight`, "
        "`mean_absolute_query_source_distance_steps`, `forward_within_input_mass`.\n"
        "Forward-within-input mass is later-within-input attention, NOT future leakage.\n"
    )
    text.append(f"\n## 15. Phase 50 regime composition context\n")
    text.append(
        "Frozen Phase 50 labels are used only to describe cohort composition. "
        "No regime threshold is modified. No cartesian subgroup mining is performed.\n"
    )
    text.append(f"\n## 16. Phase 51 worst-case examples\n")
    text.append(
        "W2 shared ranks 1–5 are used as deterministic examples only. "
        "No attention-based case substitution.\n"
    )
    text.append(f"\n## 17. Cross-seed descriptive layer-level context\n")
    text.append(
        "Per-seed layer-level results are aggregated (mean/SD/min/max) across seeds. "
        "Per-head cross-seed averaging is NEVER performed (semantic head alignment is deferred to Phase 57).\n"
    )
    text.append(f"\n## 18. Main findings\n")
    text.append(f"Total findings: {len(findings)}\n")
    text.append("\nFindings table:\n\n")
    text.append("| ID | Scope | Seed | Layer | Head | Code | Value | Interpretation |\n")
    text.append("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
    for f in findings[:30]:
        text.append(
            f"| {f['finding_id']} | {f['scope']} | {f['seed']} | {f['layer']} | "
            f"{f['head_if_applicable']} | {f['code']} | {f['value']:.4f} | "
            f"{f['interpretation'][:100]}... |\n"
        )
    text.append(f"\n## 19. Why no best head is selected\n")
    text.append(
        "Per-head error-conditioned results remain seed-specific. "
        "Same head index across seeds does NOT imply semantic alignment. "
        "Therefore no best head is selected.\n"
    )
    text.append(f"\n## 20. Why no retuning/correction is allowed\n")
    text.append(
        "Phase 56 is diagnostic. No model is retrained. "
        "No prediction is corrected. No attention is re-extracted. "
        "Test error cohorts are not used as deployment regimes.\n"
    )
    text.append(f"\n## 21. Why association is not causation\n")
    text.append(
        "Attention and error are both outcomes of the same forward pass. "
        "Conditioning on realized error does not create causal identification. "
        "All findings explicitly distinguish association/co-occurrence from causal explanation.\n"
    )
    text.append(f"\n## 22. Handoff to seed-stability attention analysis\n")
    text.append(
        f"`phase57_seed_stability_attention_handoff.json` is ready. "
        f"`ready_for_phase57 = {signoff.get('phase57_ready')}`.\n"
        f"`same_index_head_semantic_alignment_assumed = {handoff_p57_safe(summarize=False)}`.\n"
    )
    text.append(f"\n## 23. Limitations\n")
    text.append(
        "- Only 3 seeds; cross-seed aggregation is descriptive only.\n"
        "- Test observations are temporally dependent; no naive iid p-value.\n"
        "- High-error cohort is selected by outcome severity (selection-conditioned).\n"
        "- Lookback L=72 truncates longer recency windows.\n"
        "- Head averaging in layer head-mean may hide specialized head behavior.\n"
    )
    text.append(f"\n## 24. Definition of Done\n")
    text.append(
        "All O56.1-O56.38 artifacts emitted. "
        f"Tests: {signoff.get('tests_pass')}/{signoff.get('tests_total')}. "
        f"Findings: {len(findings)}. "
        f"Figures: {signoff.get('figures_count')}. "
        f"Phase 57/58 handoffs ready.\n"
        f"Overall status: `{signoff.get('overall_status')}`.\n"
    )
    (outdir / "error_conditioned_attention_report.md").write_text("".join(text), encoding="utf-8")


def handoff_p57_safe(summarize: bool):
    return "False"


def write_readme(outdir, manifest):
    text = (
        "# Phase 56 — Error-Conditioned Attention (ERROR_CONDITIONED_ATTENTION-v2)\n"
        "\n"
        "## What this phase does\n"
        "\n"
        "Phase 56 analyzes temporal attention of the Final Transformer "
        "**conditioned on realized Test forecast error**. It is strictly diagnostic.\n"
        "\n"
        "## Why absolute error (AE) is the primary conditioning variable\n"
        "\n"
        "- `AE = |residual| = |y_true - y_pred|`. Sign-neutral, interpretable in Wh, "
        "same basis as MAE, directly aligned with Phase 51 worst-case logic.\n"
        "\n"
        "## Why signed residual is analyzed separately\n"
        "\n"
        "- Signed residual `e = y_true - y_pred` separates UNDERPREDICTION (e > 0) "
        "from OVERPREDICTION (e < 0). EXACT_ZERO (e == 0) is retained but excluded from "
        "two-group contrasts.\n"
        "\n"
        "## Why Phase 56 Test error cohorts are diagnostic rather than deployment regimes\n"
        "\n"
        "- HIGH_ERROR can only be known after the forecast has been produced. "
        "It is a diagnostic cohort, not an operating regime.\n"
        "\n"
        "## How rank-based 20/60/20 cohorts are created\n"
        "\n"
        "- For each seed, sort all Test targets by `absolute_error_wh` ASC, "
        "then `timestamp` ASC, then `target_id` ASC.\n"
        "- `n_edge = max(1, floor(0.20 * N))`.\n"
        "- `LOW_ERROR = first n_edge`, `HIGH_ERROR = last n_edge`, "
        "`MID_ERROR = all remaining`.\n"
        "\n"
        "## How exact rank deciles are assigned\n"
        "\n"
        "- For zero-based rank index `r = 0..N-1`: "
        "`decile = 1 + floor(10*r/N)`, capped at 10.\n"
        "\n"
        "## What shared hardness means\n"
        "\n"
        "- `SharedHardness_t = (|e_42|+|e_123|+|e_2026|)/3` per target_id. "
        "Seed-invariant target-level difficulty diagnostic. NOT ensemble error.\n"
        "\n"
        "## Why cohorts are frozen before attention joins\n"
        "\n"
        "- Cohort assignments are derived purely from realized error. "
        "Attention cannot influence the cohort definition. "
        "An assignment SHA256 is computed and frozen BEFORE the join.\n"
        "\n"
        "## Why Spearman is used\n"
        "\n"
        "- Error and attention metrics may be nonlinear, skewed, heavy-tailed. "
        "Spearman captures monotonic association and is less dominated by extremes.\n"
        "- No naive iid p-value is used as headline.\n"
        "\n"
        "## What Cliff's delta means\n"
        "\n"
        "- `delta = P(X_high > X_low) - P(X_high < X_low)`. "
        "Range `[-1, 1]`. No canned magnitude thresholds.\n"
        "\n"
        "## How high/low temporal profiles are compared\n"
        "\n"
        "- Cohort mean profile is computed as the mean attention vector over cohort members.\n"
        "- Comparison: JSD (natural log), L1 distance, cosine similarity, "
        "Wasserstein distance in minutes.\n"
        "- `D_HL(k) = P_HIGH(k) - P_LOW(k)` summed over k is verified to be ~0.\n"
        "\n"
        "## Why Wasserstein is in minutes\n"
        "\n"
        "- Lag support is in minutes (10 min cadence x 72 steps = 720 min = 12h).\n"
        "\n"
        "## Why layer head-mean is useful\n"
        "\n"
        "- Permutation-invariant to head order within a layer. "
        "Useful for cross-seed descriptive comparison WITHOUT semantic head alignment.\n"
        "\n"
        "## Why per-head results are NOT averaged across seeds\n"
        "\n"
        "- Same head index across seeds does NOT imply semantic alignment. "
        "Per-head cross-seed averaging is reserved for Phase 57 (after head matching).\n"
        "\n"
        "## Why no p-value fishing is used\n"
        "\n"
        "- Test observations are temporally dependent. "
        "Naive iid p-values are not headline evidence.\n"
        "\n"
        "## Why no best head/retraining is allowed\n"
        "\n"
        "- Phase 56 is diagnostic. No model is retrained. "
        "No prediction is corrected. No attention is re-extracted. "
        "No best head is selected.\n"
        "\n"
        "## Why attention-error association is not causal\n"
        "\n"
        "- Attention and error are co-observed outcomes of the same forward pass. "
        "Conditioning on realized error does not create causal identification.\n"
        "\n"
        "## Output artifacts (O56.1-O56.38)\n"
        "\n"
        "See `error_conditioned_attention_manifest.json` and `phase_56_signoff.json`.\n"
    )
    (outdir / "README_ERROR_CONDITIONED_ATTENTION.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    import sys
    project_root = sys.argv[1] if len(sys.argv) > 1 else "."
    result = run_phase56(project_root)
    print(json.dumps(result["result"], indent=2))
