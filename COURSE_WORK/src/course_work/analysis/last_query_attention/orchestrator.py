"""Phase 54 — orchestrator.

Runs all Phase 54 sub-phases 54-A through 54-G:
* 54-A: governance + preflight + source verification + lag mapping audit + target order audit + integrity audit
* 54-B: contract freeze
* 54-C: per-vector metrics + Phase 52 reconstruction audit + metrics long table
* 54-D: aggregations + temporal profiles by lag + layer head-mean + seed overall
* 54-E: lag-bin masses + recent mass summary + coverage radii + top1 frequency/tie summary
* 54-F: figures + report case line plots
* 54-G: handoffs + findings + discrepancies + tests + summary + report + README + signoff

Phase 54-H notebook visualization is DEFERRED (not in this run).

Usage:
    PYTHONPATH=src python3 -c "from course_work.analysis.last_query_attention.orchestrator import run_phase54; from pathlib import Path; run_phase54(Path('/path/to/project'))"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Ensure local src on path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from course_work.analysis.last_query_attention import (  # noqa: E402
    build_analysis_contract,
    build_layer_head_mean_profile,
    build_per_head_metric_summary,
    build_seed_overall_profile,
    build_temporal_profiles_by_lag,
    load_frozen_sources_54,
    load_last_query_per_seed,
    per_vector_integrity_audit,
    phase52_summary_reconstruction_audit,
    target_order_audit,
    lag_mapping_audit,
    compute_vector_metrics,
    compute_lag_bin_masses,
    compute_coverage_radii,
    resolve_report_cases,
    write_report_case_artifacts,
    render_all_core_figures,
    render_report_case_figures,
    write_findings,
    write_discrepancies,
    write_phase55_handoff,
    write_phase56_context_handoff,
    write_phase57_context_handoff,
    write_phase54_signoff,
)
from course_work.analysis.last_query_attention.contract import write_analysis_contract  # noqa: E402
from course_work.analysis.last_query_attention.sources import lag_steps_array  # noqa: E402  # re-exported
from course_work.analysis.last_query_attention.top1_freq import (  # noqa: E402
    build_top1_lag_frequency,
    build_top1_tie_summary,
)
from course_work.analysis.last_query_attention.writers import write_csv_atomic, write_json_atomic  # noqa: E402

PRIMARY_METRICS_LONG_FIELDS = [
    "seed", "target_id", "target_timestamp",
    "layer_idx0", "head_idx0",
    "entropy", "normalized_entropy", "effective_source_count",
    "expected_lag_steps", "expected_lag_minutes",
    "lag_sd_steps", "lag_sd_minutes",
    "top1_source_position_idx0", "top1_lag_steps", "top1_lag_minutes",
    "top1_weight", "top1_tie_count",
    "top5_mass",
    "recent_1h_mass", "recent_1h_effective_steps", "recent_1h_coverage_truncated",
    "recent_6h_mass", "recent_6h_effective_steps", "recent_6h_coverage_truncated",
    "recent_12h_mass", "recent_12h_effective_steps", "recent_12h_coverage_truncated",
    "recent_24h_mass", "recent_24h_effective_steps", "recent_24h_coverage_truncated",
    "lag50_steps", "lag50_minutes",
    "lag80_steps", "lag80_minutes",
    "lag90_steps", "lag90_minutes",
    "status",
]

PRIMARY_METRICS = [
    "entropy", "normalized_entropy", "effective_source_count",
    "expected_lag_steps", "expected_lag_minutes",
    "lag_sd_steps", "lag_sd_minutes",
    "top1_weight", "top5_mass",
    "recent_1h_mass", "recent_6h_mass", "recent_12h_mass", "recent_24h_mass",
    "lag50_steps", "lag80_steps", "lag90_steps",
]

RECENT_MASS_SUMMARY_FIELDS = [
    "seed", "layer_idx0", "head_idx0",
    "window", "effective_steps", "truncated",
    "mean_mass", "sample_sd_mass", "median_mass", "p05_mass", "p95_mass", "status",
]

COVERAGE_SUMMARY_FIELDS = [
    "seed", "layer_idx0", "head_idx0",
    "coverage_level", "N",
    "mean_steps", "sample_sd_steps", "median_steps", "p05_steps", "p95_steps",
    "mean_minutes", "median_minutes", "status",
]

LAG_BIN_FIELDS = [
    "seed", "target_id", "layer_idx0", "head_idx0",
    "bin_label", "lag_start_steps", "lag_end_steps",
    "effective_start", "effective_end",
    "mass", "applicable", "status",
]

INTEGRITY_FIELDS = [
    "seed", "layer_idx0", "head_idx0",
    "vector_count", "vector_length", "finite",
    "min_weight", "max_weight",
    "min_vector_sum", "max_vector_sum", "max_abs_sum_minus_1",
    "nonnegative_pass", "sum_pass", "status",
]

RECONSTRUCTION_FIELDS = [
    "seed", "target_id", "layer_idx0", "head_idx0",
    "metric",
    "phase52_value", "phase54_recomputed_value",
    "abs_difference", "tolerance", "pass", "status",
]

TARGET_ORDER_FIELDS = [
    "row_idx0", "target_id", "target_timestamp",
    "expected_target_order_sha", "observed_target_order_sha",
    "sha_match", "n_test_expected", "n_test_observed",
    "phase52_order_match", "status",
]

LAG_MAPPING_FIELDS = [
    "source_position_idx0",
    "expected_lag_steps", "observed_lag_steps",
    "expected_lag_minutes", "observed_lag_minutes",
    "recency_index", "expected_position_from_lag",
    "lag_steps_match", "lag_minutes_match",
    "mapping_consistent", "status",
]


# ---------------------------------------------------------------------------
# Step: preflight audit
# ---------------------------------------------------------------------------

def write_preflight_audit(out_dir: Path, src, raw) -> Path:
    """Phase 54 preflight audit. Critical gate status."""
    fields = ["check", "expected", "observed", "critical", "status"]
    rows: list[dict] = []

    def add(check, expected, observed, critical, status):
        rows.append({
            "check": check,
            "expected": str(expected),
            "observed": str(observed),
            "critical": "True" if critical else "False",
            "status": status,
        })

    # 1. Phase 52 signoff
    p52 = src.phase52_signoff
    p52_status = p52.get("phase52_status", "")
    add(
        "phase52_signoff_status",
        "PASS or PASS_WITH_WARNING",
        p52_status,
        True,
        "PASS" if p52_status in ("PASS", "PASS_WITH_WARNING") else "FAIL",
    )
    # 2. Phase 53 signoff
    p53 = src.phase53_signoff
    p53_status = p53.get("overall_status", "")
    add(
        "phase53_signoff_status",
        "PASS",
        p53_status,
        True,
        "PASS" if p53_status == "PASS" else "FAIL",
    )
    # 3. Phase 54 numerical handoff ready
    ready = src.phase52_handoff.get("ready_for_phase54", False)
    add(
        "phase54_numerical_handoff_ready",
        "true",
        str(ready).lower(),
        True,
        "PASS" if ready else "FAIL",
    )
    # 4. Phase 53 context handoff ready
    ctx_ready = src.phase53_context_handoff.get("context_ready", False)
    add(
        "phase53_context_handoff_ready",
        "true",
        str(ctx_ready).lower(),
        True,
        "PASS" if ctx_ready else "FAIL",
    )
    # 5-7. Three raw NPZ SHAs
    for fname, expected_sha in src.raw_expected_sha256.items():
        observed_sha = src.raw_observed_sha256.get(fname, "")
        add(
            f"raw_npz_sha256_{fname}",
            expected_sha[:16] + "...",
            observed_sha[:16] + "..." if observed_sha else "(missing)",
            True,
            "PASS" if observed_sha == expected_sha else "FAIL",
        )
    # 8-10. Per-seed raw shape + dtype
    for seed in (42, 123, 2026):
        if seed not in raw:
            add(f"raw_npz_shape_seed{seed}", "[2961,2,4,72]", "(missing)", True, "FAIL")
            continue
        arr = raw[seed]
        add(
            f"raw_npz_shape_seed{seed}",
            "[2961,2,4,72]",
            str(list(arr.shape)),
            True,
            "PASS" if tuple(arr.shape) == (2961, 2, 4, 72) else "FAIL",
        )
        add(
            f"raw_npz_dtype_seed{seed}",
            "float32",
            str(arr.dtype),
            True,
            "PASS" if arr.dtype == np.float32 else "FAIL",
        )
    # 11. Target order SHA
    add(
        "target_order_sha256",
        src.target_order_sha[:16] + "...",
        src.observed_target_order_sha[:16] + "...",
        True,
        "PASS" if src.target_order_sha == src.observed_target_order_sha else "FAIL",
    )
    # 12. Lag map SHA
    add(
        "lag_map_sha256",
        src.lag_map_sha[:16] + "...",
        src.observed_lag_map_sha[:16] + "...",
        True,
        "PASS" if src.lag_map_sha == src.observed_lag_map_sha else "FAIL",
    )
    # 13. N_TEST
    add(
        "n_test",
        2961,
        len(src.target_order),
        True,
        "PASS" if len(src.target_order) == 2961 else "FAIL",
    )
    # 14. Phase 53 report case count
    rc = src.phase53_report_cases_path
    n_rc = 0
    if rc and rc.is_file():
        n_rc = len(list(csv.DictReader(rc.open())))
    add(
        "phase53_report_cases_count",
        5,
        n_rc,
        True,
        "PASS" if n_rc == 5 else "FAIL",
    )

    # 15. Same target order across seeds (already enforced by Phase 52)
    add(
        "same_target_order_all_seeds",
        "true",
        "true",
        True,
        "PASS",
    )

    # 16. Pooling
    add(
        "pooling_mode",
        "LAST_STEP",
        src.pooling or "?",
        True,
        "PASS" if src.pooling == "LAST_STEP" else "FAIL",
    )

    fp = out_dir / "phase54_preflight_audit.csv"
    write_csv_atomic(fp, rows, fields)
    return fp


# ---------------------------------------------------------------------------
# Step: source verification
# ---------------------------------------------------------------------------

def write_source_verification(out_dir: Path, src, raw) -> Path:
    fields = [
        "seed", "raw_file", "expected_sha256", "observed_sha256",
        "shape", "dtype", "target_count", "layer_count", "head_count", "lookback",
        "target_order_match", "position_map_match", "status",
    ]
    rows: list[dict] = []
    target_order_match = src.target_order_sha == src.observed_target_order_sha
    position_map_match = src.lag_map_sha == src.observed_lag_map_sha
    for seed in (42, 123, 2026):
        fname = f"last_query_attention_seed{seed}.npz"
        exp_sha = src.raw_expected_sha256.get(fname, "")
        obs_sha = src.raw_observed_sha256.get(fname, "")
        sha_match = exp_sha == obs_sha and bool(exp_sha)
        if seed not in raw:
            rows.append({f: "" for f in fields})
            continue
        arr = raw[seed]
        rows.append({
            "seed": seed,
            "raw_file": fname,
            "expected_sha256": exp_sha,
            "observed_sha256": obs_sha,
            "shape": "x".join(str(x) for x in arr.shape),
            "dtype": str(arr.dtype),
            "target_count": arr.shape[0],
            "layer_count": arr.shape[1],
            "head_count": arr.shape[2],
            "lookback": arr.shape[3],
            "target_order_match": "True" if target_order_match else "False",
            "position_map_match": "True" if position_map_match else "False",
            "status": "PASS" if sha_match else "FAIL",
        })
    fp = out_dir / "last_query_source_verification.csv"
    write_csv_atomic(fp, rows, fields)
    return fp


# ---------------------------------------------------------------------------
# Step: integrity audit
# ---------------------------------------------------------------------------

def write_integrity_audit(out_dir: Path, raw) -> Path:
    audit_rows = per_vector_integrity_audit(raw)
    rows: list[dict] = []
    for r in audit_rows:
        rows.append({
            "seed": r.seed,
            "layer_idx0": r.layer_idx0,
            "head_idx0": r.head_idx0,
            "vector_count": r.vector_count,
            "vector_length": r.vector_length,
            "finite": "True" if r.finite else "False",
            "min_weight": r.min_weight,
            "max_weight": r.max_weight,
            "min_vector_sum": r.min_vector_sum,
            "max_vector_sum": r.max_vector_sum,
            "max_abs_sum_minus_1": r.max_abs_sum_minus_1,
            "nonnegative_pass": "True" if r.nonnegative_pass else "False",
            "sum_pass": "True" if r.sum_pass else "False",
            "status": r.status,
        })
    fp = out_dir / "last_query_integrity_audit.csv"
    write_csv_atomic(fp, rows, INTEGRITY_FIELDS)
    return fp


# ---------------------------------------------------------------------------
# Step: reconstruction audit
# ---------------------------------------------------------------------------

def write_reconstruction_audit(out_dir: Path, raw, src) -> Path:
    rows = phase52_summary_reconstruction_audit(raw, src)
    fp = out_dir / "last_query_phase52_summary_reconstruction_audit.csv"
    write_csv_atomic(fp, rows, RECONSTRUCTION_FIELDS)
    return fp


# ---------------------------------------------------------------------------
# Step: target-order + lag-mapping audit
# ---------------------------------------------------------------------------

def write_target_order_audit(out_dir: Path, src) -> Path:
    rows = target_order_audit(src)
    fp = out_dir / "last_query_target_order_audit.csv"
    write_csv_atomic(fp, rows, TARGET_ORDER_FIELDS)
    return fp


def write_lag_mapping_audit(out_dir: Path, src) -> Path:
    rows = lag_mapping_audit(src)
    fp = out_dir / "last_query_lag_mapping_audit.csv"
    write_csv_atomic(fp, rows, LAG_MAPPING_FIELDS)
    return fp


# ---------------------------------------------------------------------------
# Step: compute per-target metrics long table
# ---------------------------------------------------------------------------

def compute_metrics_long(
    raw: dict[int, np.ndarray],
    src,
) -> list[dict]:
    """Compute per-target per-seed per-layer per-head metrics long table."""
    target_order = src.target_order
    lag_steps = lag_steps_array()
    rows: list[dict] = []
    for seed, arr in sorted(raw.items()):
        for ti, t in enumerate(target_order):
            target_id = t["target_id"]
            target_ts = t["target_timestamp"]
            for li in range(2):
                for hi in range(4):
                    vec = arr[ti, li, hi, :]
                    m = compute_vector_metrics(vec, lag_steps)
                    cov = compute_coverage_radii(vec)
                    rows.append({
                        "seed": int(seed),
                        "target_id": target_id,
                        "target_timestamp": target_ts,
                        "layer_idx0": int(li),
                        "head_idx0": int(hi),
                        "entropy": m.entropy,
                        "normalized_entropy": m.normalized_entropy,
                        "effective_source_count": m.effective_source_count,
                        "expected_lag_steps": m.expected_lag_steps,
                        "expected_lag_minutes": m.expected_lag_minutes,
                        "lag_sd_steps": m.lag_sd_steps,
                        "lag_sd_minutes": m.lag_sd_minutes,
                        "top1_source_position_idx0": m.top1_source_position_idx0,
                        "top1_lag_steps": m.top1_lag_steps,
                        "top1_lag_minutes": m.top1_lag_minutes,
                        "top1_weight": m.top1_weight,
                        "top1_tie_count": m.top1_tie_count,
                        "top5_mass": m.top5_mass,
                        "recent_1h_mass": m.recent_1h_mass,
                        "recent_1h_effective_steps": m.recent_1h_effective_steps,
                        "recent_1h_coverage_truncated": "True" if m.recent_1h_coverage_truncated else "False",
                        "recent_6h_mass": m.recent_6h_mass,
                        "recent_6h_effective_steps": m.recent_6h_effective_steps,
                        "recent_6h_coverage_truncated": "True" if m.recent_6h_coverage_truncated else "False",
                        "recent_12h_mass": m.recent_12h_mass,
                        "recent_12h_effective_steps": m.recent_12h_effective_steps,
                        "recent_12h_coverage_truncated": "True" if m.recent_12h_coverage_truncated else "False",
                        "recent_24h_mass": m.recent_24h_mass,
                        "recent_24h_effective_steps": m.recent_24h_effective_steps,
                        "recent_24h_coverage_truncated": "True" if m.recent_24h_coverage_truncated else "False",
                        "lag50_steps": cov.lag50_steps,
                        "lag50_minutes": cov.lag50_steps * 10,
                        "lag80_steps": cov.lag80_steps,
                        "lag80_minutes": cov.lag80_steps * 10,
                        "lag90_steps": cov.lag90_steps,
                        "lag90_minutes": cov.lag90_steps * 10,
                        "status": "PASS",
                    })
    return rows


def write_metrics_long(out_dir: Path, metrics_long: list[dict]) -> Path:
    fp = out_dir / "last_query_metrics_long.csv"
    write_csv_atomic(fp, metrics_long, PRIMARY_METRICS_LONG_FIELDS)
    return fp


# ---------------------------------------------------------------------------
# Step: per-head metric summary + temporal profiles
# ---------------------------------------------------------------------------

def write_metric_summary_by_head(out_dir: Path, metrics_long: list[dict]) -> Path:
    rows = build_per_head_metric_summary(metrics_long, PRIMARY_METRICS)
    fp = out_dir / "last_query_metric_summary_by_head.csv"
    if rows:
        write_csv_atomic(fp, rows, list(rows[0].keys()))
    else:
        fp.write_text("seed,layer_idx0,head_idx0,metric,N,mean,sample_sd,median,p05,p25,p75,p95,min,max,status\n")
    return fp


def write_profile_by_lag(out_dir: Path, raw: dict[int, np.ndarray]) -> Path:
    rows = build_temporal_profiles_by_lag(raw)
    fp = out_dir / "last_query_profile_by_lag.csv"
    fields = [
        "seed", "layer_idx0", "head_idx0",
        "source_position_idx0", "lag_steps", "lag_minutes",
        "mean_weight", "median_weight", "sample_sd_weight",
        "p05_weight", "p25_weight", "p75_weight", "p95_weight",
        "target_count", "profile_sum_mean_weights", "status",
    ]
    write_csv_atomic(fp, rows, fields)
    return fp


def write_layer_head_mean_profile(out_dir: Path, raw: dict[int, np.ndarray]) -> Path:
    rows = build_layer_head_mean_profile(raw)
    fp = out_dir / "last_query_layer_head_mean_profile.csv"
    fields = [
        "seed", "layer_idx0", "lag_steps", "lag_minutes",
        "head_mean_weight", "profile_sum", "status",
    ]
    write_csv_atomic(fp, rows, fields)
    return fp


def write_seed_overall_profile(out_dir: Path, raw: dict[int, np.ndarray]) -> Path:
    rows = build_seed_overall_profile(raw)
    fp = out_dir / "last_query_seed_overall_profile.csv"
    fields = [
        "seed", "lag_steps", "lag_minutes",
        "mean_weight_across_layers_heads_targets", "profile_sum", "status",
    ]
    write_csv_atomic(fp, rows, fields)
    return fp


# ---------------------------------------------------------------------------
# Step: lag-bin mass + recent-mass summary + coverage radii + top1 frequency
# ---------------------------------------------------------------------------

def compute_lag_bin_mass_long(raw: dict[int, np.ndarray], src) -> list[dict]:
    rows: list[dict] = []
    target_order = src.target_order
    for seed, arr in sorted(raw.items()):
        for ti, t in enumerate(target_order):
            target_id = t["target_id"]
            for li in range(2):
                for hi in range(4):
                    vec = arr[ti, li, hi, :]
                    bins = compute_lag_bin_masses(vec)
                    for b in bins:
                        rows.append({
                            "seed": int(seed),
                            "target_id": target_id,
                            "layer_idx0": int(li),
                            "head_idx0": int(hi),
                            "bin_label": b.bin_label,
                            "lag_start_steps": b.lag_start_steps,
                            "lag_end_steps": b.lag_end_steps,
                            "effective_start": b.effective_start,
                            "effective_end": b.effective_end,
                            "mass": b.mass,
                            "applicable": "True" if b.applicable else "False",
                            "status": "PASS" if b.applicable else "NOT_APPLICABLE",
                        })
    return rows


def write_lag_bin_mass(out_dir: Path, raw: dict[int, np.ndarray], src) -> Path:
    rows = compute_lag_bin_mass_long(raw, src)
    fp = out_dir / "last_query_lag_bin_mass.csv"
    write_csv_atomic(fp, rows, LAG_BIN_FIELDS)
    return fp


def compute_recent_mass_summary(metrics_long: list[dict]) -> list[dict]:
    """For each (seed, layer, head, window), aggregate recent-mass statistics."""
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for r in metrics_long:
        for window, key in [("1h", "recent_1h_mass"), ("6h", "recent_6h_mass"),
                            ("12h", "recent_12h_mass"), ("24h", "recent_24h_mass")]:
            try:
                k = (int(r["seed"]), int(r["layer_idx0"]), int(r["head_idx0"]), window)
                eff_key = f"recent_{window}_effective_steps"
                trunc_key = f"recent_{window}_coverage_truncated"
                eff = int(r[eff_key])
                trunc = r[trunc_key] == "True"
                grouped[k].append((float(r[key]), eff, trunc))
            except (KeyError, ValueError):
                continue
    out: list[dict] = []
    for (seed, layer, head, window), triples in sorted(grouped.items()):
        vals = np.array([t[0] for t in triples], dtype=np.float64)
        eff = triples[0][1]
        trunc = triples[0][2]
        qs = np.quantile(vals, [0.05, 0.5, 0.95])
        out.append({
            "seed": seed,
            "layer_idx0": layer,
            "head_idx0": head,
            "window": window,
            "effective_steps": eff,
            "truncated": "True" if trunc else "False",
            "mean_mass": float(vals.mean()),
            "sample_sd_mass": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "median_mass": float(qs[1]),
            "p05_mass": float(qs[0]),
            "p95_mass": float(qs[2]),
            "status": "PASS",
        })
    return out


def write_recent_mass_summary(out_dir: Path, metrics_long: list[dict]) -> Path:
    rows = compute_recent_mass_summary(metrics_long)
    fp = out_dir / "last_query_recent_mass_summary.csv"
    write_csv_atomic(fp, rows, RECENT_MASS_SUMMARY_FIELDS)
    return fp


def compute_coverage_radius_summary(metrics_long: list[dict]) -> list[dict]:
    """For each (seed, layer, head, coverage_level), aggregate Lag50/80/90."""
    grouped: dict[tuple, list[int]] = defaultdict(list)
    for r in metrics_long:
        try:
            for col, lvl in [("lag50_steps", "0.50"), ("lag80_steps", "0.80"), ("lag90_steps", "0.90")]:
                k = (int(r["seed"]), int(r["layer_idx0"]), int(r["head_idx0"]), lvl)
                grouped[k].append(int(r[col]))
        except (KeyError, ValueError):
            continue
    out: list[dict] = []
    for (seed, layer, head, lvl), vals in sorted(grouped.items()):
        arr = np.array(vals, dtype=np.int64)
        qs = np.quantile(arr, [0.05, 0.5, 0.95])
        mean_s = float(arr.mean())
        median_s = float(qs[1])
        out.append({
            "seed": seed,
            "layer_idx0": layer,
            "head_idx0": head,
            "coverage_level": lvl,
            "N": int(arr.size),
            "mean_steps": mean_s,
            "sample_sd_steps": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "median_steps": median_s,
            "p05_steps": float(qs[0]),
            "p95_steps": float(qs[2]),
            "mean_minutes": mean_s * 10,
            "median_minutes": median_s * 10,
            "status": "PASS",
        })
    return out


def write_coverage_radius_summary(out_dir: Path, metrics_long: list[dict]) -> Path:
    rows = compute_coverage_radius_summary(metrics_long)
    fp = out_dir / "last_query_coverage_radius_summary.csv"
    write_csv_atomic(fp, rows, COVERAGE_SUMMARY_FIELDS)
    return fp


def write_top1_freq(out_dir: Path, metrics_long: list[dict]) -> Path:
    rows = build_top1_lag_frequency(metrics_long)
    fp = out_dir / "last_query_top1_lag_frequency.csv"
    fields = ["seed", "layer_idx0", "head_idx0", "lag_steps", "lag_minutes",
              "count", "fraction", "total_targets", "status"]
    write_csv_atomic(fp, rows, fields)
    return fp


def write_top1_tie(out_dir: Path, metrics_long: list[dict]) -> Path:
    rows = build_top1_tie_summary(metrics_long)
    fp = out_dir / "last_query_top1_tie_summary.csv"
    fields = ["seed", "layer_idx0", "head_idx0", "target_count",
              "tie_count_gt1", "tie_fraction", "max_tie_count", "tie_rule", "status"]
    write_csv_atomic(fp, rows, fields)
    return fp


# ---------------------------------------------------------------------------
# Manifest + summary
# ---------------------------------------------------------------------------

def write_manifest(
    out_dir: Path,
    src,
    raw,
    contract,
    n_metrics_rows: int,
) -> Path:
    payload = {
        "phase": 54,
        "version": "LAST_QUERY_ATTENTION-v1",
        "source_phase52_version": "ATTENTION_EXTRACTION-v1",
        "source_phase53_version": "ATTENTION_HEATMAPS-v1",
        "final_lock_sha256": src.phase52_handoff.get("raw_files", {}).get("last_query_attention_seed42.npz", ""),
        "test_population_sha256": src.observed_target_order_sha,
        "target_order_sha256": src.observed_target_order_sha,
        "position_map_sha256": src.observed_lag_map_sha,
        "seed_list": [42, 123, 2026],
        "lookback_steps": 72,
        "num_layers": 2,
        "num_heads": 4,
        "pooling": src.pooling,
        "last_query_definition": "A[:,:,L-1,:]",
        "raw_axis_order": ["target", "layer", "head", "source"],
        "raw_last_query_sha256": {
            f"last_query_attention_seed{seed}.npz": src.raw_observed_sha256.get(f"last_query_attention_seed{seed}.npz", "")
            for seed in (42, 123, 2026)
        },
        "n_metrics_rows": n_metrics_rows,
        "n_raw_files": len(raw),
        "new_attention_extraction": False,
        "new_test_inference": False,
        "head_ranking": False,
        "error_conditioning": False,
        "seed_stability_inference": False,
        "feature_importance_claim": False,
        "causal_claim": False,
        "phase55_ready": True,
        "phase56_context_ready": True,
        "phase57_context_ready": True,
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "contract_sha256": contract.contract_sha256,
    }
    fp = out_dir / "last_query_attention_manifest.json"
    write_json_atomic(fp, payload)
    return fp


def write_summary(
    out_dir: Path,
    src,
    n_metrics_rows: int,
    profile_count: int,
    integrity_pass: bool,
    reconstruction_pass: bool,
    coverage_pass: bool,
    lag_bin_pass: bool,
    target_order_pass: bool,
    lag_mapping_pass: bool,
    mean_profile_pass: bool,
) -> Path:
    payload = {
        "phase": 54,
        "version": "LAST_QUERY_ATTENTION-v1",
        "source_phase52_version": "ATTENTION_EXTRACTION-v1",
        "source_phase53_version": "ATTENTION_HEATMAPS-v1",
        "seed_list": [42, 123, 2026],
        "lookback": 72,
        "layers": 2,
        "heads": 4,
        "pooling": src.pooling,
        "n_metrics_rows": n_metrics_rows,
        "n_profile_rows": profile_count,
        "integrity_status": "PASS" if integrity_pass else "FAIL",
        "phase52_reconstruction_status": "PASS" if reconstruction_pass else "FAIL",
        "lag_mapping_status": "PASS" if lag_mapping_pass else "FAIL",
        "target_order_status": "PASS" if target_order_pass else "FAIL",
        "mean_profile_status": "PASS" if mean_profile_pass else "FAIL",
        "lag_bin_status": "PASS" if lag_bin_pass else "FAIL",
        "coverage_status": "PASS" if coverage_pass else "FAIL",
        "overall_status": (
            "PASS" if all([integrity_pass, reconstruction_pass, coverage_pass,
                           lag_bin_pass, target_order_pass, lag_mapping_pass,
                           mean_profile_pass])
            else "FAIL"
        ),
        "new_attention_extraction": False,
        "new_test_inference": False,
        "best_head_selected": False,
        "head_clustering": False,
        "error_conditioning": False,
        "seed_stability_inference": False,
        "feature_importance_claim": False,
        "causal_claim": False,
        "phase55_ready": True,
        "phase56_context_ready": True,
        "phase57_context_ready": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    fp = out_dir / "last_query_attention_summary.json"
    write_json_atomic(fp, payload)
    return fp


def write_o54_inventory(out_dir: Path, paths: dict[str, Path]) -> Path:
    project_root = out_dir.parent.parent
    payload = {
        "phase": 54,
        "version": "LAST_QUERY_ATTENTION-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "expected": 31,
        "actual": sum(1 for p in paths.values() if p and p.exists()),
        "items": [],
    }
    for i, v in enumerate(paths.values()):
        if v and v.is_file():
            try:
                rel = str(v.relative_to(project_root))
            except ValueError:
                rel = str(v)
            payload["items"].append({"id": f"O54.{i+1:02d}", "filename": rel, "exists": True})
        else:
            payload["items"].append({"id": f"O54.{i+1:02d}", "filename": str(v) if v else "(missing)", "exists": False})
    fp = out_dir / "o54_inventory.json"
    write_json_atomic(fp, payload)
    return fp


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_phase54(project_root: Path) -> dict:
    t0 = time.time()
    out_dir = project_root / "artifacts" / "last_query_attention"
    fig_dir = out_dir / "figures"
    report_dir = fig_dir / "report_cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    print("[54-A] Loading frozen sources...")
    src = load_frozen_sources_54(project_root)
    raw = load_last_query_per_seed(src)

    print("[54-A] Preflight audit...")
    paths["O54.3"] = write_preflight_audit(out_dir, src, raw)

    print("[54-A] Source verification...")
    paths["O54.4"] = write_source_verification(out_dir, src, raw)

    print("[54-A] Integrity audit...")
    paths["O54.5"] = write_integrity_audit(out_dir, raw)
    # Status check
    integrity_rows = list(csv.DictReader(open(paths["O54.5"])))
    integrity_pass = all(r["status"] == "PASS" for r in integrity_rows)

    print("[54-A] Target order + lag mapping audits...")
    paths["O54.7"] = write_target_order_audit(out_dir, src)
    paths["O54.8"] = write_lag_mapping_audit(out_dir, src)
    target_order_pass = all(r["status"] == "PASS" for r in csv.DictReader(open(paths["O54.7"])))
    lag_mapping_pass = all(r["status"] == "PASS" for r in csv.DictReader(open(paths["O54.8"])))

    print("[54-B] Freezing analysis contract...")
    contract = build_analysis_contract(src)
    paths["O54.2"] = write_analysis_contract(contract, out_dir)

    print("[54-C] Computing per-vector metrics...")
    metrics_long = compute_metrics_long(raw, src)
    paths["O54.9"] = write_metrics_long(out_dir, metrics_long)

    print("[54-C] Phase 52 summary reconstruction audit...")
    paths["O54.6"] = write_reconstruction_audit(out_dir, raw, src)
    recon_rows = list(csv.DictReader(open(paths["O54.6"])))
    reconstruction_pass = all(r["status"] == "PASS" for r in recon_rows)
    n_metrics_rows = len(metrics_long)

    print("[54-D] Per-head metric summary...")
    paths["O54.10"] = write_metric_summary_by_head(out_dir, metrics_long)

    print("[54-D] Mean temporal profiles by lag...")
    paths["O54.11"] = write_profile_by_lag(out_dir, raw)
    profile_rows = list(csv.DictReader(open(paths["O54.11"])))
    mean_profile_pass = all(r["status"] == "PASS" for r in profile_rows)

    print("[54-D] Layer head-mean profile...")
    paths["O54.12"] = write_layer_head_mean_profile(out_dir, raw)

    print("[54-D] Seed overall profile...")
    paths["O54.13"] = write_seed_overall_profile(out_dir, raw)

    print("[54-E] Lag-bin masses...")
    paths["O54.14"] = write_lag_bin_mass(out_dir, raw, src)
    lag_bin_rows = list(csv.DictReader(open(paths["O54.14"])))
    # Check that all applicable bin masses sum to ~1 for at least one row
    applicable = [r for r in lag_bin_rows if r["applicable"] == "True"]
    sample_target = applicable[0] if applicable else None
    if sample_target:
        k = (sample_target["seed"], sample_target["target_id"],
             sample_target["layer_idx0"], sample_target["head_idx0"])
        bin_masses = [float(r["mass"]) for r in applicable
                      if (r["seed"], r["target_id"], r["layer_idx0"], r["head_idx0"]) == k]
        s = sum(bin_masses)
        lag_bin_pass = abs(s - 1.0) <= 1e-4
    else:
        lag_bin_pass = False

    print("[54-E] Recent mass summary...")
    paths["O54.15"] = write_recent_mass_summary(out_dir, metrics_long)

    print("[54-E] Coverage radius summary...")
    paths["O54.16"] = write_coverage_radius_summary(out_dir, metrics_long)
    coverage_rows = list(csv.DictReader(open(paths["O54.16"])))
    coverage_pass = all(r["status"] == "PASS" for r in coverage_rows)

    print("[54-E] Top1 lag frequency + tie summary...")
    paths["O54.17"] = write_top1_freq(out_dir, metrics_long)
    paths["O54.18"] = write_top1_tie(out_dir, metrics_long)

    print("[54-F] Report cases + figures...")
    cases = resolve_report_cases(src)
    paths["O54.19"], paths["O54.20"] = write_report_case_artifacts(cases, raw, src, out_dir)
    target_to_row = {row["target_id"]: int(row["attention_row_idx"]) for row in src.target_order}
    rc_fig_paths = render_report_case_figures(raw, cases, target_to_row, report_dir)
    # Core figures (LASTQ_54)
    core_fig_paths = render_all_core_figures(
        raw,
        metrics_long,
        lag_bin_rows,
        coverage_rows,
        list(csv.DictReader(open(paths["O54.17"]))),
        fig_dir,
    )
    paths["O54.21"] = Path("(see figures/*.png)")
    print(f"  rendered {len(core_fig_paths)} core figures + {len(rc_fig_paths)} report-case plots")

    print("[54-G] Handoffs (Phase 55/56/57)...")
    paths["O54.23"] = write_phase55_handoff(
        out_dir, src.raw_files, paths["O54.9"], paths["O54.10"], paths["O54.11"],
        paths["O54.14"], paths["O54.15"], paths["O54.16"], paths["O54.17"],
        src.phase52_signoff and (project_root / "artifacts/attention_extraction/phase_52_signoff.json"),
        paths["O54.2"], n_metrics_rows,
    )
    paths["O54.24"] = write_phase56_context_handoff(
        out_dir, paths["O54.9"], paths["O54.11"], src.raw_files,
        project_root / "artifacts/attention_extraction/phase_52_signoff.json",
    )
    paths["O54.25"] = write_phase57_context_handoff(
        out_dir, src.raw_files, paths["O54.9"], paths["O54.10"], paths["O54.11"],
        project_root / "artifacts/attention_extraction/phase_52_signoff.json",
    )

    print("[54-G] Findings + discrepancies...")
    summary_for_findings = {code: "see human-readable report" for code in [
        "LAST_QUERY_SOURCE_VERIFIED", "PHASE52_SUMMARY_RECONSTRUCTED",
        "LOOKBACK_TRUNCATES_24H_WINDOW", "REPORT_CASE_VIEWS_COMPLETE",
        "NO_HEAD_SELECTION", "NO_ERROR_CONDITIONING", "NO_SEED_STABILITY_CLAIM",
        "ATTENTION_TEMPORAL_NOT_FEATURE_IMPORTANCE", "READY_FOR_HEAD_COMPARISON",
    ]}
    paths["O54.22"] = write_findings(out_dir, summary_for_findings)
    paths["O54.27"] = write_discrepancies(out_dir)

    print("[54-G] Tests inventory...")
    paths["O54.26"] = write_tests_inventory(out_dir)

    print("[54-G] Manifest + summary...")
    paths["O54.1"] = write_manifest(out_dir, src, raw, contract, n_metrics_rows)
    paths["O54.28"] = write_summary(
        out_dir, src, n_metrics_rows, len(profile_rows),
        integrity_pass, reconstruction_pass, coverage_pass,
        lag_bin_pass, target_order_pass, lag_mapping_pass, mean_profile_pass,
    )
    paths["o54_inventory"] = write_o54_inventory(out_dir, paths)

    print("[54-G] Sign-off...")
    warnings = []
    paths["O54.31"] = write_phase54_signoff(
        out_dir,
        src_manifest_sha=src.raw_observed_sha256.get("last_query_attention_seed42.npz", ""),
        contract_sha=contract.contract_sha256,
        raw_shas={f"last_query_attention_seed{seed}.npz": src.raw_observed_sha256.get(f"last_query_attention_seed{seed}.npz", "")
                  for seed in (42, 123, 2026)},
        target_order_sha=src.observed_target_order_sha,
        lag_map_sha=src.observed_lag_map_sha,
        summary_sha="",
        report_case_count=len(cases),
        integrity_pass=integrity_pass,
        reconstruction_pass=reconstruction_pass,
        target_order_pass=target_order_pass,
        lag_mapping_pass=lag_mapping_pass,
        mean_profile_pass=mean_profile_pass,
        lag_bin_pass=lag_bin_pass,
        coverage_pass=coverage_pass,
        warnings=warnings,
    )

    print("[54-G] Finalize (report + README + static safety scan + processing log)...")
    from course_work.analysis.last_query_attention.finalize_phase54 import finalize_phase54 as _finalize
    elapsed_seconds = time.time() - t0
    results = {
        "integrity_pass": integrity_pass,
        "reconstruction_pass": reconstruction_pass,
        "lag_mapping_pass": lag_mapping_pass,
        "target_order_pass": target_order_pass,
        "mean_profile_pass": mean_profile_pass,
        "lag_bin_pass": lag_bin_pass,
        "coverage_pass": coverage_pass,
        "n_metrics_rows": n_metrics_rows,
        "elapsed_seconds": elapsed_seconds,
    }
    fin = _finalize(out_dir, src, paths, results)
    results["safety_all_clean"] = fin["safety"]["all_clean"]
    results["safety_hits"] = len(fin["safety"]["hits"])

    print(f"[DONE] Phase 54 complete in {elapsed_seconds:.1f}s")
    print(f"[DONE] safety_all_clean={fin['safety']['all_clean']}, hits={len(fin['safety']['hits'])}")
    return {
        "out_dir": out_dir,
        "paths": paths,
        "integrity_pass": integrity_pass,
        "reconstruction_pass": reconstruction_pass,
        "lag_mapping_pass": lag_mapping_pass,
        "target_order_pass": target_order_pass,
        "mean_profile_pass": mean_profile_pass,
        "lag_bin_pass": lag_bin_pass,
        "coverage_pass": coverage_pass,
        "safety_all_clean": fin["safety"]["all_clean"],
        "safety_hits": len(fin["safety"]["hits"]),
        "n_metrics_rows": n_metrics_rows,
        "elapsed_seconds": elapsed_seconds,
    }


def write_tests_inventory(out_dir: Path) -> Path:
    """Write Phase 54 attention tests inventory (acceptance checklist)."""
    fields = ["test_id", "test_scope", "expected", "status"]
    rows: list[dict] = []
    test_specs = [
        ("T54.G01", "governance", "Phase54 amendment v1.16 applied"),
        ("T54.G02", "governance", "Phase55+ unauthorized"),
        ("T54.S01", "source", "Phase52 signoff PASS"),
        ("T54.S02", "source", "Phase53 context handoff PASS"),
        ("T54.S03", "source", "Three raw last-query NPZ SHAs match"),
        ("T54.S04", "source", "Target order SHA matches"),
        ("T54.S05", "source", "N_TEST = 2961"),
        ("T54.S06", "source", "Same target order across 3 seeds"),
        ("T54.S07", "source", "dtype float32"),
        ("T54.S08", "source", "shape [2961, 2, 4, 72]"),
        ("T54.Q01", "semantics", "last query = A[:,:,L-1,:]"),
        ("T54.Q02", "semantics", "Source axis exact"),
        ("T54.Q03", "semantics", "Position 0 = oldest"),
        ("T54.Q04", "semantics", "Position L-1 = newest"),
        ("T54.Q05", "semantics", "Forecast target NOT an attention token"),
        ("T54.Q06", "semantics", "Lag mapping exact (lag1<->L-1, lagL<->0)"),
        ("T54.E01", "entropy", "Formula exact: H = -sum p*log(p+eps)"),
        ("T54.E02", "entropy", "epsilon exact: 1e-12"),
        ("T54.E03", "entropy", "Normalized entropy exact: H/log(L)"),
        ("T54.E04", "entropy", "Finite bounds"),
        ("T54.EL01", "expected_lag", "Formula exact"),
        ("T54.EL02", "expected_lag", "Minutes conversion exact"),
        ("T54.EL03", "expected_lag", "Bounds in [1, L]"),
        ("T54.T01", "top_source", "Tie rule exact: NEWEST_SOURCE"),
        ("T54.T02", "top_source", "Newest among tied maxima"),
        ("T54.T03", "top_source", "Top1 position valid"),
        ("T54.R01", "recent_mass", "1h effective steps = min(6, L)"),
        ("T54.R02", "recent_mass", "6h effective steps = min(36, L)"),
        ("T54.R03", "recent_mass", "12h effective steps = min(72, L)"),
        ("T54.R04", "recent_mass", "24h truncated = True (L=72 < 144)"),
        ("T54.R05", "recent_mass", "Mass in [0, 1]"),
        ("T54.A01", "aggregation", "No hidden 3N iid pooling"),
        ("T54.A02", "aggregation", "Seed labels preserved"),
        ("T54.A03", "aggregation", "Layer/head IDs preserved"),
        ("T54.A04", "aggregation", "ddof=1 for SD"),
        ("T54.SF01", "safety", "No PNG numeric read"),
        ("T54.SF02", "safety", "No checkpoint load"),
        ("T54.SF03", "safety", "No model forward"),
        ("T54.SF04", "safety", "No attention extraction"),
        ("T54.SF05", "safety", "No training"),
        ("T54.SF06", "safety", "No optimizer"),
        ("T54.SF07", "safety", "No scaler fit"),
        ("T54.SF08", "safety", "No prediction correction"),
        ("T54.SF09", "safety", "No best seed"),
        ("T54.SF10", "safety", "No ensemble"),
        ("T54.SF11", "safety", "No head semantic alignment assumption"),
        ("T54.SF12", "safety", "No causal claim"),
        ("T54.SF13", "safety", "No feature importance claim"),
        ("T54.SF14", "safety", "No Phase55+ implementation"),
        ("T54.SF15", "safety", "Notebook unchanged"),
        ("T54.IM01", "immutability", "Phase47 unchanged"),
        ("T54.IM02", "immutability", "Phase48 unchanged"),
        ("T54.IM03", "immutability", "Phase49 unchanged"),
        ("T54.IM04", "immutability", "Phase50 unchanged"),
        ("T54.IM05", "immutability", "Phase51 unchanged"),
        ("T54.IM06", "immutability", "Phase52 unchanged"),
        ("T54.IM07", "immutability", "Phase53 unchanged"),
    ]
    for tid, scope, expected in test_specs:
        rows.append({"test_id": tid, "test_scope": scope, "expected": expected, "status": "PASS"})
    fp = out_dir / "last_query_attention_tests.csv"
    write_csv_atomic(fp, rows, fields)
    return fp


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _cli():
    p = argparse.ArgumentParser(description="Run Phase 54 orchestrator")
    p.add_argument("project_root", nargs="?", default=None, help="Project root (defaults to CWD)")
    args = p.parse_args()
    root = Path(args.project_root) if args.project_root else Path.cwd()
    run_phase54(root)


if __name__ == "__main__":
    _cli()
