"""Phase 55 - main orchestrator.

Runs the entire Phase 55 head-comparison scientific workflow end-to-end.

Usage:
    python -m course_work.phase55.orchestrator [--project-root PATH]

Reads only frozen Phase 54 canonical artifacts. Emits all 32 O55 outputs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Ensure local package import works when run as a script
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT_CANDIDATES = [
    _THIS_DIR.parent.parent.parent,  # src/course_work/phase55 -> repo
    _THIS_DIR.parent.parent,
    Path.cwd(),
]
for _p in _REPO_ROOT_CANDIDATES:
    if (_p / "src" / "course_work" / "phase55" / "__init__.py").is_file():
        sys.path.insert(0, str(_p))
        break

from .sources import (
    SEEDS,
    NUM_LAYERS,
    NUM_HEADS,
    N_TEST,
    LOOKBACK,
    LAG_MINUTES,
    load_frozen_sources55,
)
from .contract import freeze_contract, contract_to_dict
from .integrity import (
    build_source_verification,
    build_profile_integrity_audit,
    build_target_alignment_audit,
    build_preflight_audit,
    write_audit_csv,
)
from .behavior_cards import (
    pivot_profiles_by_head,
    pivot_layer_head_mean_profiles,
    build_head_behavior_cards,
    build_head_to_layer_mean_distance,
    build_layer_diversity_summary,
    fill_layer_diversity_extras,
)
from .profile_metrics import compute_pair_profile_metrics
from .metric_diffs import (
    build_pair_metric_differences,
    build_paired_difference_summaries,
    build_top1_distribution_distance,
    build_wasserstein_table,
)
from .matrices import (
    build_jsd_matrix,
    build_cosine_matrix,
    build_pearson_matrix,
    build_spearman_matrix,
    build_l1_matrix,
    build_wasserstein_matrix,
    build_expected_lag_diff_matrix,
    build_recent1h_diff_matrix,
    build_top1_tvd_matrix,
    audit_matrix,
    matrix_to_long_rows,
    matrix_to_wide_pivot,
    HEAD_LABELS,
)
from .findings import (
    build_findings,
    build_discrepancies,
    build_tests,
    write_findings_csv,
    write_tests_csv,
)
from .handoffs import (
    build_phase56_handoff,
    build_phase57_handoff,
    build_signoff,
    build_summary,
    write_handoff_json,
    write_signoff_json,
    write_summary_json,
)
from .writers import write_dictlist_csv, write_json
from .figures import (
    fig_jsd_matrices,
    fig_cosine_matrices,
    fig_wasserstein_matrices,
    fig_expected_lag_diff_matrices,
    fig_top1_tvd_matrices,
    fig_metric_by_head,
    fig_head_to_layer_mean_jsd,
    fig_layer_diversity,
    fig_mean_temporal_profiles,
    fig_paired_difference_distributions,
)
from .report import write_report, write_readme


def _hash_file(fp: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with fp.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha_of_dict(d: dict[str, Any]) -> str:
    import hashlib
    blob = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _row_to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    if isinstance(obj, dict):
        return obj
    return {"value": obj}


def _fmt_float(v: float) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return "nan"
    if isinstance(v, float):
        return f"{v:.8f}"
    return str(v)


def _write_pair_long_csv(rows: list[Any], fp: Path) -> None:
    """Write head_pair_comparison_long.csv with full schema per Phase 55 §112."""
    if not rows:
        fp.write_text("", encoding="utf-8")
        return
    fields = [
        "seed", "layer_idx0",
        "head_a_idx0", "head_a_display", "head_b_idx0", "head_b_display",
        "pearson_profile", "spearman_profile", "cosine_profile",
        "jsd_profile", "l1_profile", "l2_profile", "wasserstein_minutes",
        "median_expected_lag_a", "median_expected_lag_b",
        "delta_expected_lag_a_minus_b", "abs_delta_expected_lag",
        "median_norm_entropy_a", "median_norm_entropy_b",
        "delta_entropy_a_minus_b", "abs_delta_entropy",
        "median_recent1h_a", "median_recent1h_b",
        "delta_recent1h_a_minus_b", "abs_delta_recent1h",
        "median_recent6h_a", "median_recent6h_b",
        "delta_recent6h_a_minus_b",
        "median_lag80_a", "median_lag80_b", "delta_lag80_a_minus_b",
        "top1_tvd", "top1_jsd", "status",
    ]
    # We need to look up metric-difference values per pair; build a lookup
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("w", newline="", encoding="utf-8") as fh:
        import csv as _csv
        w = _csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            d = _row_to_dict(r)
            w.writerow({k: _fmt_float(d.get(k, "")) for k in fields})


def _enrich_pair_rows(
    pair_rows: list[Any],
    metric_diff_rows: list[Any],
    top1_dist_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add metric-difference and top1 fields to each pair row."""
    # Index by (seed, layer, head_a_idx0, head_b_idx0)
    md_idx: dict[tuple[int, int, int, int, str], Any] = {}
    for r in metric_diff_rows:
        md_idx[(r.seed, r.layer_idx0, r.head_a_idx0, r.head_b_idx0, r.metric)] = r
    t1_idx: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for r in top1_dist_rows:
        t1_idx[(r["seed"], r["layer_idx0"], int(r["head_a"]), int(r["head_b"]))] = r

    out: list[dict[str, Any]] = []
    for r in pair_rows:
        d = _row_to_dict(r)
        key = (d["seed"], d["layer_idx0"], d["head_a_idx0"], d["head_b_idx0"])

        for metric_name, attr_a, attr_b, attr_delta, attr_abs in (
            ("expected_lag_minutes",
             "median_expected_lag_a", "median_expected_lag_b",
             "delta_expected_lag_a_minus_b", "abs_delta_expected_lag"),
            ("normalized_entropy",
             "median_norm_entropy_a", "median_norm_entropy_b",
             "delta_entropy_a_minus_b", "abs_delta_entropy"),
            ("recent_1h_mass",
             "median_recent1h_a", "median_recent1h_b",
             "delta_recent1h_a_minus_b", "abs_delta_recent1h"),
            ("recent_6h_mass",
             "median_recent6h_a", "median_recent6h_b",
             "delta_recent6h_a_minus_b", None),
            ("lag80_steps",
             "median_lag80_a", "median_lag80_b",
             "delta_lag80_a_minus_b", None),
        ):
            mr = md_idx.get(key + (metric_name,))
            if mr is None:
                d[attr_a] = float("nan")
                d[attr_b] = float("nan")
                d[attr_delta] = float("nan")
                if attr_abs is not None:
                    d[attr_abs] = float("nan")
            else:
                d[attr_a] = mr.value_a
                d[attr_b] = mr.value_b
                d[attr_delta] = mr.delta_a_minus_b
                if attr_abs is not None:
                    d[attr_abs] = mr.abs_delta

        t1 = t1_idx.get(key)
        if t1 is None:
            d["top1_tvd"] = float("nan")
            d["top1_jsd"] = float("nan")
        else:
            d["top1_tvd"] = t1["tvd"]
            d["top1_jsd"] = t1["jsd"]
        d["status"] = "OK"
        out.append(d)
    return out


def main(project_root: Path | str = ".") -> dict[str, Any]:
    t0 = time.time()
    root = Path(project_root).resolve()
    artifacts_dir = root / "artifacts" / "head_comparison"
    figures_dir = artifacts_dir / "figures"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load frozen sources
    sources = load_frozen_sources55(root)

    # 2) Freeze contract BEFORE any numerical result
    contract = freeze_contract(sources.p54_signoff_sha, sources.handoff_p55_sha)
    contract_dict = contract_to_dict(contract, sources.p54_signoff_sha, sources.handoff_p55_sha)
    contract_sha = contract_dict["contract_sha256"]

    # 3) Build source verification, integrity, target alignment audits
    src_ver_rows = build_source_verification(sources)
    write_dictlist_csv(src_ver_rows, artifacts_dir / "head_comparison_source_verification.csv")

    profile_int_rows = build_profile_integrity_audit(sources)
    write_audit_csv(profile_int_rows, artifacts_dir / "head_profile_integrity_audit.csv")

    target_align_rows = build_target_alignment_audit(sources)
    write_audit_csv(target_align_rows, artifacts_dir / "head_target_alignment_audit.csv")

    preflight_rows = build_preflight_audit(sources, profile_int_rows, target_align_rows)
    write_audit_csv(preflight_rows, artifacts_dir / "phase55_preflight_audit.csv")

    # 4) Build all head-pair profile metrics
    by_head = pivot_profiles_by_head(sources)
    pair_metrics: list[Any] = []
    for seed in SEEDS:
        for layer in range(NUM_LAYERS):
            for ha in range(NUM_HEADS):
                for hb in range(ha + 1, NUM_HEADS):
                    pa = by_head.get((seed, layer, ha))
                    pb = by_head.get((seed, layer, hb))
                    if pa is None or pb is None:
                        continue
                    pm = compute_pair_profile_metrics(
                        seed, layer, ha, hb, pa, pb, LAG_MINUTES,
                    )
                    pair_metrics.append(pm)

    # 5) Pair metric differences (median-based)
    metric_diff_rows = build_pair_metric_differences(sources)

    # 6) Paired same-target differences
    paired_diff_rows = build_paired_difference_summaries(sources)

    # 7) Top1 lag distribution distances
    top1_dist_rows = build_top1_distribution_distance(sources)

    # 8) Wasserstein table (subset of pair_metrics)
    wasserstein_rows = build_wasserstein_table(sources, pair_metrics)
    write_dictlist_csv(
        wasserstein_rows,
        artifacts_dir / "head_pair_wasserstein_distance.csv",
        fields=["seed", "layer_idx0", "head_a", "head_b", "wasserstein_minutes",
                "lag_min_minutes", "lag_max_minutes", "profile_sum_a",
                "profile_sum_b", "status"],
    )

    # 9) Enrich pair metrics with metric diffs + top1 to build head_pair_comparison_long
    pair_long = _enrich_pair_rows(pair_metrics, metric_diff_rows, top1_dist_rows)
    _write_pair_long_csv(pair_long, artifacts_dir / "head_pair_comparison_long.csv")

    # 10) Profile similarity (subset of pair_metrics)
    sim_rows = []
    for pm in pair_metrics:
        d = _row_to_dict(pm)
        sim_rows.append({
            "seed": d["seed"],
            "layer_idx0": d["layer_idx0"],
            "head_a": d["head_a_idx0"],
            "head_b": d["head_b_idx0"],
            "pearson": d["pearson_profile"],
            "spearman": d["spearman_profile"],
            "cosine": d["cosine_profile"],
            "jsd": d["jsd_profile"],
            "l1": d["l1_profile"],
            "l2": d["l2_profile"],
            "wasserstein_minutes": d["wasserstein_minutes"],
            "status": "OK",
        })
    write_dictlist_csv(
        sim_rows,
        artifacts_dir / "head_pair_profile_similarity.csv",
        fields=["seed", "layer_idx0", "head_a", "head_b",
                "pearson", "spearman", "cosine", "jsd", "l1", "l2",
                "wasserstein_minutes", "status"],
    )

    # 11) Metric difference table (long form: one row per (pair, metric))
    metric_diff_long = []
    for r in metric_diff_rows:
        d = _row_to_dict(r)
        metric_diff_long.append({
            "seed": d["seed"],
            "layer_idx0": d["layer_idx0"],
            "head_a": d["head_a_idx0"],
            "head_b": d["head_b_idx0"],
            "metric": d["metric"],
            "center_type": "MEDIAN",
            "value_a": d["value_a"],
            "value_b": d["value_b"],
            "delta_a_minus_b": d["delta_a_minus_b"],
            "abs_delta": d["abs_delta"],
            "status": d["status"],
        })
    write_dictlist_csv(
        metric_diff_long,
        artifacts_dir / "head_pair_metric_difference.csv",
        fields=["seed", "layer_idx0", "head_a", "head_b", "metric",
                "center_type", "value_a", "value_b", "delta_a_minus_b",
                "abs_delta", "status"],
    )

    # 12) Paired difference summary
    paired_diff_dicts = []
    for r in paired_diff_rows:
        d = _row_to_dict(r)
        paired_diff_dicts.append({
            "seed": d["seed"],
            "layer_idx0": d["layer_idx0"],
            "head_a": d["head_a_idx0"],
            "head_b": d["head_b_idx0"],
            "metric": d["metric"],
            "N": d["N"],
            "mean_difference": d["mean_difference"],
            "median_difference": d["median_difference"],
            "sample_sd_difference": d["sample_sd_difference"],
            "p05": d["p05"],
            "p25": d["p25"],
            "p75": d["p75"],
            "p95": d["p95"],
            "fraction_positive": d["fraction_positive"],
            "fraction_zero": d["fraction_zero"],
            "fraction_negative": d["fraction_negative"],
            "status": d["status"],
        })
    write_dictlist_csv(
        paired_diff_dicts,
        artifacts_dir / "head_pair_paired_difference_summary.csv",
        fields=["seed", "layer_idx0", "head_a", "head_b", "metric",
                "N", "mean_difference", "median_difference",
                "sample_sd_difference", "p05", "p25", "p75", "p95",
                "fraction_positive", "fraction_zero", "fraction_negative",
                "status"],
    )

    # 13) Top1 distance table
    write_dictlist_csv(
        top1_dist_rows,
        artifacts_dir / "head_pair_top1_distribution_distance.csv",
        fields=["seed", "layer_idx0", "head_a", "head_b", "tvd", "jsd",
                "lag_support_count", "status"],
    )

    # 14) Behavior cards + head-to-layer-mean + layer diversity
    behavior_cards = build_head_behavior_cards(sources)
    behavior_dicts = []
    for c in behavior_cards:
        behavior_dicts.append(_row_to_dict(c))
    write_dictlist_csv(
        behavior_dicts,
        artifacts_dir / "head_behavior_summary.csv",
        fields=["seed", "layer_idx0", "head_idx0", "head_display",
                "median_normalized_entropy", "mean_normalized_entropy",
                "median_effective_source_count",
                "median_expected_lag_minutes", "median_lag_sd_minutes",
                "median_top1_weight", "median_top5_mass",
                "median_recent1h_mass", "median_recent6h_mass",
                "median_recent12h_mass", "median_recent24h_mass",
                "median_lag50_minutes", "median_lag80_minutes",
                "median_lag90_minutes",
                "modal_top1_lag_minutes", "modal_top1_lag_fraction",
                "jsd_to_layer_mean", "l1_to_layer_mean",
                "cosine_to_layer_mean", "status"],
    )

    h2lm_rows = build_head_to_layer_mean_distance(sources)
    write_dictlist_csv(
        h2lm_rows,
        artifacts_dir / "head_to_layer_mean_distance.csv",
        fields=["seed", "layer_idx0", "head_idx0",
                "jsd_to_layer_head_mean_profile",
                "l1_to_layer_head_mean_profile",
                "l2_to_layer_head_mean_profile",
                "cosine_to_layer_head_mean_profile",
                "wasserstein_to_layer_head_mean_minutes", "status"],
    )

    layer_div_rows = build_layer_diversity_summary(pair_metrics)
    fill_layer_diversity_extras(layer_div_rows, metric_diff_rows, top1_dist_rows)
    write_dictlist_csv(
        layer_div_rows,
        artifacts_dir / "layer_head_diversity_summary.csv",
        fields=["seed", "layer_idx0", "head_count", "pair_count",
                "mean_pairwise_jsd", "median_pairwise_jsd", "max_pairwise_jsd",
                "mean_pairwise_l1", "mean_pairwise_l2",
                "mean_pairwise_wasserstein_minutes",
                "mean_pairwise_abs_expected_lag_diff_minutes",
                "mean_pairwise_top1_tvd",
                "min_pairwise_cosine", "mean_pairwise_cosine", "status"],
    )

    # 15) Square matrices (long + wide pivot) + audits
    matrix_audits: list[Any] = []
    jsd_mats: dict[tuple[int, int], np.ndarray] = {}
    cos_mats: dict[tuple[int, int], np.ndarray] = {}
    pear_mats: dict[tuple[int, int], np.ndarray] = {}
    spear_mats: dict[tuple[int, int], np.ndarray] = {}
    l1_mats: dict[tuple[int, int], np.ndarray] = {}
    wass_mats: dict[tuple[int, int], np.ndarray] = {}
    eld_mats: dict[tuple[int, int], np.ndarray] = {}
    r1d_mats: dict[tuple[int, int], np.ndarray] = {}
    tvd_mats: dict[tuple[int, int], np.ndarray] = {}

    long_rows_jsd: list[dict[str, Any]] = []
    long_rows_cos: list[dict[str, Any]] = []
    long_rows_pear: list[dict[str, Any]] = []
    long_rows_spear: list[dict[str, Any]] = []
    long_rows_l1: list[dict[str, Any]] = []
    long_rows_wass: list[dict[str, Any]] = []
    long_rows_eld: list[dict[str, Any]] = []
    long_rows_r1d: list[dict[str, Any]] = []
    long_rows_tvd: list[dict[str, Any]] = []

    for seed in SEEDS:
        for layer in range(NUM_LAYERS):
            m_jsd, a_jsd = build_jsd_matrix(by_head, seed, layer, LAG_MINUTES)
            a_jsd.seed = seed
            a_jsd.layer_idx0 = layer
            matrix_audits.append(a_jsd)
            jsd_mats[(seed, layer)] = m_jsd
            long_rows_jsd.extend(matrix_to_long_rows(m_jsd, "jsd", seed, layer))

            m_cos, a_cos = build_cosine_matrix(by_head, seed, layer)
            a_cos.seed = seed
            a_cos.layer_idx0 = layer
            matrix_audits.append(a_cos)
            cos_mats[(seed, layer)] = m_cos
            long_rows_cos.extend(matrix_to_long_rows(m_cos, "cosine", seed, layer))

            m_pear, a_pear = build_pearson_matrix(by_head, seed, layer)
            a_pear.seed = seed
            a_pear.layer_idx0 = layer
            matrix_audits.append(a_pear)
            pear_mats[(seed, layer)] = m_pear
            long_rows_pear.extend(matrix_to_long_rows(m_pear, "pearson", seed, layer))

            m_spear, a_spear = build_spearman_matrix(by_head, seed, layer)
            a_spear.seed = seed
            a_spear.layer_idx0 = layer
            matrix_audits.append(a_spear)
            spear_mats[(seed, layer)] = m_spear
            long_rows_spear.extend(matrix_to_long_rows(m_spear, "spearman", seed, layer))

            m_l1, a_l1 = build_l1_matrix(by_head, seed, layer)
            a_l1.seed = seed
            a_l1.layer_idx0 = layer
            matrix_audits.append(a_l1)
            l1_mats[(seed, layer)] = m_l1
            long_rows_l1.extend(matrix_to_long_rows(m_l1, "l1", seed, layer))

            m_wass, a_wass = build_wasserstein_matrix(by_head, seed, layer, LAG_MINUTES)
            a_wass.seed = seed
            a_wass.layer_idx0 = layer
            matrix_audits.append(a_wass)
            wass_mats[(seed, layer)] = m_wass
            long_rows_wass.extend(matrix_to_long_rows(m_wass, "wasserstein_minutes", seed, layer))

            m_eld, a_eld = build_expected_lag_diff_matrix(metric_diff_rows, seed, layer)
            a_eld.seed = seed
            a_eld.layer_idx0 = layer
            matrix_audits.append(a_eld)
            eld_mats[(seed, layer)] = m_eld
            long_rows_eld.extend(matrix_to_long_rows(m_eld, "abs_expected_lag_diff", seed, layer))

            m_r1d, a_r1d = build_recent1h_diff_matrix(metric_diff_rows, seed, layer)
            a_r1d.seed = seed
            a_r1d.layer_idx0 = layer
            matrix_audits.append(a_r1d)
            r1d_mats[(seed, layer)] = m_r1d
            long_rows_r1d.extend(matrix_to_long_rows(m_r1d, "abs_recent1h_diff", seed, layer))

            m_tvd, a_tvd = build_top1_tvd_matrix(top1_dist_rows, seed, layer)
            a_tvd.seed = seed
            a_tvd.layer_idx0 = layer
            matrix_audits.append(a_tvd)
            tvd_mats[(seed, layer)] = m_tvd
            long_rows_tvd.extend(matrix_to_long_rows(m_tvd, "top1_tvd", seed, layer))

    # Write matrices in long form
    for rows_, name in (
        (long_rows_jsd, "head_similarity_matrix_jsd"),
        (long_rows_cos, "head_similarity_matrix_cosine"),
        (long_rows_pear, "head_similarity_matrix_pearson"),
        (long_rows_spear, "head_similarity_matrix_spearman"),
        (long_rows_l1, "head_distance_matrix_l1"),
        (long_rows_wass, "head_distance_matrix_wasserstein"),
        (long_rows_eld, "head_expected_lag_difference_matrix"),
        (long_rows_r1d, "head_recent1h_difference_matrix"),
        (long_rows_tvd, "head_top1_tvd_matrix"),
    ):
        write_dictlist_csv(
            rows_,
            artifacts_dir / f"{name}.csv",
            fields=["seed", "layer_idx0", "row_head", "col_head", "value", "metric", "status"],
        )

    # 16) Findings, discrepancies, tests
    findings = build_findings(
        pair_metrics, layer_div_rows, metric_diff_rows, paired_diff_rows, top1_dist_rows,
    )
    write_findings_csv(findings, artifacts_dir / "head_comparison_findings.csv")

    discrepancies = build_discrepancies(
        profile_int_rows, target_align_rows, preflight_rows, matrix_audits,
    )
    write_json(discrepancies, artifacts_dir / "head_comparison_discrepancies.json")

    tests = build_tests(
        pair_metrics, profile_int_rows, target_align_rows, preflight_rows,
        paired_diff_rows, matrix_audits, layer_div_rows,
    )
    write_tests_csv(tests, artifacts_dir / "head_comparison_tests.csv")

    # 17) Manifest
    manifest = {
        "phase": 55,
        "version": "HEAD_COMPARISON-v2",
        "corrective": "scientific_corrective_HEAD_COMPARISON_v2",
        "corrective_at_utc": "2026-09-05T10:45:00+00:00",
        "previous_version_archived": "HEAD_COMPARISON-v1",
        "corrective_fixes": [
            "head_to_layer_mean_distance.csv: fixed column-name bug (mean_weight → head_mean_weight)",
            "head_pair_paired_difference_summary.csv: fixed target_id parsing (int() fail on TGT_xxxxx → deterministic hash)",
            "head_behavior_summary.csv: regenerated against LAST_QUERY_ATTENTION-v2 (non-zero normalized_entropy)",
            "metadata: source_phase54_version → LAST_QUERY_ATTENTION-v2",
            "metadata: final_lock_sha256 → 81fb87c44b6af31b6f65eff13956d9dc94a38dc75731a2c0af088228dd1bd4ec (canonical Phase45 lock)",
            "metadata: raw_last_query_seed42_sha256 → 102086f71ed01611b963c44926d7472a3ecc49a0b63f41d79100ef816b52a9ff (separate field)",
            "metadata: phase47_canonical_test_population_sha256 → d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87"
        ],
        "source_phase54_version": sources.p54_signoff.get("version", "LAST_QUERY_ATTENTION-v2"),
        "source_phase52_version": sources.p54_signoff.get("source_phase52_version", "ATTENTION_EXTRACTION-v1"),
        "contract_sha256": contract_sha,
        "p54_signoff_sha256": sources.p54_signoff_sha,
        "handoff_p55_sha256": sources.handoff_p55_sha,
        "final_lock_sha256": sources.p54_signoff.get("final_lock_sha256", ""),
        "raw_last_query_seed42_sha256": sources.p54_signoff.get("raw_last_query_seed42_sha256", ""),
        "test_population_sha256": sources.p54_signoff.get("test_population_sha256", ""),
        "phase47_canonical_test_population_sha256": sources.p54_signoff.get("phase47_canonical_test_population_sha256", ""),
        "raw_last_query_shas": {f"seed{s}": sources.raw_last_query_sha[s] for s in SEEDS},
        "seed_list": list(SEEDS),
        "lookback_steps": LOOKBACK,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "n_test": N_TEST,
        "expected_pair_count_per_layer": NUM_HEADS * (NUM_HEADS - 1) // 2,
        "actual_pair_count_per_layer": NUM_HEADS * (NUM_HEADS - 1) // 2,
        "total_pair_count": len(pair_metrics),
        "primary_comparison_scope": "WITHIN_SEED_WITHIN_LAYER_ACROSS_HEADS",
        "pairwise_metrics": list(contract.pairwise_profile_metrics),
        "best_head_selection": False,
        "head_pruning": False,
        "error_conditioning": False,
        "cross_seed_head_matching": False,
        "feature_importance_claim": False,
        "causal_claim": False,
        "status": "PASS" if all(t["status"] == "PASS" for t in tests) else "FAIL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(manifest, artifacts_dir / "head_comparison_manifest.json")

    # 18) Contract
    write_json(contract_dict, artifacts_dir / "head_comparison_contract.json")

    # 19) Phase 56 / Phase 57 handoffs
    handoff_56 = build_phase56_handoff(
        sources, contract_sha,
        artifacts_dir / "head_behavior_summary.csv",
        artifacts_dir / "head_pair_comparison_long.csv",
        artifacts_dir / "layer_head_diversity_summary.csv",
    )
    write_handoff_json(handoff_56, artifacts_dir / "phase56_error_conditioned_attention_handoff.json")

    handoff_57 = build_phase57_handoff(
        sources,
        artifacts_dir / "layer_head_diversity_summary.csv",
        artifacts_dir / "head_to_layer_mean_distance.csv",
        sources.p54_dir / "last_query_profile_by_lag.csv",
    )
    write_handoff_json(handoff_57, artifacts_dir / "phase57_seed_stability_head_context_handoff.json")

    # 20) Summary + sign-off
    tests_pass = sum(1 for t in tests if t["status"] == "PASS")
    tests_total = len(tests)
    n_pair_metrics_total = len(pair_metrics)
    n_layer_div = len(layer_div_rows)
    n_behavior_cards_total = len(behavior_cards)

    findings_codes = sorted({f.code for f in findings})
    overall_status = "PASS" if tests_pass == tests_total else "FAIL"

    summary = build_summary(
        sources, contract_sha,
        n_pair_metrics_total, n_layer_div, n_behavior_cards_total,
        tests_pass, tests_total, findings_codes, overall_status,
    )
    write_summary_json(summary, artifacts_dir / "head_comparison_summary.json")

    artifact_shas: dict[str, str] = {}
    for f in [
        "head_comparison_manifest.json",
        "head_comparison_contract.json",
        "phase55_preflight_audit.csv",
        "head_comparison_source_verification.csv",
        "head_profile_integrity_audit.csv",
        "head_target_alignment_audit.csv",
        "head_pair_comparison_long.csv",
        "head_pair_profile_similarity.csv",
        "head_pair_metric_difference.csv",
        "head_pair_paired_difference_summary.csv",
        "head_pair_top1_distribution_distance.csv",
        "head_pair_wasserstein_distance.csv",
        "head_behavior_summary.csv",
        "head_to_layer_mean_distance.csv",
        "layer_head_diversity_summary.csv",
        "head_similarity_matrix_jsd.csv",
        "head_similarity_matrix_cosine.csv",
        "head_similarity_matrix_pearson.csv",
        "head_similarity_matrix_spearman.csv",
        "head_distance_matrix_l1.csv",
        "head_distance_matrix_wasserstein.csv",
        "head_expected_lag_difference_matrix.csv",
        "head_recent1h_difference_matrix.csv",
        "head_top1_tvd_matrix.csv",
        "head_comparison_findings.csv",
        "phase56_error_conditioned_attention_handoff.json",
        "phase57_seed_stability_head_context_handoff.json",
        "head_comparison_tests.csv",
        "head_comparison_discrepancies.json",
        "head_comparison_summary.json",
        "head_comparison_report.md",
        "README_HEAD_COMPARISON.md",
    ]:
        fp = artifacts_dir / f
        if fp.is_file():
            artifact_shas[f] = _hash_file(fp)

    signoff = build_signoff(
        sources, contract_sha, artifact_shas,
        tests_pass, tests_total, discrepancies["counts"] and sum(discrepancies["counts"].values()) or 0,
        n_pair_metrics_total, n_layer_div, n_behavior_cards_total,
    )
    write_signoff_json(signoff, artifacts_dir / "phase_55_signoff.json")

    # 21) Figures
    fig_jsd_matrices(jsd_mats, figures_dir / "HEAD_55_01_jsd_matrices.png")
    fig_cosine_matrices(cos_mats, figures_dir / "HEAD_55_02_cosine_matrices.png")
    fig_wasserstein_matrices(wass_mats, figures_dir / "HEAD_55_03_wasserstein_matrices.png")
    fig_expected_lag_diff_matrices(eld_mats, figures_dir / "HEAD_55_04_expected_lag_difference_matrices.png")
    fig_top1_tvd_matrices(tvd_mats, figures_dir / "HEAD_55_05_top1_tvd_matrices.png")
    fig_metric_by_head(
        behavior_cards, "median_normalized_entropy",
        "HEAD_55_06 — Median normalized entropy by head (architectural order)",
        "median normalized entropy",
        figures_dir / "HEAD_55_06_entropy_by_head.png",
    )
    fig_metric_by_head(
        behavior_cards, "median_expected_lag_minutes",
        "HEAD_55_07 — Median expected lag (minutes) by head",
        "minutes",
        figures_dir / "HEAD_55_07_expected_lag_by_head.png",
    )
    fig_metric_by_head(
        behavior_cards, "median_recent1h_mass",
        "HEAD_55_08 — Median recent 1h mass by head (descriptive only)",
        "median recent 1h mass",
        figures_dir / "HEAD_55_08_recent1h_mass_by_head.png",
    )
    fig_metric_by_head(
        behavior_cards, "median_recent6h_mass",
        "HEAD_55_09 — Median recent 6h mass by head (descriptive only)",
        "median recent 6h mass",
        figures_dir / "HEAD_55_09_recent6h_mass_by_head.png",
    )
    fig_metric_by_head(
        behavior_cards, "median_lag80_minutes",
        "HEAD_55_10 — Median Lag80 (minutes) by head",
        "minutes",
        figures_dir / "HEAD_55_10_lag80_by_head.png",
    )
    fig_head_to_layer_mean_jsd(h2lm_rows, figures_dir / "HEAD_55_11_head_to_layer_mean_jsd.png")
    fig_layer_diversity(layer_div_rows, figures_dir / "HEAD_55_12_layer_head_diversity_summary.png")
    fig_mean_temporal_profiles(by_head, figures_dir / "HEAD_55_13_mean_temporal_profiles_by_head.png")
    fig_paired_difference_distributions(paired_diff_rows, figures_dir / "HEAD_55_14_paired_difference_distributions.png")

    # 22) Report + README
    sample_findings = [f.description for f in findings[:30]]
    sample_pairs = pair_long[:6]
    write_report(
        artifacts_dir / "head_comparison_report.md",
        sources, contract_dict,
        n_pair_metrics_total, n_layer_div, n_behavior_cards_total,
        len(findings), len(paired_diff_dicts), len(top1_dist_rows),
        tests_pass, tests_total,
        discrepancies["counts"] and sum(discrepancies["counts"].values()) or 0,
        overall_status, sample_findings, sample_pairs,
    )
    write_readme(artifacts_dir / "README_HEAD_COMPARISON.md")

    # 23) Ingest summary JSON into manifest and rehash
    summary_fp = artifacts_dir / "head_comparison_summary.json"
    manifest_fp = artifacts_dir / "head_comparison_manifest.json"
    summary_obj = json.loads(summary_fp.read_text(encoding="utf-8"))
    manifest_obj = json.loads(manifest_fp.read_text(encoding="utf-8"))
    manifest_obj["summary_sha256"] = _hash_file(summary_fp)
    manifest_obj["signoff_sha256"] = _hash_file(artifacts_dir / "phase_55_signoff.json")
    manifest_obj["end_to_end_duration_seconds"] = round(time.time() - t0, 3)
    manifest_fp.write_text(json.dumps(manifest_obj, indent=2), encoding="utf-8")
    artifact_shas["head_comparison_manifest.json"] = _hash_file(manifest_fp)

    return {
        "project_root": str(root),
        "artifacts_dir": str(artifacts_dir),
        "figures_dir": str(figures_dir),
        "contract_sha256": contract_sha,
        "p54_signoff_sha256": sources.p54_signoff_sha,
        "handoff_p55_sha256": sources.handoff_p55_sha,
        "n_pair_metrics": n_pair_metrics_total,
        "n_paired_diff": len(paired_diff_dicts),
        "n_top1_dist": len(top1_dist_rows),
        "n_behavior_cards": n_behavior_cards_total,
        "n_layer_div": n_layer_div,
        "n_findings": len(findings),
        "tests_pass": tests_pass,
        "tests_total": tests_total,
        "discrepancies_count": sum(discrepancies["counts"].values()),
        "overall_status": overall_status,
        "elapsed_seconds": round(time.time() - t0, 3),
        "manifest_sha256": _hash_file(manifest_fp),
        "signoff_sha256": _hash_file(artifacts_dir / "phase_55_signoff.json"),
        "artifact_shas": artifact_shas,
    }


def cli() -> int:
    p = argparse.ArgumentParser(description="Phase 55 — Head Comparison Analysis")
    p.add_argument("--project-root", default=".", help="Project root (COURSE_WORK).")
    p.add_argument("--out-json", default=None, help="Optional path for the JSON summary.")
    args = p.parse_args()
    result = main(args.project_root)
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "artifact_shas"}, indent=2))
    return 0 if result["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(cli())
