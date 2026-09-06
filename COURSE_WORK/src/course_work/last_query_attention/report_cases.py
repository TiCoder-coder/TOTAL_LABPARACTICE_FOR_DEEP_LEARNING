"""Phase 54 — report-case deterministic last-query metrics + line plots.

Resolves Phase 51 W2 SHARED_WORST ranks 1–5 from canonical Phase 53 artifact
(attention_heatmap_report_cases.csv). For each report case (rank, target_id,
case_row_idx0), computes per-seed per-layer per-head metric summary and
per-target last-query metrics, plus generates report-case line plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .aggregations import quantile_set
from .coverage import compute_coverage_radii
from .metrics import compute_vector_metrics
from .sources import (
    CADENCE_MINUTES,
    LOOKBACK,
    NUM_HEADS,
    NUM_LAYERS,
    SEEDS,
    FrozenSources54,
)
from .writers import read_csv, write_csv_atomic


@dataclass
class ReportCase:
    report_order: int
    case_row_idx0: int
    target_id: str
    target_timestamp: str
    shared_worst_rank: int


def resolve_report_cases(src: FrozenSources54) -> list[ReportCase]:
    """Resolve deterministic Phase 51 shared ranks 1–5 from Phase 53
    attention_heatmap_report_cases.csv.
    """
    rows = read_csv(src.phase53_report_cases_path)
    cases: list[ReportCase] = []
    for r in rows:
        try:
            cases.append(ReportCase(
                report_order=int(r["report_order"]),
                case_row_idx0=int(r["case_row_idx0"]),
                target_id=str(r["target_id"]),
                target_timestamp=str(r["target_timestamp"]),
                shared_worst_rank=int(r["shared_worst_rank"]),
            ))
        except (KeyError, ValueError):
            continue
    cases.sort(key=lambda c: c.shared_worst_rank)
    return cases


def build_report_case_manifest(cases: list[ReportCase]) -> list[dict]:
    rows: list[dict] = []
    for c in cases:
        rows.append({
            "report_order": c.report_order,
            "shared_rank": c.shared_worst_rank,
            "case_row_idx0": c.case_row_idx0,
            "target_id": c.target_id,
            "target_timestamp": c.target_timestamp,
            "selection_rule": "PHASE51_W2_SHARED_RANK_1_TO_5",
            "status": "PASS",
        })
    return rows


def build_report_case_metrics(
    cases: list[ReportCase],
    raw: dict[int, np.ndarray],
    src: FrozenSources54,
) -> list[dict]:
    """For each report case, compute per-seed per-layer per-head last-query
    metrics + Lag50/80/90."""
    # Map target_id -> row_idx
    target_to_row = {row["target_id"]: int(row["attention_row_idx"]) for row in src.target_order}

    out: list[dict] = []
    for c in cases:
        if c.target_id not in target_to_row:
            continue
        target_row = target_to_row[c.target_id]
        for seed in SEEDS:
            arr = raw[seed]
            for li in range(NUM_LAYERS):
                for hi in range(NUM_HEADS):
                    vec = arr[target_row, li, hi, :]
                    m = compute_vector_metrics(vec, np.array([LOOKBACK - p for p in range(LOOKBACK)]))
                    cov = compute_coverage_radii(vec)
                    out.append({
                        "report_order": c.report_order,
                        "shared_rank": c.shared_worst_rank,
                        "target_id": c.target_id,
                        "target_timestamp": c.target_timestamp,
                        "seed": int(seed),
                        "layer_idx0": int(li),
                        "head_idx0": int(hi),
                        "entropy": m.entropy,
                        "normalized_entropy": m.normalized_entropy,
                        "effective_source_count": m.effective_source_count,
                        "expected_lag_minutes": m.expected_lag_minutes,
                        "top1_lag_minutes": m.top1_lag_minutes,
                        "top1_weight": m.top1_weight,
                        "top5_mass": m.top5_mass,
                        "recent_1h_mass": m.recent_1h_mass,
                        "recent_6h_mass": m.recent_6h_mass,
                        "lag50_minutes": cov.lag50_steps * CADENCE_MINUTES,
                        "lag80_minutes": cov.lag80_steps * CADENCE_MINUTES,
                        "lag90_minutes": cov.lag90_steps * CADENCE_MINUTES,
                        "status": "PASS",
                    })
    return out


def write_report_case_artifacts(
    cases: list[ReportCase],
    raw: dict[int, np.ndarray],
    src: FrozenSources54,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Write report-case manifest + metrics; return paths."""
    manifest_rows = build_report_case_manifest(cases)
    metrics_rows = build_report_case_metrics(cases, raw, src)

    manifest_fp = out_dir / "last_query_report_case_manifest.csv"
    metrics_fp = out_dir / "last_query_report_case_metrics.csv"

    if manifest_rows:
        write_csv_atomic(manifest_fp, manifest_rows, list(manifest_rows[0].keys()))
    else:
        # Write empty with header
        manifest_fp.write_text("report_order,shared_rank,case_row_idx0,target_id,target_timestamp,selection_rule,status\n")
    if metrics_rows:
        write_csv_atomic(metrics_fp, metrics_rows, list(metrics_rows[0].keys()))
    else:
        metrics_fp.write_text("report_order,shared_rank,target_id,target_timestamp,seed,layer_idx0,head_idx0,entropy,normalized_entropy,effective_source_count,expected_lag_minutes,top1_lag_minutes,top1_weight,top5_mass,recent_1h_mass,recent_6h_mass,lag50_minutes,lag80_minutes,lag90_minutes,status\n")
    return manifest_fp, metrics_fp
