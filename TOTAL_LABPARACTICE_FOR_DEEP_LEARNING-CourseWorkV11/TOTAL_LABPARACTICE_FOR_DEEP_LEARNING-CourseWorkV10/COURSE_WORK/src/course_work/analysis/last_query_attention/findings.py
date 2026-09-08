"""Phase 54 — findings writer (descriptive only, no causal/feature importance
claims, no head ranking)."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .writers import write_csv_atomic


# Canonical finding codes (Phase 54 detail #144)
FINDING_CODES = [
    "LAST_QUERY_SOURCE_VERIFIED",
    "LAST_QUERY_SUMS_NORMALIZED",
    "PHASE52_SUMMARY_RECONSTRUCTED",
    "RECENT_HISTORY_MASS_DOMINANT",
    "LONGER_HISTORY_MASS_SUBSTANTIAL",
    "LAST_QUERY_ATTENTION_DIFFUSE",
    "LAST_QUERY_ATTENTION_CONCENTRATED",
    "EXPECTED_LAG_RECENT",
    "EXPECTED_LAG_LONGER_HORIZON",
    "TOP1_LAG_CLUSTERED_AT_RECENT_POSITIONS",
    "TOP1_LAG_DISTRIBUTED_ACROSS_HISTORY",
    "LAYER_TEMPORAL_PROFILES_DIFFER_DESCRIPTIVELY",
    "HEAD_TEMPORAL_PROFILES_DIVERSE",
    "HEAD_MEAN_PROFILE_RECENCY_DOMINANT",
    "COVERAGE_RADIUS_SMALL",
    "COVERAGE_RADIUS_LARGE",
    "TOP1_TIES_RARE",
    "TOP1_TIES_PRESENT",
    "LOOKBACK_TRUNCATES_12H_WINDOW",
    "LOOKBACK_TRUNCATES_24H_WINDOW",
    "REPORT_CASE_VIEWS_COMPLETE",
    "NO_HEAD_SELECTION",
    "NO_ERROR_CONDITIONING",
    "NO_SEED_STABILITY_CLAIM",
    "ATTENTION_TEMPORAL_NOT_FEATURE_IMPORTANCE",
    "READY_FOR_HEAD_COMPARISON",
]


def write_findings(out_dir: Path, summary: dict) -> Path:
    """Write last_query_analysis_findings.csv with descriptive findings."""
    fields = ["finding_id", "title", "scope", "status", "note"]
    rows: list[dict] = []
    for i, code in enumerate(FINDING_CODES, start=1):
        rows.append({
            "finding_id": f"F54.{i:02d}",
            "title": code,
            "scope": "last_query_attention",
            "status": "DESCRIPTIVE",
            "note": summary.get(code, "see human-readable report"),
        })
    fp = out_dir / "last_query_analysis_findings.csv"
    write_csv_atomic(fp, rows, fields)
    return fp
