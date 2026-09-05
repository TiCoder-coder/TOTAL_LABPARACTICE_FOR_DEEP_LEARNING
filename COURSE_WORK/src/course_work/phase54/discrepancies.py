"""Phase 54 — discrepancy log writer."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .writers import write_json_atomic


def write_discrepancies(out_dir: Path) -> Path:
    """Write canonical discrepancies JSON. Only resolved defects may be
    recorded; the file is initialized empty (no defects)."""
    payload = {
        "phase": 54,
        "version": "LAST_QUERY_ATTENTION-v1",
        "items": [],
        "resolved": [],
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "schema": [
            "PHASE52_NOT_APPROVED",
            "PHASE54_HANDOFF_NOT_READY",
            "RAW_LAST_QUERY_FILE_MISSING",
            "RAW_SHA_MISMATCH",
            "RAW_SHAPE_MISMATCH",
            "RAW_DTYPE_MISMATCH",
            "TARGET_ORDER_MISMATCH",
            "TARGET_DUPLICATE",
            "TARGET_MISSING",
            "LAYER_COUNT_MISMATCH",
            "HEAD_COUNT_MISMATCH",
            "LOOKBACK_MISMATCH",
            "POSITION_MAP_MISMATCH",
            "LAST_QUERY_AXIS_MISMATCH",
            "RAW_VECTOR_NONFINITE",
            "RAW_VECTOR_NEGATIVE",
            "RAW_VECTOR_SUM_MISMATCH",
            "ATTENTION_RENORMALIZED",
            "PHASE52_SUMMARY_RECONSTRUCTION_MISMATCH",
            "LAG_DIRECTION_REVERSED",
            "LAG1_MAPPING_ERROR",
            "RECENCY_REINDEX_ERROR",
            "EXPECTED_LAG_FORMULA_ERROR",
            "ENTROPY_FORMULA_DRIFT",
            "EFFECTIVE_SOURCE_COUNT_FORMULA_ERROR",
            "TOP1_TIE_RULE_DRIFT",
            "RECENT_WINDOW_DEFINITION_DRIFT",
            "TRUNCATED_WINDOW_MISLABELED",
            "LAG50_DEFINITION_ERROR",
            "LAG80_DEFINITION_ERROR",
            "LAG90_DEFINITION_ERROR",
            "LAG_BIN_OVERLAP",
            "LAG_BIN_GAP",
            "LAG_BIN_MASS_SUM_MISMATCH",
            "AVERAGE_PROFILE_SUM_MISMATCH",
            "HEAD_METRICS_SORTED_AS_RANKING",
            "BEST_HEAD_SELECTED",
            "HEAD_CLUSTERING_SCOPE_CREEP",
            "ERROR_CONDITIONING_SCOPE_CREEP",
            "SEED_STABILITY_SCOPE_CREEP",
            "HEAD_ABLATION_ATTEMPT",
            "NEW_ATTENTION_EXTRACTION_ATTEMPT",
            "NEW_TEST_INFERENCE_ATTEMPT",
            "HEATMAP_IMAGE_USED_AS_NUMERIC_SOURCE",
            "RAW_FEATURE_IMPORTANCE_CLAIM",
            "CAUSAL_ATTRIBUTION_CLAIM",
            "SAME_INDEX_HEAD_SEMANTIC_EQUIVALENCE_ASSUMED",
            "REPORT_CASE_CHANGED",
            "OTHER",
        ],
    }
    fp = out_dir / "last_query_attention_discrepancies.json"
    write_json_atomic(fp, payload)
    return fp
