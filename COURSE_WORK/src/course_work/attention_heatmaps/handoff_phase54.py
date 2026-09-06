"""Phase 53 — Phase54 context handoff.

`phase54_last_query_attention_context_handoff.json` per canonical schema §156:

* Phase54 numeric source MUST remain Phase52 RAW LAST_QUERY NPZ (NOT PNG)
* Phase53 only supplies visualization context/index
* Phase54 is UNAUTHORIZED in this run
* do_not_extract_last_query_from_png = true
* ready_for_phase54_context = true
* phase54_authorized = false
"""

from __future__ import annotations

import json
from pathlib import Path

from .sources import (
    LOOKBACK,
    NUM_HEADS,
    NUM_LAYERS,
    SEEDS,
)


def write_phase54_handoff(
    output_fp: Path,
    catalog_summary: dict,
    orientation_passed: bool,
    case_scale_count: int,
    report_case_count: int,
    v1_grid_count: int,
    v2_grid_count: int,
    fixed_prob_count: int,
    n_images: int,
    image_shas_present: bool,
    phase53_signoff_pass: bool,
) -> None:
    """Write `phase54_last_query_attention_context_handoff.json`."""
    payload = {
        "phase": 53,
        "downstream_phase": 54,
        "downstream_name": "Last-Query Attention Analysis",
        "context_ready": True,
        "phase53_status": "PASS" if phase53_signoff_pass else "FAIL",
        "phase54_authorized": False,
        "do_not_extract_last_query_from_png": True,
        "phase54_numeric_source": "PHASE52_RAW_LAST_QUERY_NPZ",
        "phase54_numeric_source_detail": (
            "Phase54 last-query numeric analysis MUST be computed from the "
            "frozen Phase52 raw last-query attention NPZ "
            "(`last_query_attention_seed42.npz`, "
            "`last_query_attention_seed123.npz`, "
            "`last_query_attention_seed2026.npz`) — shape [2961, "
            f"{NUM_LAYERS}, {NUM_HEADS}, {LOOKBACK}] per seed. PNG pixels MUST "
            "NOT be used as numerical input, even though the heatmap bottom row "
            "is q=L-1."
        ),
        "context": {
            "phase53_orientation_passed": orientation_passed,
            "phase53_case_scale_count": case_scale_count,
            "phase53_report_case_count": report_case_count,
            "phase53_v1_grid_count": v1_grid_count,
            "phase53_v2_grid_count": v2_grid_count,
            "phase53_fixed_prob_view_count": fixed_prob_count,
            "phase53_n_images": n_images,
            "phase53_image_checksums_present": image_shas_present,
            "phase53_catalog_summary": catalog_summary,
        },
        "phase54_hard_reminders": [
            "Phase54 numeric source = Phase52 RAW LAST_QUERY NPZ (NOT PNG)",
            "do_not_extract_last_query_from_png = true",
            "Phase54 implementation requires separate Human approval",
            "phase54_authorized = false",
        ],
        "phase52_last_query_files": {
            seed: f"artifacts/attention_extraction/raw/last_query_attention_seed{seed}.npz"
            for seed in SEEDS
        },
        "approval_state": "CONTEXT_READY",
    }
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    output_fp.write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit("Phase 53 handoff_phase54 is a library — import it from orchestrator.")
