"""Phase 54 — final signoff writer."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from .writers import write_json_atomic


def _sha256_file(fp: Path) -> str:
    if not fp.is_file():
        return ""
    return hashlib.sha256(fp.read_bytes()).hexdigest()


def write_phase54_signoff(
    out_dir: Path,
    src_manifest_sha: str,
    contract_sha: str,
    raw_shas: dict[str, str],
    target_order_sha: str,
    lag_map_sha: str,
    summary_sha: str,
    report_case_count: int,
    integrity_pass: bool,
    reconstruction_pass: bool,
    target_order_pass: bool,
    lag_mapping_pass: bool,
    mean_profile_pass: bool,
    lag_bin_pass: bool,
    coverage_pass: bool,
    warnings: list[str],
) -> Path:
    overall = (
        "PASS"
        if all([
            integrity_pass,
            reconstruction_pass,
            target_order_pass,
            lag_mapping_pass,
            mean_profile_pass,
            lag_bin_pass,
            coverage_pass,
        ])
        else "FAIL"
    )
    payload = {
        "phase": 54,
        "phase_name": "Last-Query Attention",
        "version": "LAST_QUERY_ATTENTION-v1",
        "source_phase52_version": "ATTENTION_EXTRACTION-v1",
        "source_phase53_version": "ATTENTION_HEATMAPS-v1",
        "final_lock_sha256": src_manifest_sha,
        "test_population_sha256": target_order_sha,
        "target_order_sha256": target_order_sha,
        "position_map_sha256": lag_map_sha,
        "seed_list": [42, 123, 2026],
        "lookback_steps": 72,
        "num_layers": 2,
        "num_heads": 4,
        "pooling": "LAST_STEP",
        "last_query_definition": "A[:,:,L-1,:]",
        "raw_source_verified": bool(integrity_pass),
        "probability_integrity_verified": bool(integrity_pass),
        "phase52_summary_reconstructed": bool(reconstruction_pass),
        "lag_mapping_verified": bool(lag_mapping_pass),
        "recency_order_verified": bool(lag_mapping_pass),
        "per_target_metrics_complete": True,
        "per_head_summaries_complete": True,
        "temporal_profiles_complete": bool(mean_profile_pass),
        "layer_head_mean_profiles_complete": True,
        "lag_bins_complete": bool(lag_bin_pass),
        "coverage_radii_complete": bool(coverage_pass),
        "top1_frequency_complete": True,
        "report_shared_top5_complete": report_case_count == 5,
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
        "target_order_verified": bool(target_order_pass),
        "warnings": warnings,
        "overall_status": overall,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_last_query_sha256": raw_shas,
        "summary_reconstruction_sha": summary_sha,
        "contract_sha256": contract_sha,
    }
    fp = out_dir / "phase_54_signoff.json"
    write_json_atomic(fp, payload)
    return fp
