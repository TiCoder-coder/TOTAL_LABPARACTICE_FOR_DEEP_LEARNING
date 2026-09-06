"""Phase 54 — downstream handoff writers.

Writes Phase 55 head-comparison handoff, Phase 56 error-conditioned context
handoff, and Phase 57 seed-stability context handoff.

These handoff files pin the numeric source to Phase 52 raw last-query NPZ
(NOT PNG), preserve no head ranking, no error conditioning, no head matching.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .writers import write_json_atomic


def _sha256_file(fp: Path) -> str:
    return hashlib.sha256(fp.read_bytes()).hexdigest()


def write_phase55_handoff(
    out_dir: Path,
    raw: dict[int, Path],
    metrics_long_path: Path,
    head_summary_path: Path,
    profile_path: Path,
    lag_bin_path: Path,
    recent_mass_path: Path,
    coverage_path: Path,
    top1_freq_path: Path,
    phase52_signoff_path: Path,
    contract_path: Path,
    n_metrics_rows: int,
) -> Path:
    """Phase 55 head-comparison handoff. Provides standardized head metrics and
    profiles; no head ranking, no head matching, no causal claim."""
    raw_shas = {f"last_query_attention_seed{seed}.npz": _sha256_file(fp) for seed, fp in raw.items()}
    phase52_sha = _sha256_file(phase52_signoff_path)
    contract_sha = _sha256_file(contract_path)

    payload = {
        "phase": 54,
        "downstream_phase": 55,
        "downstream_name": "Head Comparison",
        "source_phase54_version": "LAST_QUERY_ATTENTION-v1",
        "source_phase52_version": "ATTENTION_EXTRACTION-v1",
        "final_lock_sha256": phase52_sha,
        "test_population_sha256": _sha256_file(metrics_long_path),
        "seed_list": [42, 123, 2026],
        "lookback": 72,
        "layers": 2,
        "heads": 4,
        "last_query_metrics_long": str(metrics_long_path),
        "metric_summary_by_head": str(head_summary_path),
        "profile_by_lag": str(profile_path),
        "lag_bin_mass": str(lag_bin_path),
        "recent_mass_summary": str(recent_mass_path),
        "coverage_radius_summary": str(coverage_path),
        "top1_lag_frequency": str(top1_freq_path),
        "head_order": "ARCHITECTURAL_INDEX",
        "head_semantic_alignment_across_seeds": False,
        "head_ranking_performed": False,
        "ready_for_phase55": True,
        "phase55_authorized": False,
        "n_metrics_rows": n_metrics_rows,
        "raw_last_query_sha256": raw_shas,
        "contract_sha256": contract_sha,
    }
    fp = out_dir / "phase55_head_comparison_handoff.json"
    write_json_atomic(fp, payload)
    return fp


def write_phase56_context_handoff(
    out_dir: Path,
    metrics_long_path: Path,
    profile_path: Path,
    raw: dict[int, Path],
    phase52_signoff_path: Path,
) -> Path:
    """Phase 56 error-conditioned attention context handoff. Numeric source
    remains Phase 52 raw last-query NPZ. Phase 54 performs NO error
    conditioning."""
    raw_shas = {f"last_query_attention_seed{seed}.npz": _sha256_file(fp) for seed, fp in raw.items()}
    payload = {
        "phase": 54,
        "downstream_phase": 56,
        "downstream_name": "Error-Conditioned Attention",
        "source_phase54_version": "LAST_QUERY_ATTENTION-v1",
        "last_query_metrics_long": str(metrics_long_path),
        "profile_by_lag": str(profile_path),
        "raw_last_query_refs": "PHASE52_RAW_LAST_QUERY_NPZ",
        "raw_last_query_sha256": raw_shas,
        "phase49_residual_refs": "artifacts/residual_analysis/residual_long_table.csv",
        "phase50_regime_refs": "artifacts/error_by_regime/test_regime_assignment.csv",
        "phase51_worst_case_refs": "artifacts/worst_error_analysis/worst_error_analysis_case_selection.csv",
        "error_conditioning_performed_in_phase54": False,
        "ready_for_phase56_context": True,
        "phase56_authorized": False,
        "phase52_signoff_sha256": _sha256_file(phase52_signoff_path),
    }
    fp = out_dir / "phase56_error_conditioned_attention_context_handoff.json"
    write_json_atomic(fp, payload)
    return fp


def write_phase57_context_handoff(
    out_dir: Path,
    raw: dict[int, Path],
    metrics_long_path: Path,
    head_summary_path: Path,
    profile_path: Path,
    phase52_signoff_path: Path,
) -> Path:
    """Phase 57 seed-stability attention context handoff. NO head matching in
    Phase 54. Same-index heads across seeds are NOT assumed semantically
    equivalent."""
    raw_shas = {f"last_query_attention_seed{seed}.npz": _sha256_file(fp) for seed, fp in raw.items()}
    payload = {
        "phase": 54,
        "downstream_phase": 57,
        "downstream_name": "Seed-Stability Attention",
        "source_phase54_version": "LAST_QUERY_ATTENTION-v1",
        "per_seed_head_profiles": str(profile_path),
        "per_seed_head_metrics": str(head_summary_path),
        "raw_last_query_refs": "PHASE52_RAW_LAST_QUERY_NPZ",
        "raw_last_query_sha256": raw_shas,
        "same_target_order": True,
        "same_layers_heads": True,
        "same_index_semantic_alignment_not_guaranteed": True,
        "head_matching_not_performed_in_phase54": True,
        "seed_stability_inference_performed": False,
        "ready_for_phase57_context": True,
        "phase57_authorized": False,
        "phase52_signoff_sha256": _sha256_file(phase52_signoff_path),
        "metrics_long_ref": str(metrics_long_path),
    }
    fp = out_dir / "phase57_seed_stability_attention_context_handoff.json"
    write_json_atomic(fp, payload)
    return fp
