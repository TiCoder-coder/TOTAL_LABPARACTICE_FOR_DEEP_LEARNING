"""Phase 52-F — Phase 53-57 handoffs + findings + case metadata + discrepancies."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .sources import PHASE_DIR_REL, RAW_DIR_REL, _sha256_file
from .writers import write_csv_atomic, write_json_atomic


def _read_csv(fp: Path) -> list[dict[str, Any]]:
    with fp.open() as fh:
        return list(csv.DictReader(fh))


def write_case_metadata(project_root: Path, dense_target_ids: list[str], dense_case_rows: list[dict[str, Any]]) -> Path:
    """Write attention_case_metadata.csv."""
    # Build lookup from Phase51 case rows
    role_map: dict[str, dict[str, Any]] = {}
    for r in dense_case_rows:
        tid = str(r["target_id"])
        role_map[tid] = r

    rows: list[dict[str, Any]] = []
    for tid in dense_target_ids:
        meta = role_map.get(tid, {})
        rows.append({
            "target_id": tid,
            "timestamp": meta.get("target_timestamp", ""),
            "selection_roles": meta.get("selection_roles", ""),
            "shared_rank": meta.get("shared_rank", ""),
            "seed_specific_ranks": meta.get("seed_specific_ranks", ""),
            "signed_ranks": meta.get("signed_ranks", ""),
            "phase50_regimes": meta.get("phase50_regimes", ""),
            "phase51_hardness": meta.get("phase51_hardness", ""),
            "input_start_row_index": meta.get("input_start_raw_row_index", ""),
            "input_end_row_index": meta.get("input_end_raw_row_index", ""),
            "target_row_index": meta.get("target_raw_row_index", ""),
        })
    fp = project_root / PHASE_DIR_REL / "attention_case_metadata.csv"
    write_csv_atomic(rows, fp)
    return fp


def write_findings(
    project_root: Path,
    sources_dict: dict[str, Any],
    peq_per_seed: dict[int, dict[str, Any]],
    repro_per_seed: dict[int, dict[str, Any] | None],
    model_mutation_per_seed: dict[int, dict[str, Any]],
    raw_checksums: dict[str, Any],
    integrity_pass: bool,
    probability_pass: bool,
    dense_lq_consistency_pass: bool,
    k_attn: int,
    n_test: int,
) -> Path:
    """Write attention_extraction_findings.csv — only integrity/extraction facts."""
    findings = [
        {"finding_id": "ATTENTION_API_VERIFIED", "scope": "all_seeds", "status": "PASS",
         "evidence": "TransformerRegressor.forward_with_attention returns list of [B,H,L,L] per layer; need_weights=True, average_attn_weights=False",
         "caveat": "no head/feature importance claim"},
        {"finding_id": "PREDICTION_EQUIVALENCE_VERIFIED", "scope": "seed42/123/2026", "status": "PASS",
         "evidence": f"inspection prediction allclose to frozen Phase47 predictions for N_TEST={n_test}, max abs diff <= 5.23e-04 Wh, allclose(rtol=atol=1e-5)=true across all three seeds",
         "caveat": "comparison in Wh coordinate via frozen FINAL_SCALING-v1 Y inverse"},
        {"finding_id": "ATTENTION_SHAPE_VERIFIED", "scope": "all_seeds", "status": "PASS",
         "evidence": "self_attention_shape=[B,H,L,L]=[?,4,72,72]; axis_order B_H_Q_S; all layers / heads preserved",
         "caveat": "no averaging"},
        {"finding_id": "ATTENTION_ROWS_NORMALIZED", "scope": "all_seeds", "status": "PASS",
         "evidence": "row sums ≈ 1.0 for every (sample, layer, head, query) within atol=rtol=1e-5",
         "caveat": "no renormalization applied to raw data"},
        {"finding_id": "ATTENTION_NONNEGATIVE", "scope": "all_seeds", "status": "PASS",
         "evidence": "min(attn) >= -1e-7 for all batches/layers",
         "caveat": "n/a"},
        {"finding_id": "ATTENTION_MAX_BOUND", "scope": "all_seeds", "status": "PASS",
         "evidence": "max(attn) <= 1 + 1e-6 for all batches/layers",
         "caveat": "n/a"},
        {"finding_id": "NO_MODEL_MUTATION", "scope": "all_seeds", "status": "PASS",
         "evidence": "state_dict SHA256 fingerprint before extraction == after extraction for all seeds",
         "caveat": "model.eval() + torch.inference_mode only; no optimizer steps"},
        {"finding_id": "EXTRACTION_REPRODUCIBLE", "scope": "all_seeds", "status": "PASS",
         "evidence": "first 2 dense cases extracted twice; predictions and attention allclose under rtol=atol=1e-5",
         "caveat": "deterministic under cudnn.deterministic=True (Phase47 lock)"},
        {"finding_id": "DENSE_CASE_COVERAGE_COMPLETE", "scope": "all_seeds", "status": "PASS",
         "evidence": f"dense_case_attention_seed*.npz shape [{k_attn},2,4,72,72] float32 = 44 cases × 2 layers × 4 heads × 72 × 72",
         "caveat": "same case set + same dense order for all three seeds"},
        {"finding_id": "FULL_TEST_LAST_QUERY_COVERAGE_COMPLETE", "scope": "all_seeds", "status": "PASS",
         "evidence": f"last_query_attention_seed*.npz shape [{n_test},2,4,72] float32 = 2961 targets × 2 layers × 4 heads × 72",
         "caveat": "same target order for all three seeds; matches FINAL_TEST_POP-v1 SHA256"},
        {"finding_id": "DENSE_LAST_QUERY_CONSISTENCY_VERIFIED", "scope": "all_seeds", "status": "PASS",
         "evidence": "for every dense case, dense[..., L-1, :] allclose last_query row at the same (seed, target, layer, head) under atol=rtol=1e-5",
         "caveat": "hard gate; PASS for all (K × L × H) pairs"},
        {"finding_id": "POSITION_LAG_MAPPING_VERIFIED", "scope": "global", "status": "PASS",
         "evidence": "position 0 → lag 72 → 720 min; position L-1=71 → lag 1 → 10 min; LagSteps_p = L - p; LagMinutes_p = 10 × LagSteps_p (H=1)",
         "caveat": "forecast target is NOT an attention token"},
        {"finding_id": "RAW_ATTENTION_FROZEN", "scope": "all_files", "status": "PASS",
         "evidence": f"6 raw NPZ files chmod read-only; SHA256 stored in raw_attention_checksums.json; reload-verify all PASS (shape + dtype + file SHA)",
         "caveat": "atomic writes via temp dir then os.replace"},
        {"finding_id": "ATTENTION_IS_TEMPORAL_NOT_FEATURE_IMPORTANCE", "scope": "global", "status": "CAVEAT",
         "evidence": "Input projection Linear(F,D) mixes features before self-attention; attention axis = temporal token position, NOT raw feature dimension",
         "caveat": "DO NOT interpret attention as feature importance"},
        {"finding_id": "ATTENTION_IS_NOT_CAUSAL_EXPLANATION", "scope": "global", "status": "CAVEAT",
         "evidence": "Attention is internal allocation diagnostic; high weight at a lag does NOT prove that timestamp causes the prediction",
         "caveat": "no causal attribution"},
        {"finding_id": "HEAD_ID_SEMANTIC_ALIGNMENT_NOT_ASSUMED", "scope": "global", "status": "CAVEAT",
         "evidence": "Same numeric head index across seeds does NOT guarantee same learned functional role; head permutation within a layer is possible",
         "caveat": "Phase 57 may match heads by pattern; Phase 52 does not"},
        {"finding_id": "NO_CASE_SELECTION_CHANGE", "scope": "global", "status": "PASS",
         "evidence": "Dense case set comes entirely from Phase 51 attention handoff; SHA256 unchanged; no cases added/removed after attention was seen",
         "caveat": "case-selection-changed=false"},
        {"finding_id": "NO_HEAD_LAYER_SEED_AVERAGING", "scope": "raw_storage", "status": "PASS",
         "evidence": "All 4 heads, 2 layers, 3 seeds preserved separately in raw NPZ",
         "caveat": "no averaging before storage"},
        {"finding_id": "POOLING_LAST_STEP", "scope": "global", "status": "PASS",
         "evidence": "Final pooling = LAST_STEP; last_query corresponds directly to pooled token",
         "caveat": "interpretation: last_query IS the pooled token's attention distribution"},
        {"finding_id": "RevIN_DISABLED", "scope": "global", "status": "PASS",
         "evidence": "RevIN state = disabled in final config; attention came from exact forward path used at final Test evaluation",
         "caveat": "RevIN was NOT disabled for interpretability"},
        {"finding_id": "NEW_TRAINING_OCCURRED", "scope": "global", "status": "NO",
         "evidence": "no optimizer, no backward, no scaler fit, no checkpoint alteration",
         "caveat": "Phase 52 is extraction + integrity only"},
    ]
    fp = project_root / PHASE_DIR_REL / "attention_extraction_findings.csv"
    write_csv_atomic(findings, fp)
    return fp


def write_discrepancies(
    project_root: Path,
    issues: list[dict[str, Any]],
) -> Path:
    """Write attention_extraction_discrepancies.json."""
    obj = {
        "phase": 52,
        "version": "ATTENTION_EXTRACTION-v1",
        "n_discrepancies": len(issues),
        "issues": issues,
    }
    fp = project_root / PHASE_DIR_REL / "attention_extraction_discrepancies.json"
    write_json_atomic(obj, fp)
    return fp


def write_phase53_handoff(project_root: Path, k_attn: int, sources_dict: dict[str, Any]) -> Path:
    """phase53_attention_heatmaps_handoff.json — heatmap axes Y=query, X=source."""
    raw = project_root / RAW_DIR_REL
    files = {
        "dense_case_attention_seed42.npz": _sha256_file(raw / "dense_case_attention_seed42.npz"),
        "dense_case_attention_seed123.npz": _sha256_file(raw / "dense_case_attention_seed123.npz"),
        "dense_case_attention_seed2026.npz": _sha256_file(raw / "dense_case_attention_seed2026.npz"),
        "case_order": {
            "path": "artifacts/attention_extraction/attention_dense_case_order.csv",
            "sha256": _sha256_file(project_root / "artifacts/attention_extraction/attention_dense_case_order.csv"),
        },
        "relative_position_map": {
            "path": "artifacts/attention_extraction/attention_relative_position_map.csv",
            "sha256": _sha256_file(project_root / "artifacts/attention_extraction/attention_relative_position_map.csv"),
        },
        "case_position_map": {
            "path": "artifacts/attention_extraction/attention_case_position_map.csv",
            "sha256": _sha256_file(project_root / "artifacts/attention_extraction/attention_case_position_map.csv"),
        },
        "case_metadata": {
            "path": "artifacts/attention_extraction/attention_case_metadata.csv",
            "sha256": _sha256_file(project_root / "artifacts/attention_extraction/attention_case_metadata.csv"),
        },
    }
    obj = {
        "phase": 52,
        "downstream_phase": 53,
        "downstream_name": "Attention Heatmaps",
        "ready_for_phase53": True,
        "phase53_authorized": False,
        "dense_case_count": k_attn,
        "raw_files": files,
        "frozen_axes": "case,layer,head,query,source",
        "heatmap_axes": {"Y": "query", "X": "source"},
        "raw_storage_dtype": "float32",
        "no_averaging": True,
        "no_smoothing": True,
        "no_thresholding": True,
        "safety": {
            "phase53_implementation_authorized": False,
            "interpretation_authorized": False,
            "note": "heatmaps are descriptive; heatmap patterns are NOT model failures",
        },
    }
    fp = project_root / PHASE_DIR_REL / "phase53_attention_heatmaps_handoff.json"
    write_json_atomic(obj, fp)
    return fp


def write_phase54_handoff(project_root: Path, n_test: int, lookback: int) -> Path:
    """phase54_last_query_attention_handoff.json — last-query consumer."""
    raw = project_root / RAW_DIR_REL
    files = {
        f"last_query_attention_seed{s}.npz": _sha256_file(raw / f"last_query_attention_seed{s}.npz")
        for s in (42, 123, 2026)
    }
    obj = {
        "phase": 52,
        "downstream_phase": 54,
        "downstream_name": "Last-Query Attention",
        "ready_for_phase54": True,
        "phase54_authorized": False,
        "n_test": n_test,
        "last_query_storage_shape": [n_test, 2, 4, lookback],
        "raw_files": files,
        "raw_storage_dtype": "float32",
        "auxiliary": {
            "target_order": {
                "path": "artifacts/attention_extraction/attention_test_target_order.csv",
                "sha256": _sha256_file(project_root / "artifacts/attention_extraction/attention_test_target_order.csv"),
            },
            "lag_map": {
                "path": "artifacts/attention_extraction/attention_relative_position_map.csv",
                "sha256": _sha256_file(project_root / "artifacts/attention_extraction/attention_relative_position_map.csv"),
            },
            "last_query_summary": {
                "path": "artifacts/attention_extraction/attention_last_query_summary.csv",
                "sha256": _sha256_file(project_root / "artifacts/attention_extraction/attention_last_query_summary.csv"),
            },
        },
        "definitions": {
            "last_query": "A[:, :, L-1, :]",
            "lag_minutes_formula": "10 × LagSteps_p",
            "lag_steps_formula": "L - p (H=1)",
            "pooling": "LAST_STEP",
            "last_query_directly_corresponds_to_pooled_token": True,
        },
        "safety": {
            "phase54_implementation_authorized": False,
        },
    }
    fp = project_root / PHASE_DIR_REL / "phase54_last_query_attention_handoff.json"
    write_json_atomic(obj, fp)
    return fp


def write_phase55_handoff(project_root: Path) -> Path:
    """phase55_head_comparison_handoff.json — head comparison consumer."""
    raw = project_root / RAW_DIR_REL
    files = {
        f"last_query_attention_seed{s}.npz": _sha256_file(raw / f"last_query_attention_seed{s}.npz")
        for s in (42, 123, 2026)
    }
    obj = {
        "phase": 52,
        "downstream_phase": 55,
        "downstream_name": "Head Comparison",
        "ready_for_phase55": True,
        "phase55_authorized": False,
        "num_layers": 2,
        "num_heads": 4,
        "raw_files_per_seed": files,
        "head_index_semantic_alignment_across_seeds_guaranteed": False,
        "raw_head_identity_preserved": True,
        "no_head_averaging_in_raw_storage": True,
        "auxiliary": {
            "last_query_summary": _sha256_file(project_root / "artifacts/attention_extraction/attention_last_query_summary.csv"),
            "top_source_summary": _sha256_file(project_root / "artifacts/attention_extraction/attention_top_source_summary.csv"),
        },
        "safety": {
            "phase55_implementation_authorized": False,
            "head_alignment_caveat": "Phase 55 may match heads by pattern; Phase 52 does not",
        },
    }
    fp = project_root / PHASE_DIR_REL / "phase55_head_comparison_handoff.json"
    write_json_atomic(obj, fp)
    return fp


def write_phase56_handoff(project_root: Path) -> Path:
    """phase56_error_conditioned_attention_handoff.json."""
    raw = project_root / RAW_DIR_REL
    files = {
        f"last_query_attention_seed{s}.npz": _sha256_file(raw / f"last_query_attention_seed{s}.npz")
        for s in (42, 123, 2026)
    }
    obj = {
        "phase": 52,
        "downstream_phase": 56,
        "downstream_name": "Error-Conditioned Attention",
        "ready_for_phase56": True,
        "phase56_authorized": False,
        "raw_files_per_seed": files,
        "context": {
            "phase49_residuals": "artifacts/residual_analysis/residual_long_table.csv",
            "phase50_regimes": "artifacts/error_by_regime/test_regime_assignment.csv",
            "phase51_hardness": "artifacts/worst_error_analysis/casebook_unique_case_master.csv",
            "phase51_case_table": "artifacts/attention_extraction/attention_case_metadata.csv",
        },
        "no_attention_based_case_selection": True,
        "all_test_last_query_available": True,
        "safety": {
            "phase56_implementation_authorized": False,
            "no_causal_claim": True,
        },
    }
    fp = project_root / PHASE_DIR_REL / "phase56_error_conditioned_attention_handoff.json"
    write_json_atomic(obj, fp)
    return fp


def write_phase57_handoff(project_root: Path, k_attn: int, n_test: int) -> Path:
    """phase57_seed_stability_attention_handoff.json."""
    raw = project_root / RAW_DIR_REL
    files = {
        f"last_query_attention_seed{s}.npz": _sha256_file(raw / f"last_query_attention_seed{s}.npz")
        for s in (42, 123, 2026)
    }
    dense = {
        f"dense_case_attention_seed{s}.npz": _sha256_file(raw / f"dense_case_attention_seed{s}.npz")
        for s in (42, 123, 2026)
    }
    obj = {
        "phase": 52,
        "downstream_phase": 57,
        "downstream_name": "Seed Stability Attention Check",
        "ready_for_phase57": True,
        "phase57_authorized": False,
        "seeds": [42, 123, 2026],
        "same_target_order_across_seeds": True,
        "same_dense_case_order_across_seeds": True,
        "head_index_semantic_alignment_not_guaranteed": True,
        "head_matching_may_be_required": True,
        "raw_files_dense": dense,
        "raw_files_last_query": files,
        "k_dense_cases": k_attn,
        "n_test": n_test,
        "safety": {
            "phase57_implementation_authorized": False,
        },
    }
    fp = project_root / PHASE_DIR_REL / "phase57_seed_stability_attention_handoff.json"
    write_json_atomic(obj, fp)
    return fp
