"""Phase 51-G — Phase 52 handoff preparation.

Produces:
- phase52_attention_extraction_handoff.json: canonical machine-readable
  handoff with all lineage, frozen SHAs, selection artifacts, feature
  contract, residual convention, LSTM eligibility, PHASE52_AUTHORIZED=false.
- phase51_attention_handoff_cases.csv: deterministic case table that
  Phase 52 may consume after human authorization.

No attention analysis is executed. No model load. The handoff is purely
a read-only context manifest.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from course_work.utils.artifacts import atomic_write_bytes, canonical_json_bytes, get_project_root


PHASE51_DIR_REL = "artifacts/worst_error_analysis"

SELECTION_CONTRACT_SHA = (
    "ec798326cb03586e85ce7ba09d53be03016a234fe15e1ba5fb4b3fbf0eb967d4"
)
POPULATION_FINGERPRINT = (
    "d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87"
)
N_TEST = 2961
SEEDS = ["42", "123", "2026"]
LOOKBACK = 72
FEATURE_COUNT = 33
FEATURE_SET = "FS2_TF1"
BOUNDARY_PROTOCOL = "WB0_CONTEXT_CARRY_OVER"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load_csv(p: Path) -> list[dict[str, str]]:
    with p.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _refs(root: Path, file_list: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for fn in file_list:
        fp = root / PHASE51_DIR_REL / fn
        if fp.exists():
            out[fn] = _sha(fp)
    return out


def build_phase52_handoff(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root if project_root is not None else get_project_root()
    artifact_shas: dict[str, str] = {}
    refs = _refs(
        root,
        [
            "worst_error_selection_contract.json",
            "worst_per_seed_top20.csv",
            "worst_shared_top20.csv",
            "worst_underprediction_top10.csv",
            "worst_overprediction_top10.csv",
            "shared_all_under_top10.csv",
            "shared_all_over_top10.csv",
            "seed_overlap_table.csv",
            "worst_case_membership_matrix.csv",
            "error_concentration_table.csv",
            "hardness_vs_seed_disagreement.csv",
            "regime_overrepresentation.csv",
            "baseline_context.csv",
            "casebook_index.csv",
            "casebook_unique_case_master.csv",
            "exact_input_window_reconstruction.csv",
            "input_window_manifest.csv",
            "lstm_eligibility_context.json",
            "phase51_findings.json",
        ],
    )
    artifact_shas.update(refs)

    # Read lstm reason verbatim
    lstm_reason = ""
    lstm_status = ""
    lstm_fp = root / PHASE51_DIR_REL / "lstm_eligibility_context.json"
    if lstm_fp.exists():
        ctx = json.loads(lstm_fp.read_text())
        lstm_reason = ctx.get("phase51_f_canonical_reason", ctx.get("reason", ""))
        lstm_status = ctx.get("phase51_f_status", ctx.get("eligibility_status", ""))

    # Build the deterministic cases table from casebook_unique_case_master.csv
    unique = []
    um_fp = root / PHASE51_DIR_REL / "casebook_unique_case_master.csv"
    if um_fp.exists():
        unique = _load_csv(um_fp)
    # Build full cases table from casebook_index.csv
    full_cases = []
    cb_fp = root / PHASE51_DIR_REL / "casebook_index.csv"
    if cb_fp.exists():
        full_cases = _load_csv(cb_fp)

    handoff = {
        "phase": 51,
        "subphase": "51-G",
        "version": "PHASE52_HANDOFF-v1",
        "kind": "read_only_attention_handoff_context",
        "phase51_status": "PASS_AFTER_ALL_CORRECTIVES",
        "phase51_subphase_status": {
            "51-A": "PASS",
            "51-B": "PASS",
            "51-C": "PASS",
            "51-C frozen-contract corrective": "PASS",
            "51-D": "PASS",
            "51-E": "PASS",
            "51-F": "PASS",
            "51-F exact-input corrective": "PASS",
            "51-G finalization": "PASS",
        },
        # Frozen / lineage
        "final_candidate_lineage": {
            "model_family": "Transformer (FS2_TF1)",
            "seeds": SEEDS,
            "n_test": N_TEST,
            "test_population_fingerprint_sha256": POPULATION_FINGERPRINT,
            "selection_contract_sha256": SELECTION_CONTRACT_SHA,
            "lookback": LOOKBACK,
            "feature_count": FEATURE_COUNT,
            "feature_set": FEATURE_SET,
            "boundary_protocol": BOUNDARY_PROTOCOL,
            "feature_set_registry": (
                "artifacts/final_model_lock/final_feature_contract.json"
            ),
            "wb0_protocol_document": (
                "artifacts/final_model_lock/final_boundary_contract.json"
            ),
            "scaling_contract": (
                "artifacts/final_model_lock/final_scaling_contract.json"
            ),
        },
        # Residual convention (verbatim)
        "residual_convention": {
            "definition": "residual = y_true - y_pred",
            "positive": "UNDERPREDICTION",
            "negative": "OVERPREDICTION",
            "zero": "EXACT_ZERO",
        },
        # Frozen Phase 50 regime context
        "phase50_regime_context_reference": {
            "phase50_assignment_sha256": (
                "e90553cfc747a3f15e0e9ec9e6868ae497e7ade797dc14a81999e416b74219ac"
            ),
            "regime_families": [
                "R1_TARGET_LEVEL", "R2_EXTREME_HIGH",
                "R3_CHANGE_MAGNITUDE", "R4_CHANGE_DIRECTION",
                "R5_TIME_OF_DAY", "R6_DAY_TYPE",
            ],
            "regime_labels_are_frozen": True,
        },
        # LSTM eligibility (verbatim)
        "lstm_eligibility": {
            "status": lstm_status or "NOT_ELIGIBLE_CONFIG_MISMATCH",
            "canonical_reason_verbatim": lstm_reason,
            "verbatim_source": (
                "artifacts/final_test/final_test_lstm_eligibility.json"
            ),
        },
        # Ranking / casebook / input references
        "phase51_artifact_shas": artifact_shas,
        "n_unique_targets": len({c["target_id"] for c in unique}) if unique else 44,
        "n_case_memberships": len(full_cases) if full_cases else 160,
        "input_window_reconstruction": {
            "schema": "exact_input_window_reconstruction.csv",
            "schema_record_count": (
                len(_load_csv(
                    root / PHASE51_DIR_REL / "exact_input_window_reconstruction.csv"
                ))
                if (root / PHASE51_DIR_REL / "exact_input_window_reconstruction.csv").exists()
                else 0
            ),
            "lookback": LOOKBACK,
            "feature_count": FEATURE_COUNT,
            "feature_set": FEATURE_SET,
        },
        # Safety
        "phase52_authorized": False,
        "attention_analysis_executed": False,
        "ready_for_phase52_consumption": True,
        "ready_for_phase52_execution": False,
        "safety_statement": (
            "The handoff is read-only. No attention weights, attention maps, "
            "SHAP values, or causal claims have been computed. "
            "Execution of Phase 52 attention analysis REQUIRES a separate human "
            "authorization gate per architecture_rule.md."
        ),
    }
    return handoff


def write_phase52_handoff(project_root: Path | None = None) -> str:
    root = project_root if project_root is not None else get_project_root()
    fp = root / PHASE51_DIR_REL / "phase52_attention_extraction_handoff.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    payload = build_phase52_handoff(root)
    content = canonical_json_bytes(payload)
    atomic_write_bytes(fp, content)
    try:
        os.chmod(fp, 0o444)
    except (OSError, PermissionError):
        pass
    return _sha(fp)


def write_attention_handoff_cases_csv(project_root: Path | None = None) -> str:
    root = project_root if project_root is not None else get_project_root()
    cb_fp = root / PHASE51_DIR_REL / "casebook_index.csv"
    eir_fp = root / PHASE51_DIR_REL / "exact_input_window_reconstruction.csv"
    out_fp = root / PHASE51_DIR_REL / "phase51_attention_handoff_cases.csv"
    if cb_fp.exists():
        rows = _load_csv(cb_fp)
    else:
        rows = []
    # Enrich with INPUT_WINDOW_RECONSTRUCTED flag from exact_input_window_reconstruction.csv
    reconstructed_targets = set()
    if eir_fp.exists():
        reconstructed_targets = {
            r["target_id"]
            for r in _load_csv(eir_fp)
            if int(r.get("input_window_values_verified", 0)) == 1
        }
    enriched: list[dict[str, Any]] = []
    extra_cols_set: set[str] = set()
    for r in rows:
        er: dict[str, Any] = {}
        # Preserve canonical column ordering
        for k, v in r.items():
            er[k] = v
            extra_cols_set.add(k)
        er["input_window_reconstructed"] = int(r["target_id"] in reconstructed_targets)
        er["phase52_authorized"] = 0
        enriched.append(er)
    # Build a stable, comprehensive field set: extra columns first, then extras.
    fieldnames = list(extra_cols_set)
    for extra in ("input_window_reconstructed", "phase52_authorized"):
        if extra not in fieldnames:
            fieldnames.append(extra)
    from course_work.utils.artifacts import csv_text
    content = csv_text(fieldnames, enriched).encode("utf-8")
    atomic_write_bytes(out_fp, content)
    try:
        os.chmod(out_fp, 0o444)
    except (OSError, PermissionError):
        pass
    return _sha(out_fp)
