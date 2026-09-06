"""Phase 51-G — O51.1–O51.38 inventory audit.

Reads the canonical Phase51 plan output inventory and verifies which
artifacts exist in the repository, with schema sanity checks where
applicable. Emits a CSV + JSON inventory + a completeness summary.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from ..utils.artifacts import get_project_root


PHASE51_DIR_REL = "artifacts/worst_error_analysis"

# O51 logical inventory (canonical, derived from plan §114 / §115-145).
O51_INVENTORY: list[dict[str, Any]] = [
    {"id": "O51.1",  "name": "analysis_manifest", "phase": "51-B", "applicability": "required", "filename": "worst_error_analysis_manifest.json"},
    {"id": "O51.2",  "name": "analysis_contract", "phase": "51-B", "applicability": "required", "filename": "worst_error_selection_contract.json"},
    {"id": "O51.3",  "name": "preflight_audit", "phase": "51-A", "applicability": "required", "filename": "phase51_alignment_audit.csv"},
    {"id": "O51.4",  "name": "source_verification", "phase": "51-A", "applicability": "required", "filename": "phase51_alignment_audit.json"},
    {"id": "O51.5",  "name": "frozen_selection_contract", "phase": "51-B", "applicability": "required", "filename": "worst_error_selection_contract.json"},
    {"id": "O51.6",  "name": "selection_contract_fingerprint", "phase": "51-B", "applicability": "required", "filename": "selection_contract_fingerprint.json"},
    {"id": "O51.7",  "name": "alignment_audit", "phase": "51-B", "applicability": "required", "filename": "phase51_alignment_audit.csv"},
    {"id": "O51.8",  "name": "per_seed_top20", "phase": "51-C", "applicability": "required", "filename": "worst_per_seed_top20.csv"},
    {"id": "O51.9",  "name": "shared_hardness_all_test", "phase": "51-C", "applicability": "required", "filename": "hardness_vs_seed_disagreement.csv"},
    {"id": "O51.10", "name": "shared_top20", "phase": "51-C", "applicability": "required", "filename": "worst_shared_top20.csv"},
    {"id": "O51.11", "name": "worst_underprediction", "phase": "51-C", "applicability": "required", "filename": "worst_underprediction_top10.csv"},
    {"id": "O51.12", "name": "worst_overprediction", "phase": "51-C", "applicability": "required", "filename": "worst_overprediction_top10.csv"},
    {"id": "O51.13", "name": "shared_all_under_top10", "phase": "51-C", "applicability": "required", "filename": "shared_all_under_top10.csv"},
    {"id": "O51.14", "name": "shared_all_over_top10", "phase": "51-C", "applicability": "required", "filename": "shared_all_over_top10.csv"},
    {"id": "O51.15", "name": "seed_overlap", "phase": "51-D", "applicability": "required", "filename": "seed_overlap_table.csv"},
    {"id": "O51.16", "name": "membership_matrix", "phase": "51-D", "applicability": "required", "filename": "worst_case_membership_matrix.csv"},
    {"id": "O51.17", "name": "error_concentration", "phase": "51-D", "applicability": "required", "filename": "error_concentration_table.csv"},
    {"id": "O51.18", "name": "regime_composition", "phase": "51-E", "applicability": "required", "filename": "regime_composition.csv"},
    {"id": "O51.19", "name": "regime_overrepresentation", "phase": "51-E", "applicability": "required", "filename": "regime_overrepresentation.csv"},
    {"id": "O51.20", "name": "hardness_vs_seed_disagreement", "phase": "51-D", "applicability": "required", "filename": "hardness_vs_seed_disagreement.csv"},
    {"id": "O51.21", "name": "baseline_context", "phase": "51-E", "applicability": "required", "filename": "baseline_context.csv"},
    {"id": "O51.22", "name": "casebook_index", "phase": "51-F", "applicability": "required", "filename": "casebook_index.csv"},
    {"id": "O51.23", "name": "casebook_markdown", "phase": "51-G", "applicability": "required", "filename": "worst_case_casebook.md"},
    {"id": "O51.24", "name": "local_temporal_context", "phase": "51-F", "applicability": "required", "filename": "local_temporal_context.csv"},
    {"id": "O51.25", "name": "input_window_manifest", "phase": "51-F", "applicability": "required", "filename": "input_window_manifest.csv"},
    {"id": "O51.26", "name": "model_visible_feature_summary", "phase": "51-F", "applicability": "required", "filename": "exact_input_windows_per_feature_summary.csv"},
    {"id": "O51.27", "name": "target_history_summary", "phase": "51-F", "applicability": "required", "filename": "target_history_context.csv"},
    {"id": "O51.28", "name": "context_integrity_audit", "phase": "51-F", "applicability": "required", "filename": "context_integrity_audit.csv"},
    {"id": "O51.29", "name": "figures", "phase": "51-G", "applicability": "required", "filename": "figures/"},
    {"id": "O51.30", "name": "attention_handoff_case_table", "phase": "51-G", "applicability": "required", "filename": "phase51_attention_handoff_cases.csv"},
    {"id": "O51.31", "name": "findings", "phase": "51-G", "applicability": "required", "filename": "phase51_findings.json"},
    {"id": "O51.32", "name": "phase52_handoff", "phase": "51-G", "applicability": "required", "filename": "phase52_attention_extraction_handoff.json"},
    {"id": "O51.33", "name": "tests", "phase": "51-G", "applicability": "required", "filename": "phase51_tests_summary.csv"},
    {"id": "O51.34", "name": "discrepancies", "phase": "51-G", "applicability": "required", "filename": "phase51_discrepancies.json"},
    {"id": "O51.35", "name": "summary_json", "phase": "51-G", "applicability": "required", "filename": "phase51_summary.json"},
    {"id": "O51.36", "name": "human_readable_report", "phase": "51-G", "applicability": "required", "filename": "phase51_report.md"},
    {"id": "O51.37", "name": "readme", "phase": "51-G", "applicability": "required", "filename": "README_WORST_ERROR_ANALYSIS.md"},
    {"id": "O51.38", "name": "signoff", "phase": "51-G", "applicability": "required", "filename": "phase51_signoff.json"},
]


def _sha256(p: Path) -> str:
    if p.is_dir():
        return ""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def audit_o51_inventory(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root if project_root is not None else get_project_root()
    rows: list[dict[str, Any]] = []
    required_count = 0
    found_count = 0
    missing: list[str] = []
    for entry in O51_INVENTORY:
        fp = root / PHASE51_DIR_REL / entry["filename"]
        exists = fp.exists()
        if exists:
            sha = _sha256(fp)
            size = fp.stat().st_size
            try:
                if fp.is_dir():
                    file_count = sum(1 for x in fp.rglob("*") if x.is_file())
                else:
                    file_count = 1
            except OSError:
                file_count = 0
        else:
            sha = ""
            size = 0
            file_count = 0
        if entry["applicability"] == "required":
            required_count += 1
            if exists:
                found_count += 1
            else:
                missing.append(entry["id"])
        rows.append({
            "id": entry["id"],
            "name": entry["name"],
            "phase": entry["phase"],
            "applicability": entry["applicability"],
            "expected_filename": entry["filename"],
            "exists": int(exists),
            "file_size_bytes": size,
            "file_count": file_count,
            "sha256": sha,
        })
    summary = {
        "n_required": required_count,
        "n_found": found_count,
        "n_missing": len(missing),
        "missing_o51_ids": missing,
        "completeness_ratio": (
            found_count / required_count if required_count else 1.0
        ),
        "overall_status": "PASS" if not missing else "FAIL",
    }
    return {
        "rows": rows,
        "summary": summary,
        "fieldnames": [
            "id", "name", "phase", "applicability",
            "expected_filename", "exists",
            "file_size_bytes", "file_count", "sha256",
        ],
    }


def write_o51_inventory_csv(
    audit: dict[str, Any],
    root: Path | None = None,
) -> str:
    root = root if root is not None else get_project_root()
    from ..utils.artifacts import atomic_write_bytes, csv_text
    import os
    fp = root / PHASE51_DIR_REL / "o51_inventory.csv"
    fp.parent.mkdir(parents=True, exist_ok=True)
    content = csv_text(audit["fieldnames"], audit["rows"]).encode("utf-8")
    atomic_write_bytes(fp, content)
    os.chmod(fp, 0o444)
    return _sha256(fp)


def write_o51_inventory_summary_json(
    audit: dict[str, Any],
    root: Path | None = None,
) -> str:
    root = root if root is not None else get_project_root()
    from ..utils.artifacts import atomic_write_bytes, canonical_json_bytes
    import os
    fp = root / PHASE51_DIR_REL / "o51_inventory_summary.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes({
        "phase": 51,
        "subphase": "51-G",
        "audit_kind": "O51_INVENTORY",
        "summary": audit["summary"],
    })
    atomic_write_bytes(fp, content)
    os.chmod(fp, 0o444)
    return _sha256(fp)
