"""Phase 51-B — atomic writers for selection contract and audit artifacts.

All writers enforce:
  - parent directory creation
  - atomic write via tempfile + fsync + os.replace
  - JSON canonical (sort_keys=True)
  - CSV deterministic (LF terminators, no extras)
  - post-write chmod 0444 (read-only) for frozen artifacts
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

from ..utils.artifacts import (
    atomic_write_bytes,
    canonical_json_bytes,
    csv_text,
    get_project_root,
)

# Frozen directory
PHASE51_DIR_REL = "artifacts/worst_error_analysis"

# Filenames
SELECTION_CONTRACT_REL = "worst_error_selection_contract.json"
SELECTION_CONTRACT_FP_REL = "selection_contract_fingerprint.json"
PHASE51_MANIFEST_REL = "worst_error_analysis_manifest.json"
ALIGNMENT_AUDIT_REL = "phase51_alignment_audit.json"
ALIGNMENT_AUDIT_CSV_REL = "phase51_alignment_audit.csv"
WORKING_TABLE_REL = "phase51_target_level_working_table.csv"


def _project_root(project_root: Path | None = None) -> Path:
    if project_root is not None:
        return project_root
    return get_project_root()


def _ensure_phase51_dir(project_root: Path | None = None) -> Path:
    root = _project_root(project_root)
    out = root / PHASE51_DIR_REL
    out.mkdir(parents=True, exist_ok=True)
    return out


def _chmod_0444(path: Path) -> None:
    """Set file mode to read-only."""
    try:
        os.chmod(path, 0o444)
    except (OSError, PermissionError):
        pass  # best-effort; not all filesystems support this


def write_selection_contract(
    contract_payload: dict[str, Any],
    project_root: Path | None = None,
) -> str:
    """Write worst_error_selection_contract.json atomically (idempotent).

    After the contract has been frozen (i.e. once it has been written at
    least once), this function refuses to overwrite it. Subsequent
    invocations of ``materialize_phase51_b`` MUST NOT change the frozen
    contract bytes. Use ``sha256_file`` on the resulting path to confirm
    it remains at the human-approved SHA256.

    Returns SHA256 of the file (verified against any pre-existing copy).
    """
    from ..utils.artifacts import write_bytes_once_or_verify
    out = _ensure_phase51_dir(project_root)
    path = out / SELECTION_CONTRACT_REL

    payload = dict(contract_payload)
    payload.setdefault("contract_frozen", True)
    payload.setdefault("selection_executed", False)
    payload.setdefault("worst_error_ranking_executed", False)
    payload.setdefault("individual_case_inspection", False)
    payload.setdefault("phase52_authorized", False)

    content = canonical_json_bytes(payload)
    # Idempotent: if file exists, verify exactly, do not overwrite.
    write_bytes_once_or_verify(path, content)
    _chmod_0444(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_selection_contract_fingerprint(
    contract_sha256: str,
    contract_path: str,
    project_root: Path | None = None,
) -> str:
    """Write selection_contract_fingerprint.json with SHA256 of the frozen contract."""
    out = _ensure_phase51_dir(project_root)
    path = out / SELECTION_CONTRACT_FP_REL

    payload = {
        "version": "SELECTION_CONTRACT_FINGERPRINT-v1",
        "phase": 51,
        "subphase": "51-B",
        "selection_contract_path": contract_path,
        "selection_contract_sha256": contract_sha256,
        "contract_frozen": True,
        "selection_executed": False,
        "worst_error_ranking_executed": False,
        "individual_case_inspection": False,
        "phase52_authorized": False,
        "created_at_utc": _utc_now_iso(),
    }
    atomic_write_bytes(path, canonical_json_bytes(payload))
    _chmod_0444(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_phase51_manifest(
    manifest_payload: dict[str, Any],
    project_root: Path | None = None,
) -> str:
    """Write worst_error_analysis_manifest.json (atomic + chmod 0444)."""
    out = _ensure_phase51_dir(project_root)
    path = out / PHASE51_MANIFEST_REL

    payload = dict(manifest_payload)
    content = canonical_json_bytes(payload)
    atomic_write_bytes(path, content)
    _chmod_0444(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_alignment_audit(
    audit_dict: dict[str, Any],
    project_root: Path | None = None,
) -> str:
    """Write phase51_alignment_audit.json (atomic + chmod 0444)."""
    out = _ensure_phase51_dir(project_root)
    path = out / ALIGNMENT_AUDIT_REL

    payload = dict(audit_dict)
    payload.setdefault("phase", 51)
    payload.setdefault("subphase", "51-B")
    payload.setdefault("ranking_executed", False)
    payload.setdefault("created_at_utc", _utc_now_iso())

    atomic_write_bytes(path, canonical_json_bytes(payload))
    _chmod_0444(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_alignment_audit_csv(
    audit_dict: dict[str, Any],
    project_root: Path | None = None,
) -> str:
    """Write phase51_alignment_audit.csv as a flat list of check rows."""
    out = _ensure_phase51_dir(project_root)
    path = out / ALIGNMENT_AUDIT_CSV_REL

    fieldnames = ["check", "expected", "observed", "status"]
    rows = []

    rows.append({
        "check": "n_test",
        "expected": str(audit_dict.get("n_test")),
        "observed": "2961",
        "status": "PASS" if audit_dict.get("n_test") == 2961 else "FAIL",
    })
    rows.append({
        "check": "n_residual_long",
        "expected": "8883",
        "observed": str(audit_dict.get("n_residual_long")),
        "status": "PASS" if audit_dict.get("n_residual_long") == 8883 else "FAIL",
    })
    rows.append({
        "check": "n_seeds",
        "expected": "3",
        "observed": str(audit_dict.get("n_seeds")),
        "status": "PASS" if audit_dict.get("n_seeds") == 3 else "FAIL",
    })
    for seed in ("42", "123", "2026"):
        rows.append({
            "check": f"residual_rows_seed_{seed}",
            "expected": "2961",
            "observed": str(audit_dict.get("residual_rows_per_seed", {}).get(seed, "?")),
            "status": "PASS" if audit_dict.get("residual_rows_per_seed", {}).get(seed) == 2961 else "FAIL",
        })
    rows.append({
        "check": "timestamps_exact",
        "expected": "True",
        "observed": str(audit_dict.get("timestamps_exact")),
        "status": "PASS" if audit_dict.get("timestamps_exact") else "FAIL",
    })
    rows.append({
        "check": "y_true_exact",
        "expected": "True",
        "observed": str(audit_dict.get("y_true_exact")),
        "status": "PASS" if audit_dict.get("y_true_exact") else "FAIL",
    })
    rows.append({
        "check": "residual_convention_verified",
        "expected": "True",
        "observed": str(audit_dict.get("residual_convention_verified")),
        "status": "PASS" if audit_dict.get("residual_convention_verified") else "FAIL",
    })
    rows.append({
        "check": "abs_error_verified",
        "expected": "True",
        "observed": str(audit_dict.get("abs_error_verified")),
        "status": "PASS" if audit_dict.get("abs_error_verified") else "FAIL",
    })
    rows.append({
        "check": "squared_error_verified",
        "expected": "True",
        "observed": str(audit_dict.get("squared_error_verified")),
        "status": "PASS" if audit_dict.get("squared_error_verified") else "FAIL",
    })
    rows.append({
        "check": "residual_sign_verified",
        "expected": "True",
        "observed": str(audit_dict.get("residual_sign_verified")),
        "status": "PASS" if audit_dict.get("residual_sign_verified") else "FAIL",
    })
    rows.append({
        "check": "phase50_assignment_join_lossless",
        "expected": "True",
        "observed": str(audit_dict.get("phase50_assignment_join_lossless")),
        "status": "PASS" if audit_dict.get("phase50_assignment_join_lossless") else "FAIL",
    })
    rows.append({
        "check": "phase50_regime_labels_unchanged",
        "expected": "True",
        "observed": str(audit_dict.get("phase50_regime_labels_unchanged")),
        "status": "PASS" if audit_dict.get("phase50_regime_labels_unchanged") else "FAIL",
    })
    rows.append({
        "check": "phase48_seed_spread_join_lossless",
        "expected": "True",
        "observed": str(audit_dict.get("phase48_seed_spread_join_lossless")),
        "status": "PASS" if audit_dict.get("phase48_seed_spread_join_lossless") else "FAIL",
    })
    rows.append({
        "check": "no_forbidden_columns",
        "expected": "True",
        "observed": str(audit_dict.get("no_forbidden_columns")),
        "status": "PASS" if audit_dict.get("no_forbidden_columns") else "FAIL",
    })
    rows.append({
        "check": "no_ranking_executed",
        "expected": "True",
        "observed": str(audit_dict.get("no_ranking_executed")),
        "status": "PASS" if audit_dict.get("no_ranking_executed") else "FAIL",
    })

    content = csv_text(fieldnames, rows)
    atomic_write_bytes(path, content.encode("utf-8"))
    _chmod_0444(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_working_table(
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    project_root: Path | None = None,
) -> str:
    """Write phase51_target_level_working_table.csv (atomic + chmod 0444)."""
    out = _ensure_phase51_dir(project_root)
    path = out / WORKING_TABLE_REL

    content = csv_text(fieldnames, rows)
    atomic_write_bytes(path, content.encode("utf-8"))
    _chmod_0444(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
