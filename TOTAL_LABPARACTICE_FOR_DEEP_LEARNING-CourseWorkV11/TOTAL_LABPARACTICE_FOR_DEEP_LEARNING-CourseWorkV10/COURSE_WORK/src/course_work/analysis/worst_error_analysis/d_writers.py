"""Phase 51-D — Atomic writers for descriptive Phase 51-D artifacts.

Implements:
  O51.15  seed_overlap_table.csv
  O51.16  worst_case_membership_matrix.csv
  O51.17  error_concentration_table.csv
  O51.20  hardness_vs_seed_disagreement.csv (+ group summary section)

All writers:
  - Use atomic_write_bytes
  - Apply chmod 0444 on completion
  - Use deterministic CSV / JSON serialization
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

from course_work.utils.artifacts import (
    atomic_write_bytes,
    canonical_json_bytes,
    csv_text,
    get_project_root,
)


PHASE51_DIR_REL = "artifacts/worst_error_analysis"

O51_D_ARTIFACTS = {
    "seed_overlap_table.csv": "O51.15",
    "worst_case_membership_matrix.csv": "O51.16",
    "error_concentration_table.csv": "O51.17",
    "hardness_vs_seed_disagreement.csv": "O51.20",
    "hardness_group_summary.csv": "O51.20-summary",
    "phase51_d_manifest.json": "O51.D-MANIFEST",
}


def _project_root(project_root: Path | None = None) -> Path:
    if project_root is not None:
        return project_root
    return get_project_root()


def _ensure_dir(project_root: Path | None = None) -> Path:
    root = _project_root(project_root)
    out = root / PHASE51_DIR_REL
    out.mkdir(parents=True, exist_ok=True)
    return out


def _chmod_0444(path: Path) -> None:
    try:
        import os
        os.chmod(path, 0o444)
    except (OSError, PermissionError):
        pass


def write_csv(
    rows: list[dict[str, Any]] | list[list[Any]],
    fieldnames: list[str],
    rel: str,
    project_root: Path | None = None,
) -> str:
    """Write a deterministic CSV. Returns SHA256."""
    out = _ensure_dir(project_root)
    path = out / rel
    # csv_text only accepts dict-rows; convert list-rows to dict using fieldnames
    if rows and isinstance(rows[0], dict):
        dict_rows: list[dict[str, Any]] = rows  # type: ignore[assignment]
    else:
        dict_rows = [dict(zip(fieldnames, r)) for r in rows]
    content = csv_text(fieldnames, dict_rows)
    atomic_write_bytes(path, content.encode("utf-8"))
    _chmod_0444(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(
    payload: dict[str, Any],
    rel: str,
    project_root: Path | None = None,
) -> str:
    out = _ensure_dir(project_root)
    path = out / rel
    content = canonical_json_bytes(payload)
    atomic_write_bytes(path, content)
    _chmod_0444(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()
