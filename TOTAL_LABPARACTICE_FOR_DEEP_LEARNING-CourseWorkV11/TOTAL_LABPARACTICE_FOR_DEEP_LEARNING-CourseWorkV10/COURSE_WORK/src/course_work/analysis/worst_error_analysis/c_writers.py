"""Phase 51-C — atomic writers for ranking artifacts.

All writers are deterministic:
  - LF line terminators
  - sort_keys JSON
  - chmod 0444 on frozen artifacts
  - atomic write via tempfile + os.replace
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

from course_work.utils.artifacts import (
    atomic_write_bytes,
    canonical_json_bytes,
    csv_text,
    get_project_root,
)


PHASE51_DIR_REL = "artifacts/worst_error_analysis"

# Filenames per O51 inventory
W1_PER_SEED_TOP20_REL = "worst_per_seed_top20.csv"
W2_SHARED_TOP20_REL = "worst_shared_top20.csv"
W3_UNDERPREDICTION_TOP10_REL = "worst_underprediction_top10.csv"
W4_OVERPREDICTION_TOP10_REL = "worst_overprediction_top10.csv"
W3_SH_SHARED_ALL_UNDER_REL = "shared_all_under_top10.csv"
W4_SH_SHARED_ALL_OVER_REL = "shared_all_over_top10.csv"
RANKING_AUDIT_REL = "phase51_ranking_audit.json"
RANKING_AUDIT_CSV_REL = "phase51_ranking_audit.csv"
PHASE51_C_MANIFEST_REL = "worst_error_ranking_manifest.json"


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
        os.chmod(path, 0o444)
    except (OSError, PermissionError):
        pass


def write_csv(
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    rel: str,
    project_root: Path | None = None,
) -> str:
    """Write a ranking CSV atomically with chmod 0444. Returns SHA256."""
    out = _ensure_dir(project_root)
    path = out / rel
    content = csv_text(fieldnames, rows)
    atomic_write_bytes(path, content.encode("utf-8"))
    _chmod_0444(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(
    payload: dict[str, Any],
    rel: str,
    project_root: Path | None = None,
) -> str:
    """Write a JSON atomically with chmod 0444. Returns SHA256."""
    out = _ensure_dir(project_root)
    path = out / rel
    content = canonical_json_bytes(payload)
    atomic_write_bytes(path, content)
    _chmod_0444(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()
