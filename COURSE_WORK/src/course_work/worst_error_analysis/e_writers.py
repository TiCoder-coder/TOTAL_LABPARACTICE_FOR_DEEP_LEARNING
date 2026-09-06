"""Phase 51-E — atomic writers for regime / persistence / LSTM / sign-consensus
context + enrichment artifacts.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..utils.artifacts import (
    atomic_write_bytes,
    canonical_json_bytes,
    csv_text,
    get_project_root,
)


PHASE51_DIR_REL = "artifacts/worst_error_analysis"


def _project_root(project_root: Path | None) -> Path:
    return project_root if project_root is not None else get_project_root()


def _ensure_dir(project_root: Path) -> Path:
    out = _project_root(project_root) / PHASE51_DIR_REL
    out.mkdir(parents=True, exist_ok=True)
    return out


def _chmod_0444(path: Path) -> None:
    try:
        import os
        os.chmod(path, 0o444)
    except (OSError, PermissionError):
        pass


def write_csv(
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    rel: str,
    project_root: Path,
) -> str:
    out = _ensure_dir(project_root)
    path = out / rel
    content = csv_text(fieldnames, rows)
    atomic_write_bytes(path, content.encode("utf-8"))
    _chmod_0444(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(
    payload: dict[str, Any],
    rel: str,
    project_root: Path,
) -> str:
    out = _ensure_dir(project_root)
    path = out / rel
    content = canonical_json_bytes(payload)
    atomic_write_bytes(path, content)
    _chmod_0444(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()
