"""Phase 50-C orchestrator: Test regime assignment (Stage B) + freeze (Stage C)."""
from __future__ import annotations
from pathlib import Path

from . import contract
from . import sources
from .regime_assignment import (
    materialize_phase50_c,
    build_test_regime_assignment,
    write_assignment_audit,
    write_assignment_fingerprint,
    write_train_regime_assignment,
)


def main(project_root: Path | None = None) -> dict:
    return materialize_phase50_c(project_root)
