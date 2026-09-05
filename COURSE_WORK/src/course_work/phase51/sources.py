"""Phase 51 — canonical source loaders with SHA verification.

Loads ONLY frozen Phase 47 / 48 / 49 / 50 artifacts.
Does NOT recompute predictions, residuals, or regime labels.
Does NOT rank errors.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from ..utils.artifacts import get_project_root

# ── Artifact path constants ────────────────────────────────────────────────────

# Phase 47
PHASE47_SIGNOFF_REL = "artifacts/final_test/phase_47_signoff.json"
PHASE47_PRED_DIR_REL = "artifacts/final_test/predictions"
PHASE47_PRED_SEED42_REL = "artifacts/final_test/predictions/final_test_predictions_seed42.csv"
PHASE47_PRED_SEED123_REL = "artifacts/final_test/predictions/final_test_predictions_seed123.csv"
PHASE47_PRED_SEED2026_REL = "artifacts/final_test/predictions/final_test_predictions_seed2026.csv"
PHASE47_PRED_PERSISTENCE_REL = (
    "artifacts/final_test/predictions/final_test_predictions_persistence.csv"
)
PHASE47_LSTM_ELIGIBILITY_REL = "artifacts/final_test/final_test_lstm_eligibility.json"

# Phase 48
PHASE48_SIGNOFF_REL = "artifacts/prediction_analysis/phase_48_signoff.json"
PHASE48_SEED_SPREAD_REL = "artifacts/prediction_analysis/prediction_seed_spread.csv"
PHASE48_P51_CONTEXT_HANDOFF_REL = (
    "artifacts/prediction_analysis/phase51_worst_error_context_handoff.json"
)

# Phase 49
PHASE49_SIGNOFF_REL = "artifacts/residual_analysis/phase_49_signoff.json"
PHASE49_RESIDUAL_LONG_REL = "artifacts/residual_analysis/residual_long_table.csv"
PHASE49_RESIDUAL_WIDE_REL = "artifacts/residual_analysis/residual_wide_table.csv"
PHASE49_SIGN_CONSENSUS_REL = (
    "artifacts/residual_analysis/phase49_cross_seed_sign_consensus.csv"
)

# Phase 50
PHASE50_SIGNOFF_REL = "artifacts/error_by_regime/phase_50_signoff.json"
PHASE50_TEST_ASSIGN_REL = "artifacts/error_by_regime/test_regime_assignment.csv"
PHASE50_THRESHOLDS_REL = "artifacts/error_by_regime/regime_thresholds_train_only.json"
PHASE50_TRAIN_MANIFEST_REL = (
    "artifacts/error_by_regime/regime_reference_train_manifest.json"
)
PHASE50_TEST_ASSIGN_FP_REL = (
    "artifacts/error_by_regime/test_regime_assignment_fingerprint.json"
)
PHASE50_P51_HANDOFF_REL = "artifacts/error_by_regime/phase51_handoff.json"

# Phase 47 → Phase 51 handoff
FT_P51_HANDOFF_REL = "artifacts/final_test/phase51_worst_error_handoff.json"


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _project_root(project_root: Path | None = None) -> Path:
    if project_root is not None:
        return project_root
    return get_project_root()


# ── Source verification helpers ─────────────────────────────────────────────────

class SourceVerificationError(RuntimeError):
    """Raised when a canonical source fails SHA256 verification."""
    pass


def verify_sha256(path: Path, expected: str, label: str) -> None:
    """Verify file SHA256 matches expected value. Raises SourceVerificationError."""
    actual = _sha256(path)
    if actual != expected:
        raise SourceVerificationError(
            f"SHA256 mismatch for {label}: "
            f"expected={expected[:16]}… actual={actual[:16]}…"
        )


# ── Phase 47 loaders ───────────────────────────────────────────────────────────

def load_phase47_signoff(project_root: Path | None = None) -> dict[str, Any]:
    root = _project_root(project_root)
    return _read_json(root / PHASE47_SIGNOFF_REL)


def load_phase47_lstm_eligibility(project_root: Path | None = None) -> dict[str, Any]:
    root = _project_root(project_root)
    return _read_json(root / PHASE47_LSTM_ELIGIBILITY_REL)


def load_phase47_persistence_bundle(
    project_root: Path | None = None,
) -> list[dict[str, str]]:
    root = _project_root(project_root)
    return _read_csv(root / PHASE47_PRED_PERSISTENCE_REL)


# ── Phase 48 loaders ───────────────────────────────────────────────────────────

def load_phase48_signoff(project_root: Path | None = None) -> dict[str, Any]:
    root = _project_root(project_root)
    return _read_json(root / PHASE48_SIGNOFF_REL)


def load_phase48_seed_spread(
    project_root: Path | None = None,
) -> list[dict[str, str]]:
    root = _project_root(project_root)
    return _read_csv(root / PHASE48_SEED_SPREAD_REL)


# ── Phase 49 loaders ─────────────────────────────────────────────────────────

def load_phase49_signoff(project_root: Path | None = None) -> dict[str, Any]:
    root = _project_root(project_root)
    return _read_json(root / PHASE49_SIGNOFF_REL)


def load_residual_long(
    project_root: Path | None = None,
) -> list[dict[str, str]]:
    """Load Phase 49 residual long table (one row per seed per target)."""
    root = _project_root(project_root)
    return _read_csv(root / PHASE49_RESIDUAL_LONG_REL)


def load_residual_wide(
    project_root: Path | None = None,
) -> list[dict[str, str]]:
    """Load Phase 49 residual wide table (one row per target, all seeds)."""
    root = _project_root(project_root)
    return _read_csv(root / PHASE49_RESIDUAL_WIDE_REL)


def load_phase49_sign_consensus(
    project_root: Path | None = None,
) -> list[dict[str, str]] | None:
    """Load Phase 49 sign consensus table if present."""
    root = _project_root(project_root)
    path = root / PHASE49_SIGN_CONSENSUS_REL
    if path.exists():
        return _read_csv(path)
    return None


# ── Phase 50 loaders ───────────────────────────────────────────────────────────

def load_phase50_signoff(project_root: Path | None = None) -> dict[str, Any]:
    root = _project_root(project_root)
    return _read_json(root / PHASE50_SIGNOFF_REL)


def load_test_regime_assignment(
    project_root: Path | None = None,
) -> list[dict[str, str]]:
    """Load frozen Phase 50 test regime assignment (one row per test target)."""
    root = _project_root(project_root)
    return _read_csv(root / PHASE50_TEST_ASSIGN_REL)


def load_phase50_thresholds(project_root: Path | None = None) -> dict[str, Any]:
    """Load frozen Phase 50 TRAIN-derived thresholds."""
    root = _project_root(project_root)
    return _read_json(root / PHASE50_THRESHOLDS_REL)


def load_phase50_train_manifest(project_root: Path | None = None) -> dict[str, Any]:
    root = _project_root(project_root)
    return _read_json(root / PHASE50_TRAIN_MANIFEST_REL)


def load_phase50_test_assign_fingerprint(
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = _project_root(project_root)
    return _read_json(root / PHASE50_TEST_ASSIGN_FP_REL)


# ── Handoff loaders ────────────────────────────────────────────────────────────

def load_phase51_ft_handoff(project_root: Path | None = None) -> dict[str, Any]:
    root = _project_root(project_root)
    return _read_json(root / FT_P51_HANDOFF_REL)


def load_phase51_er_handoff(project_root: Path | None = None) -> dict[str, Any]:
    root = _project_root(project_root)
    return _read_json(root / PHASE50_P51_HANDOFF_REL)


# ── SHA verification ─────────────────────────────────────────────────────────

def verify_all_sources(project_root: Path | None = None) -> dict[str, bool]:
    """Verify SHA256 of all canonical Phase 47/49/50 sources.

    Returns dict mapping source label → True (PASS).
    Raises SourceVerificationError on any mismatch.
    """
    root = _project_root(project_root)

    checks = {
        # Phase 47
        "phase47_seed42_predictions": (
            root / PHASE47_PRED_SEED42_REL,
            "246ee0d725af972bd621ce9cf4dbc550d8c02ec7c9dc1214b373807c99bf73f2",
        ),
        "phase47_seed123_predictions": (
            root / PHASE47_PRED_SEED123_REL,
            "1bb55c445ffe132d3cfdc22e09439d77c76a2defd2f918bfb8f47029f666a08b",
        ),
        "phase47_seed2026_predictions": (
            root / PHASE47_PRED_SEED2026_REL,
            "bfb575357dd6a8e3a23fd08230c69f8e3d3297582ea73c16cf006601fcce79d8",
        ),
        "phase47_persistence_predictions": (
            root / PHASE47_PRED_PERSISTENCE_REL,
            "7115af1c479b89575f2f7ed6c065a68d214e44d336a0c681c033a8015bd9ee9b",
        ),
        # Phase 49
        "phase49_residual_long": (
            root / PHASE49_RESIDUAL_LONG_REL,
            "8418a99110bfda7047bd27c49c1c7a9769313b1ce1fa6dc66925f5286d120038",
        ),
        "phase49_residual_wide": (
            root / PHASE49_RESIDUAL_WIDE_REL,
            "931ff9109aef236d9517d8964d29168316c7d5903633441443c1ae5e71e8f8bd",
        ),
        # Phase 50
        "phase50_test_regime_assignment": (
            root / PHASE50_TEST_ASSIGN_REL,
            "e90553cfc747a3f15e0e9ec9e6868ae497e7ade797dc14a81999e416b74219ac",
        ),
        "phase50_thresholds": (
            root / PHASE50_THRESHOLDS_REL,
            "2fe9ad4f873e3b3e42013fe3b2d630e377e4e568120769bd2234a76d6974b109",
        ),
    }

    results = {}
    for label, (path, expected) in checks.items():
        verify_sha256(path, expected, label)
        results[label] = True
    return results
