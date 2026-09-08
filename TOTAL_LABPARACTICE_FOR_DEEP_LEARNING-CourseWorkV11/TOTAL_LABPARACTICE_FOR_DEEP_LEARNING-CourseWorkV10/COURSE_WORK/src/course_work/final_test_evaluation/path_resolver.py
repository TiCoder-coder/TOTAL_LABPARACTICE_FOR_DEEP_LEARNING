"""Phase 47 — Phase46/45 Artifact Path Resolver.

CANONICAL SOURCE OF TRUTH for all Phase46/45 artifact paths.

This module resolves all Phase46 → Phase47 artifact paths from the repository,
centralizing path logic so it is defined in ONE place.

CANONICAL LAYOUT:
    artifacts/
        final_model_lock/          ← Phase45 lock artifacts
            phase_45_signoff.json          ✓
            phase46_three_seed_handoff.json  ✓
            final_model_scientific_config.json  ✓
            ... (Phase45 artifacts, read-only from Phase47)
        three_seed_final_runs/      ← Phase46 final-run artifacts
            phase_46_signoff.json             ✓ CANONICAL for Phase47
            phase47_test_release.json         ✓ CANONICAL for Phase47
            phase47_final_test_evaluation_handoff.json  ✓ CANONICAL for Phase47
            three_seed_manifest.json
            three_seed_contract.json
            ... (Phase46 outputs)
        scaling/final_dev/
            final_scaler_registry.json       ✓ CANONICAL for Phase47
        windows/
            window_index.csv                ✓ CANONICAL for Phase47
        feature_sets/
            feature_set_registry.json        ✓ CANONICAL for Phase47
        features/
            feature_view.parquet             ✓ CANONICAL for Phase47

IMPORTANT RULES:
1. Phase_46_signoff.json is at artifacts/three_seed_final_runs/, NOT artifacts/final_model_lock/
2. Phase_45_signoff.json is at artifacts/final_model_lock/, NOT artifacts/three_seed_final_runs/
3. phase47_test_release.json is at artifacts/three_seed_final_runs/
4. phase47_final_test_evaluation_handoff.json is at artifacts/three_seed_final_runs/
5. Phase47 NEVER writes to artifacts/final_model_lock/ or artifacts/three_seed_final_runs/
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FINAL_MODEL_LOCK_DIR = "artifacts/final_model_lock"

THREE_SEED_FINAL_RUNS_DIR = "artifacts/three_seed_final_runs"

FINAL_SCALING_DIR = "artifacts/scaling/final_dev"
WINDOWS_DIR = "artifacts/windows"
FEATURE_SETS_DIR = "artifacts/feature_sets"
FEATURES_DIR = "artifacts/features"


def get_phase46_signoff_path(root: Path | None = None) -> Path:
    root = root or _default_root()
    return root / THREE_SEED_FINAL_RUNS_DIR / "phase_46_signoff.json"

def get_phase47_release_path(root: Path | None = None) -> Path:
    root = root or _default_root()
    return root / THREE_SEED_FINAL_RUNS_DIR / "phase47_test_release.json"

def get_phase47_handoff_path(root: Path | None = None) -> Path:
    root = root or _default_root()
    return root / THREE_SEED_FINAL_RUNS_DIR / "phase47_final_test_evaluation_handoff.json"

def get_phase45_signoff_path(root: Path | None = None) -> Path:
    root = root or _default_root()
    return root / FINAL_MODEL_LOCK_DIR / "phase_45_signoff.json"


def get_phase46_three_seed_handoff_path(root: Path | None = None) -> Path:
    root = root or _default_root()
    return root / FINAL_MODEL_LOCK_DIR / "phase46_three_seed_handoff.json"


def get_final_scaling_registry_path(root: Path | None = None) -> Path:
    root = root or _default_root()
    return root / FINAL_SCALING_DIR / "final_scaler_registry.json"

def get_window_index_path(root: Path | None = None) -> Path:
    root = root or _default_root()
    return root / WINDOWS_DIR / "window_index.csv"


def get_feature_set_registry_path(root: Path | None = None) -> Path:
    root = root or _default_root()
    return root / FEATURE_SETS_DIR / "feature_set_registry.json"

def get_feature_view_path(root: Path | None = None) -> Path:
    root = root or _default_root()
    return root / FEATURES_DIR / "feature_view.parquet"

def get_transformer_checkpoint_path(seed: int, root: Path | None = None) -> Path:
    root = root or _default_root()
    paths = {
        42: root / THREE_SEED_FINAL_RUNS_DIR / "official_checkpoints" / "seed_42" / "seed_42_FINAL_REFIT.pt",
        123: root / THREE_SEED_FINAL_RUNS_DIR / "official_checkpoints" / "seed_123" / "seed_123_FINAL_REFIT.pt",
        2026: root / THREE_SEED_FINAL_RUNS_DIR / "official_checkpoints" / "seed_2026" / "seed_2026_FINAL_REFIT.pt",
    }
    if seed not in paths:
        raise ValueError(f"No canonical checkpoint for seed {seed}. Canonical seeds: {list(paths.keys())}")
    return paths[seed]


def _default_root() -> Path:
    from course_work.utils.artifacts import get_project_root
    return get_project_root()



class Phase46PathError(Exception):
    """Raised when a required Phase46 artifact path is invalid or missing."""
    pass


@dataclass(frozen=True)
class Phase46ReleaseState:
    """Verified Phase46 release state for Phase47."""
    phase46_signoff: dict[str, Any]
    phase47_release: dict[str, Any]
    phase47_handoff: dict[str, Any]
    root: Path
    phase46_signoff_path: Path
    phase47_release_path: Path
    phase47_handoff_path: Path


def verify_phase46_release_for_phase47(
    root: Path | None = None,
    strict: bool = True,
) -> Phase46ReleaseState:
    """Verify Phase46 release state for Phase47.

    This function:
    1. Locates the canonical phase_46_signoff.json (three_seed_final_runs/, NOT final_model_lock/)
    2. Verifies it exists and is valid JSON
    3. Verifies status = PASS
    4. Verifies ready_for_phase47 = true
    5. Verifies phase47_released = true
    6. Verifies candidate = TR_C2_ALT_LOOKBACK
    7. Verifies 3 canonical seeds
    8. Locates phase47_test_release.json
    9. Verifies released = true
    10. Locates phase47_final_test_evaluation_handoff.json
    11. Verifies handoff exists

    Args:
        root: COURSE_WORK root.
        strict: If True, verify all gates. If False, just load files.

    Returns:
        Phase46ReleaseState with all loaded artifacts.

    Raises:
        Phase46PathError: If any required artifact is missing or invalid.
    """
    root = root or _default_root()

    p46_signoff_path = get_phase46_signoff_path(root)
    if not p46_signoff_path.exists():
        raise Phase46PathError(
            f"Phase46 signoff not found at canonical path: {p46_signoff_path}\n"
            f"NOTE: phase_46_signoff.json is at THREE_SEED_FINAL_RUNS_DIR, "
            f"not final_model_lock/ (which contains phase_45_signoff.json)"
        )

    try:
        p46_signoff = json.loads(p46_signoff_path.read_text())
    except Exception as e:
        raise Phase46PathError(f"Phase46 signoff is not valid JSON: {e}")

    if strict:
        if p46_signoff.get("status") != "PASS":
            raise Phase46PathError(
                f"Phase46 status is {p46_signoff.get('status')}, expected PASS"
            )

        if not p46_signoff.get("ready_for_phase47"):
            raise Phase46PathError(
                "Phase46 ready_for_phase47 is not True"
            )

        if not p46_signoff.get("phase47_released"):
            raise Phase46PathError(
                "Phase46 phase47_released is not True"
            )

        if p46_signoff.get("candidate_id") != "TR_C2_ALT_LOOKBACK":
            raise Phase46PathError(
                f"Phase46 candidate is {p46_signoff.get('candidate_id')}, "
                f"expected TR_C2_ALT_LOOKBACK"
            )

        seeds = p46_signoff.get("seed_list", [])
        if set(seeds) != {42, 123, 2026}:
            raise Phase46PathError(
                f"Phase46 seeds are {seeds}, expected [42, 123, 2026]"
            )

    p47_release_path = get_phase47_release_path(root)
    if not p47_release_path.exists():
        raise Phase46PathError(
            f"Phase47 release not found at: {p47_release_path}"
        )

    try:
        p47_release = json.loads(p47_release_path.read_text())
    except Exception as e:
        raise Phase46PathError(f"Phase47 release is not valid JSON: {e}")

    if strict:
        if not p47_release.get("released"):
            raise Phase46PathError(
                "phase47_test_release.released is not True"
            )

        if p47_release.get("status") != "PASS":
            raise Phase46PathError(
                f"phase47_test_release.status is {p47_release.get('status')}, expected PASS"
            )

        gates = p47_release.get("gates", {})
        failed = [k for k, v in gates.items() if not v]
        if failed:
            raise Phase46PathError(
                f"phase47_test_release gates failed: {failed}"
            )

    p47_handoff_path = get_phase47_handoff_path(root)
    if not p47_handoff_path.exists():
        raise Phase46PathError(
            f"Phase47 handoff not found at: {p47_handoff_path}"
        )

    try:
        p47_handoff = json.loads(p47_handoff_path.read_text())
    except Exception as e:
        raise Phase46PathError(f"Phase47 handoff is not valid JSON: {e}")

    return Phase46ReleaseState(
        phase46_signoff=p46_signoff,
        phase47_release=p47_release,
        phase47_handoff=p47_handoff,
        root=root,
        phase46_signoff_path=p46_signoff_path,
        phase47_release_path=p47_release_path,
        phase47_handoff_path=p47_handoff_path,
    )
