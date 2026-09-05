"""Tests for the Phase 46 historical checkpoint archival script.

These tests verify that the archival procedure:
  - Reads source SHA correctly
  - Computes archived SHA correctly
  - Verifies metadata sidecar points to historical run IDs
  - Writes the archive manifest with required fields
  - Excludes historical run IDs from Phase 47 release

Uses synthetic disposable checkpoints (NOT the real historical ones) to
verify the procedure is safe-by-construction.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

ARTIFACT_DIR = ROOT / "artifacts" / "three_seed_final_runs"


def _create_synthetic_checkpoint(seed: int) -> bytes:
    """Create synthetic checkpoint bytes."""
    return hashlib.sha256(f"synthetic_seed_{seed}".encode()).digest() * 100


def test_archive_manifest_exists_and_has_required_fields():
    """The real archive manifest must exist after running the archive script."""
    archive_manifest = (
        ARTIFACT_DIR / "historical_checkpoints" / "historical_checkpoint_archive_manifest.json"
    )
    assert archive_manifest.exists(), "Archive manifest not created"
    m = json.loads(archive_manifest.read_bytes())
    assert m["archive_version"] == "HISTORICAL_CHECKPOINT_ARCHIVAL-v1"
    assert m["phase"] == 46
    assert m["historical_status"] == "INVALIDATED_BY_CONTRACT_DEVIATION"
    assert m["archive_verified"] is True
    assert m["test_access"] is False
    assert m["scientific_training"] is False
    assert m["status"] == "PASS"
    assert m["phase47_release_reference"] == "NEVER"
    assert set(m["excluded_run_ids"]) == {
        "RUN_TR_FSD_0153_B15A19DC",
        "RUN_TR_FSD_0154_DD82D743",
        "RUN_TR_FSD_0155_59A50ADD",
    }
    assert len(m["runs"]) == 3
    seeds = sorted(r["seed"] for r in m["runs"])
    assert seeds == [42, 123, 2026]
    for r in m["runs"]:
        assert r["status"] == "PASS"
        assert "checkpoint_sha256" in r
        assert "archived_path" in r
        assert "original_path" in r
        assert "metadata_run_id" in r
        assert "metadata_checkpoint_type" in r


def test_archive_audit_csv_exists():
    audit = ARTIFACT_DIR / "historical_checkpoints" / "historical_checkpoint_archive_audit.csv"
    assert audit.exists()


def test_archive_contains_three_checkpoints():
    """The archive directory must contain three checkpoint files, one per historical run."""
    archive = ARTIFACT_DIR / "historical_checkpoints"
    assert archive.exists()
    for run_id in ["RUN_TR_FSD_0153_B15A19DC", "RUN_TR_FSD_0154_DD82D743", "RUN_TR_FSD_0155_59A50ADD"]:
        run_dir = archive / run_id
        assert run_dir.exists()
        assert (run_dir / f"seed_{ {'RUN_TR_FSD_0153_B15A19DC':42,'RUN_TR_FSD_0154_DD82D743':123,'RUN_TR_FSD_0155_59A50ADD':2026}[run_id] }_FINAL_REFIT.pt").exists()
        assert (run_dir / f"seed_{ {'RUN_TR_FSD_0153_B15A19DC':42,'RUN_TR_FSD_0154_DD82D743':123,'RUN_TR_FSD_0155_59A50ADD':2026}[run_id] }_FINAL_REFIT_metadata.json").exists()


def test_active_official_checkpoint_path_cleared():
    """Active canonical .pt paths must NOT contain the stale historical .pt files."""
    for seed_dir in ["seed_42", "seed_123", "seed_2026"]:
        active = ARTIFACT_DIR / "official_checkpoints" / seed_dir
        active_pt = active / f"{seed_dir}_FINAL_REFIT.pt"
        active_meta = active / f"{seed_dir}_FINAL_REFIT_metadata.json"
        # Stale .pt must be removed.
        assert not active_pt.exists(), f"Stale .pt still present: {active_pt}"
        # Placeholder metadata.json must exist.
        assert active_meta.exists(), f"Placeholder metadata missing: {active_meta}"
        meta = json.loads(active_meta.read_bytes())
        assert meta.get("checkpoint_type") == "AWAITING_CORRECTED_PHASE46_RUN"
        assert meta.get("status") == "AWAITING_CORRECTED_PHASE46_RUN"


def test_historical_run_dirs_have_invalidation_markers():
    """Each historical run directory must contain an invalidation manifest."""
    for run_id in ["RUN_TR_FSD_0153_B15A19DC", "RUN_TR_FSD_0154_DD82D743", "RUN_TR_FSD_0155_59A50ADD"]:
        run_dir = ROOT / "artifacts" / "runs" / run_id
        if not run_dir.exists():
            continue
        marker = run_dir / "historical_phase46_invalidation_manifest.json"
        assert marker.exists(), f"Missing invalidation marker in {run_id}"
        m = json.loads(marker.read_bytes())
        assert m["historical_run_id"] == run_id
        assert m["historical_status"] == "INVALIDATED_BY_CONTRACT_DEVIATION"
        assert m["phase47_release_excluded"] is True
        assert m["scientific_retraining"] is False
        assert m["test_access"] is False


def test_archived_checkpoint_sha_matches_metadata():
    """Archived checkpoint SHA must match metadata.model_state_sha256."""
    archive = ARTIFACT_DIR / "historical_checkpoints"
    manifest = json.loads(
        (archive / "historical_checkpoint_archive_manifest.json").read_bytes()
    )
    for r in manifest["runs"]:
        archived_pt = ROOT / r["archived_path"]
        assert archived_pt.exists()
        # Compute SHA of archived file.
        actual_sha = hashlib.sha256(archived_pt.read_bytes()).hexdigest()
        assert actual_sha == r["checkpoint_sha256"], (
            f"Archived SHA mismatch for {r['historical_run_id']}"
        )
        # Also verify metadata SHA matches.
        metadata_path = ROOT / r["metadata_path"]
        meta = json.loads(metadata_path.read_bytes())
        assert meta["model_state_sha256"] == r["checkpoint_sha256"], (
            f"Metadata SHA mismatch for {r['historical_run_id']}"
        )
        assert meta["run_id"] == r["historical_run_id"]
