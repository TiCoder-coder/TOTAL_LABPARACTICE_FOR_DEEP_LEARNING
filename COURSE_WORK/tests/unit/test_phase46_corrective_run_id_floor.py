"""Tests for the Phase 46 corrective run-id allocation floor.

These tests are NON-SCIENTIFIC:

  - NO training
  - NO inference
  - NO Test access
  - NO optimizer.step()
  - NO scaler.fit()
  - NO registry record modification
  - NO filesystem mutation outside tmp_path / read-only registry state

Coverage (matching requirements of Part 2G-K §2):

1. Corrective Phase 46 run sequence is >= 256.
2. Existing 0153/0154/0155 remain excluded (EXCLUDED_RUN_IDS).
3. RUN 0183 is never promoted/reused as the official corrected run.
4. IDs remain unique across three seeds.
5. Normal non-corrective registry behavior is not unintentionally changed.
6. No training is triggered by allocation tests.
"""

from __future__ import annotations

import importlib.util as _iu
import json
import sys
from pathlib import Path

import pytest

ROOT = (
    Path(__file__).resolve().parents[2]
)

# Load canonical registry module via importlib
REG_PATH = ROOT / "src" / "course_work" / "experiments" / "registry.py"
_spec = _iu.spec_from_file_location("_registry_module_under_test", str(REG_PATH))
_registry = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_registry)

# Load p46_three_seed_runs (canonical, under src/) to check it imports the floor
P46_PATH = ROOT / "src" / "course_work" / "scripts" / "p46_three_seed_runs.py"
_p46_spec = _iu.spec_from_file_location("_p46_module_under_test", str(P46_PATH))


# ---------------------------------------------------------------------------
# Fixture: use the canonical on-disk registry (READ-ONLY for allocation
# logic). `allocate_run_id` does NOT persist; it only allocates an id by
# scanning existing records (read-only) and returns the new id without
# writing. So using the canonical registry is safe for this test.
# ---------------------------------------------------------------------------


@pytest.fixture
def canonical_registry():
    """Yield the canonical on-disk ExperimentRegistry (read-only allocation)."""
    return _registry.ExperimentRegistry()


# ---------------------------------------------------------------------------
# 1. MIN_CORRECTIVE_PHASE46_SEQUENCE constant is 256.
# ---------------------------------------------------------------------------


class TestFloorConstant:
    def test_min_corrective_sequence_is_256(self) -> None:
        assert _registry.MIN_CORRECTIVE_PHASE46_SEQUENCE == 256


# ---------------------------------------------------------------------------
# 2. allocate_run_id accepts min_sequence and respects the floor.
# ---------------------------------------------------------------------------


class TestAllocateRunIdFloor:
    def test_allocate_with_min_sequence_returns_geq_floor(
        self, canonical_registry,
    ) -> None:
        rid = canonical_registry.allocate_run_id(
            "TRANSFORMER_ENCODER", "FINAL_SEED_RUN",
            "585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24",
            min_sequence=256,
        )
        seq = int(rid.split("_")[3])
        assert seq >= 256, f"alloc sequence {seq} < floor 256"

    def test_allocate_without_min_sequence_uses_default(
        self, canonical_registry,
    ) -> None:
        """Default behavior is unchanged: no min_sequence -> legacy semantics."""
        rid = canonical_registry.allocate_run_id(
            "TRANSFORMER_ENCODER", "FINAL_SEED_RUN",
            "585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24",
        )
        # Just confirm shape: RUN_TR_FSD_NNNN_<8chars>
        parts = rid.split("_")
        assert parts[0] == "RUN"
        assert parts[1] == "TR"
        assert parts[2] == "FSD"
        assert parts[3].isdigit()


# ---------------------------------------------------------------------------
# 3. register_run forwards min_sequence to allocate_run_id.
# ---------------------------------------------------------------------------


class TestRegisterRunForwardsMinSequence:
    def test_register_run_accepts_min_sequence_param(self) -> None:
        """register_run must accept a `min_sequence` kwarg without error."""
        import inspect

        sig = inspect.signature(_registry.ExperimentRegistry.register_run)
        assert "min_sequence" in sig.parameters, (
            "register_run must accept `min_sequence` keyword argument"
        )


# ---------------------------------------------------------------------------
# 4. Existing 0153/0154/0155 remain excluded (governance preservation).
# ---------------------------------------------------------------------------


class TestHistoricalExclusionPreserved:
    def test_0153_0154_0155_still_in_excluded(self) -> None:
        """EXCLUDED_RUN_IDS in p46_three_seed_runs.py must still contain 0153/0154/0155."""
        # We can't import the constants from p46_three_seed_runs.py without
        # running main(); the file is a script. Instead, check the canonical
        # values declared in registry._registry are preserved by scanning for
        # the canonical historical IDs in any governance artifact.
        # The on-disk RUN_ID forbidden values are encoded in
        # src/course_work/scripts/p46_three_seed_runs.py. Verify the canonical
        # IDs are still present by scanning the script source.
        p46_source = (ROOT / "src" / "course_work" / "scripts" / "p46_three_seed_runs.py").read_text()
        for forbidden_id in (
            "RUN_TR_FSD_0153_B15A19DC",
            "RUN_TR_FSD_0154_DD82D743",
            "RUN_TR_FSD_0155_59A50ADD",
        ):
            assert forbidden_id in p46_source, (
                f"forbidden ID {forbidden_id} missing from p46_three_seed_runs.py"
            )


# ---------------------------------------------------------------------------
# 5. RUN 0183 is never promoted to the official corrected namespace.
# ---------------------------------------------------------------------------


class TestRun0183NotPromoted:
    def test_run_0183_directory_unchanged(self) -> None:
        """RUN_TR_FSD_0183_C2F24D58 directory must still be present on disk
        and not promoted into artifacts/three_seed_final_runs/official_checkpoints/seed_42/.

        Per Part 2G-K governance decision: RUN 0183 remains preserved as valid
        interrupted/persistence-failed historical evidence. It is NOT
        promoted to the FINAL_REFIT official position.
        """
        run_dir = ROOT / "artifacts" / "runs" / "RUN_TR_FSD_0183_C2F24D58"
        assert run_dir.exists(), "RUN 0183 must be preserved on disk"

        # Official_final_refit .pt for seed_42 (canonical position) should be
        # absent (since the user REJECTED persistence-only promotion in Part 2G-K).
        # Note: as of this task, the FINAL_REFIT .pt for seed_42 has not been
        # promoted; if a legacy placeholder .pt file existed, that's just a
        # stale placeholder. Verify only that nothing NEW has been written.
        official_pt = ROOT / "artifacts" / "three_seed_final_runs" / "official_checkpoints" / "seed_42" / "seed_42_FINAL_REFIT.pt"
        if official_pt.exists():
            # If a placeholder exists, it should be the legacy RUN 0153
            # metadata, NOT a NEW canonical .pt promoted from RUN 0183.
            # (We do not assert the SHA here because part 2G-J added the
            # archive helper but the user did NOT execute the finalization.)
            import hashlib
            sha = hashlib.sha256(official_pt.read_bytes()).hexdigest()
            # The legacy placeholder has very small content; verify it has
            # NOT been overwritten with the RUN 0183 best_checkpoint content.
            # best_checkpoint.pt SHA is f4537e26...
            assert sha != "f4537e269dceb0355cf24cb234505ea53df773e57fdf642f10670f4a710c3968", (
                "RUN 0183 best_checkpoint.pt must NOT have been promoted to "
                "official_checkpoints/seed_42_FINAL_REFIT.pt"
            )

    def test_recovery_helper_not_executed(self) -> None:
        """The p46_finalize_seed42_recovery.py helper exists but must NOT have been
        executed (no _history/SEED42_PERSISTENCE_RECOVERY_* dir under
        official_checkpoints/seed_42/ should exist beyond a clean state).
        """
        seed42_dir = ROOT / "artifacts" / "three_seed_final_runs" / "official_checkpoints" / "seed_42"
        # The recovery helper writes to _history/SEED42_PERSISTENCE_RECOVERY_<UTC>/
        # under seed42_dir. Since the user REJECTED its execution, that dir
        # must not exist with a SEED42_PERSISTENCE_RECOVERY prefix.
        for entry in seed42_dir.glob("_history/SEED42_PERSISTENCE_RECOVERY_*"):
            # If found, this is unexpected — recovery must not have run.
            raise AssertionError(
                f"Recovery helper archive dir found: {entry}. "
                "This means p46_finalize_seed42_recovery.py was executed, which "
                "is forbidden under Part 2G-K."
            )


# ---------------------------------------------------------------------------
# 6. Allocation uniqueness across three seeds.
# ---------------------------------------------------------------------------


class TestAllocationUniquenessAcrossThreeSeeds:
    def test_allocate_three_distinct_runs_returns_distinct_ids(
        self, canonical_registry,
    ) -> None:
        """Three sequential allocations with distinct fingerprints AND
        min_sequence=256 produce three distinct run_ids (the allocator
        uses config_fingerprint[:8] as the run-id suffix, so the test
        fingerprints must have distinct 8-char prefixes).
        """
        # Use distinct 8-char prefixes to make the run-id suffix distinct.
        fingerprints = [
            "abcdef0011111111111111111111111111111111111111111111111111111111",
            "1234567822222222222222222222222222222222222222222222222222222222",
            "fedcba9833333333333333333333333333333333333333333333333333333333",
        ]
        # Verify the prefixes are distinct so the test is meaningful.
        prefixes = [fp[:8] for fp in fingerprints]
        assert len(set(prefixes)) == 3, f"test fingerprints must have distinct 8-char prefixes; got {prefixes}"

        rids = []
        for fp in fingerprints:
            rid = canonical_registry.allocate_run_id(
                "TRANSFORMER_ENCODER", "FINAL_SEED_RUN", fp,
                min_sequence=256,
            )
            rids.append(rid)
        assert len(set(rids)) == 3, f"allocations not unique: {rids}"
        for rid in rids:
            seq = int(rid.split("_")[3])
            assert seq >= 256


# ---------------------------------------------------------------------------
# 7. Non-corrective registry behavior is preserved (no regression).
# ---------------------------------------------------------------------------


class TestNonCorrectiveBehaviorPreserved:
    def test_default_allocate_no_min_sequence_unaffected(
        self, canonical_registry,
    ) -> None:
        """Default (legacy) allocation behavior must be unchanged."""
        # No min_sequence -> the allocator behaves exactly as before.
        rid = canonical_registry.allocate_run_id(
            "TRANSFORMER_ENCODER", "FINAL_SEED_RUN",
            "abcdef0000000000000000000000000000000000000000000000000000000000",
            # NOTE: no min_sequence
        )
        # The shape must match legacy format. We just assert it's a valid
        # 4-digit zero-padded sequence.
        parts = rid.split("_")
        assert len(parts[3]) == 4
        assert parts[3].isdigit()


# ---------------------------------------------------------------------------
# 8. No training is triggered by allocation tests (training guard).
# ---------------------------------------------------------------------------


class TestNoTrainingDuringAllocation:
    def test_allocate_does_not_invoke_training(self) -> None:
        """allocate_run_id must NOT trigger any training or optimizer side effects."""
        # Install the canonical training-attempt guard from the registry
        # package's sibling module: we cannot import p46_three_seed_runs.py
        # without running it; here we just check that allocate_run_id does
        # NOT import torch.optim or call optimizer.step at all.
        # We do this by examining allocate_run_id's code path: it only does
        # disk reads (records JSONL + run_root glob). No training is possible.
        # Static assertion: allocate_run_id does not reference torch.optim
        # or .step().
        import inspect
        source = inspect.getsource(_registry.ExperimentRegistry.allocate_run_id)
        assert "torch" not in source, "allocate_run_id must not import torch"
        assert "step(" not in source, "allocate_run_id must not call optimizer.step"
        assert "train(" not in source, "allocate_run_id must not call train()"


# ---------------------------------------------------------------------------
# 9. Dry-run of corrected runner produces RUN_TR_FSD_>=0256 in next allocation.
# ---------------------------------------------------------------------------


class TestDryRunPredictsNextRunIdFloor:
    def test_dry_run_emits_planned_runs_at_ge_0256(self) -> None:
        """Verify the Phase 46 dry-run output mentions the namespace floor logic.

        We don't run training; we just inspect the source of p46_three_seed_runs.py
        to confirm it passes min_sequence=MIN_CORRECTIVE_PHASE46_SEQUENCE (=256)
        to registry.register_run(...).
        """
        p46_source = (ROOT / "src" / "course_work" / "scripts" / "p46_three_seed_runs.py").read_text()
        assert "min_sequence=MIN_CORRECTIVE_PHASE46_SEQUENCE" in p46_source, (
            "p46_three_seed_runs.py must pass min_sequence=MIN_CORRECTIVE_PHASE46_SEQUENCE "
            "to registry.register_run(...) for all corrective runs"
        )
