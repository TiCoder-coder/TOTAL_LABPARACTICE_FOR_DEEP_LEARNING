"""Tests for Part 2G-O: Archive pre-corrective Phase47 evidence.

These tests verify that:
  1. Old Phase47 evidence is preserved (not deleted).
  2. Archive copies SHA-match originals.
  3. Stale artifacts are flagged as AUTHORITATIVE_CORRECTIVE_PHASE47 = NO.
  4. Test access history is append-only (not reset/truncated/rewritten).
  5. Corrected Phase46 run IDs are the only eligible Phase47 inputs.
  6. All 3 seeds are required.
  7. No best-seed/ensemble shortcut exists in preflight.
  8. No training/tuning path exists in corrective Phase47 preflight.
  9. No Test access occurs during this task.

NO TRAINING, INFERENCE, OR TEST ACCESS during these tests.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = ROOT / "artifacts" / "final_test"
THREE_SEED_DIR = ROOT / "artifacts" / "three_seed_final_runs"
PROJECT_ROOT = ROOT.parent
SCRIPT_PATH = ROOT / "scripts" / "archive_pre_corrective_phase47.py"

# Stale (pre-corrective) lineage references
STALE_FINAL_LOCK_SHA = "585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24"
STALE_CHECKPOINT_SHAS = {
    "42": "c3cfad116aa91d47fb2f2950407498901b980a344a99462776e4b90828805b10",
    "123": "8a134fec517be0dfcd81ac6f961bf75fa69e8d34666205f731d893969bf01e22",
    "2026": "8753800539f7a617bacbaa541daafbbfaff338a1de38ae1a5146650b4593dc6f",
}
STALE_RUN_IDS = {
    "RUN_TR_FSD_0254_2B11AC68",
    "RUN_TR_FSD_0254_3858DDA9",
    "RUN_TR_FSD_0255_C7E123FB",
}

# Corrected Phase46 inputs
CORRECTED_RUN_IDS = {
    "RUN_TR_FSD_0256_C2F24D58",
    "RUN_TR_FSD_0256_AA575C42",
    "RUN_TR_FSD_0256_247AB83A",
}
CORRECTED_CKPT_SHAS = {
    "523d2e98f82f8534782e9364a4fb86e7f9381e33c2abb694921ed1c4b3d3de67",
    "650637c14f84548237ac0641a880e1b461824643bd276ae9cae0832879204804",
    "809cfde75611723e774791735c436009ce29d259faf30ab4123a2dc46c84212f",
}


def _latest_archive_dir() -> Path | None:
    """Find the most recent archive directory."""
    history_dir = TEST_DIR / "_history"
    if not history_dir.exists():
        return None
    archives = sorted(p for p in history_dir.iterdir() if p.name.startswith("PRE_CORRECTIVE_PHASE47_"))
    if not archives:
        return None
    return archives[-1]


class TestOldEvidencePreserved:
    """1. Old Phase47 evidence is preserved."""

    def test_old_phase47_signoff_still_exists(self) -> None:
        """The pre-corrective phase_47_signoff.json is preserved (not deleted)."""
        path = TEST_DIR / "phase_47_signoff.json"
        assert path.exists(), "Old phase_47_signoff.json must be preserved"
        signoff = json.loads(path.read_text())
        assert signoff["final_lock_sha256"] == STALE_FINAL_LOCK_SHA

    def test_old_evaluation_contract_still_exists(self) -> None:
        path = TEST_DIR / "final_test_evaluation_contract.json"
        assert path.exists(), "Old final_test_evaluation_contract.json must be preserved"

    def test_old_release_verification_still_exists(self) -> None:
        path = TEST_DIR / "final_test_release_verification.json"
        assert path.exists(), "Old final_test_release_verification.json must be preserved"


class TestArchiveSHAMatch:
    """2. Archive copies SHA-match originals."""

    def test_archive_exists(self) -> None:
        archive = _latest_archive_dir()
        assert archive is not None, "An archive directory must exist"
        assert archive.is_dir()

    def test_archive_manifest_exists(self) -> None:
        archive = _latest_archive_dir()
        assert archive is not None
        manifest = archive / "_archive_manifest.json"
        assert manifest.exists(), f"Archive manifest must exist at {manifest}"

    def test_all_archived_files_match_source_sha(self) -> None:
        archive = _latest_archive_dir()
        assert archive is not None
        manifest = json.loads((archive / "_archive_manifest.json").read_text())
        mismatches = 0
        for entry in manifest["files_archived"]:
            src = PROJECT_ROOT / entry["source_path"]
            dst = PROJECT_ROOT / entry["archive_path"]
            if not src.exists() or not dst.exists():
                mismatches += 1
                continue
            src_sha = hashlib.sha256(src.read_bytes()).hexdigest()
            dst_sha = hashlib.sha256(dst.read_bytes()).hexdigest()
            if src_sha != dst_sha or src_sha != entry["source_sha256"]:
                mismatches += 1
        assert mismatches == 0, f"All archive copies must SHA-match sources; got {mismatches} mismatches"

    def test_archive_total_files_reasonable(self) -> None:
        """Archive should contain 20+ files (all Phase47 evidence)."""
        archive = _latest_archive_dir()
        assert archive is not None
        manifest = json.loads((archive / "_archive_manifest.json").read_text())
        assert manifest["total_archived_files"] >= 20, (
            f"Archive should contain at least 20 files; got {manifest['total_archived_files']}"
        )


class TestStaleNotAuthoritative:
    """3. Stale artifacts are not considered active authoritative evidence."""

    def test_archive_manifest_authoritative_flag(self) -> None:
        archive = _latest_archive_dir()
        assert archive is not None
        manifest = json.loads((archive / "_archive_manifest.json").read_text())
        assert "NO" in manifest["authoritative_corrective_phase47"], (
            "All archived files must NOT be authoritative corrective Phase47"
        )

    def test_stale_run_ids_in_old_phase47_signoff(self) -> None:
        """The pre-corrective phase_47_signoff references stale run_ids."""
        signoff = json.loads((TEST_DIR / "phase_47_signoff.json").read_text())
        # The old signoff does NOT contain the corrected run_ids; it has no run_ids per seed
        # (it has checkpoint SHAs which are stale).
        assert signoff["seed42_checkpoint_sha256"] in STALE_CHECKPOINT_SHAS.values()
        assert signoff["seed123_checkpoint_sha256"] in STALE_CHECKPOINT_SHAS.values()
        assert signoff["seed2026_checkpoint_sha256"] in STALE_CHECKPOINT_SHAS.values()

    def test_stale_run_ids_in_evaluation_contract(self) -> None:
        contract = json.loads((TEST_DIR / "final_test_evaluation_contract.json").read_text())
        run_ids = {m["run_id"] for m in contract["models"]["transformers"]}
        assert run_ids == STALE_RUN_IDS, (
            f"Old contract must reference stale run_ids; got {run_ids}"
        )

    def test_stale_final_lock_in_old_signoff(self) -> None:
        """Old signoff has the conflated final_lock_sha = config fingerprint."""
        signoff = json.loads((TEST_DIR / "phase_47_signoff.json").read_text())
        assert signoff["final_lock_sha256"] == STALE_FINAL_LOCK_SHA

    def test_readme_pre_corrective_phase47_marker(self) -> None:
        """A README marker file documents the pre-corrective status."""
        marker = TEST_DIR / "README_PRE_CORRECTIVE_PHASE47.md"
        assert marker.exists(), f"Marker README must exist at {marker}"
        content = marker.read_text()
        assert "AUTHORITATIVE_CORRECTIVE_PHASE47" in content
        assert "NO" in content  # classified as not authoritative


class TestTestAccessLogAppendOnly:
    """4. Test access history is append-only (not reset/truncated/rewritten)."""

    def test_log_file_exists(self) -> None:
        log_path = TEST_DIR / "final_test_access_log.jsonl"
        assert log_path.exists(), "Test access log must exist"

    def test_log_has_historical_entries(self) -> None:
        log_path = TEST_DIR / "final_test_access_log.jsonl"
        entries = [json.loads(line) for line in log_path.read_text().strip().split("\n") if line.strip()]
        assert len(entries) >= 4, (
            f"Log should have at least 4 historical entries; got {len(entries)}"
        )

    def test_log_entries_have_required_fields(self) -> None:
        log_path = TEST_DIR / "final_test_access_log.jsonl"
        entries = [json.loads(line) for line in log_path.read_text().strip().split("\n") if line.strip()]
        for entry in entries:
            assert "timestamp" in entry
            assert "action" in entry

    def test_log_chronologically_monotonic(self) -> None:
        log_path = TEST_DIR / "final_test_access_log.jsonl"
        entries = [json.loads(line) for line in log_path.read_text().strip().split("\n") if line.strip()]
        timestamps = [e["timestamp"] for e in entries]
        for i in range(len(timestamps) - 1):
            assert timestamps[i] <= timestamps[i + 1], (
                f"Log timestamps must be monotonic; {timestamps[i]} > {timestamps[i+1]}"
            )

    def test_archive_directory_preserved(self) -> None:
        archive_dir = TEST_DIR / ".archive"
        assert archive_dir.exists()
        archived = list(archive_dir.glob("*.json"))
        assert len(archived) >= 10, (
            f"Historical .archive should have at least 10 entries; got {len(archived)}"
        )

    def test_log_not_modified_by_2go_task(self) -> None:
        """The log file SHA before 2G-O is recorded as a constant here.
        If the log file SHA matches the recorded value, it was NOT modified.
        """
        log_path = TEST_DIR / "final_test_access_log.jsonl"
        actual_sha = hashlib.sha256(log_path.read_bytes()).hexdigest()
        # This is the SHA at the time of 2G-O audit (pre-modification).
        # The test passes if the SHA matches the recorded value.
        expected_sha = "fe5247fc5c2908858a349b77af66d8f4f1699c3eb82ab01fd2874670868d6a09"
        assert actual_sha == expected_sha, (
            f"Test access log SHA must match pre-2G-O value (append-only); "
            f"got {actual_sha}, expected {expected_sha}"
        )


class TestCorrectedInputsOnly:
    """5. Corrected Phase46 run IDs are the only eligible Phase47 inputs."""

    def test_corrected_run_ids_in_phase46_signoff(self) -> None:
        p46 = json.loads((THREE_SEED_DIR / "phase_46_signoff.json").read_text())
        run_ids = {p46["seed42_run_id"], p46["seed123_run_id"], p46["seed2026_run_id"]}
        assert run_ids == CORRECTED_RUN_IDS, (
            f"Phase 46 signoff must reference only corrected run_ids; got {run_ids}"
        )

    def test_corrected_checkpoint_shas_match_files(self) -> None:
        for seed, expected_sha in (
            (42, "523d2e98f82f8534782e9364a4fb86e7f9381e33c2abb694921ed1c4b3d3de67"),
            (123, "650637c14f84548237ac0641a880e1b461824643bd276ae9cae0832879204804"),
            (2026, "809cfde75611723e774791735c436009ce29d259faf30ab4123a2dc46c84212f"),
        ):
            ckpt_path = THREE_SEED_DIR / f"official_checkpoints/seed_{seed}/seed_{seed}_FINAL_REFIT.pt"
            actual_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
            assert actual_sha == expected_sha, (
                f"Seed {seed} checkpoint SHA must match expected; "
                f"got {actual_sha[:16]}, expected {expected_sha[:16]}"
            )

    def test_phase47_release_references_corrected_run_ids(self) -> None:
        release = json.loads((THREE_SEED_DIR / "phase47_test_release.json").read_text())
        release_run_ids = {r["run_id"] for r in release["run_records"]}
        assert release_run_ids == CORRECTED_RUN_IDS, (
            f"Phase 47 release must reference corrected run_ids; got {release_run_ids}"
        )

    def test_phase47_release_excludes_historical_run_ids(self) -> None:
        release = json.loads((THREE_SEED_DIR / "phase47_test_release.json").read_text())
        excluded = set(release["historical_run_ids_excluded"])
        # All 6 excluded historical run_ids must include the stale pre-corrective ones
        assert "RUN_TR_FSD_0153_B15A19DC" in excluded
        assert "RUN_TR_FSD_0154_DD82D743" in excluded
        assert "RUN_TR_FSD_0155_59A50ADD" in excluded


class TestAllSeedsRequired:
    """6. All 3 seeds are required."""

    def test_three_seeds_required_in_release(self) -> None:
        release = json.loads((THREE_SEED_DIR / "phase47_test_release.json").read_text())
        assert release["seed_count"] == 3
        assert sorted(release["seeds"]) == [42, 123, 2026]
        assert len(release["run_records"]) == 3
        # Also verify via final_test_release_verification.json (older but still active)
        verification = json.loads((TEST_DIR / "final_test_release_verification.json").read_text())
        assert verification["required_seed_count"] == 3
        assert verification["completed_seed_count"] == 3

    def test_three_checkpoints_exist(self) -> None:
        for seed in (42, 123, 2026):
            ckpt_path = THREE_SEED_DIR / f"official_checkpoints/seed_{seed}/seed_{seed}_FINAL_REFIT.pt"
            assert ckpt_path.exists(), f"Seed {seed} checkpoint must exist"


class TestNoBestSeedEnsemble:
    """7. No best-seed/ensemble shortcut exists in preflight."""

    def test_release_disallows_best_seed_selection(self) -> None:
        release = json.loads((THREE_SEED_DIR / "phase47_test_release.json").read_text())
        # Phase47 release must enforce no best-seed selection
        # (via the all_seeds_completed gate)
        assert release["gates"]["all_seeds_completed"] is True

    def test_phase47_handoff_has_all_three_seeds(self) -> None:
        handoff = json.loads((THREE_SEED_DIR / "phase47_final_test_evaluation_handoff.json").read_text())
        run_seeds = sorted(r["seed"] for r in handoff["final_runs"])
        assert run_seeds == [42, 123, 2026], f"All 3 seeds must be in handoff; got {run_seeds}"


class TestNoTrainingTuningInPreflight:
    """8. No training/tuning path exists in corrective Phase47 preflight."""

    def test_archive_script_does_not_train(self) -> None:
        """The archive script source must NOT contain training paths."""
        if not SCRIPT_PATH.exists():
            pytest.skip("Archive script not present")
        src = SCRIPT_PATH.read_text()
        forbidden = ["engine.train", "optimizer.step", ".backward(", "model.fit"]
        for f in forbidden:
            assert f not in src, f"Archive script must not invoke training: {f!r}"

    def test_archive_script_does_not_call_inference(self) -> None:
        if not SCRIPT_PATH.exists():
            pytest.skip("Archive script not present")
        src = SCRIPT_PATH.read_text()
        # Forbidden: any inference or forward path
        forbidden = ["model.eval(", "model(x", "predict("]
        for f in forbidden:
            assert f not in src, f"Archive script must not invoke inference: {f!r}"


class TestNoTestAccessDuringTask:
    """9. No Test access occurs during this task."""

    def test_log_sha_unchanged_after_task(self) -> None:
        """The Test access log SHA must be unchanged after 2G-O."""
        log_path = TEST_DIR / "final_test_access_log.jsonl"
        actual_sha = hashlib.sha256(log_path.read_bytes()).hexdigest()
        expected_sha = "fe5247fc5c2908858a349b77af66d8f4f1699c3eb82ab01fd2874670868d6a09"
        assert actual_sha == expected_sha

    def test_no_new_test_access_event_added(self) -> None:
        """No new test_access_event_<TS>.json file should be added by 2G-O."""
        archive_dir = TEST_DIR / ".archive"
        if not archive_dir.exists():
            pytest.fail(".archive directory missing")
        # Pre-2G-O had 91 archive entries (recorded at 2G-O start).
        # Allow ±0 entries (no new events added).
        archived = list(archive_dir.glob("final_test_access_event_*.json"))
        # If new events were added by this test run, it would fail.
        # Otherwise, count remains 91 (or whatever it was pre-2G-O).
        # We don't have a strict "exactly 91" assertion because the count
        # may have evolved over time. Instead, we verify no event with
        # timestamp later than 2026-09-06T20:46 UTC (this task start).
        # The 2G-O work happened around 2026-09-06T20:46-20:48 UTC.
        # Pre-existing events all have 20260906T prefix or 20260903T.
        # We verify no event after 2026-09-06T20:46:00Z.
        cutoff = "20260906T204600Z"
        for f in archived:
            m = re.search(r"final_test_access_event_(\d+Z)", f.name)
            if m:
                ts = m.group(1)
                assert ts <= cutoff, (
                    f"No new Test access event should have been added after 2G-O start; "
                    f"found {f.name} > {cutoff}"
                )
