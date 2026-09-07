#!/usr/bin/env python3
"""Phase 47 — Archive Current Test Evidence (Pre-Corrective Preservation).

Mirrors the Phase 46 historical-checkpoint archival pattern
(`p46_archive_historical_checkpoints.py`) for the Phase 47 evidence directory.

ARCHIVES THE CURRENT PHASE 47 EVIDENCE PRIOR TO THE CORRECTIVE PHASE 47 RUN.

Per the approved corrective plan
(`docs/plan/plan_before_process/phase_47_corrective_post_audit_test_re_evaluation_plan.md` §6),
the prior Phase 47 evidence is preserved verbatim under
`artifacts/final_test/_history/PHASE47_PRE_CORRECTIVE_<UTC>/` before any
corrective Test access occurs.

WHAT IS ARCHIVED
================
Each file in §6.2 of the approved plan is copied (not moved) to the archive
destination. SHA256 of source vs archived copy is verified. Any mismatch →
ROLLBACK (raise).

WHICH FILES
===========
- phase_47_signoff.json
- final_test_summary.json
- final_test_evaluation_contract.json
- final_test_evaluation_manifest.json
- final_test_release_verification.json
- final_test_population_manifest.json (FROZEN — must NOT change after archive)
- final_test_access_event.json
- final_test_access_log.jsonl
- final_test_lstm_eligibility.json
- final_test_discrepancies.json
- prediction_checksums.json
- phase48_prediction_analysis_handoff.json
- phase49_residual_analysis_handoff.json
- phase50_error_regime_handoff.json
- phase51_worst_error_handoff.json
- phase52_attention_extraction_handoff.json
- final_test_report.md
- README_FINAL_TEST_EVALUATION.md

NOT ARCHIVED (NEVER MODIFIED)
=============================
- artifacts/final_test/.archive/   — historical Test access events
- artifacts/final_test/predictions/   — generated during corrective run
- Any checkpoint binaries / phase 46 evidence

PROCEDURE
=========
1. Compute SHA256 of source.
2. Copy (not move) into archive destination.
3. Compute SHA256 of archived copy.
4. Require source_sha == archived_sha.
5. Append to _archive_manifest.json with reason = "PHASE47_PRE_CORRECTIVE".
6. STOP on any SHA mismatch.

NOT EXECUTED IN PART 2F — this file defines the archive procedure only.
The corrective Phase 47 execution (which is the trigger to call this script)
requires separate explicit human approval per Rule #76.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from course_work.utils.artifacts import sha256_file


FINAL_TEST_DIR = ROOT / "artifacts" / "final_test"
HISTORY_DIR = FINAL_TEST_DIR / "_history"

# UTC timestamp suffix used for archive directory name.
UTC_SUFFIX = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

# Files to archive (per Phase 47 corrective plan §6.2).
# Each entry: (relative_path, archive_status, notes)
ARCHIVE_TARGETS = [
    ("phase_47_signoff.json", "stale-numerically-consistent-lineage-invalid",
     "CURRENT (numerically consistent but lineage-invalid); preserved."),
    ("final_test_summary.json", "stale",
     "CURRENT; preserved."),
    ("final_test_evaluation_contract.json", "stale-with-known-bug",
     "CURRENT; preserved as evidence of the final_lock_sha256 conflation bug."),
    ("final_test_evaluation_manifest.json", "stale",
     "CURRENT; preserved."),
    ("final_test_release_verification.json", "stale-references-missing",
     "CURRENT (references missing phase47_test_release.json); preserved."),
    ("final_test_population_manifest.json", "frozen-applies-to-both",
     "FROZEN, applies to both prior and corrective runs; preserved unchanged."),
    ("final_test_access_event.json", "first-access-event",
     "CURRENT (event_id=TEST_FIRST_ACCESS_EVENT); preserved (NEW corrective event recorded separately)."),
    ("final_test_access_log.jsonl", "append-only",
     "CURRENT (4 inference events); preserved (append-only)."),
    ("final_test_lstm_eligibility.json", "preserved-verbatim",
     "CURRENT (NOT_ELIGIBLE_CONFIG_MISMATCH); preserved verbatim."),
    ("final_test_discrepancies.json", "stale-not-valid-post-corrective",
     "CURRENT (discrepancy_count=0 — not valid post-corrective); preserved."),
    ("prediction_checksums.json", "references-missing-csvs",
     "CURRENT (referenced but CSV files absent); preserved."),
    ("phase48_prediction_analysis_handoff.json", "stale",
     "CURRENT; preserved."),
    ("phase49_residual_analysis_handoff.json", "stale",
     "CURRENT; preserved."),
    ("phase50_error_regime_handoff.json", "stale",
     "CURRENT; preserved."),
    ("phase51_worst_error_handoff.json", "stale",
     "CURRENT; preserved."),
    ("phase52_attention_extraction_handoff.json", "stale",
     "CURRENT; preserved."),
    ("final_test_report.md", "stale",
     "CURRENT; preserved."),
    ("README_FINAL_TEST_EVALUATION.md", "stale",
     "CURRENT; preserved."),
]

# Forbidden — never archive these
NEVER_ARCHIVE_PATTERNS = [
    ".archive/",  # historical Test access events — never touched
    "predictions/",  # generated during corrective run
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def archive_one(rel_path: str, status: str, note: str) -> dict | None:
    """Archive a single file safely.

    Returns:
        A record describing the archived file, or None if source is absent
        (treated as informational skip — NOT a failure).
    Raises:
        RuntimeError on SHA mismatch (rollback semantics).
        FileNotFoundError only if the source path is required but absent.
    """
    source = FINAL_TEST_DIR / rel_path
    if not source.exists():
        # Source absent → informational skip; not a hard failure.
        return None

    # Determine the archive destination.
    archive_root = HISTORY_DIR / f"PHASE47_PRE_CORRECTIVE_{UTC_SUFFIX}"
    archive_root.mkdir(parents=True, exist_ok=True)
    dest = archive_root / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Compute SHA256 of source.
    source_sha = sha256_file(source)

    # Copy (not move).
    shutil.copy2(source, dest)

    # Verify SHA256 of archived copy.
    archived_sha = sha256_file(dest)
    if archived_sha != source_sha:
        # ROLLBACK: remove the corrupted archive copy.
        if dest.exists():
            dest.unlink()
        raise RuntimeError(
            f"Archive SHA mismatch for {rel_path}: source={source_sha} "
            f"archived={archived_sha}. ROLLBACK executed."
        )

    return {
        "path": rel_path,
        "source_sha256": source_sha,
        "archived_sha256": archived_sha,
        "source_size_bytes": source.stat().st_size,
        "status": status,
        "note": note,
        "archived_at": now_iso(),
    }


def write_archive_manifest(records: list[dict]) -> Path:
    archive_root = HISTORY_DIR / f"PHASE47_PRE_CORRECTIVE_{UTC_SUFFIX}"
    manifest = {
        "archive_version": "PHASE47_PRE_CORRECTIVE_EVIDENCE-v1",
        "phase": 47,
        "prior_phase47_status": "INVALIDATED_BY_PHASE46_LINEAGE_DRIFT",
        "archived_at": now_iso(),
        "archive_root": str(archive_root.relative_to(ROOT)),
        "utc_suffix": UTC_SUFFIX,
        "files_archived": len([r for r in records if r is not None]),
        "files_skipped_absent": len([r for r in records if r is None]),
        "test_access": False,
        "scientific_training": False,
        "phase47_execution": False,
        "approval_status": "AWAITING_HUMAN_APPROVAL_FOR_CORRECTIVE_EXECUTION",
        "records": [r for r in records if r is not None],
        "created_at": now_iso(),
    }
    manifest_path = archive_root / "_archive_manifest.json"
    manifest_path.write_bytes(json.dumps(manifest, indent=2).encode("utf-8"))
    return manifest_path


def main() -> int:
    """Execute the Phase 47 pre-corrective evidence archive.

    NOT EXECUTED IN PART 2F — the corrective Phase 47 run (which is the
    trigger for this archive) requires separate explicit human approval per
    Rule #76. This script is defined for completeness and provides a runnable
    entry point; calling it would archive the current evidence.
    """
    print("=" * 70)
    print("PHASE 47 — PRE-CORRECTIVE EVIDENCE ARCHIVE")
    print("=" * 70)
    print(f"Archive destination: {HISTORY_DIR / f'PHASE47_PRE_CORRECTIVE_{UTC_SUFFIX}'}")
    print()
    print("WARNING: This script archives the CURRENT Phase 47 evidence before")
    print("         the corrective Phase 47 Test re-evaluation.")
    print()
    print("Per Rule #76, this script must NOT be executed until explicit human")
    print("approval for the corrective Phase 47 Test re-evaluation is granted.")
    print()
    print("If you are seeing this output, the script is being executed.")
    print("=" * 70)
    print()

    if not FINAL_TEST_DIR.exists():
        print(f"ERROR: {FINAL_TEST_DIR} does not exist. No Phase 47 evidence to archive.")
        return 1

    records: list[dict | None] = []
    for rel_path, status, note in ARCHIVE_TARGETS:
        print(f"  Archiving {rel_path} ({status})...")
        try:
            rec = archive_one(rel_path, status, note)
            if rec is None:
                print(f"    SKIP (source absent)")
            else:
                print(f"    OK (sha={rec['source_sha256'][:16]}...)")
            records.append(rec)
        except RuntimeError as e:
            print(f"    ROLLBACK: {e}")
            return 2

    manifest_path = write_archive_manifest(records)
    print()
    print(f"Archive manifest written: {manifest_path}")
    print()
    print(f"Files archived: {len([r for r in records if r is not None])}")
    print(f"Files skipped (absent): {len([r for r in records if r is None])}")
    print()
    print("Archive complete. Prior Phase 47 evidence preserved.")
    print("Corrective Phase 47 Test re-evaluation may now proceed (separate human approval required).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
