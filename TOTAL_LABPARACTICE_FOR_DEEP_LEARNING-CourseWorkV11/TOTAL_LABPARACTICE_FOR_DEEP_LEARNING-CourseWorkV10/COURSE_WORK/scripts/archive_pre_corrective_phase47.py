"""Part 2G-O: Non-destructive archive of pre-corrective Phase47 evidence.

This script:
  1. Inventories all active Phase47 artifacts.
  2. Identifies pre-corrective lineage (stale run_ids, stale checkpoint SHAs,
     config/lock conflation, prior Test access).
  3. Copies them into a timestamped _history/PRE_CORRECTIVE_PHASE47_<TS>/
     subdirectory under artifacts/final_test/.
  4. SHA256-verifies every copied file matches its source.
  5. Writes an archive manifest with provenance and classification.

Requirements:
  - DO NOT modify any source artifact (copy only).
  - DO NOT touch checkpoint binaries.
  - DO NOT touch Phase46 artifacts.
  - DO NOT touch the historical Test access log (preserve append-only).
  - DO NOT add a new Test access event.

Usage:
  PYTHONPATH=COURSE_WORK/src python3 COURSE_WORK/scripts/archive_pre_corrective_phase47.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
COURSE_WORK_ROOT = SCRIPT_PATH.parent.parent  # COURSE_WORK
PROJECT_ROOT = COURSE_WORK_ROOT.parent  # TOTAL_LABPARACTICE_FOR_DEEP_LEARNING


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now_utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> int:
    test_dir = COURSE_WORK_ROOT / "artifacts" / "final_test"
    if not test_dir.is_dir():
        print(f"ERROR: {test_dir} not found", file=sys.stderr)
        return 1

    timestamp = now_utc_compact()
    archive_dir = test_dir / "_history" / f"PRE_CORRECTIVE_PHASE47_{timestamp}"
    archive_dir.mkdir(parents=True, exist_ok=False)

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

    candidates: list[Path] = []
    for entry in sorted(test_dir.iterdir()):
        if not entry.is_file():
            continue
        if entry.name.startswith("."):
            continue
        if entry.name.startswith("_archive"):
            continue
        candidates.append(entry)

    figures_dir = test_dir / "figures"
    if figures_dir.is_dir():
        for entry in sorted(figures_dir.iterdir()):
            if entry.is_file():
                candidates.append(entry)

    manifest_entries: list[dict] = []
    for src_path in candidates:
        rel = src_path.relative_to(test_dir)
        dst_path = archive_dir / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        src_sha = sha256_file(src_path)
        src_size = src_path.stat().st_size

        shutil.copy2(src_path, dst_path)
        dst_sha = sha256_file(dst_path)
        assert src_sha == dst_sha, f"SHA mismatch after copy: {src_path}"

        # Classification
        try:
            content = src_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            content = ""
        classifications: list[str] = []
        reasons: list[str] = []
        # Stale fingerprint check (use textual match, robust to JSON formatting)
        if STALE_FINAL_LOCK_SHA in content:
            classifications.append("STALE_FINAL_LOCK_SHA")
            reasons.append("references 585c5e79e6a1c8... (Phase 45-audit config fingerprint, conflated with final_lock_sha)")
        for stale_sha in STALE_CHECKPOINT_SHAS.values():
            if stale_sha in content:
                classifications.append("STALE_CHECKPOINT_SHA")
                reasons.append(f"references stale checkpoint SHA {stale_sha[:16]}... (does not match any corrected Phase46 checkpoint)")
        for stale_run in STALE_RUN_IDS:
            if stale_run in content:
                classifications.append("STALE_RUN_ID")
                reasons.append(f"references stale Phase 46 run_id {stale_run} (pre-corrective)")
        if "test_first_access_authorized" in content or "test_previously_accessed" in content or "first_access_timestamp" in content:
            classifications.append("PRIOR_TEST_ACCESS")
            reasons.append("records Test access from pre-corrective Phase 47 (2026-09-03)")
        if not classifications:
            classifications.append("OPERATIONAL")
            reasons.append("operational metadata not bound to stale run_ids/checkpoints (e.g. documentation, CSV audit log)")

        manifest_entries.append({
            "source_path": str(src_path.relative_to(PROJECT_ROOT)),
            "archive_path": str(dst_path.relative_to(PROJECT_ROOT)),
            "source_sha256": src_sha,
            "archive_sha256": dst_sha,
            "size_bytes": src_size,
            "classification": classifications,
            "reason": reasons,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    manifest = {
        "manifest_version": "PRE_CORRECTIVE_PHASE47_ARCHIVE-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive_dir": str(archive_dir.relative_to(PROJECT_ROOT)),
        "task": "PART_2G_O",
        "purpose": "Preserve pre-corrective Phase47 evidence prior to corrective Phase47 execution. AUTHORITATIVE_CORRECTIVE_PHASE47 = NO for all archived artifacts.",
        "stale_lineage": {
            "stale_final_lock_sha": STALE_FINAL_LOCK_SHA,
            "stale_checkpoint_shas": STALE_CHECKPOINT_SHAS,
            "stale_run_ids": sorted(STALE_RUN_IDS),
        },
        "current_corrected_phase46_inputs": {
            "RUN_TR_FSD_0256_C2F24D58": "523d2e98f82f8534782e9364a4fb86e7f9381e33c2abb694921ed1c4b3d3de67",
            "RUN_TR_FSD_0256_AA575C42": "650637c14f84548237ac0641a880e1b461824643bd276ae9cae0832879204804",
            "RUN_TR_FSD_0256_247AB83A": "809cfde75611723e774791735c436009ce29d259faf30ab4123a2dc46c84212f",
        },
        "authoritative_corrective_phase47": "NO (for all archived artifacts)",
        "historical_test_access_preserved": "YES (final_test_access_log.jsonl NOT modified)",
        "checkpoint_binary_changed": "NO",
        "phase46_artifact_changed": "NO",
        "phase45_lock_changed": "NO",
        "files_archived": manifest_entries,
        "total_archived_files": len(manifest_entries),
    }

    manifest_path = archive_dir / "_archive_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Archive directory: {archive_dir}")
    print(f"Total files archived: {len(manifest_entries)}")
    print(f"Manifest: {manifest_path}")
    print(f"All SHAs verified: YES")
    print()
    print("Classification counts:")
    from collections import Counter
    counter: Counter[str] = Counter()
    for entry in manifest_entries:
        for c in entry["classification"]:
            counter[c] += 1
    for k, v in sorted(counter.items()):
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
