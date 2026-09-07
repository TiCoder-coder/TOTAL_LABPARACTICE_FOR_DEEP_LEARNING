#!/usr/bin/env python3
"""Phase 46 — SAFE Preservation of Stale Evidence (No Overwrite).

ARCHIVES the CURRENT (INVALIDATED) Phase 46 evidence to
artifacts/three_seed_final_runs/_history/PHASE46_PRE_CORRECTIVE_<UTC>/

BEFORE the corrective Phase 46 training can begin.

This script is INTENTIONALLY DIFFERENT from
`p46_archive_historical_checkpoints.py`:

  - The approved archive script (`p46_archive_historical_checkpoints.py`) was
    written assuming historical checkpoint .pt binaries exist at the active
    canonical paths. The Part 2C audit confirmed they are ABSENT in this
    repository (only metadata sidecars are present, with the documented
    `average_rmse_wh: 0.0` and the historical run_ids RUN_TR_FSD_0153/0154/0155).

  - Per the Part 2G user instruction:
      "Do not overwrite corrected/canonical scientific results with fake placeholders"
      "Do not delete historical evidence"
      "On ANY mismatch: STOP and rollback preservation changes"

    Writing fake placeholder .pt binaries or replacing the metadata sidecars
    would violate these instructions. So this script ONLY:

      1. Copies the stale evidence files (manifests, signoff, metadata sidecars)
         to a _history/ archive directory.
      2. Computes SHA256 of each source file (the documented historical SHA).
      3. Computes SHA256 of each archived copy.
      4. Requires exact SHA equality. On mismatch: rollback, STOP.
      5. Records the absence of .pt binaries (does NOT generate placeholder files).
      6. Writes an audit manifest.

  - The active canonical paths (`official_checkpoints/seed_*/seed_*_FINAL_REFIT*`)
    are LEFT UNTOUCHED. Future corrected Phase 46 training will overwrite
    these paths when it runs.

  - The plan §5.2 list of files is fully preserved here.
  - The plan §5.2 requires actual `.pt` checkpoint binaries. Since they are
    absent from the active directory, this archive records that absence
    with `checkpoint_binary: ABSENT` in the manifest. They cannot be
    fabricated or fabricated placeholders.

  - The historical run_ids (RUN_TR_FSD_0153/0154/0155) remain excluded from
    any future Phase 47 release per the approved plan §4 G15.

NO Test access. NO training. NO inference. NO correction.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _resolve_root() -> Path:
    p = ROOT
    for _ in range(8):
        if (p / "artifacts").exists() and (p / "src").exists():
            return p
        p = p.parent
    return ROOT


PROJECT_ROOT = _resolve_root()
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "three_seed_final_runs"
OFFICIAL_CHECKPOINTS_DIR = ARTIFACT_DIR / "official_checkpoints"
HISTORY_DIR = ARTIFACT_DIR / "_history"

UTC_SUFFIX = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
ARCHIVE_ROOT = HISTORY_DIR / f"PHASE46_PRE_CORRECTIVE_{UTC_SUFFIX}"

# Files to preserve (per approved plan §5.2). Each entry:
#   (relative_path_under_artifact_dir, archive_status, note)
# These are the stale evidence files; their content remains unchanged.
ARCHIVE_TARGETS = [
    ("phase_46_signoff.json", "stale-TR_C0_PRIMARY-50ep-rmse0.0",
     "STALE (TR_C0_PRIMARY / 50 epochs / average_rmse_wh=0.0); preserved, NOT deleted."),
    ("three_seed_manifest.json", "stale",
     "STALE; preserved."),
    ("three_seed_contract.json", "stale",
     "STALE; preserved."),
    ("three_seed_final_runs_summary.json", "real-per-seed-rmse",
     "Real per-seed rmse 41.71 / 39.62 / 39.89; preserved."),
    ("final_lock_verification.json", "stale-a711a9b8-universe",
     "STALE (verifies a711a9b8... universe); preserved."),
    ("phase46_reuse_cache.json", "placeholder-rmse",
     "Placeholder rmse; preserved."),
    ("phase47_final_test_evaluation_handoff.json", "stale",
     "STALE; preserved."),
    ("final_dev_population_manifest.json", "stale",
     "STALE; preserved."),
    ("phase_46_corrections_log.json", "log",
     "Corrections log; preserved."),
    ("phase_46_recovery_notes.json", "notes",
     "Recovery notes; preserved."),
    ("phase_46_active_recovery_record.json", "record",
     "Active recovery record; preserved."),
]

# Per-seed metadata sidecars (always present even when .pt is absent).
PER_SEED_METADATA = [
    ("seed_42/seed_42_FINAL_REFIT_metadata.json", 42, "RUN_TR_FSD_0153_B15A19DC"),
    ("seed_123/seed_123_FINAL_REFIT_metadata.json", 123, "RUN_TR_FSD_0154_DD82D743"),
    ("seed_2026/seed_2026_FINAL_REFIT_metadata.json", 2026, "RUN_TR_FSD_0155_59A50ADD"),
]

PER_SEED_PT = [
    ("seed_42/seed_42_FINAL_REFIT.pt", 42, "RUN_TR_FSD_0153_B15A19DC"),
    ("seed_123/seed_123_FINAL_REFIT.pt", 123, "RUN_TR_FSD_0154_DD82D743"),
    ("seed_2026/seed_2026_FINAL_REFIT.pt", 2026, "RUN_TR_FSD_0155_59A50ADD"),
]

EXCLUDED_HISTORICAL_RUN_IDS = {
    "RUN_TR_FSD_0153_B15A19DC",
    "RUN_TR_FSD_0154_DD82D743",
    "RUN_TR_FSD_0155_59A50ADD",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_bytes(obj) -> bytes:
    return json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")


def archive_one(relative_path: str, status: str, note: str) -> dict | None:
    """Archive a single file safely. Returns None if source absent (informational skip).

    On any SHA mismatch between source and archived copy: removes the archived
    copy and raises RuntimeError to trigger rollback.
    """
    source = ARTIFACT_DIR / relative_path
    if not source.exists():
        return None

    dest = ARCHIVE_ROOT / relative_path
    dest.parent.mkdir(parents=True, exist_ok=True)

    source_sha = _sha256_file(source)

    # Copy (NOT move).
    shutil.copy2(source, dest)

    archived_sha = _sha256_file(dest)
    if archived_sha != source_sha:
        # ROLLBACK: remove the bad archive copy.
        if dest.exists():
            dest.unlink()
        raise RuntimeError(
            f"Archive SHA mismatch for {relative_path}: "
            f"source={source_sha} archived={archived_sha}. ROLLBACK executed."
        )

    return {
        "path": relative_path,
        "source_sha256": source_sha,
        "archived_sha256": archived_sha,
        "source_size_bytes": source.stat().st_size,
        "status": status,
        "note": note,
        "archived_at": _utc_now(),
    }


def archive_per_seed_metadata() -> list[dict]:
    records: list[dict] = []
    for rel, seed, run_id in PER_SEED_METADATA:
        rec = archive_one(
            f"official_checkpoints/{rel}",
            "metadata-sidecar-historical-invalid",
            f"Stale historical metadata for {run_id} (seed={seed}); "
            f"declared model_state_sha256 is recorded; .pt binary status recorded separately.",
        )
        if rec is not None:
            # Cross-reference the declared model_state_sha256 from the metadata.
            meta = json.loads((ARTIFACT_DIR / "official_checkpoints" / rel).read_bytes())
            rec["declared_model_state_sha256"] = meta.get("model_state_sha256")
            rec["declared_rmse_wh"] = meta.get("rmse_wh")
            rec["declared_run_id"] = meta.get("run_id")
        records.append(rec)
    return records


def audit_per_seed_pt_presence() -> list[dict]:
    """Record whether each historical .pt binary is present at the active canonical path.

    DOES NOT copy .pt files (they are absent). DOES NOT write placeholder .pt files.
    This is a presence audit only.
    """
    out: list[dict] = []
    for rel, seed, run_id in PER_SEED_PT:
        path = OFFICIAL_CHECKPOINTS_DIR / rel
        meta_path = OFFICIAL_CHECKPOINTS_DIR / rel.replace(".pt", "_metadata.json")
        declared_sha = None
        if meta_path.exists():
            meta = json.loads(meta_path.read_bytes())
            declared_sha = meta.get("model_state_sha256")
        record = {
            "seed": seed,
            "historical_run_id": run_id,
            "canonical_path": str(path.relative_to(PROJECT_ROOT)),
            "checkpoint_binary_present": path.exists(),
            "checkpoint_binary_status": "PRESENT" if path.exists() else "ABSENT",
            "declared_model_state_sha256": declared_sha,
            "actual_file_sha256": _sha256_file(path) if path.exists() else None,
        }
        if path.exists() and declared_sha is not None:
            record["declared_vs_actual_sha_match"] = (
                _sha256_file(path) == declared_sha
            )
        out.append(record)
    return out


def write_archive_manifest(
    target_records: list[dict | None],
    metadata_records: list[dict | None],
    pt_audit: list[dict],
) -> Path:
    n_files = sum(1 for r in target_records if r is not None)
    n_meta = sum(1 for r in metadata_records if r is not None)
    n_absent = sum(1 for r in target_records if r is None)
    n_pt_present = sum(1 for r in pt_audit if r["checkpoint_binary_present"])
    n_pt_absent = sum(1 for r in pt_audit if not r["checkpoint_binary_present"])

    manifest = {
        "archive_version": "PHASE46_PRE_CORRECTIVE_SAFE_PRESERVATION-v1",
        "phase": 46,
        "prior_phase46_status": "INVALIDATED_BY_CONTRACT_DEVIATION",
        "archived_at": _utc_now(),
        "archive_root": str(ARCHIVE_ROOT.relative_to(PROJECT_ROOT)),
        "utc_suffix": UTC_SUFFIX,
        "test_access": False,
        "scientific_training": False,
        "phase47_execution": False,
        "phase47_release_reference": "NEVER",
        "active_canonical_paths_modified": False,
        "placeholder_files_written": False,
        "scientific_test_artifacts_touched": False,
        "approval_status": "AWAITING_HUMAN_APPROVAL_FOR_CORRECTIVE_EXECUTION",
        "invalidation_reasons": [
            "Stale TR_C0_PRIMARY candidate with lookback 36 instead of locked TR_C2_ALT_LOOKBACK L72",
            "Stale 50-epoch final_refit_epochs instead of locked 30",
            "Stale average_rmse_wh = 0.0 for all three seeds (documented bug)",
            "Missing actual .pt checkpoint binaries at canonical paths",
            "Missing phase47_test_release.json",
            "config_fingerprint mismatch across historical runs (per-seed values differ)",
        ],
        "excluded_historical_run_ids": sorted(EXCLUDED_HISTORICAL_RUN_IDS),
        "counts": {
            "files_archived": n_files + n_meta,
            "files_skipped_absent": n_absent,
            "checkpoint_binaries_present": n_pt_present,
            "checkpoint_binaries_absent": n_pt_absent,
        },
        "plan_section_reference": "docs/plan/plan_before_process/phase_46_corrective_post_audit_reimplementation_plan.md §5",
        "target_files_archived": [r for r in target_records if r is not None],
        "per_seed_metadata_archived": [r for r in metadata_records if r is not None],
        "checkpoint_binary_presence_audit": pt_audit,
        "notes": [
            "Active canonical paths under official_checkpoints/ were NOT modified.",
            "No placeholder .pt files were written.",
            "Future corrected Phase 46 training will overwrite the canonical paths with fresh FINAL_REFIT checkpoints.",
            "Historical run_ids remain EXCLUDED from any future phase47_test_release.json (per plan §4 G15).",
        ],
    }
    manifest_path = ARCHIVE_ROOT / "_archive_manifest.json"
    manifest_path.write_bytes(_canonical_json_bytes(manifest))
    return manifest_path


def write_archive_audit_csv(
    target_records: list[dict | None],
    metadata_records: list[dict | None],
    pt_audit: list[dict],
) -> Path:
    csv_path = ARCHIVE_ROOT / "_archive_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "kind", "path", "source_sha256", "archived_sha256",
            "source_size_bytes", "status", "note",
        ])
        for r in target_records:
            if r is None:
                continue
            writer.writerow([
                "file", r["path"], r["source_sha256"], r["archived_sha256"],
                r["source_size_bytes"], r["status"], r["note"],
            ])
        for r in metadata_records:
            if r is None:
                continue
            writer.writerow([
                "per_seed_metadata", r["path"], r["source_sha256"], r["archived_sha256"],
                r["source_size_bytes"], r["status"], r["note"],
            ])
        for r in pt_audit:
            writer.writerow([
                "checkpoint_binary_audit",
                r["canonical_path"],
                r["actual_file_sha256"] or "ABSENT",
                "N/A",
                "N/A",
                r["checkpoint_binary_status"],
                f"declared={r['declared_model_state_sha256']}",
            ])
    return csv_path


def write_run_id_exclusion_marker() -> Path | None:
    """Add an additive (non-destructive) marker noting the historical run_ids
    are excluded from any future Phase 47 release. Does NOT modify any
    existing run dir file."""
    marker_path = ARTIFACT_DIR / "historical_run_ids_excluded.json"
    if marker_path.exists():
        # Append-only semantics: do NOT overwrite if it already exists.
        return None
    payload = {
        "marker_version": "PHASE46_HISTORICAL_RUN_IDS_EXCLUDED-v1",
        "phase": 46,
        "excluded_run_ids": sorted(EXCLUDED_HISTORICAL_RUN_IDS),
        "reason": "These three run_ids are EXCLUDED from any future phase47_test_release.json per plan §4 G15.",
        "created_at": _utc_now(),
        "test_access": False,
        "scientific_training": False,
    }
    marker_path.write_bytes(_canonical_json_bytes(payload))
    return marker_path


def main() -> int:
    print("=" * 70)
    print("PHASE 46 — SAFE PRE-CORRECTIVE EVIDENCE PRESERVATION")
    print("(no overwrite; no fake placeholders)")
    print("=" * 70)
    print(f"Archive destination: {ARCHIVE_ROOT}")
    print()

    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)

    # Phase 1: archive target files (safe copy).
    print("Phase 1: Archiving target files...")
    target_records: list[dict | None] = []
    for rel, status, note in ARCHIVE_TARGETS:
        try:
            rec = archive_one(rel, status, note)
            if rec is None:
                print(f"  SKIP (absent): {rel}")
            else:
                print(f"  OK     sha={rec['source_sha256'][:16]}... {rel}")
            target_records.append(rec)
        except RuntimeError as exc:
            print(f"  ROLLBACK: {exc}")
            return 2

    # Phase 2: archive per-seed metadata sidecars.
    print()
    print("Phase 2: Archiving per-seed metadata sidecars...")
    metadata_records = archive_per_seed_metadata()
    for rec in metadata_records:
        if rec is None:
            print("  SKIP (absent)")
        else:
            print(
                f"  OK     sha={rec['source_sha256'][:16]}... "
                f"{rec['path']} (declared model_state_sha256="
                f"{rec['declared_model_state_sha256'][:16] if rec.get('declared_model_state_sha256') else None}...)"
            )

    # Phase 3: checkpoint binary presence audit (does NOT copy or fabricate).
    print()
    print("Phase 3: Checkpoint binary presence audit (no copy, no placeholders)...")
    pt_audit = audit_per_seed_pt_presence()
    for r in pt_audit:
        if r["checkpoint_binary_present"]:
            print(f"  PRESENT sha={r['actual_file_sha256'][:16]}... {r['canonical_path']}")
        else:
            print(f"  ABSENT  (declared model_state_sha256="
                  f"{r['declared_model_state_sha256'][:16] if r['declared_model_state_sha256'] else None}...) "
                  f"{r['canonical_path']}")

    # Phase 4: write archive manifest + audit CSV.
    print()
    print("Phase 4: Writing archive manifest + audit CSV...")
    manifest_path = write_archive_manifest(target_records, metadata_records, pt_audit)
    csv_path = write_archive_audit_csv(target_records, metadata_records, pt_audit)
    print(f"  manifest: {manifest_path}")
    print(f"  csv:      {csv_path}")

    # Phase 5: additive run-id exclusion marker (no destructive mutation).
    print()
    print("Phase 5: Writing additive run-id exclusion marker...")
    marker_path = write_run_id_exclusion_marker()
    if marker_path:
        print(f"  marker: {marker_path}")
    else:
        print(f"  marker: already present (not overwritten)")

    # Summary
    print()
    print("=" * 70)
    print("PRESERVATION SUMMARY")
    print("=" * 70)
    print(f"Target files archived:      {sum(1 for r in target_records if r is not None)}")
    print(f"Per-seed metadata archived: {sum(1 for r in metadata_records if r is not None)}")
    print(f"Checkpoint binaries present: {sum(1 for r in pt_audit if r['checkpoint_binary_present'])}")
    print(f"Checkpoint binaries absent:  {sum(1 for r in pt_audit if not r['checkpoint_binary_present'])}")
    print(f"Active canonical paths modified: NO")
    print(f"Placeholder .pt files written:   NO")
    print(f"Test access:                     NO")
    print(f"Scientific training:             NO")
    print(f"Phase 47 execution:              NO")
    print()
    print("Archive PASS. Prior Phase 46 evidence is preserved verbatim.")
    print("Corrective Phase 46 training may now proceed (separate human approval required).")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
