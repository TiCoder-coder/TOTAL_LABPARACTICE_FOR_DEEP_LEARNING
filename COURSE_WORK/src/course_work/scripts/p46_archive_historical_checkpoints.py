"""Phase 46 historical checkpoint archival.

Safely archives the three INVALIDATED historical Phase 46 checkpoints
under the canonical official checkpoint paths so that future corrected
Phase 46 scientific training can write fresh FINAL_REFIT checkpoints
to the same paths.

Procedure (safe by construction):

  For each of the three historical runs:
    1. Compute SHA256 of the source .pt file.
    2. Read the source metadata sidecar.
    3. Copy (not move) the .pt and metadata.json into
       artifacts/three_seed_final_runs/historical_checkpoints/<run_id>/
    4. Compute SHA256 of the archived .pt file.
    5. Require source_sha == archived_sha == metadata.model_state_sha256.
    6. Stop on any mismatch.

  Only after ALL three checkpoints are checksum-verified in the archive:
    - Write the archive manifest and audit CSV.
    - Write additive historical_phase46_invalidation_manifest.json into each
      historical run directory (no destructive mutation of the run dir).
    - Replace the canonical active .pt file with a placeholder metadata.json
      that declares status = AWAITING_CORRECTED_PHASE46_RUN. The .pt file is
      REMOVED only after archive verification PASSES.

  This script performs NO scientific training and accesses NO Test data.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from course_work.utils.artifacts import canonical_json_bytes, sha256_file

ARTIFACT_DIR = ROOT / "artifacts" / "three_seed_final_runs"
HISTORICAL_CHECKPOINTS_DIR = ARTIFACT_DIR / "historical_checkpoints"
OFFICIAL_CHECKPOINTS_DIR = ARTIFACT_DIR / "official_checkpoints"

HISTORICAL_RUNS = [
    {
        "seed": 42,
        "run_id": "RUN_TR_FSD_0153_B15A19DC",
        "checkpoint_filename": "seed_42_FINAL_REFIT.pt",
        "metadata_filename": "seed_42_FINAL_REFIT_metadata.json",
    },
    {
        "seed": 123,
        "run_id": "RUN_TR_FSD_0154_DD82D743",
        "checkpoint_filename": "seed_123_FINAL_REFIT.pt",
        "metadata_filename": "seed_123_FINAL_REFIT_metadata.json",
    },
    {
        "seed": 2026,
        "run_id": "RUN_TR_FSD_0155_59A50ADD",
        "checkpoint_filename": "seed_2026_FINAL_REFIT.pt",
        "metadata_filename": "seed_2026_FINAL_REFIT_metadata.json",
    },
]

PHASE_46_INVALIDATION_REASONS = [
    "TRAIN-only bug: Phase 46 used datasets_dict['TRAIN'] instead of FINAL_DEV_REGION-v1 = TRAIN+VALIDATION",
    "Phase 9 scaler reuse bug: official Phase 46 X/Y scalers were TRAIN-only Phase 9 scalers, not FINAL_SCALING-v1",
    "Stale RMSE metadata: best_validation_rmse_wh was 0.0 for all three historical runs",
    "best_epoch metadata issue: official_epoch did not equal FINAL_REFIT_EPOCHS=50",
    "Missing phase47_test_release.json: the Phase 47 release artifact was not generated",
    "Historical runs are PRESERVED as evidence but cannot satisfy any corrected Phase 47 release gate",
]

EXCLUDED_HISTORICAL_RUN_IDS = [run["run_id"] for run in HISTORICAL_RUNS]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict) -> None:
    path.write_bytes(canonical_json_bytes(data))


def sha256_str(sha_bytes_or_path) -> str:
    return sha256_file(sha_bytes_or_path)


def archive_one(run: dict) -> dict:
    """Archive one historical checkpoint safely.

    Returns a dict describing the archived run, or raises on any SHA mismatch.
    """
    seed = run["seed"]
    run_id = run["run_id"]
    ckpt_filename = run["checkpoint_filename"]
    meta_filename = run["metadata_filename"]

    seed_dir_name = f"seed_{seed}"
    source_ckpt = OFFICIAL_CHECKPOINTS_DIR / seed_dir_name / ckpt_filename
    source_meta = OFFICIAL_CHECKPOINTS_DIR / seed_dir_name / meta_filename

    if not source_ckpt.exists():
        raise FileNotFoundError(f"Source checkpoint missing: {source_ckpt}")
    if not source_meta.exists():
        raise FileNotFoundError(f"Source metadata missing: {source_meta}")

    source_sha = sha256_file(source_ckpt)
    source_meta_obj = json.loads(source_meta.read_bytes())
    source_meta_sha = source_meta_obj.get("model_state_sha256")
    if source_meta_sha != source_sha:
        raise RuntimeError(
            f"Source metadata model_state_sha256 {source_meta_sha} does not "
            f"match source file SHA {source_sha} for {run_id}"
        )

    dest_dir = HISTORICAL_CHECKPOINTS_DIR / run_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_ckpt = dest_dir / ckpt_filename
    dest_meta = dest_dir / meta_filename

    shutil.copy2(source_ckpt, dest_ckpt)
    shutil.copy2(source_meta, dest_meta)

    archived_sha = sha256_file(dest_ckpt)
    if archived_sha != source_sha:
        raise RuntimeError(
            f"Archived SHA {archived_sha} != source SHA {source_sha} for {run_id}"
        )

    return {
        "seed": seed,
        "historical_run_id": run_id,
        "original_path": str(source_ckpt.relative_to(ROOT)),
        "archived_path": str(dest_ckpt.relative_to(ROOT)),
        "metadata_path": str(dest_meta.relative_to(ROOT)),
        "checkpoint_sha256": archived_sha,
        "metadata_run_id": source_meta_obj.get("run_id"),
        "metadata_seed": source_meta_obj.get("seed"),
        "metadata_phase": source_meta_obj.get("phase"),
        "metadata_checkpoint_type": source_meta_obj.get("checkpoint_type"),
        "original_created_at": source_meta_obj.get("created_at"),
        "archived_at": now_iso(),
        "status": "PASS",
    }


def write_archive_manifest(records: list[dict]) -> dict:
    manifest = {
        "archive_version": "HISTORICAL_CHECKPOINT_ARCHIVAL-v1",
        "phase": 46,
        "historical_status": "INVALIDATED_BY_CONTRACT_DEVIATION",
        "archive_root": "artifacts/three_seed_final_runs/historical_checkpoints",
        "runs": records,
        "excluded_run_ids": EXCLUDED_HISTORICAL_RUN_IDS,
        "invalidation_reasons": PHASE_46_INVALIDATION_REASONS,
        "archive_verified": True,
        "test_access": False,
        "scientific_training": False,
        "phase47_release_reference": "NEVER",
        "created_at": now_iso(),
        "status": "PASS",
    }
    path = HISTORICAL_CHECKPOINTS_DIR / "historical_checkpoint_archive_manifest.json"
    write_json(path, manifest)
    return manifest


def write_archive_audit(records: list[dict]) -> None:
    rows = []
    for r in records:
        rows.append([
            r["seed"],
            r["historical_run_id"],
            r["original_path"],
            r["archived_path"],
            r["checkpoint_sha256"],
            r["metadata_run_id"],
            r["metadata_checkpoint_type"],
            r["original_created_at"],
            r["archived_at"],
            r["status"],
        ])
    path = HISTORICAL_CHECKPOINTS_DIR / "historical_checkpoint_archive_audit.csv"
    lines = [
        "seed,historical_run_id,original_path,archived_path,checkpoint_sha256,metadata_run_id,metadata_checkpoint_type,original_created_at,archived_at,status",
    ]
    for row in rows:
        lines.append(",".join(f"\"{v}\"" if isinstance(v, str) and ("\"" in v or "," in v) else str(v) for v in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_invalidation_markers(records: list[dict]) -> None:
    """Add additive invalidation markers to historical run directories.

    Does NOT delete or mutate any existing run dir file.
    """
    for r in records:
        run_dir = ROOT / "artifacts" / "runs" / r["historical_run_id"]
        if not run_dir.exists():
            continue
        marker = {
            "artifact_version": "HISTORICAL_PHASE46_INVALIDATION-v1",
            "phase": 46,
            "historical_run_id": r["historical_run_id"],
            "historical_status": "INVALIDATED_BY_CONTRACT_DEVIATION",
            "invalidation_reasons": PHASE_46_INVALIDATION_REASONS,
            "archived_checkpoint_path": r["archived_path"],
            "checkpoint_sha256_preserved": r["checkpoint_sha256"],
            "test_access": False,
            "scientific_retraining": False,
            "phase47_release_excluded": True,
            "created_at": now_iso(),
        }
        marker_path = run_dir / "historical_phase46_invalidation_manifest.json"
        write_json(marker_path, marker)


def clear_active_canonical_path(run: dict) -> None:
    """Remove the stale .pt from the active canonical path after archive.

    Writes a placeholder metadata.json declaring AWAITING_CORRECTED_PHASE46_RUN.
    The .pt is removed ONLY if archive verification passed for this run.
    """
    seed = run["seed"]
    ckpt_filename = run["checkpoint_filename"]
    meta_filename = run["metadata_filename"]
    seed_dir = OFFICIAL_CHECKPOINTS_DIR / f"seed_{seed}"
    active_ckpt = seed_dir / ckpt_filename
    active_meta = seed_dir / meta_filename

    # Remove the stale .pt file (archived copy is verified).
    if active_ckpt.exists():
        active_ckpt.unlink()

    placeholder = {
        "checkpoint_path": f"artifacts/three_seed_final_runs/official_checkpoints/seed_{seed}/{ckpt_filename}",
        "checkpoint_type": "AWAITING_CORRECTED_PHASE46_RUN",
        "phase": 46,
        "seed": seed,
        "status": "AWAITING_CORRECTED_PHASE46_RUN",
        "historical_run_id": run["run_id"],
        "archived_to": f"artifacts/three_seed_final_runs/historical_checkpoints/{run['run_id']}/{ckpt_filename}",
        "note": "Stale INVALIDATED checkpoint archived. Future corrected Phase 46 scientific training will write a new FINAL_REFIT checkpoint to this path.",
        "created_at": now_iso(),
    }
    write_json(active_meta, placeholder)


def main() -> int:
    print("=" * 70)
    print("Phase 46 Historical Checkpoint Archival")
    print("=" * 70)

    HISTORICAL_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: archive every run safely.
    records: list[dict] = []
    for run in HISTORICAL_RUNS:
        print(f"[ARCHIVE] Seed {run['seed']}: archiving {run['run_id']}")
        record = archive_one(run)
        records.append(record)
        print(f"           source SHA = {record['checkpoint_sha256'][:16]}...")
        print(f"           archived   = {record['archived_path']}")

    # Phase 2: write archive manifest and audit CSV.
    write_archive_manifest(records)
    write_archive_audit(records)
    print(f"[ARCHIVE] Manifest: {HISTORICAL_CHECKPOINTS_DIR / 'historical_checkpoint_archive_manifest.json'}")
    print(f"[ARCHIVE] Audit CSV: {HISTORICAL_CHECKPOINTS_DIR / 'historical_checkpoint_archive_audit.csv'}")

    # Phase 3: additive invalidation markers in historical run dirs.
    write_invalidation_markers(records)
    print(f"[ARCHIVE] Invalidation markers written to historical run dirs.")

    # Phase 4: clear stale .pt from active canonical path (after archive verified).
    for run in HISTORICAL_RUNS:
        clear_active_canonical_path(run)
    print(f"[ARCHIVE] Active official .pt cleared, placeholder metadata written.")

    print("\n[ARCHIVE] COMPLETE")
    print(f"[ARCHIVE] Status: PASS")
    print(f"[ARCHIVE] Test access: NO")
    print(f"[ARCHIVE] Scientific training: NO")
    print(f"[ARCHIVE] Excluded historical run IDs from Phase 47 release:")
    for rid in EXCLUDED_HISTORICAL_RUN_IDS:
        print(f"           {rid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
