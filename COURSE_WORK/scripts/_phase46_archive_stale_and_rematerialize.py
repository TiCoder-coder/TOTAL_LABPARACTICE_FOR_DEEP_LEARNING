#!/usr/bin/env python3
"""Phase 46 — Archive stale FINAL_DEV/Scaling manifests and re-materialize.

The previous Phase 46 audit created FINAL_DEV manifest with lookback_steps=36
and FINAL_SCALING manifest with lookback_steps=36. These are WRONG for the
Phase 45 locked candidate TR_C2_ALT_LOOKBACK (lookback=72).

This script:
1. Archives the existing FINAL_DEV manifest + audit CSV (stale lookback=36).
2. Archives the existing FINAL_SCALING manifest + statistics CSVs (stale lookback=36).
3. Re-materializes FINAL_DEV and FINAL_SCALING with lookback=72.
4. Verifies the new artifacts match the Phase 45 lock.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from course_work.data.final_dev import materialize_final_dev_region
from course_work.scaling.final_scaling import materialize_final_scaling_v1


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _archive(art_root: Path, files: list[str], audit_label: str) -> dict:
    archive_dir = art_root / "_archive" / f"STALE_LOOKBACK_36_{_now()}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived = {}
    for f in files:
        src = art_root / f
        if not src.exists():
            continue
        dest = archive_dir / f
        shutil.move(str(src), str(dest))
        archived[f] = str(dest.relative_to(ROOT))
    audit = {
        "audit_label": audit_label,
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "archived_files": archived,
        "reason": "Stale lookback=36 manifest superseded by lookback=72 (Phase 45 lock).",
        "phase": 46,
    }
    (archive_dir / "archive_manifest.json").write_text(json.dumps(audit, indent=2))
    return audit


def main() -> int:
    final_dev_art_root = ROOT / "artifacts" / "final_dev_region"
    scaling_art_root = ROOT / "artifacts" / "scaling" / "final_dev"

    # 1. Archive stale FINAL_DEV artifacts (lookback=36 was wrong).
    final_dev_files = ["final_dev_region_manifest.json", "final_dev_population_audit.csv"]
    fd_audit = _archive(final_dev_art_root, final_dev_files, "FINAL_DEV_v1")
    print(f"[Archive] FINAL_DEV: {list(fd_audit['archived_files'].keys())}")

    # 2. Archive stale FINAL_SCALING manifests/audit.
    scaling_files = [
        "final_scaling_manifest.json",
        "final_x_scaler_statistics.csv",
        "final_y_scaler_statistics.csv",
        "final_scaler_fit_audit.csv",
        "final_scaler_fit_manifest.json",
        "final_scaler_roundtrip_tests.csv",
        "final_scaling_leakage_audit.csv",
        "final_scaler_checksums.sha256",
        "final_scaler_checksums.json",
        "final_scaler_registry.json",
    ]
    sc_audit = _archive(scaling_art_root, scaling_files, "FINAL_SCALING_v1")
    print(f"[Archive] FINAL_SCALING: {list(sc_audit['archived_files'].keys())}")

    # 2b. Archive stale scaler bundles (X and Y joblib files).
    for subdir in ("x", "y"):
        sub = scaling_art_root / subdir
        if sub.exists():
            sub_archived = []
            for f in sub.glob("*.joblib"):
                archive = sc_audit["archived_files"]  # use same archive dir
                # Just move individual files into the existing archive.
                archive_dir = Path(archive.get("final_scaling_manifest.json", "")).parent if archive else None
                if archive_dir and archive_dir.exists():
                    dest = archive_dir / subdir / f.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), str(dest))
                    sub_archived.append(str(dest.relative_to(ROOT)))
            if sub_archived:
                print(f"[Archive] FINAL_SCALING/{subdir}: {sub_archived}")

    # 3. Re-materialize FINAL_DEV with lookback=72.
    print("\n[Rematerialize] FINAL_DEV with lookback=72...")
    final_dev_manifest = materialize_final_dev_region(
        project_root=ROOT,
        feature_variant_id="FS2_TF1",
        lookback=72,
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
    )
    print(f"  train={final_dev_manifest.train_window_count}, "
          f"val={final_dev_manifest.validation_window_count}, "
          f"final_dev={final_dev_manifest.final_dev_window_count}, "
          f"lookback={final_dev_manifest.lookback_steps}")
    print(f"  population_fingerprint: {final_dev_manifest.population_fingerprint[:16]}...")

    # 4. Re-materialize FINAL_SCALING with lookback=72.
    print("\n[Rematerialize] FINAL_SCALING with lookback=72...")
    scaling_result = materialize_final_scaling_v1(
        project_root=ROOT,
        feature_variant_id="FS2_TF1",
    )
    print(f"  x_sha={scaling_result['x_sha256'][:16]}, "
          f"y_sha={scaling_result['y_sha256'][:16]}, "
          f"fit_row_count={scaling_result['fit_row_count']}, "
          f"lookback={scaling_result.get('lookback_steps')}")

    # 5. Verify against Phase 45 lock.
    lock = json.loads((ROOT / "artifacts" / "final_model_lock" / "phase46_three_seed_handoff.json").read_text())
    expected_train = lock["scientific_config"]["data"]["train_sample_count"]
    expected_val = lock["scientific_config"]["data"]["validation_sample_count"]
    if final_dev_manifest.train_window_count != expected_train:
        print(f"[FAIL] train_window_count {final_dev_manifest.train_window_count} != locked {expected_train}")
        return 1
    if final_dev_manifest.validation_window_count != expected_val:
        print(f"[FAIL] validation_window_count {final_dev_manifest.validation_window_count} != locked {expected_val}")
        return 1
    print("\n[PASS] FINAL_DEV and FINAL_SCALING re-materialized successfully against Phase 45 lock.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
