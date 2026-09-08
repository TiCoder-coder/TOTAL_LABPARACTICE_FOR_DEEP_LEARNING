"""Phase 46 — Quarantine orphan RUNNING records from prior failed attempts.

Earlier Phase 46 attempts left RUN_TR_FSD_* records in status=RUNNING. These
block the pre-train gate because they represent in-flight scientific runs
that were never completed. This script moves their directories to
``artifacts/runs/_quarantine_phase46_orphans/`` and marks them as QUARANTINED
in the registry so the pre-train gate can pass.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    runs_root = ROOT / "artifacts" / "runs"
    if not runs_root.exists():
        print("[INFO] No runs/ directory.")
        return 0

    orphan_ids = []
    for run_dir in sorted(runs_root.glob("RUN_TR_FSD_*")):
        registry_path = run_dir / "status.json"
        if not registry_path.exists():
            continue
        try:
            data = json.loads(registry_path.read_text())
        except Exception:
            continue
        if data.get("status") == "RUNNING":
            orphan_ids.append(run_dir.name)
    print(f"[Scan] {len(orphan_ids)} orphan RUN_TR_FSD_* RUNNING records")

    if not orphan_ids:
        print("[OK] No orphan records to quarantine.")
        return 0

    registry_path = ROOT / "artifacts" / "experiments" / "experiment_registry.jsonl"
    if registry_path.exists():
        records = []
        for line in registry_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
        for rec in records:
            if rec.get("run_id") in orphan_ids and rec.get("status") == "RUNNING":
                rec["status"] = "QUARANTINED"
                rec["quarantined_at"] = _now_iso()
                rec["quarantine_reason"] = "Phase 46 orphan RUNNING record from prior failed attempt"
        tmp_path = registry_path.with_suffix(".tmp")
        tmp_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        tmp_path.replace(registry_path)
        print(f"[Registry] Updated {len(orphan_ids)} records to QUARANTINED in experiment_registry.jsonl")

    archive = runs_root / "_quarantine_phase46_orphans"
    archive.mkdir(parents=True, exist_ok=True)
    manifest_path = archive / "quarantine_manifest.json"
    manifest = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    for run_id in orphan_ids:
        src = runs_root / run_id
        dest = archive / run_id
        try:
            shutil.move(str(src), str(dest))
            print(f"[Moved] {run_id} -> {dest.relative_to(ROOT)}")
        except Exception as exc:
            print(f"[WARN] Could not move {run_id}: {exc}")
            continue
        manifest.append({
            "run_id": run_id,
            "quarantined_at": _now_iso(),
            "reason": "Phase 46 orphan RUNNING record from prior failed attempt",
        })

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n[OK] Quarantined {len(orphan_ids)} orphan records.")
    print(f"     Manifest: {manifest_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
