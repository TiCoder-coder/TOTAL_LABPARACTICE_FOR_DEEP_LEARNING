"""Quarantine the 3 orphaned RUN_TR_ROB_0001/0002/0003_D9585D90 runs that
crashed during the human's first two official attempts.

The first run crashed at the first Transformer forward pass (FORWARD_PASS_ERROR,
already quarantined by previous session).

The second and third runs each completed 1 epoch of training but crashed
during evaluation with `YS1 scaler bundle contract mismatch`. The runs are
in canonical_registry with status=RUNNING (orphan).

The canonical registry contract requires that a run be marked FAILED when an
exception occurs after registration. This script marks all three as FAILED
with FORWARD_PASS_ERROR / SCALER_CONTRACT_ERROR so they can NEVER be reused
as canonical scientific evidence.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from course_work.experiments.registry import (
    ExperimentRegistry, FailureType,
)

ORPHAN_RUN_IDS = [
    "RUN_TR_ROB_0002_D9585D90",
    "RUN_TR_ROB_0003_D9585D90",
]

CANONICAL_REGISTRY_ROOT = PROJECT_ROOT / "artifacts" / "registry"
CANONICAL_RUN_ROOT = PROJECT_ROOT / "artifacts" / "runs"


def main() -> int:
    print("=" * 78)
    print(f"PHASE 44 — QUARANTINE ORPHANED RUNS (0002, 0003)")
    print("=" * 78)

    if not CANONICAL_REGISTRY_ROOT.exists():
        print(f"  [INFO] canonical registry {CANONICAL_REGISTRY_ROOT} does not exist")
        return 0

    registry = ExperimentRegistry(
        project_root=PROJECT_ROOT,
        registry_root=CANONICAL_REGISTRY_ROOT,
        run_root=CANONICAL_RUN_ROOT,
    )

    records = registry._load_records()

    for orphan_id in ORPHAN_RUN_IDS:
        record = next((r for r in records if r["run_id"] == orphan_id), None)
        if record is None:
            print(f"  [INFO] {orphan_id} not in registry")
            continue

        current_status = record.get("status")
        print(f"\n  {orphan_id}:")
        print(f"    current status: {current_status}")
        print(f"    started_at:     {record.get('started_at')}")
        print(f"    failure:        {record.get('failure')}")

        if current_status == "FAILED":
            print(f"    [OK] already FAILED")
            continue

        if current_status == "RUNNING":
            failure_message = (
                "ValueError: YS1 scaler bundle contract mismatch during "
                "training evaluation. Fold-local target scaler schema "
                "did not match transform_target/inverse_transform_target "
                "contract after 1 epoch completed. Run is quarantined: "
                "must NOT be reused as canonical Phase 44 evidence."
            )
            registry.fail_run(
                orphan_id,
                failure_type=FailureType.SCALER_CONTRACT_ERROR.value,
                failure_stage="TRAINING",
                failure_message=failure_message,
                exception_class="ValueError",
                recoverable=False,
                rerun_recommended=True,
            )
            print(f"    [OK] marked FAILED (SCALER_CONTRACT_ERROR)")

    records = registry._load_records()
    print()
    print(f"[FINAL STATUS]")
    for orphan_id in ["RUN_TR_ROB_0001_D9585D90", "RUN_TR_ROB_0002_D9585D90", "RUN_TR_ROB_0003_D9585D90"]:
        rec = next((r for r in records if r["run_id"] == orphan_id), None)
        if rec:
            print(f"  {orphan_id}: status={rec.get('status')}, failure_type={rec.get('failure', {}).get('failure_type') if rec.get('failure') else None}")

    return 0


if __name__ == "__main__":
    sys.exit(main())