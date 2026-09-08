"""Quarantine the orphaned RUN_TR_ROB_0001_D9585D90 that crashed during
the human's first official attempt.

The run is in canonical_registry with status=RUNNING (orphan). The
canonical registry contract requires that a run be marked FAILED when an
exception occurs after registration. This script:

  - opens the canonical ExperimentRegistry (no mutation of upstream files)
  - calls registry.fail_run() with FailureType.FORWARD_PASS_ERROR
  - preserves status.json / config.json (no deletion)
  - sets the run's failure payload so it can NEVER be reused as canonical
    scientific evidence
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from course_work.experiments.registry import (
    ExperimentRegistry, FailureType,
)

ORPHAN_RUN_ID = "RUN_TR_ROB_0001_D9585D90"

CANONICAL_REGISTRY_ROOT = PROJECT_ROOT / "artifacts" / "registry"
CANONICAL_RUN_ROOT = PROJECT_ROOT / "artifacts" / "runs"


def main() -> int:
    print("=" * 78)
    print(f"PHASE 44 — QUARANTINE ORPHANED RUN {ORPHAN_RUN_ID}")
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
    record = next((r for r in records if r["run_id"] == ORPHAN_RUN_ID), None)
    if record is None:
        print(f"  [INFO] {ORPHAN_RUN_ID} is not in the registry (already cleaned up)")
        return 0

    print(f"  current status: {record.get('status')}")
    print(f"  started_at:    {record.get('started_at')}")
    print(f"  failure:       {record.get('failure')}")

    if record.get("status") == "FAILED":
        print(f"  [OK] {ORPHAN_RUN_ID} is already FAILED — nothing to do")
        return 0

    failure_message = (
        "ValueError: x must have shape [B, L, F] during Transformer forward "
        "pass. Loader produced tensor without sequence axis. "
        "Run is quarantined: must NOT be reused as canonical Phase 44 evidence."
    )
    registry.fail_run(
        ORPHAN_RUN_ID,
        failure_type=FailureType.FORWARD_PASS_ERROR.value,
        failure_stage="TRAINING",
        failure_message=failure_message,
        exception_class="ValueError",
        recoverable=False,
        rerun_recommended=True,
    )
    print(f"  [OK] {ORPHAN_RUN_ID} marked FAILED (FORWARD_PASS_ERROR)")

    records = registry._load_records()
    record = next((r for r in records if r["run_id"] == ORPHAN_RUN_ID), None)
    print(f"  new status:   {record.get('status') if record else 'MISSING'}")
    print(f"  failure_type: {record.get('failure', {}).get('failure_type') if record else 'MISSING'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())