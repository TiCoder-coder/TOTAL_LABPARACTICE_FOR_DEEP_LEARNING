"""
Phase 39 Unauthorized Run Cleanup

Invalidates unauthorized S17 runs created during debugging.

Canonical mechanism: registry.cancel_run() (RUNNING -> CANCELLED)
followed by registry.invalidate_run() (CANCELLED -> ARCHIVED transition not allowed,
so we keep them in CANCELLED state).

Per project rules:
- Do NOT delete run directories
- Do NOT manually edit registry JSONL
- Use only repository-supported mechanisms
"""

import sys
from pathlib import Path

ROOT = Path("/Users/vientu/Deep Learning/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK")
sys.path.insert(0, str(ROOT / "src"))

from course_work.experiments.registry import (
    ExperimentRegistry,
    RunStatus,
)

UNAUTHORIZED_RUNS = [
    "RUN_TR_S17_0026_082F7FF5",
    "RUN_TR_S17_0027_082F7FF5",
    "RUN_TR_S17_0028_082F7FF5",
]

CANCEL_REASON = (
    "UNAUTHORIZED_PRE_TRAINING_DEBUG_RUN: Created during Phase 39 "
    "GC0 registry corrective debugging before the override fix was applied. "
    "No scientific training occurred. No checkpoints exist. No metrics recorded. "
    "Test not accessed. Cancelled per Phase 39 cleanup checkpoint."
)


def main():
    registry = ExperimentRegistry(ROOT)

    results = []
    for run_id in UNAUTHORIZED_RUNS:
        print(f"=== Processing {run_id} ===")
        try:
            record = registry._load_records()
            target = next((r for r in record if r["run_id"] == run_id), None)
            if target is None:
                print(f"  Not found in registry")
                results.append({"run_id": run_id, "status": "NOT_FOUND", "action": "skipped"})
                continue

            current_status = target.get("status")
            print(f"  Current status: {current_status}")

            if current_status == RunStatus.RUNNING.value:
                result = registry.cancel_run(run_id, CANCEL_REASON)
                print(f"  Cancelled: {result['status']}")
                results.append({"run_id": run_id, "status": result["status"], "action": "cancelled"})
            elif current_status in {RunStatus.CANCELLED.value, RunStatus.FAILED.value,
                                     RunStatus.INVALIDATED.value, RunStatus.ARCHIVED.value}:
                print(f"  Already in terminal state: {current_status}")
                results.append({"run_id": run_id, "status": current_status, "action": "already_clean"})
            else:
                print(f"  Unexpected status: {current_status}")
                results.append({"run_id": run_id, "status": current_status, "action": "unexpected"})

        except Exception as e:
            print(f"  Error: {e}")
            results.append({"run_id": run_id, "status": "ERROR", "action": str(e)})

    print("\n=== Cleanup Summary ===")
    for r in results:
        print(f"  {r['run_id']}: {r.get('action', '?')} -> {r['status']}")

    return results


if __name__ == "__main__":
    main()
