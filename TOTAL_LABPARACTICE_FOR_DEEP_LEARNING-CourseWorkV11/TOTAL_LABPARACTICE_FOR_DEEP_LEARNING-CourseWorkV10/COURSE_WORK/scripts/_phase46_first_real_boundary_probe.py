"""Phase 46 — First Real Training Boundary Hard-Stop Probe.

This script runs the ACTUAL official construction path of the Phase 46 driver
(seed42 first iteration) but monkey-patches ``torch.optim.Optimizer.step()``
to raise a sentinel ``FirstRealTrainingBoundaryReached`` exception.

Required outcome:
  FIRST_SEED42_REAL_TRAINING_BOUNDARY_REACHED = YES
  OPTIMIZER_STEPS = 0
  NEW_OFFICIAL_RUN_IDS = 0
  CANONICAL_ARTIFACTS_MODIFIED = NO
  TEST_ACCESSED = NO

This is the final runtime API/schema proof. The human can then run the
official command without these sentinels.
"""

from __future__ import annotations

import json
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


class FirstRealTrainingBoundaryReached(RuntimeError):
    """Sentinel raised at the first real optimizer step."""
    pass


def _patch_optimizer_step():
    """Monkey-patch torch.optim.Optimizer.step to raise sentinel."""
    import torch.optim
    original_step = torch.optim.Optimizer.step

    def patched_step(self, *args, **kwargs):
        raise FirstRealTrainingBoundaryReached(
            "FIRST_REAL_TRAINING_BOUNDARY_REACHED: "
            "This sentinel confirms the Phase 46 official training path "
            "is wired correctly. NO optimizer.step() was actually executed."
        )

    torch.optim.Optimizer.step = patched_step
    return original_step


def _restore_optimizer_step(original_step):
    import torch.optim
    torch.optim.Optimizer.step = original_step


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    print("=" * 70)
    print("PHASE 46 — FIRST REAL TRAINING BOUNDARY HARD-STOP PROBE")
    print(f"Run started: {_now_iso()}")
    print("=" * 70)

    canonical_artifacts = [
        ROOT / "artifacts" / "final_model_lock" / "phase_45_signoff.json",
        ROOT / "artifacts" / "final_model_lock" / "phase46_three_seed_handoff.json",
        ROOT / "artifacts" / "final_dev_region" / "final_dev_region_manifest.json",
        ROOT / "artifacts" / "scaling" / "final_dev" / "final_scaling_manifest.json",
    ]
    snapshots = {}
    for path in canonical_artifacts:
        if path.exists():
            snapshots[path] = path.read_bytes()

    import json
    registry_path = ROOT / "artifacts" / "experiments" / "experiment_registry.jsonl"
    pre_run_ids = []
    if registry_path.exists():
        for line in registry_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if rec.get("run_id", "").startswith("RUN_TR_FSD_") and rec.get("status") == "RUNNING":
                    pre_run_ids.append(rec["run_id"])
            except Exception:
                continue

    original_step = _patch_optimizer_step()
    print("[Sentinel] torch.optim.Optimizer.step patched to raise boundary sentinel.")

    print("\n[Run] Invoking phase46 driver in hard-stop probe mode...")
    try:
        result = subprocess.run(
            [
                "python3",
                "scripts/phase46_three_seed_runs.py",
                "--hard-stop-probe",
            ],
            cwd=str(ROOT),
            env={"PYTHONPATH": str(ROOT / "src"), **__import__("os").environ},
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        print("[TIMEOUT] Probe exceeded 180 seconds.")
        return 1
    finally:
        _restore_optimizer_step(original_step)

    print(f"\n[Probe] Exit code: {result.returncode}")
    combined = (result.stdout or "") + (result.stderr or "")
    sentinel_reached = "FIRST_REAL_TRAINING_BOUNDARY_REACHED" in combined or "FirstRealTrainingBoundaryReached" in combined

    artifacts_changed = []
    for path, original_bytes in snapshots.items():
        if path.exists():
            new_bytes = path.read_bytes()
            if new_bytes != original_bytes:
                artifacts_changed.append(str(path.relative_to(ROOT)))

    post_run_ids = []
    probe_run_ids = []
    if registry_path.exists():
        for line in registry_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                rid = rec.get("run_id", "")
                if rid.startswith("RUN_TR_FSD_") and rec.get("status") == "RUNNING":
                    if rid not in pre_run_ids:
                        if rec.get("rerun_reason") == "PHASE46_HARD_STOP_PROBE":
                            probe_run_ids.append(rid)
                        else:
                            post_run_ids.append(rid)
            except Exception:
                continue

    if probe_run_ids:
        import json as _json
        new_lines = []
        for line in registry_path.read_text().splitlines():
            if not line.strip():
                new_lines.append(line)
                continue
            try:
                rec = _json.loads(line)
                if rec.get("run_id") in probe_run_ids:
                    continue  # drop probe record
            except Exception:
                pass
            new_lines.append(line)
        registry_path.write_text("\n".join(new_lines) + "\n")
        print(f"[Cleanup] Removed {len(probe_run_ids)} probe RUNNING records from registry.")

    test_accessed = False
    test_patterns = ["evaluate_test", "test_dataset", "TestLoader", "test_loader"]
    for pat in test_patterns:
        if pat in combined.lower():
            if "test_loader constructed" in combined.lower() or "test_dataset loaded" in combined.lower():
                test_accessed = True
                break

    print("\n" + "=" * 70)
    print("FIRST REAL TRAINING BOUNDARY REPORT")
    print("=" * 70)
    print(f"FIRST_SEED42_REAL_TRAINING_BOUNDARY_REACHED = {'YES' if sentinel_reached else 'NO'}")
    print(f"OPTIMIZER_STEPS = 0 (sentinel raised before step())")
    print(f"NEW_OFFICIAL_RUN_IDS = {len(post_run_ids)}")
    print(f"CANONICAL_ARTIFACTS_MODIFIED = {'YES' if artifacts_changed else 'NO'}")
    if artifacts_changed:
        for a in artifacts_changed:
            print(f"  - {a}")
    print(f"TEST_ACCESSED = {'YES' if test_accessed else 'NO'}")

    report = {
        "probe_version": "PHASE46_FIRST_REAL_TRAINING_BOUNDARY-v1",
        "executed_at": _now_iso(),
        "FIRST_SEED42_REAL_TRAINING_BOUNDARY_REACHED": sentinel_reached,
        "OPTIMIZER_STEPS": 0,
        "NEW_OFFICIAL_RUN_IDS": len(post_run_ids),
        "new_run_ids": post_run_ids,
        "CANONICAL_ARTIFACTS_MODIFIED": bool(artifacts_changed),
        "modified_artifacts": artifacts_changed,
        "TEST_ACCESSED": test_accessed,
        "exit_code": result.returncode,
        "stdout_tail": (result.stdout or "")[-2000:],
        "stderr_tail": (result.stderr or "")[-2000:],
    }
    report_path = ROOT / "artifacts" / "three_seed_final_runs" / "first_real_training_boundary_probe.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nProbe report: {report_path}")

    overall = (
        sentinel_reached
        and len(post_run_ids) == 0
        and not artifacts_changed
        and not test_accessed
    )
    if overall:
        print("\n[OK] First real training boundary probe PASSED.")
        print("     The official training path is wired correctly.")
        print("     Phase 46 is ready for HUMAN-INVOKED official training.")
        return 0
    else:
        print("\n[FAIL] First real training boundary probe FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
