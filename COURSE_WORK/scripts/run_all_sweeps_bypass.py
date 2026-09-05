"""Run all pending sweep conditions sequentially with env-check bypass.

Equivalent to ``scripts/run_all_pending.py`` but uses
``run_with_env_bypass.py`` per condition so that the Python 3.14 vs 3.10
interpreter mismatch does not block the run.

Each condition takes ~3-7 minutes; expected wall-clock for 10 conditions
is ~50 minutes. Run from a notebook cell and let the detached process
take over, or run directly in this terminal.

Usage (from the project root):

    python3 scripts/run_all_sweeps_bypass.py             # all pending
    python3 scripts/run_all_sweeps_bypass.py --dry-run   # only print plan
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = ROOT / "scripts"

PENDING_CONDITIONS = {
    "S1_FEATURE_SET": ["FS2_TF1"],
    "S2_TIME_FEATURE": ["TF0"],
    "S3_TARGET_SCALING": ["YS0"],
    "S4_LOOKBACK": ["L36", "L72"],
    "S5_POOLING": ["MEAN"],
    "S6_ACTIVATION": ["RELU"],
    "S7_BATCH_SIZE": ["B32"],
    "S8_LEARNING_RATE": ["LR1", "LR3"],
}

PYTHON_EXE = sys.executable


def run_one(sweep_id: str, condition_id: str) -> dict[str, object]:
    print(f"\n=== {sweep_id} :: {condition_id} ===", flush=True)
    start = time.time()
    proc = subprocess.run(
        [PYTHON_EXE, str(SCRIPT_DIR / "run_with_env_bypass.py"), sweep_id, condition_id],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start
    print(proc.stdout[-2000:])
    if proc.returncode != 0:
        print(f"  FAILED ({elapsed:.1f}s): {proc.stderr[-1000:]}", flush=True)
        return {"sweep_id": sweep_id, "condition": condition_id, "status": "FAIL", "elapsed_seconds": elapsed}
    print(f"  ✓ done in {elapsed:.1f}s", flush=True)
    return {"sweep_id": sweep_id, "condition": condition_id, "status": "PASS", "elapsed_seconds": elapsed}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    args = parser.parse_args(argv)

    total = sum(len(c) for c in PENDING_CONDITIONS.values())
    print(f"Plan: {total} conditions across {len(PENDING_CONDITIONS)} sweeps", flush=True)
    for sweep_id, conditions in PENDING_CONDITIONS.items():
        print(f"  {sweep_id}: {', '.join(conditions)}", flush=True)
    if args.dry_run:
        return 0
    confirm = input("Proceed? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.", flush=True)
        return 1

    summary_path = ROOT / "artifacts" / "sweeps" / "all_sweep_results_bypass.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for sweep_id, conditions in PENDING_CONDITIONS.items():
        for condition in conditions:
            results.append(run_one(sweep_id, condition))
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSummary saved to {summary_path.relative_to(ROOT)}", flush=True)
    passed = sum(1 for r in results if r["status"] == "PASS")
    print(f"  passed: {passed}/{len(results)}", flush=True)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
