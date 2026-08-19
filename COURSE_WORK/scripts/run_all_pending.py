"""Run all 13 pending sweep conditions sequentially."""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

# Import shared logic
from scripts.run_single_condition import run_condition, ROOT


# Pending conditions per sweep
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


def main():
    output_file = ROOT / "artifacts" / "sweeps" / "live_sweep_results.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total = sum(len(c) for c in PENDING_CONDITIONS.values())
    print(f"Will run {total} conditions", flush=True)

    completed = []
    failed = []
    idx = 0
    for sweep_id, conditions in PENDING_CONDITIONS.items():
        for cond in conditions:
            idx += 1
            print(f"\n[{idx}/{total}] {sweep_id} :: {cond}", flush=True)
            try:
                record = run_condition(sweep_id, cond)
                completed.append(record)
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
                failed.append({"sweep_id": sweep_id, "condition": cond, "error": str(e)})

    print(f"\n=== Summary ===", flush=True)
    print(f"  completed: {len(completed)}/{total}", flush=True)
    print(f"  failed: {len(failed)}/{total}", flush=True)
    if failed:
        print(f"  failures:", flush=True)
        for f in failed:
            print(f"    {f['sweep_id']}::{f['condition']} - {f['error']}", flush=True)

    # Save final summary
    summary_path = ROOT / "artifacts" / "sweeps" / "all_sweep_results.json"
    summary_path.write_text(json.dumps({"completed": completed, "failed": failed}, indent=2))
    print(f"\nSummary saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()