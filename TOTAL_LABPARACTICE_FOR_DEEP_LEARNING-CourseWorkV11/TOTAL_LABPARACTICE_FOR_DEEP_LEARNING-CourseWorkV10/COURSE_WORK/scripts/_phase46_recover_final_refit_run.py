"""Phase 46 — FINAL_REFIT no-training recovery helper.

This script finalizes a RUNNING Phase 46 FINAL_REFIT run that completed all
training epochs but crashed during artifact/metric registration.

It performs ZERO optimizer steps. It only:
  - reads evidence from the existing run directory
  - registers artifacts in the registry
  - registers FINAL_DEV metrics
  - marks the run COMPLETED

The run must already have:
  - checkpoints/best_checkpoint.pt
  - checkpoints/last_checkpoint.pt
  - training_history.csv (with 30 rows for FINAL_REFIT mode)
  - metrics/best_validation_metrics.json (with split_id=FINAL_DEV)
  - predictions/best_validation_predictions.csv
  - training.log

Any Phase 46 lock contract deviation aborts the recovery.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="RUN_TR_FSD_* to recover")
    parser.add_argument("--best-epoch", type=int, default=30, help="best_epoch value (default 30 for FINAL_REFIT)")
    parser.add_argument("--dry-run", action="store_true", help="Inspect without mutating")
    args = parser.parse_args()

    run_id = args.run_id
    best_epoch = args.best_epoch
    dry_run = args.dry_run

    from course_work.experiments.registry import ExperimentRegistry
    from course_work.utils.artifacts import get_project_root

    registry = ExperimentRegistry(ROOT)
    record = registry.get_run(run_id)
    cfg = record.get("config", {})
    if not bool(cfg.get("training", {}).get("final_refit_mode", False)):
        print(f"[FAIL] {run_id} is not in FINAL_REFIT mode.")
        return 1
    if record["status"] != "RUNNING":
        print(f"[FAIL] {run_id} status={record['status']} (expected RUNNING).")
        return 1

    run_dir = registry.run_root / run_id
    required = [
        "checkpoints/best_checkpoint.pt",
        "checkpoints/last_checkpoint.pt",
        "training_history.csv",
        "metrics/best_validation_metrics.json",
        "predictions/best_validation_predictions.csv",
        "training.log",
    ]
    missing = [r for r in required if not (run_dir / r).exists()]
    if missing:
        print(f"[FAIL] {run_id} missing on-disk evidence: {missing}")
        return 1

    metrics = json.loads((run_dir / "metrics" / "best_validation_metrics.json").read_text())
    mr = metrics.get("metric_result", {})
    if mr.get("split_id") != "FINAL_DEV":
        print(f"[FAIL] {run_id} metric split_id={mr.get('split_id')} (expected FINAL_DEV).")
        return 1
    expected_n = int(cfg["data"]["train_sample_count"]) + int(cfg["data"]["validation_sample_count"])
    if int(mr.get("n_samples", 0)) != expected_n:
        print(f"[FAIL] {run_id} metric n_samples={mr.get('n_samples')} expected={expected_n}.")
        return 1
    fingerprint = mr.get("population_fingerprint", "")
    expected_fp = cfg.get("lineage", {}).get("final_dev_population_fingerprint", "")
    if fingerprint != expected_fp:
        print(f"[FAIL] {run_id} metric population_fingerprint={fingerprint!r} expected={expected_fp!r}.")
        return 1

    history_lines = (run_dir / "training_history.csv").read_text().splitlines()
    history_rows = len(history_lines) - 1  
    expected_epochs = int(cfg["training"]["max_epochs"])
    if history_rows < expected_epochs:
        print(f"[FAIL] {run_id} training_history rows={history_rows} expected={expected_epochs}.")
        return 1

    if dry_run:
        print(f"[DRY-RUN] {run_id} would be recovered as COMPLETED with {history_rows} history rows.")
        return 0

    print(f"[Recovery] Finalizing {run_id} (best_epoch={best_epoch}, history_rows={history_rows})...")
    updated = registry.recover_final_refit_run(run_id, best_epoch=best_epoch)
    print(f"[Recovery] Status: {updated['status']}")
    print(f"[Recovery] best_epoch: {updated.get('best_epoch')}")
    print(f"[Recovery] best_validation_rmse_wh: {updated.get('best_validation_rmse_wh'):.4f}")
    print(f"[Recovery] completed_at: {updated.get('completed_at')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
