"""Phase 46 corrected FINAL_DEV real-path dry-run.

This script:
  - Instantiates the CORRECTED FINAL_DEV dataset using build_final_dev_dataset
  - Constructs an actual DataLoader and fetches a batch
  - Verifies FINAL_DEV length = 16,630
  - Verifies TRAIN-origin count = 13,670
  - Verifies VALIDATION-origin count = 2,960
  - Verifies TEST-origin count = 0
  - Verifies batch x.shape = [B, 36, 33]
  - Verifies batch y_model.shape = [B, 1]
  - Verifies all values are finite
  - Does NOT create official run IDs
  - Does NOT access Test
  - Does NOT train

Usage:
    PYTHONPATH=COURSE_WORK/src python3.11 COURSE_WORK/tests/integration/dry_run_final_dev.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from course_work.data.datasets import (
    build_final_dev_dataset,
)


def main():
    print("=" * 70)
    print("Phase 46 FINAL_DEV Corrected Real-Path Dry-Run")
    print("=" * 70)
    print(f"Root: {ROOT}")

    # Load FINAL_SCALING-v1 scaler bundles from artifacts.
    reg_path = ROOT / "artifacts" / "scaling" / "final_dev" / "final_scaler_registry.json"
    if not reg_path.exists():
        print(f"FAIL: FINAL_SCALING-v1 registry not found at {reg_path}")
        sys.exit(1)

    import json
    registry = json.loads(reg_path.read_bytes())
    x_entry = registry["x_bundles"]["FS2_TF1"]
    y_entry = registry["target_bundles"]["YS1"]
    x_bundle = joblib.load(ROOT / x_entry["artifact_path"])
    y_bundle = joblib.load(ROOT / y_entry["artifact_path"])

    # Build the corrected FINAL_DEV dataset.
    print("\n[1] Building corrected FINAL_DEV dataset...")
    fd_dataset = build_final_dev_dataset(
        project_root=ROOT,
        variant_id="FS2_TF1",
        lookback=36,
        target_option="YS1",
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
        target_scaler_bundle=y_bundle,
    )

    n_total = len(fd_dataset)
    n_train = int((fd_dataset._records["target_split_id"] == "TRAIN").sum())
    n_val = int((fd_dataset._records["target_split_id"] == "VALIDATION").sum())
    n_test = int((fd_dataset._records["target_split_id"] == "TEST").sum())

    print(f"\n[2] FINAL_DEV dataset invariants:")
    print(f"   Total windows: {n_total} (expected: 16630) {'PASS' if n_total == 16630 else 'FAIL'}")
    print(f"   TRAIN-origin:  {n_train} (expected: 13670) {'PASS' if n_train == 13670 else 'FAIL'}")
    print(f"   VALIDATION:    {n_val} (expected: 2960) {'PASS' if n_val == 2960 else 'FAIL'}")
    print(f"   TEST:          {n_test} (expected: 0) {'PASS' if n_test == 0 else 'FAIL'}")

    if not (n_total == 16630 and n_train == 13670 and n_val == 2960 and n_test == 0):
        print("FAIL: FINAL_DEV invariants failed")
        sys.exit(1)

    # Construct a DataLoader.
    print(f"\n[3] Constructing DataLoader...")
    loader = DataLoader(fd_dataset, batch_size=8, shuffle=False)
    print(f"   DataLoader created successfully with batch_size=8")

    # Fetch one batch.
    print(f"\n[4] Fetching one batch...")
    batch = next(iter(loader))

    x_shape = tuple(batch["x"].shape)
    y_shape = tuple(batch["y_model"].shape)
    x_finite = bool(torch.isfinite(batch["x"]).all())
    y_finite = bool(torch.isfinite(batch["y_model"]).all())
    print(f"   x.shape = {x_shape} (expected: (8, 36, 33))")
    print(f"   y_model.shape = {y_shape} (expected: (8, 1))")
    print(f"   x finite: {x_finite}")
    print(f"   y_model finite: {y_finite}")

    if x_shape != (8, 36, 33):
        print(f"FAIL: batch x shape mismatch")
        sys.exit(1)
    if y_shape != (8, 1):
        print(f"FAIL: batch y_model shape mismatch")
        sys.exit(1)
    if not (x_finite and y_finite):
        print(f"FAIL: batch contains non-finite values")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED")
    print("=" * 70)
    print(f"Scientific training executed: NO")
    print(f"New official run IDs: NO")
    print(f"Test access: NO")
    print(f"FINAL_DEV runtime count: {n_total}")
    print(f"FINAL_DEV dataset config: {fd_dataset.config.split_id}")
    print(f"FINAL_DEV access mode: {fd_dataset.config.target_access_mode}")
    print(f"FINAL_SCALING-v1 X SHA: {x_entry['artifact_sha256'][:16]}...")
    print(f"FINAL_SCALING-v1 Y SHA: {y_entry['artifact_sha256'][:16]}...")


if __name__ == "__main__":
    main()