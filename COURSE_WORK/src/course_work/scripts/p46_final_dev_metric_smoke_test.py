"""
Phase 46 FINAL_DEV_DIAGNOSTIC smoke test — NO TRAINING, NO OFFICIAL RUN ID.

Executes the exact metric evaluation path using the REAL corrected FINAL_DEV dataset
(16,630 windows) with a disposable model. Verifies:

- observed n = 16,630
- expected n = 16,630
- population coverage PASS
- metric computation PASS
- evaluation_mode = FINAL_DEV_DIAGNOSTIC

No official run ID. No optimizer step. No Test access.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "COURSE_WORK/src")

from course_work.data.datasets import build_final_dev_dataset
from course_work.evaluation.metrics import (
    EvaluationMode,
    compute_regression_metrics,
    expected_sample_indices,
)
from course_work.training.engine import build_model_from_run_config
from course_work.utils.reproducibility import set_seed as set_global_seed
from course_work.scaling.final_scaling import load_final_dev_target_scaler

ROOT = Path("COURSE_WORK")
FINAL_DEV_VERSION = "FINAL_DEV_REGION-v1"
LOOKBACK = 36
HORIZON = 1
SEED = 2026  # disposable seed


def run_smoke_test() -> bool:
    print("=" * 70)
    print("Phase 46 FINAL_DEV_DIAGNOSTIC Metric Smoke Test")
    print("NO TRAINING  |  NO OFFICIAL RUN ID  |  NO TEST ACCESS")
    print("=" * 70)

    set_global_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed: {SEED}")
    print()

    # ── 1. Load REAL FINAL_DEV dataset ───────────────────────────────────────
    print("[1] Loading REAL FINAL_DEV dataset (TRAIN+VALIDATION=16630)...")
    try:
        y_scaler_bundle = load_final_dev_target_scaler(project_root=ROOT)
        final_dev_ds = build_final_dev_dataset(
            project_root=ROOT,
            variant_id="FS2_TF1",
            lookback=LOOKBACK,
            target_option="YS1",
            boundary_protocol="WB0_CONTEXT_CARRY_OVER",
            target_scaler_bundle=y_scaler_bundle,
            audit_mode=False,
        )
    except Exception as e:
        print(f"  FAIL: Could not build FINAL_DEV dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

    n_final_dev = len(final_dev_ds)
    print(f"  FINAL_DEV dataset length: {n_final_dev}")
    assert n_final_dev == 16630, f"Expected 16630, got {n_final_dev}"
    print("  PASS: n_final_dev == 16630")
    print()

    # ── 2. Build disposable model ─────────────────────────────────────────────
    print("[3] Building disposable locked Transformer (seed={SEED})...")
    # Load the locked model config from Phase 45.
    import json as _json
    lock_cfg = _json.loads((ROOT / "artifacts/final_model_lock/final_model_scientific_config.json").read_text())
    model_cfg = lock_cfg["model"]
    model_cfg["feature_count"] = 33
    model_cfg["lookback_steps"] = LOOKBACK
    try:
        model = build_model_from_run_config({"model": model_cfg})
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"  FAIL: Could not build locked Transformer: {e}")
        import traceback
        traceback.print_exc()
        return False
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("  PASS: Model built")
    print()

    # ── 3. Build DataLoader (matching engine._evaluate_loader pattern) ──────
    loader = DataLoader(
        final_dev_ds,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    print(f"[4] Running evaluation over FULL FINAL_DEV (16,630 windows) via DataLoader...")

    sample_indices: list[int] = []
    y_true_list: list[np.ndarray] = []
    y_pred_list: list[np.ndarray] = []
    target_option = "YS1"
    target_scaler_bundle = None  # Not needed for diagnostic

    try:
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(device)
                y_model = batch["y_model"].to(device)
                y_raw = batch["y_raw_wh"].to(device)
                sample_idx = batch["sample_idx"].cpu().numpy()
                predictions = model(x)
                if predictions.shape != y_model.shape:
                    raise RuntimeError("Prediction and target shape mismatch")
                # Skip inverse-transform for smoke test; use y_raw directly.
                true_wh = y_raw.cpu().numpy().reshape(-1)
                pred_wh = predictions.cpu().numpy().reshape(-1)
                sample_indices.extend(sample_idx.tolist())
                y_true_list.append(true_wh)
                y_pred_list.append(pred_wh)
    except Exception as e:
        print(f"  FAIL: Evaluation failed: {e}")
        return False

    observed_n = len(sample_indices)
    print(f"  Observed sample count: {observed_n}")
    assert observed_n == 16630, f"Expected 16630 observed, got {observed_n}"
    print("  PASS: observed_n == 16630")
    print()

    # ── 4. Check expected FINAL_DEV population ────────────────────────────────
    print("[5] Checking expected FINAL_DEV population...")
    try:
        expected = expected_sample_indices("FINAL_DEV", LOOKBACK, ROOT)
    except Exception as e:
        print(f"  FAIL: expected_sample_indices failed: {e}")
        return False
    expected_n = len(expected)
    print(f"  Expected sample count: {expected_n}")
    assert expected_n == 16630, f"Expected 16630 expected, got {expected_n}"
    print("  PASS: expected_n == 16630")
    print()

    # ── 5. Verify population coverage ──────────────────────────────────────────
    print("[6] Verifying population coverage (strict gate)...")
    obs_set = set(sample_indices)
    exp_set = set(expected.tolist())
    if obs_set != exp_set:
        missing = exp_set - obs_set
        extra = obs_set - exp_set
        print(f"  FAIL: Population mismatch!")
        print(f"    Missing from observed: {len(missing)}")
        print(f"    Extra in observed: {len(extra)}")
        return False
    print("  PASS: Observed population exactly matches expected FINAL_DEV population")
    print()

    # ── 6. Compute regression metrics with FINAL_DEV_DIAGNOSTIC mode ─────────
    print("[7] Computing regression metrics with evaluation_mode=FINAL_DEV_DIAGNOSTIC...")
    try:
        y_true_arr = np.concatenate(y_true_list)
        y_pred_arr = np.concatenate(y_pred_list)
        sample_idx_arr = np.array(sample_indices, dtype=np.int64)
        metric_result = compute_regression_metrics(
            y_true_wh=y_true_arr,
            y_pred_wh=y_pred_arr,
            sample_idx=sample_idx_arr,
            split_id="FINAL_DEV",
            evaluation_mode=EvaluationMode.FINAL_DEV_DIAGNOSTIC.value,
            population_fingerprint="smoke_test_fp",
            run_id="SMOKE_TEST_ONLY",
            model_id="DISPOSABLE_MODEL",
            expected_sample_idx=expected,
            lookback_steps=LOOKBACK,
            horizon_steps=HORIZON,
            target_scaling_option="YS1",
        )
    except Exception as e:
        print(f"  FAIL: compute_regression_metrics raised: {e}")
        return False

    print(f"  n_samples: {metric_result.n_samples}")
    print(f"  RMSE (Wh): {metric_result.rmse_wh:.4f}")
    print(f"  MAE (Wh): {metric_result.mae_wh:.4f}")
    print(f"  R2: {metric_result.r2:.4f}")
    print(f"  status: {metric_result.status}")
    print(f"  split_id: {metric_result.split_id}")

    assert metric_result.n_samples == 16630, (
        f"Expected n_samples=16630, got {metric_result.n_samples}"
    )
    assert metric_result.split_id == "FINAL_DEV", (
        f"Expected split_id=FINAL_DEV, got {metric_result.split_id!r}"
    )
    assert metric_result.rmse_wh >= 0, "RMSE must be non-negative"
    assert metric_result.status in {"PASS", "PASS_WITH_WARNING"}, (
        f"Metric status must be PASS/PASS_WITH_WARNING, got {metric_result.status!r}"
    )
    print("  PASS: All metric assertions satisfied")
    print()

    # ── 7. Summary ───────────────────────────────────────────────────────────
    print("=" * 70)
    print("SMOKE TEST RESULT: ALL CHECKS PASSED")
    print("=" * 70)
    print(f"  evaluation_mode: FINAL_DEV_DIAGNOSTIC")
    print(f"  evaluation_population: {FINAL_DEV_VERSION}")
    print(f"  n_samples: {observed_n}")
    print(f"  RMSE: {metric_result.rmse_wh:.4f} Wh")
    print(f"  Model: disposable (not saved)")
    print(f"  Run ID: SMOKE_TEST_ONLY (no official registration)")
    print(f"  Test access: NONE")
    print("=" * 70)
    return True


if __name__ == "__main__":
    ok = run_smoke_test()
    sys.exit(0 if ok else 1)
