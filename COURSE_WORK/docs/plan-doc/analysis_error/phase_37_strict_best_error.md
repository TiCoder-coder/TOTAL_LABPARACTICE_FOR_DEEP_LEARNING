# Phase 37 Strict-BEST Verification Failure — Analysis

## Metadata

| Field | Value |
|-------|-------|
| Analysis ID | PHASE37-ERROR-001 |
| Phase | 37 (S15 Loss sweep) |
| Analysis Date | 2026-08-24 |
| Author | Cursor Main Agent |
| Status | COMPLETE |

---

## Error Summary

**Error Type:** `RuntimeError("Phase 37 Huber strict Validation metrics differ from stored BEST evidence")`

**Location:** `loss.py:verify_phase_37_huber_best()` line 807-808

**Impact:** Phase 37 BLOCKED; Phase 38 blocked

---

## Observed Metrics Comparison

| Metric | Stored | Recomputed | Delta | Tolerance |
|--------|--------|------------|-------|-----------|
| RMSE Wh | 58.680985170839044 | 58.680984596202755 | 5.746e-07 | 1e-9 |
| MAE Wh | 26.97820227679493 | 26.97820068991709 | 1.584e-06 | 1e-9 |
| R² | 0.5952706605937734 | 0.5952706685213625 | 7.927e-09 | 1e-9 |

---

## Root Cause Analysis

### Confirmed Divergence: Evaluation Device

| Path | Device | Evidence |
|------|--------|----------|
| Training-time evaluation (epoch 12) | MPS | `config["runtime"]["device_type"] = "mps"` |
| Strict-BEST verification | CPU | `loss.py:767` hardcoded `device_type="cpu"` |

### Why This Causes Numerical Difference

On Apple Silicon:
- MPS (Metal Performance Shaders) executes float32 operations on GPU
- CPU executes float32 operations using standard x86/ARM floating-point
- Same mathematical operations on different hardware can produce slightly different rounding results due to:
  - Different hardware implementations of IEEE 754 operations
  - Different instruction-level parallelism
  - Different accumulator precision

The observed delta (~5.75e-07) is consistent with float32 rounding differences between MPS and CPU.

### Metric Implementation Verification

Both paths call the SAME implementations:
- `inverse_transform_target()` from `scaling.py:282`
- `compute_regression_metrics()` from `metrics.py:376`
- Both convert to `np.float64` before metric computation
- Both use `sklearn.metrics` for RMSE, MAE, R²

### Prediction Artifact Verification

The Huber run produced `best_validation_predictions.csv` with:
- 2961 lines (header + 2960 samples)
- Full float64 precision
- Generated during epoch-12 training on MPS
- Sufficient precision to reproduce stored metrics

---

## Training-Time Provenance (epoch 12 BEST)

```
run_single_condition.py:381
    device = select_device() → torch.device("mps")
        ↓
engine.py:276-303 → _evaluate_loader(model, val_loader, device=mps)
    ├─ model(x) ← INFERENCE ON MPS (float32)
    ├─ predictions.cpu().numpy() → float32 CPU numpy
    ├─ inverse_transform_target(float32) → float64
    ├─ .tolist() → Python floats
    ├─ np.asarray(y_pred_wh) → float64
    ├─ compute_regression_metrics(float64, float64, ...)
    └─ MetricResult(rmse_wh=58.680985170839044)
        ↓
persist_run_artifacts():
    ├─ best_validation_metrics.json
    ├─ best_validation_predictions.csv
    └─ best_checkpoint.pt (epoch 12)
```

---

## Verification Provenance (FAILING)

```
loss.py:745-754 → torch.load(checkpoint, map_location="cpu")
    ↓
loss.py:759-769 → build_train_validation_loaders(device_type="cpu")
    ↓
loss.py:771-814
    ├─ model(x) ← INFERENCE ON CPU (float32) ← DIVERGENCE
    ├─ inverse_transform_target(float32) → float64
    ├─ compute_regression_metrics(float64, float64, ...)
    └─ MetricResult (different values)
    ↓
loss.py:806-808: abs(stored - recomputed) > 1e-9 → FAIL
```

---

## Required Fix

The strict-BEST verifier must reproduce the original evaluation device (MPS) instead of hardcoding CPU.

**Fix Location:** `COURSE_WORK/src/course_work/sweeps/loss.py`

**Change:** Replace hardcoded `device_type="cpu"` with device from run configuration.

---

## Related Documentation

- Corrective Plan: `docs/plan-doc/plan_to_refactor&fix/phase_37_device_consistent_fix.md`
- Preparation Plan: `docs/plan-doc/plan_before_process/phase_37_strict_best_corrective_plan.md`
