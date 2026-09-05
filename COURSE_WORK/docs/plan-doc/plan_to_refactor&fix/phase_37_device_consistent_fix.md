# Phase 37 Device-Consistent Strict-BEST Fix

## Metadata

| Field | Value |
|-------|-------|
| Fix ID | PHASE37-FIX-001 |
| Phase | 37 (S15 Loss sweep) |
| Fix Date | 2026-08-24 |
| Author | Cursor Main Agent |
| Status | IMPLEMENTED |
| Human Approval | 2026-08-24 |

---

## Problem Statement

Phase 37 strict-BEST verification fails because:
- Training evaluation: Used MPS device
- Verification evaluation: Hardcoded CPU device
- Observed delta: ~5.75e-07 (exceeds 1e-9 tolerance)

---

## Solution

Use the recorded training device (MPS) for strict-BEST verification instead of hardcoding CPU.

---

## Implementation

**File:** `COURSE_WORK/src/course_work/sweeps/loss.py`

**Location:** `verify_phase_37_huber_best()`, around line 759

**Changes:**

```python
# BEFORE:
device_type="cpu",

# AFTER:
recorded_device = config["runtime"]["device_type"]
if recorded_device not in ("cpu", "cuda", "mps"):
    raise RuntimeError(f"Phase 37 unsupported recorded device: {recorded_device}")
if recorded_device == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError("Phase 37 recorded device MPS is not available on this platform")
if recorded_device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("Phase 37 recorded device CUDA is not available on this platform")
verification_device = torch.device(recorded_device)
...
device_type=recorded_device,
...
model = model.to(verification_device)
...
batch_x = batch["x"].to(verification_device)
batch_y_model = batch["y_model"].to(verification_device)
```

---

## Verification Results

| Check | Result |
|-------|--------|
| Syntax | PASS |
| Import | PASS |
| Device tests (19 tests) | ALL PASSED |
| MPS unavailable blocks verification | YES (as designed) |

**Note:** Full same-device verification blocked because MPS is not available on the current verification platform.

---

## Scientific Invariants Preserved

| Invariant | Value | Status |
|-----------|-------|--------|
| MSE reference | RUN_TR_S14_0023_A711A9B8 | Preserved |
| Huber run | RUN_TR_S15_0024_9420CDD7 | Preserved |
| Huber best epoch | 12 | Preserved |
| Stored metrics | All exact values preserved | Preserved |
| Tolerance | 1e-9 | Unchanged |
| Phase 38 policy | Not executed | Preserved |

---

## Testing Requirements

All tests implemented in `tests/unit/test_loss.py::TestDeviceConsistentVerification`:

1. ✓ recorded CPU run requests CPU verification
2. ✓ recorded MPS run requests MPS verification
3. ✓ recorded CUDA run requests CUDA verification
4. ✓ unavailable recorded device blocks rather than silently falling back
5. ✓ strict checkpoint load remains enforced
6. ✓ 1e-9 tolerance remains enforced
7. ✓ population mismatch still fails
8. ✓ target-scaler mismatch still fails
9. ✓ Test access remains forbidden
10. ✓ device resolution does not change scientific config
11. ✓ audit-only/dry-run do not create training runs

---

## Next Steps

1. **Verify on MPS environment:** Run `verify_phase_37_huber_best('RUN_TR_S15_0024_9420CDD7')` on a machine with MPS available
2. **Expected result:** Verification PASS with delta < 1e-9
3. **If verification passes:** Proceed to Phase 37 finalization
4. **If verification fails:** New BLOCKED checkpoint required

---

## Related Documentation

- Error Analysis: `docs/plan-doc/analysis_error/phase_37_strict_best_error.md`
- Corrective Plan: `docs/plan-doc/plan_before_process/phase_37_strict_best_corrective_plan.md`
- Processing Log: `docs/save_log_in_processing/phase_37_s15_loss_corrective_log.json`
