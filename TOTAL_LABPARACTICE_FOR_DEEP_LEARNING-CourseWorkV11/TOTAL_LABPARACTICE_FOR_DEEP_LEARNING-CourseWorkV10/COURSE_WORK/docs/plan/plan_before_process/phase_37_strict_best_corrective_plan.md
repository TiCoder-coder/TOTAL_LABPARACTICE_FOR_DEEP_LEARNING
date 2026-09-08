# PHASE 37 STRICT-BEST CORRECTIVE PLAN

## Metadata

| Field | Value |
|-------|-------|
| Plan ID | CW-PHASE37-CORRECTIVE-001 |
| Phase | 37 (S15 Loss sweep) |
| Status | DRAFT |
| Author | Cursor Main Agent |
| Date | 2026-08-24 |
| Purpose | Resolve Phase 37 strict-BEST verification failure without changing scientific results |
| Scope | verification/provenance mechanics only |

---

## Blocked Issue

Phase 37 strict-BEST verification gate in `loss.py:verify_phase_37_huber_best()` raises `RuntimeError("Phase 37 Huber strict Validation metrics differ from stored BEST evidence")`.

Stored Huber RMSE: `58.680985170839044`
Recomputed RMSE: `~58.680984596202755`
Observed delta: `~5.746e-07`
Required tolerance: `1e-9`

Phase 37 cannot finalize. Phase 38 is blocked.

---

## What Must NOT Change

These are inviolable scientific and verification contract invariants:

| Invariant | Value |
|-----------|-------|
| Phase 36 winner F256 | `ffn_dim = 256` |
| MSE reference run | `RUN_TR_S14_0023_A711A9B8` |
| MSE reference RMSE | `57.69679988114431` |
| Huber run | `RUN_TR_S15_0024_9420CDD7` |
| Huber BEST epoch | `12` |
| Huber stored RMSE | `58.680985170839044` |
| Huber stored MAE | `26.97820227679493` |
| Huber stored R² | `0.5952706605937734` |
| Huber delta | `1.0` model-space |
| Huber delta raw-Wh equivalent | `106.853424078282` |
| Huber epochs executed | `22` |
| Selection metric | Validation RMSE Wh, min wins |
| Tie-breaking | Exact RMSE tie selects MSE |
| Tolerance | `1e-9` (approved contract) |
| Target scaler checksum | `b3326a79da81b092460ef8d4a140a101b21f2433ff30c36971d626d2a2491697` |
| Population fingerprint | `a40ded8802e90008535d268720bad9e9dcca5eee1436ddb359deea3ad39a1987` |
| Test access | FORBIDDEN |
| All Phase 36 hyperparameters | Frozen from Phase 36 winner |
| All earlier Phase selections | Frozen from their respective winners |

---

## Root Cause Analysis

### Confirmed: Training-Evaluation Device Path Difference

**Training path:**
```
run_single_condition.py:381 → device = select_device()
                            → returns torch.device("mps") on Apple Silicon
                            → model.to(device) → model on MPS
                            → engine._evaluate_loader():
                              model(x) ← runs on MPS
                              predictions.cpu().numpy() ← float32 MPS tensor → float32 CPU numpy
                              inverse_transform_target(float32) → float64
                              .tolist() → Python float → appended
                              np.asarray(y_pred_wh) → float64
                              compute_regression_metrics(float64, float64)
```

**Verification path:**
```
loss.py:767 → device_type="cpu" (hardcoded)
              → model on CPU
              → model(batch["x"]) ← runs on CPU
              → model.numpy() ← float32 CPU tensor
              inverse_transform_target(float32) → float64
              .astype(float) → Python float → appended
              np.asarray(y_pred_wh, dtype=np.float64) ← float64
              compute_regression_metrics(float64, float64)
```

**Conclusion:** Both paths produce float64 metrics, but the model inference occurs on different hardware (MPS vs CPU). On Apple Silicon, MPS and CPU float32 execution produce different rounding results due to different hardware implementations of the same mathematical operations.

### Confirmed: Evaluation Device Is MPS

The Huber run `RUN_TR_S15_0024_9420CDD7` has `config.json` field:

```json
"runtime": {
    "device_name": "mps",
    "device_type": "mps"
}
```

The training device (MPS) was used for both forward training and epoch-12 BEST evaluation.

### Confirmed: Verification Forces CPU

`loss.py:767`:
```python
loaders = build_train_validation_loaders(
    ...
    device_type="cpu",  # hardcoded
)
```

### Confirmed: Same Metric Implementation

Both paths call the same functions:
- `inverse_transform_target()` from `scaling.py`
- `compute_regression_metrics()` from `metrics.py`
- Both use `sklearn.metrics.mean_absolute_error`, `root_mean_squared_error`, `r2_score`
- Both eventually call `normalize_regression_vector()` which enforces `dtype=np.float64`

### Confirmed: Prediction Artifacts Exist

The Huber run produced `best_validation_predictions.csv` with 2961 rows (header + 2960 samples). This CSV was written from `best_y_true_wh` and `best_y_pred_wh` which were captured during epoch-12 training evaluation on MPS.

The strict-BEST verification at `loss.py:809-814` compares this CSV against recomputed values using `np.allclose(atol=1e-9, rtol=1e-9)`.

### Confirmed: CSV Precision Is Sufficient

CSV uses Python `float()` → CSV text. Python floats are IEEE 754 double-precision (float64). No precision loss occurs in serialization or deserialization.

### Confirmed: JSON Metric Precision Is Sufficient

Metrics stored in `best_validation_metrics.json` use Python `float` → JSON. Python floats are float64. No precision loss.

### Confirmed: BEST Checkpoint Identity

The checkpoint at `checkpoints/best_checkpoint.pt` was saved at line `engine.py:401` with `model.state_dict()` from epoch 12 (when `early_stop.update()` returned True). This is the same model state used for the training-time BEST evaluation.

The verification at `loss.py:745-754` loads this checkpoint with `strict=True` and verifies `best_epoch` matches.

### NOT ConfIRMED: MPS→CPU Is Sole Numerical Cause

While MPS vs CPU is the most probable explanation for the observed delta (~5.75e-07 ≈ float32 rounding), this cannot be definitively proven without running inference on both devices and comparing intermediate values. This remains a strong hypothesis.

---

## Detailed Provenance Trace

### Training-Time Metric Provenance (epoch 12 BEST)

```
run_single_condition.py:381
    device = select_device() → torch.device("mps")
        │
        ▼
    model.to(device) → TransformerRegressor on MPS
        │
        ▼
engine.py:276-303
    _evaluate_loader(
        model, val_loader, device=torch.device("mps"), ...)
        │
        ├─ model.eval()
        ├─ for batch in val_loader:
        │     x = batch["x"].to(device)       → MPS tensor
        │     predictions = model(x)              → float32 on MPS ← INFERENCE ON MPS
        │     pred_wh = _inverse_predictions_to_wh(
        │         predictions.cpu().numpy(),     → float32 CPU numpy
        │         target_option, scaler_bundle)
        │         → inverse_transform_target(float32) → float64
        │     .tolist() → Python floats → list
        │     y_pred_wh.extend(pred_wh.tolist())
        │     y_true_wh.extend(true_wh.tolist())
        ├─ np.asarray(y_pred_wh)               → float64
        ├─ np.asarray(y_true_wh)               → float64
        ├─ compute_regression_metrics(
        │     float64, float64, int64,
        │     split_id="VALIDATION",
        │     evaluation_mode="VALIDATION",
        │     ...)
        │     → MetricResult(rmse_wh=58.680985170839044)
        │
        ▼
    return sample_idx, y_true, y_pred, val_metric
        │
        ▼
engine.py:306-312
    if improved (epoch 12):
        best_state = model.state_dict() (CPU clone)
        best_metric_result = val_metric
        best_sample_idx = sample_idx
        best_y_true = y_true
        best_y_pred = y_pred
        │
        ▼
persist_run_artifacts():
    best_validation_metrics.json ← MetricResult (float64 → JSON)
    best_validation_predictions.csv ← best_y_true, best_y_pred (float64 → CSV)
    best_checkpoint.pt ← best_state (epoch 12)
```

### Strict-BEST Verification Provenance

```
loss.py:745-754
    torch.load(checkpoint, map_location="cpu")
    model = TransformerRegressor(config)
    model.load_state_dict(checkpoint, strict=True)
        │
        ▼
loss.py:759-769
    build_train_validation_loaders(
        device_type="cpu",       ← HARDCODED CPU
        seed=42,
        num_workers=0)
        │
        ▼
loss.py:771-814
    model.eval()
    for batch in validation_loader:
        prediction_model = model(batch["x"])  ← INFERENCE ON CPU ← DIFFERENCE
        inverse_transform_target(
            prediction_model.numpy(),  → float32 CPU tensor
            "YS1", scaler) → float64
        y_pred_wh.extend(prediction_wh.astype(float).tolist())
        y_true_wh.extend(batch["y_raw_wh"].numpy().reshape(-1).astype(float).tolist())
        │
        ▼
    compute_regression_metrics(
        np.asarray(y_true_wh, dtype=np.float64),
        np.asarray(y_pred_wh, dtype=np.float64),
        ...  → MetricResult
        )
        │
        ▼
loss.py:806-808
    tolerance = 1e-9
    if any(abs(stored - recomputed) > tolerance):
        raise RuntimeError("strict Validation metrics differ")
```

---

## Repair Candidates

### Candidate 1: Use Same Device for Verification as Training (Recommended)

**Mechanism:** Detect the original training device from `config["runtime"]["device_type"]` and use it for verification instead of hardcoding `device_type="cpu"`.

**Implementation approach:**
```python
# In loss.py:759, change:
device_type_from_training = config["runtime"]["device_type"]
# Instead of: device_type="cpu"
loaders = build_train_validation_loaders(
    ...
    device_type=device_type_from_training,  # "mps" or "cpu"
)
```

**Pros:**
- Exactly reproduces the training evaluation provenance
- Device-consistent metric recomputation
- Does not change scientific results
- Does not change tolerance
- Aligned with Phase 37 verification contract intent

**Cons:**
- Requires MPS to be available on the verification platform
- If MPS is unavailable, falls back to CPU which still has mismatch risk
- May need fallback logic for non-Apple-Silicon platforms

**Feasibility:** HIGH — MPS is recorded in config.json, device selection is deterministic.

### Candidate 2: Use Stored Predictions as Canonical Evidence

**Mechanism:** Since `best_validation_predictions.csv` was computed on MPS during epoch-12 training, and the CSV precision is sufficient, use the stored predictions as the canonical evidence instead of recomputing.

**Implementation approach:**
```python
# In loss.py:809-814, compare recomputed against stored CSV
# If stored CSV has float64 precision and matches verification population:
#   use stored CSV as authoritative metric source
stored_predictions = pd.read_csv(paths["predictions"])
recomputed_predictions = pd.DataFrame({
    "sample_idx": sample_ids,
    "y_true_wh": y_true_wh,
    "y_pred_wh": y_pred_wh,
})
# Both are float64 CSV-backed
# Use stored as authoritative
```

**Pros:**
- Eliminates device-difference problem entirely
- Uses MPS-computed predictions as ground truth
- Scientifically correct: these ARE the training-time predictions

**Cons:**
- Requires verifying CSV provenance (which we have)
- Changes verification methodology from "recompute" to "validate stored evidence"
- Requires architecture amendment if approach changes scientific evidence model
- May conflict with existing Phase 37 contract

**Feasibility:** MEDIUM — CSV exists and is valid, but verification contract assumes recomputation.

### Candidate 3: Widen Verification Tolerance for Device Boundary

**Mechanism:** Adjust tolerance to account for known float32 MPS→CPU numerical boundary (~1e-6).

**Implementation approach:** Change `tolerance = 1e-9` to `tolerance = 1e-6` in `loss.py:806`.

**Status:** REJECTED — Tolerance change requires architecture amendment and human approval. Per project rules, this is NOT the primary repair approach.

---

## Recommended Repair

**Candidate 1: Use Same Device for Verification**

This approach:
1. Preserves the 1e-9 tolerance contract
2. Exactly reproduces training-time evaluation provenance
3. Does not change scientific results
4. Is device-consistent
5. Is technically straightforward
6. Aligns with Phase 37 verification contract intent

**Fallback:** If MPS is unavailable on verification platform, fall back to CPU with documented numerical limitation, then escalate for architecture decision.

---

## Implementation Steps (DO NOT IMPLEMENT YET)

### Step 1: Modify `loss.py`

File: `COURSE_WORK/src/course_work/sweeps/loss.py`

Location: `verify_phase_37_huber_best()`, around line 759

Change:
```python
# Before:
loaders = build_train_validation_loaders(
    ...
    device_type="cpu",
)

# After:
training_device = config["runtime"]["device_type"]
loaders = build_train_validation_loaders(
    ...
    device_type=training_device,
)
```

### Step 2: Add Fallback Logic

If `training_device == "mps"` but MPS is not available:
- Attempt MPS first
- Fall back to CPU with warning
- Log the fallback decision
- Proceed with CPU verification (will have residual delta)

### Step 3: Run Strict-BEST Verification

After modification, re-run `verify_phase_37_huber_best()`.

Expected outcome:
- MPS device matches training → `abs(stored - recomputed) <= 1e-9` → PASS
- CPU fallback with delta ~5e-7 → FAIL (document and escalate)

### Step 4: If Verification Passes

Proceed to Phase 37 finalization:
- Generate sweep results
- Select winner (MSE expected)
- Write phase artifacts
- Create sign-off

### Step 5: If Verification Still Fails

Escalate with full provenance evidence:
- Training device recorded in config
- Verification device used
- Observed delta
- Recommendation for architecture amendment

---

## Risk Analysis

| Risk | Mitigation |
|------|------------|
| MPS unavailable on verification platform | Implement fallback to CPU with warning |
| Device selection nondeterministic | `select_device()` is deterministic based on environment |
| Small residual delta after device fix | Below 1e-9 expected if MPS→MPS |
| Changes verification methodology | Only internal to `verify_phase_37_huber_best()` |

---

## Files to Modify

| File | Change | Impact |
|------|--------|--------|
| `COURSE_WORK/src/course_work/sweeps/loss.py` | Add device detection and usage in verification | Only Phase 37 verification; no impact on other phases |
| `COURSE_WORK/src/course_work/sweeps/loss.py` | Add MPS availability check and CPU fallback | Graceful degradation |

**No other files modified. No scientific artifacts changed.**

---

## Pre-Implementation Validation

Before implementing, verify:

1. `config["runtime"]["device_type"]` is `"mps"` in Huber run
2. `build_train_validation_loaders()` accepts `device_type="mps"`
3. Model inference runs on MPS without error
4. MPS produces matching predictions (within 1e-9)

---

## Human Approval Required

This plan requires explicit human approval before implementation.

Per project rules:
- `working_rule.md`: Implementation Plan 10 mục, chờ Human confirm
- `rule_code.md`: USER_APPROVAL_GATE bắt buộc
- Phase 37 finalization changes require architecture awareness

---

## Signatures

**Agent:** Cursor Main Agent (Claude)
**Date:** 2026-08-24
**Status:** AWAITING HUMAN APPROVAL
