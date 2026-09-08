# PHASE 46 — FINAL_DEV_SCALING IMPLEMENTATION MISMATCH ISSUE

**Document version:** `phase_46_final_dev_scaling_implementation_mismatch_issue.md`
**Created:** 2026-09-01
**Author:** Senior ML Engineer / Scientific Reproducibility Engineer
**Classification:** Contract Deviation — Phase 46 Corrective Implementation
**Status:** DOCUMENTED — Corrective Reimplementation in Progress

---

## 1. Expected Phase 46 Contract

The approved Phase 46 plan (`Phase_46_Three-seed_final_runs.md`) and Phase 45 lock contract
(`final_scaling_contract.json`) mandate:

### 1.1 FINAL_DEV_REGION-v1

```
FINAL_DEV_REGION-v1 = TRAIN + VALIDATION
```

- TRAIN rows: 13,670 (locked window count)
- VALIDATION rows: 2,960 (locked window count)
- Combined FINAL_DEV: 16,630
- TEST: **excluded** (hard firewall)

### 1.2 FINAL_SCALING-v1

```
FINAL_SCALING-v1:
  fit_region = FINAL_DEV_REGION-v1
  X scaler: fitted once on FINAL_DEV pre-Test rows (TRAIN + VALIDATION)
  Y scaler (YS1): fitted once on FINAL_DEV target-fit population
  fit_once = true
  Test_rows_used = false
  reuse_all_seeds = true
```

This is a **separate** set of scalers from Phase 9 SCALING-v1 (which were fit on TRAIN-only).

### 1.3 FINAL_REFIT Semantics

```
Seeds: [42, 123, 2026]
FINAL_REFIT_EPOCHS: 50
validation_loader: None
early_stopping_enabled: False
BEST selection: NONE
checkpoint_type: FINAL_REFIT
official_epoch = max_epochs (= 50)
```

### 1.4 Phase 47 Release

```
phase47_test_release.json
  owned_by: Phase 46
  released = true only if: 3/3 seeds completed, all verified, no Test access
```

---

## 2. Actual Old Implementation

### 2.1 Train-Only Bug (Phase 46 Implementation Bug)

**File:** `scripts/phase46_three_seed_runs.py`, lines 511-514

```python
datasets_dict = loaders_tuple[0]
train_dataset = datasets_dict["TRAIN"]    # ← ONLY TRAIN
target_scaler = None
if target_option == "YS1":
    target_scaler = load_validated_target_scaler(ROOT)  # ← Existing TRAIN-only scaler
```

**Reality:** Training was performed on TRAIN-only dataset (13,670 samples), NOT on
FINAL_DEV_REGION-v1 (TRAIN+VALIDATION = 16,630 samples).

### 2.2 Phase 9 Scaler Reuse Bug (Phase 45 Lock Bug + Phase 46 Implementation Bug)

**File:** `scripts/phase46_three_seed_runs.py`, line 514

```python
target_scaler = load_validated_target_scaler(ROOT)  # ← Loads Phase 9 TRAIN-only scaler
```

**Reality:** The Phase 46 training used scalers fitted on TRAIN-only data (n=13,814
window rows), NOT on FINAL_DEV_REGION-v1 (n=16,630 window rows).

**Evidence:**
- Phase 9 scaler `YSCALER__YS1__SCALING-v1.joblib`: `fit_row_count: 13814`, `fit_split: TRAIN`
- Phase 9 scaler `XSCALER__FS2_TF1__SCALING-v1.joblib`: `n_samples_seen: 13814`, `fit_split: TRAIN`
- `artifacts/scaling/scaler_registry.json`: `fit_row_count: 13814` for all X/Y scalers
- `artifacts/splits/split_summary.csv`: TRAIN=13814, VALIDATION=2960, TEST=2961

**Expected FINAL_SCALING-v1:**
- Y scaler n_samples_seen ≈ 16,630
- X scaler n_samples_seen ≈ 16,630

### 2.3 Missing phase47_test_release.json (Phase 46 Implementation Bug)

**Contract requirement:** `phase47_test_release.json` is listed as required output O46.33
in Phase 46 plan (lines 1828, 3315).

**Reality:** The file was never created. However, `phase_46_signoff.json` contains
`"phase47_released": true`, which is a metadata inconsistency — no physical release
artifact exists.

### 2.4 Stale RMSE Metadata (Phase 46 Metadata Drift)

**File:** `artifacts/three_seed_final_runs/phase_46_signoff.json`
- `average_rmse_wh: 0.0` — placeholder/stale value

**File:** `artifacts/three_seed_final_runs/three_seed_run_matrix.csv`
- All `rmse_wh` values: `0.0`

**Correction:** File `artifacts/three_seed_final_runs/three_seed_final_runs_summary.json`
was updated separately with correct RMSE values (commit `1ec0c02`), but the signoff
itself was not updated to reflect this.

### 2.5 best_epoch Metadata Issue (Phase 46 Metadata Drift)

**File:** `src/course_work/training/engine.py`, line 392

```python
return TrainingResult(
    history=history,
    best_epoch=int(early_stop.best_epoch or 1),  # ← Defaults to 1 when no early stopping
    best_validation_rmse_wh=float(best_metric_result.rmse_wh),
    ...
)
```

**Reality:** When `evaluate_validation=False` (Phase 46's case), `early_stop.best_epoch`
remains `None`, so `best_epoch` defaults to `1` in the checkpoint metadata — even though
the model trained for 50 epochs and the FINAL_REFIT checkpoint saves the correct weights.

**Impact:** Checkpoint sidecar metadata (`seed_*_FINAL_REFIT_metadata.json`) contains
`best_epoch: 1` instead of `best_epoch: 50`. The actual model state is correct (final
weights), only the metadata label is wrong.

### 2.6 Phase 46 Metric Semantic Issue

**File:** `src/course_work/training/engine.py`, line 340

```python
val_metric = train_metric  # when evaluate_validation=False
```

**Reality:** When `evaluate_validation=False`, the field `validation_rmse_wh` in the
training history and status JSON actually stores **TRAIN RMSE**, not validation RMSE.
The `phase_46_signoff.json` reports this as `average_rmse_wh: 0.0` (placeholder).

**Correct semantic:** These metrics should be labeled `TRAIN_DIAGNOSTIC`, not
Validation RMSE.

---

## 3. Historical Phase 46 Runs — Preserved as Invalidated Evidence

The following run IDs and checkpoints from the old Phase 46 implementation are preserved
as historical evidence. They are classified as `HISTORICAL_INVALIDATED_BY_CONTRACT_DEVIATION`.

| Run ID | Seed | Status | Invalidation Reason |
|--------|------|--------|---------------------|
| RUN_TR_FSD_0153_B15A19DC | 42 | Historical | TRAIN-only training, TRAIN-only scalers |
| RUN_TR_FSD_0154_DD82D743 | 123 | Historical | TRAIN-only training, TRAIN-only scalers |
| RUN_TR_FSD_0155_59A50ADD | 2026 | Historical | TRAIN-only training, TRAIN-only scalers |

These checkpoint bytes are **NOT modified or deleted**. They are preserved as evidence
of the previous implementation's contract deviation.

**New corrected Phase 46 runs will create NEW run IDs with NEW checkpoints.**

---

## 4. Test Firewall — Intact

Despite the TRAIN-only bug and scaler reuse bug, the Test firewall was NOT breached:

- No Test DataLoader was constructed during Phase 46 training
- No Test target values were materialized
- No Test inference was performed
- No Test metrics were computed
- Phase 47 was NOT executed

The Test firewall remains **INTACT** in the old implementation.

---

## 5. Root Cause Classification

| Issue | Classification | Owner |
|-------|---------------|-------|
| Scaler Contract Violation | `PHASE45_LOCK_BUG + PHASE46_IMPLEMENTATION_BUG` | Phase 45 (spec) + Phase 46 (impl) |
| FINAL_DEV Training Population | `PHASE46_IMPLEMENTATION_BUG` | Phase 46 (impl) |
| Missing phase47_test_release.json | `PHASE46_IMPLEMENTATION_BUG + PHASE46_METADATA_DRIFT` | Phase 46 (impl) |
| RMSE Metadata Drift (0.0 in signoff) | `PHASE46_METADATA_DRIFT` (+ minor `PHASE46_IMPLEMENTATION_BUG`) | Phase 46 (impl) |
| best_epoch=1 in Checkpoint Payload | `PHASE46_METADATA_DRIFT` | Phase 46 (impl) |
| TRAIN_DIAGNOSTIC mislabeled as Validation | `PHASE46_IMPLEMENTATION_BUG` | Phase 46 (impl) |

---

## 6. Corrective Action Plan

Refer to: `docs/plan-doc/plan_before_process/phase_46_corrective_reimplementation_plan.md`

---

## 7. Phase 46 Corrective Classification

```
PHASE_46_STATUS_AFTER_AUDIT: FAIL
READY_FOR_PHASE47: NO

Reason:
1. FINAL_DEV_REGION-v1 was TRAIN-only, not TRAIN+VALIDATION
2. FINAL_SCALING-v1 was not created (reused TRAIN-only Phase 9 scalers)
3. phase47_test_release.json was not produced
4. Metric semantics were incorrect (TRAIN mislabeled as Validation)
5. Checkpoint metadata had wrong best_epoch

Test Firewall: INTACT (no regression)
```
