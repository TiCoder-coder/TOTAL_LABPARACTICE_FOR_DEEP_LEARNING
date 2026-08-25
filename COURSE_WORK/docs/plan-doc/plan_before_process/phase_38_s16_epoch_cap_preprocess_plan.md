# Phase 38 Pre-Process Plan — S16 Epoch-cap Sweep

## Metadata

| Field | Value |
|-------|-------|
| Plan ID | PHASE38-PREPROCESS-001 |
| Phase | 38 |
| Sweep ID | S16_EPOCH_CAP |
| Sweep Version | SWEEP_S16_EPOCHCAP-v1 |
| Plan Date | 2026-08-24 |
| Author | Cursor Main Agent |
| Status | DRAFT |

---

## 1. Phase Identity

| Field | Value |
|-------|-------|
| Phase | 38 |
| Phase Name | S16 Epoch-cap sweep |
| Sweep ID | S16_EPOCH_CAP |
| Sweep Version | SWEEP_S16_EPOCHCAP-v1 |
| Source Reference | SWEEP_S15_LOSS-v1 |

---

## 2. Objective

Determine whether extending the maximum training epoch cap from 50 to 100 improves Validation RMSE under the current frozen configuration with patience=10 early stopping.

**Core Question:** Does the 50-epoch cap prematurely terminate training, or is patience=10 sufficient to find the optimal checkpoint?

---

## 3. Upstream Dependencies

### 3.1 Required Phase 37 Artifacts

| Artifact | Path | Status |
|----------|------|--------|
| Phase 37 sign-off | `artifacts/sweeps/S15_loss/phase_37_signoff.json` | REQUIRED |
| S15 winner | `artifacts/sweeps/S15_loss/s15_loss_winner.json` | REQUIRED |
| S15 reference update | `artifacts/sweeps/S15_loss/s15_reference_update.json` | REQUIRED |

### 3.2 Phase 37 Handoff Verification

| Field | Value |
|-------|-------|
| Phase 37 status | PASS |
| Winner | MSE |
| Selected loss | MSE |
| Reference run | RUN_TR_S14_0023_A711A9B8 |
| Reference Validation RMSE | 57.69679988114431 Wh |
| approved_for_phase38 | true |

---

## 4. Downstream Dependencies

Phase 39 (S17 Gradient-clipping sweep) will inherit:
- Winning epoch cap (E50 or E100)
- All frozen configuration from S15

---

## 5. Scientific Contract

### 5.1 Swept Factor

| Field | Value |
|-------|-------|
| Factor | max_epochs |
| Unit | epochs |

### 5.2 Conditions

| Condition | max_epochs | Strategy |
|-----------|------------|----------|
| E50 | 50 | REUSE_REFERENCE |
| E100 | 100 | TRAIN_NEW |

### 5.3 Frozen Configuration (from S15)

| Parameter | Value |
|-----------|-------|
| feature_variant_id | FS2_TF1 |
| target_scaling_id | YS1 |
| lookback | 36 |
| pooling | LAST_STEP |
| activation | GELU |
| batch_size | 32 |
| learning_rate | 0.0003 |
| weight_decay | 0.001 |
| dropout | 0.1 |
| d_model | 64 |
| num_heads | 4 |
| head_dim | 16 |
| num_layers | 2 |
| ffn_dim | 256 |
| loss | MSE |
| optimizer | AdamW |
| gradient_clip | 1.0 |
| scheduler | None |
| warmup | None |
| patience | 10 |
| min_delta | 0 |
| seed | 42 |

### 5.4 Selection Rules

| Rule | Value |
|------|-------|
| Selection metric | Validation RMSE Wh |
| Selection direction | MIN |
| Tie rule | E50 wins on exact RMSE tie |

---

## 6. Execution Matrix

### 6.1 E50 Reference (REUSE_REFERENCE)

- Source: `RUN_TR_S14_0023_A711A9B8`
- max_epochs: 50
- Requires new training: NO
- Must verify exact configuration match

### 6.2 E100 Candidate (TRAIN_NEW)

- Source: NEW RUN
- max_epochs: 100
- Requires new training: YES
- Fresh model/optimizer/loader from seed 42

---

## 7. Key Scientific Distinctions

### 7.1 Max Epochs is a Ceiling, Not a Budget

```
actual_epochs_completed = min(early_stop_epoch, max_epochs)
```

E100 is NOT guaranteed to train 100 epochs. It may early-stop at any epoch ≤ 100.

### 7.2 Early Stopping Remains Fixed

Both conditions use:
- patience = 10
- min_delta = 0
- monitor = Validation RMSE Wh
- strict improvement semantics

### 7.3 No Scheduler Dependencies

- scheduler = None
- warmup = None

Max epochs change must NOT affect LR dynamics.

---

## 8. Pre-Training Dependency Audit

### 8.1 Phase 37 Sign-off Health

| Check | Required | Status |
|-------|----------|--------|
| Phase 37 overall_status | PASS | TO_VERIFY |
| approved_for_phase38 | true | TO_VERIFY |

### 8.2 E50 Reference Availability

| Check | Required | Status |
|-------|----------|--------|
| Reference run exists | RUN_TR_S14_0023_A711A9B8 | TO_VERIFY |
| Reference config matches S15 | Yes | TO_VERIFY |
| Reference BEST verified | Yes | TO_VERIFY |
| Reference Validation RMSE | 57.69679988114431 | TO_VERIFY |

### 8.3 Required Scientific Evidence

| Evidence | Required | Status |
|----------|----------|--------|
| E50 training_history.csv | Yes | TO_VERIFY |
| E50 best_validation_metrics.json | Yes | TO_VERIFY |
| E50 best_checkpoint.pt | Yes | TO_VERIFY |
| Target scaler YS1 | Yes | TO_VERIFY |
| Validation population | Yes | TO_VERIFY |

### 8.4 Experiment Registry

| Check | Required | Status |
|-------|----------|--------|
| Registry accessible | Yes | TO_VERIFY |
| E50 registered | Yes | TO_VERIFY |

### 8.5 Test Firewall

| Check | Required | Status |
|-------|----------|--------|
| Test access | FORBIDDEN | MUST_VERIFY |

---

## 9. Implementation Scope

### 9.1 Source Code (MISSING - to be created)

| File | Purpose |
|------|---------|
| `src/course_work/sweeps/epoch_cap.py` | Phase 38 sweep logic |

### 9.2 Required Implementations

1. **Configuration loading:** Load S15 winner/reference
2. **Configuration freezing:** Verify all inherited fields
3. **E50 reference verification:** Confirm exact match
4. **E100 training registration:** New run with max_epochs=100
5. **E100 training execution:** Fresh training from epoch 1
6. **E100 BEST verification:** Strict checkpoint validation
7. **Prefix reproducibility audit:** Compare E50/E100 trajectories
8. **Cap-binding analysis:** Classify E50 cap status
9. **Extra-budget utilization:** Measure E100 epochs beyond 50
10. **Winner selection:** Minimum verified BEST RMSE

---

## 10. Required Canonical Outputs

### 10.1 Required Artifacts (44 outputs per Phase 38 Detail)

Key outputs include:
- `s16_epoch_cap_sweep_manifest.json`
- `s16_epoch_cap_sweep_contract.json`
- `s16_run_matrix.csv`
- `s16_epoch_cap_metrics.csv`
- `s16_epoch_cap_effect.csv`
- `s16_epoch_cap_winner.json`
- `s16_reference_update.json`
- `phase_38_signoff.json`
- `README_S16_EPOCH_CAP_SWEEP.md`

### 10.2 Required Figures (11 figures)

- S16_01_validation_rmse_by_epoch.png
- S16_02_validation_mae_by_epoch.png
- S16_03_train_criterion_by_epoch.png
- S16_04_prefix_rmse_difference.png
- S16_05_post50_validation_rmse.png
- S16_06_gradient_clipping_fraction.png
- S16_07_best_validation_metrics.png
- S16_08_epoch_budget_utilization.png
- S16_09_runtime_vs_best_rmse.png
- S16_10_convergence_summary.png
- S16_11_generalization_gap_optional.png

---

## 11. Pre-flight Plan

### 11.1 Pre-flight Checks (21 checks before training)

1. Phase 37 sign-off verified
2. approved_for_phase38 = true
3. S15 winner valid
4. All prior selected fields frozen
5. E50 registered
6. E100 registered
7. patience = 10 (fixed)
8. min_delta = 0 (fixed)
9. early_stop_metric = Validation RMSE Wh (fixed)
10. BEST metric = Validation RMSE Wh (fixed)
11. scheduler = None
12. warmup = None
13. gradient_clip = 1.0 (fixed)
14. Loss = MSE (fixed from S15)
15. Architecture identical
16. Parameter count identical
17. Population identical
18. Seed = 42
19. History supports 100 epochs
20. Plotting supports 100 epochs
21. Test firewall verified

### 11.2 Fail-Fast Conditions

If any pre-flight check fails: **STOP and report BLOCKED**

---

## 12. Training Authorization Boundary

### 12.1 Authorized Actions

| Action | Authorized |
|--------|------------|
| E100 training (max_epochs=100) | YES (after pre-flight PASS) |
| E50 retraining | NO |
| Test access | NO |
| Phase 39 execution | NO |

### 12.2 E100 Training Parameters

```python
max_epochs = 100
patience = 10
min_delta = 0
early_stop_metric = "validation_rmse_wh"
gradient_clip = 1.0
scheduler = None
```

---

## 13. STOP Conditions

The following conditions MUST stop Phase 38 and report BLOCKED:

| # | Condition |
|----|-----------|
| 1 | Phase 37 sign-off invalid or missing |
| 2 | approved_for_phase38 != true |
| 3 | E50 reference mismatch with S15 |
| 4 | Any frozen configuration drift detected |
| 5 | Population mismatch between E50 and S15 |
| 6 | Test access attempted |
| 7 | E100 training failure |
| 8 | E100 BEST verification failure |

---

## 14. Human Approval Gate

**This plan requires explicit Human approval before:**

1. Any Phase 38 source implementation
2. Any Phase 38 training execution
3. Any Phase 38 artifact materialization

---

## 15. Phase 38 Handoff to Phase 39

Upon successful Phase 38 completion:

| Field | Value |
|-------|-------|
| Selected epoch cap | E50 or E100 (winner) |
| Phase 39 factor | gradient_clip |
| GC0 | gradient clipping OFF |
| GC1 | max_norm = 1.0 (reference) |
| Phase 39 approved | YES (after Phase 38 PASS) |

---

## 16. Related Documentation

| Document | Path |
|----------|------|
| Phase 38 Detail | `docs/plan-doc/plan_detail_for_each_phase/Phase_38_S16_Epoch-cap_sweep.md` |
| Phase 37 sign-off | `artifacts/sweeps/S15_loss/phase_37_signoff.json` |
| S15 winner | `artifacts/sweeps/S15_loss/s15_loss_winner.json` |
| S15 reference | `artifacts/sweeps/S15_loss/s15_reference_update.json` |

---

## 17. Next Steps

| Step | Action | Gate |
|------|--------|------|
| 1 | Verify Phase 37 handoff artifacts | AUTOMATIC |
| 2 | Review this pre-process plan | HUMAN |
| 3 | Approve pre-process plan | HUMAN |
| 4 | Implement Phase 38 source | HUMAN_GATE |
| 5 | Execute pre-flight checks | AUTOMATIC |
| 6 | Execute E100 training (if pre-flight PASS) | HUMAN_GATE |
| 7 | Verify E100 BEST | AUTOMATIC |
| 8 | Materialize canonical outputs | AUTOMATIC |
| 9 | Generate Phase 38 sign-off | AUTOMATIC |
| 10 | Prepare Phase 39 handoff | AUTOMATIC |

---

## 18. Current Implementation Status

| Component | Status |
|-----------|--------|
| Phase 38 Detail | READ |
| Phase 38 source | MISSING |
| Phase 38 implementation | NOT_STARTED |
| Phase 38 training | NOT_AUTHORIZED |
| Phase 38 finalization | NOT_STARTED |

---

**END OF PHASE 38 PRE-PROCESS PLAN**

---

**Human action required:** Review and explicitly approve this plan before Phase 38 implementation or training.
