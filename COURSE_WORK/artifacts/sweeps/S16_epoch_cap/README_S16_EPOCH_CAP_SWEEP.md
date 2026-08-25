# Phase 38 — S16 Epoch-cap Sweep

## Overview

**Phase:** 38  
**Sweep ID:** S16_EPOCH_CAP  
**Sweep Version:** SWEEP_S16_EPOCHCAP-v1  
**Factor:** `training.max_epochs` (Maximum Training Epoch Cap)

## Objective

Determine whether extending the maximum training epoch cap from 50 to 100 improves Validation RMSE under the current frozen configuration with patience=10 early stopping.

## Scientific Question

Does the 50-epoch cap prematurely terminate training, or is patience=10 sufficient to find the optimal checkpoint?

## Conditions

| Condition | max_epochs | Strategy |
|-----------|------------|----------|
| E50 | 50 | REUSE_REFERENCE |
| E100 | 100 | TRAIN_NEW |

## Key Semantic Distinction

> **max_epochs is a CEILING, not a guaranteed budget.**

Early stopping may terminate training before the cap is reached.

- E100 does NOT need to execute 100 epochs if early stopping triggers earlier.
- E100 may early-stop at any epoch ≤ 100.

## Inherited Configuration

From Phase 37 (S15 Loss sweep), winner = MSE:

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

## Early Stopping

Both conditions use identical early stopping:
- **patience:** 10
- **min_delta:** 0
- **monitor:** Validation RMSE Wh
- **direction:** MIN

## Selection Rules

| Rule | Value |
|------|-------|
| Primary metric | Validation RMSE Wh |
| Direction | MIN |
| Tie rule | E50 wins on exact RMSE tie |

## Why E50 is REUSE_REFERENCE

- E50 = `RUN_TR_S14_0023_A711A9B8`
- Phase 37 winner: MSE
- Reference Validation RMSE: **57.69679988114431** Wh
- Best epoch: **12**
- E50 will NOT be retrained

## Why E100 is TRAIN_NEW

- E100 requires fresh training with `max_epochs=100`
- Fresh model/optimizer/loader from seed 42
- One continuous run from epoch 1
- No continuation from E50 BEST checkpoint

## What is NOT Allowed

- E50 retraining
- E100 continuation from E50 BEST
- Patience reset at epoch 50
- Optimizer reset at epoch 50
- RNG/DataLoader reset at epoch 50
- BEST checkpoint reset at epoch 50
- Scheduler tied to max_epochs
- Hidden E150/E200 extension
- Score-based rerun
- **Test access (FORBIDDEN)**

## Prefix Equivalence

Under `scheduler=None` and full reproducibility, E50 and E100 should share identical trajectories before epoch 50. This is a powerful hidden-drift detector.

## Key Diagnostics

### E50 Cap-Binding Classification

- **NON_BINDING_EARLY_STOP:** E50 early-stopped before epoch 50
- **REACHED_CAP_BEST_EARLY:** E50 hit cap but BEST was early
- **REACHED_CAP_BEST_NEAR_BOUNDARY:** E50 hit cap and BEST was near boundary
- **REACHED_CAP_BEST_AT_BOUNDARY:** E50 hit cap and BEST was at boundary

### E100 Extra-Budget Utilization

- **NO_EXTRA_BUDGET_USED:** E100 early-stopped at or before epoch 50
- **PARTIAL_EXTRA_BUDGET_USED:** E100 trained 51-99 epochs
- **FULL_EXTRA_BUDGET_USED:** E100 completed all 100 epochs

### Post-50 Improvement

- **POST50_NOT_OBSERVED:** E100 did not train past epoch 50
- **POST50_NO_NEW_GLOBAL_BEST:** E100 trained past 50 but found no new global BEST
- **POST50_NEW_GLOBAL_BEST:** E100 found new global BEST after epoch 50

## Expected Outcomes

### If E50 wins

Extending the maximum epoch allowance from 50 to 100 did not improve the verified Validation RMSE under the fixed patience-10 early-stopping policy.

### If E100 wins with BEST > 50

Under the fixed training protocol, E100 discovered a new Validation BEST after epoch 50 and achieved lower verified RMSE than the E50 reference.

### If E100 wins with BEST ≤ 50

This suggests stochastic/reproducibility differences rather than benefit from extra epochs.

## Phase 39 Handoff

Phase 39 (S17 Gradient-clipping sweep) will:
- Inherit S16 winner's epoch cap (E50 or E100)
- Test gradient clipping (GC0 vs GC1)

## Artifacts

All Phase 38 artifacts are stored under:
```
artifacts/sweeps/S16_epoch_cap/
```

## Status

| Item | Status |
|------|--------|
| Phase 37 handoff | PASS |
| E50 reference | VERIFIED |
| Preflight | PASS |
| E100 training | NOT_AUTHORIZED |
| Phase 38 finalization | PENDING |

## Test Access

**FORBIDDEN** — Test set must not be accessed during Phase 38.

---

**Created:** 2026-08-24  
**Phase:** 38  
**Sweep:** S16 Epoch-cap  
**Inherited Warning:** H4_SOURCE_ARTIFACT_RETENTION_INCOMPLETE
