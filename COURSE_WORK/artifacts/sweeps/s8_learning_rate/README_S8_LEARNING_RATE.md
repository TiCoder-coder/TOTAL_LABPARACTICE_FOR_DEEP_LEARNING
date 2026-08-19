# S8 Learning-Rate Sweep (Phase 30)

## Overview

Phase 30 performs a one-factor sweep over learning rate to isolate its effect on optimization convergence and validation RMSE.

## Design

| Condition | Learning Rate | Status |
|-----------|--------------|--------|
| LR1 | 1e-4 | PENDING |
| LR2 | 3e-4 | REUSED |
| LR3 | 1e-3 | PENDING |

**Note**: Constant LR throughout (no scheduler/warmup).

## Hypotheses

| ID | Statement | Expected |
|----|-----------|----------|
| H1 | Learning rate affects RMSE | Varies |

## Winner

**LR2** selected with Val RMSE = 61.031 Wh (from FS1_TF1_YS1_L144_P0_A1_B64_LR2)

## Final Config Summary

All hyperparameters selected from S1-S8 sweeps:

| Factor | Value |
|--------|-------|
| feature_variant_id | FS1_TF1 |
| lookback_steps | 144 |
| target_scaling_option | YS1 |
| pooling | LAST_STEP |
| activation | GELU |
| batch_size | 64 |
| learning_rate | 3e-4 |
| weight_decay | 1e-4 |
| seed | 42 |

## Files

- `s8_learning_rate_run_contract.json` - Sweep contract
- `s8_learning_rate_summary.json` - Results summary
- `s8_learning_rate_metrics.csv` - All condition metrics
- `s8_learning_rate_pairwise.csv` - Pairwise comparisons
- `s8_learning_rate_audit.csv` - Integrity checks
- `s8_learning_rate_discrepancies.json` - Discrepancies
- `s8_reference_update.json` - Final reference
- `phase_30_signoff.json` - Phase signoff

## Downstream

Phase 31 (S9 Weight-Decay Sweep) inherits final config as frozen baseline.
