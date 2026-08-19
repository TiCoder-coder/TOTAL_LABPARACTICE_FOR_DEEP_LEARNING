# S5 Pooling Sweep (Phase 27)

## Overview

Phase 27 performs a one-factor sweep over pooling strategy to isolate its effect on sequence-to-one regression.

## Design

| Condition | Formula | Status |
|-----------|---------|--------|
| LAST_STEP | `encoded[:, -1, :]` | REUSED |
| MEAN | `encoded.mean(dim=1)` | PENDING |

Pooling occurs after the full Encoder stack. No CLS token, attention pooling, or max pooling.

## Hypotheses

| ID | Statement | Expected |
|----|-----------|----------|
| H1 | Pooling affects RMSE | Varies |

## Winner

**LAST_STEP** selected with Val RMSE = 61.031 Wh (from FS1_TF1_YS1_L144)

## Frozen Factors for Next Phase

| Factor | Value |
|--------|-------|
| feature_variant_id | FS1_TF1 |
| lookback_steps | 144 |
| target_scaling | YS1 |
| pooling | LAST_STEP |

## Files

- `s5_pooling_run_contract.json` - Sweep contract
- `s5_pooling_summary.json` - Results summary
- `s5_pooling_metrics.csv` - All condition metrics
- `s5_pooling_audit.csv` - Integrity checks
- `s5_pooling_discrepancies.json` - Discrepancies
- `s5_reference_update.json` - Reference for Phase 28
- `phase_27_signoff.json` - Phase signoff

## Downstream

Phase 28 (S6 Activation Sweep) inherits LAST_STEP pooling as frozen.
