# S3 Target-Scaling Sweep (Phase 25)

## Overview

Phase 25 performs a one-factor sweep over target scaling to isolate its effect on optimization behavior and validation RMSE.

## Design

| Option | Transform | Inverse | Units |
|--------|-----------|---------|-------|
| YS0 | identity | none | raw Wh |
| YS1 | (y - mu) / sigma | y * sigma + mu | standardized |

**Critical**: MSE losses are NOT comparable across YS0/YS1. All metrics must be computed in original Wh space.

## Hypotheses

| ID | Statement | Expected |
|----|-----------|----------|
| H1 | Target scaling affects optimization | YS0 != YS1 |

## Winner

**YS1** selected with Val RMSE = 61.031 Wh (from FS1_TF1)

## Frozen Factors for Next Phase

| Factor | Value |
|--------|-------|
| feature_variant_id | FS1_TF1 |
| target_scaling | YS1 |
| lookback_steps | 144 |

## Files

- `s3_target_scaling_run_contract.json` - Sweep contract
- `s3_target_scaling_summary.json` - Results summary
- `s3_target_scaling_metrics.csv` - All condition metrics
- `s3_target_scaling_effect.csv` - Effect comparison
- `s3_target_scaling_audit.csv` - Integrity checks
- `s3_target_scaling_discrepancies.json` - Discrepancies
- `s3_reference_update.json` - Reference for Phase 26
- `phase_25_signoff.json` - Phase signoff

## Downstream

Phase 26 (S4 Lookback Sweep) inherits FS1_TF1 with YS1 as frozen baseline.
