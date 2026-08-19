# S2 Time-Feature Sweep (Phase 24)

## Overview

Phase 24 performs a one-factor sweep over time features to isolate their effect on validation RMSE.

## Design

| Factor | TF0 | TF1 |
|--------|-----|-----|
| hour_sin, hour_cos | No | Yes |
| dow_sin, dow_cos | No | Yes |
| weekend | No | Yes |
| Feature count | 26 | 31 |

## Hypotheses

| ID | Statement | Expected |
|----|-----------|----------|
| H1 | Time features improve RMSE | TF1 < TF0 |

## Winner

**TF1** selected with Val RMSE = 61.031 Wh (from FS1_TF1)

## Frozen Factors for Next Phase

| Factor | Value |
|--------|-------|
| feature_variant_id | FS1_TF1 |
| time_features | TF1 |
| lookback_steps | 144 |
| target_scaling | YS1 |

## Files

- `s2_time_feature_run_contract.json` - Sweep contract
- `s2_time_feature_summary.json` - Results summary
- `s2_time_feature_metrics.csv` - All condition metrics
- `s2_time_feature_effect.csv` - Effect comparison
- `s2_time_feature_audit.csv` - Integrity checks
- `s2_time_feature_discrepancies.json` - Discrepancies
- `s2_reference_update.json` - Reference for Phase 25
- `phase_24_signoff.json` - Phase signoff

## Downstream

Phase 25 (S3 Target-Scaling Sweep) inherits FS1_TF1 with TF1 as frozen baseline.
