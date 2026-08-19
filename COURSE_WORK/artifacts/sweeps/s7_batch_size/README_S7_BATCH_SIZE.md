# S7 Batch-Size Sweep (Phase 29)

## Overview

Phase 29 performs a one-factor sweep over batch size to isolate its effect on optimization dynamics and validation RMSE.

## Design

| Condition | Batch Size | Steps per Epoch | Status |
|-----------|-----------|-----------------|--------|
| B32 | 32 | More | PENDING |
| B64 | 64 | Fewer | REUSED |

## Hypotheses

| ID | Statement | Expected |
|----|-----------|----------|
| H1 | Batch size affects RMSE | Varies |

## Winner

**B64** selected with Val RMSE = 61.031 Wh (from FS1_TF1_YS1_L144_P0_A1_B64)

## Frozen Factors for Next Phase

| Factor | Value |
|--------|-------|
| feature_variant_id | FS1_TF1 |
| lookback_steps | 144 |
| target_scaling | YS1 |
| pooling | LAST_STEP |
| activation | GELU |
| batch_size | 64 |

## Files

- `s7_batch_size_run_contract.json` - Sweep contract
- `s7_batch_size_summary.json` - Results summary
- `s7_batch_size_metrics.csv` - All condition metrics
- `s7_batch_size_audit.csv` - Integrity checks
- `s7_batch_size_discrepancies.json` - Discrepancies
- `s7_reference_update.json` - Reference for Phase 30
- `phase_29_signoff.json` - Phase signoff

## Downstream

Phase 30 (S8 Learning-Rate Sweep) inherits B64 batch size as frozen.
