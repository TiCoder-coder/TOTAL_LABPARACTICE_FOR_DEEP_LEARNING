# S6 Activation Sweep (Phase 28)

## Overview

Phase 28 performs a one-factor sweep over FFN activation function to isolate its effect on optimization and validation RMSE.

## Design

| Condition | Formula | Status |
|-----------|---------|--------|
| ReLU | `max(0, x)` | PENDING |
| GELU | `x * Phi(x)` | REUSED |

**Note**: Same activation applied to all Encoder FFN layers. Regression head has NO activation.

## Hypotheses

| ID | Statement | Expected |
|----|-----------|----------|
| H1 | Activation affects RMSE | Varies |

## Winner

**GELU** selected with Val RMSE = 61.031 Wh (from FS1_TF1_YS1_L144_P0_A1)

## Frozen Factors for Next Phase

| Factor | Value |
|--------|-------|
| feature_variant_id | FS1_TF1 |
| lookback_steps | 144 |
| target_scaling | YS1 |
| pooling | LAST_STEP |
| activation | GELU |

## Files

- `s6_activation_run_contract.json` - Sweep contract
- `s6_activation_summary.json` - Results summary
- `s6_activation_metrics.csv` - All condition metrics
- `s6_activation_audit.csv` - Integrity checks
- `s6_activation_discrepancies.json` - Discrepancies
- `s6_reference_update.json` - Reference for Phase 29
- `phase_28_signoff.json` - Phase signoff

## Downstream

Phase 29 (S7 Batch-Size Sweep) inherits GELU activation as frozen.
