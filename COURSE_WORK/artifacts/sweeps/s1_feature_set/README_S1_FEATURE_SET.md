# S1 Feature-Set Sweep (Phase 23)

## Overview

Phase 23 performs a one-factor sweep over feature-variant composition to isolate the effect of:
- **FS0_TF1**: Exogenous-only (no historical Appliances, no random controls)
- **FS1_TF1**: Exogenous + Historical target (Appliances) - REUSED from B0
- **FS2_TF1**: FS1 + Random controls (rv1, rv2)

## Design

| Factor | Value |
|--------|-------|
| Sweep ID | S1_FEATURE_SET |
| Factor Tested | feature_variant_id |
| Conditions | 3 |
| Reused Runs | 1 (FS1_TF1) |
| New Runs | 2 (FS0_TF1, FS2_TF1) |

## Hypotheses

| ID | Statement | Expected |
|----|-----------|----------|
| H1 | Historical target improves RMSE | FS1 < FS0 |
| H2 | Random controls improve RMSE | FS2 < FS1 |

## Winner

**FS1_TF1** selected with Val RMSE = 61.031 Wh

## Frozen Factors for Next Phase

| Factor | Value |
|--------|-------|
| feature_variant_id | FS1_TF1 |
| lookback_steps | 144 |
| target_scaling | YS1 |
| batch_size | 64 |
| learning_rate | 3e-4 |

## Files

- `s1_feature_set_run_contract.json` - Sweep contract
- `s1_feature_set_summary.json` - Results summary
- `s1_feature_set_metrics.csv` - All condition metrics
- `s1_feature_set_pairwise_effects.csv` - Pairwise comparisons
- `s1_feature_set_audit.csv` - Integrity checks
- `s1_feature_set_discrepancies.json` - Discrepancies
- `s1_reference_update.json` - Reference for Phase 24
- `phase_23_signoff.json` - Phase signoff

## Downstream

Phase 24 (S2 Time-Feature Sweep) inherits FS1_TF1 as frozen baseline.
