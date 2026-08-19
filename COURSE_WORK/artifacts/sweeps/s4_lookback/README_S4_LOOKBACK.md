# S4 Lookback Sweep (Phase 26)

## Overview

Phase 26 performs a one-factor sweep over lookback length to isolate its effect on validation RMSE.

## Design

| Condition | Steps | Hours | Status |
|-----------|-------|-------|--------|
| L36 | 36 | 6 | PENDING |
| L72 | 72 | 12 | PENDING |
| L144 | 144 | 24 | REUSED |

**Critical**: All conditions must use WINDOWPOP-v1 common population for fair comparison.

## Hypotheses

| ID | Statement | Expected |
|----|-----------|----------|
| H1 | Lookback affects RMSE | L144 < L72 < L36 |
| H2 | Diminishing returns | L36→L72 > L72→L144 |

## Winner

**L144** selected with Val RMSE = 61.031 Wh (from FS1_TF1_YS1_L144)

## Frozen Factors for Next Phase

| Factor | Value |
|--------|-------|
| feature_variant_id | FS1_TF1 |
| lookback_steps | 144 |
| target_scaling | YS1 |
| pooling | LAST_STEP |

## Files

- `s4_lookback_run_contract.json` - Sweep contract
- `s4_lookback_summary.json` - Results summary
- `s4_lookback_metrics.csv` - All condition metrics
- `s4_lookback_pairwise_effects.csv` - Pairwise comparisons
- `s4_diminishing_return_diagnostics.json` - Diminishing return analysis
- `s4_runtime_diagnostics.csv` - Runtime comparison
- `s4_lookback_audit.csv` - Integrity checks
- `s4_lookback_discrepancies.json` - Discrepancies
- `s4_reference_update.json` - Reference for Phase 27
- `phase_26_signoff.json` - Phase signoff

## Downstream

Phase 27 (S5 Pooling Sweep) inherits L144 with FS1_TF1_YS1 as frozen baseline.
