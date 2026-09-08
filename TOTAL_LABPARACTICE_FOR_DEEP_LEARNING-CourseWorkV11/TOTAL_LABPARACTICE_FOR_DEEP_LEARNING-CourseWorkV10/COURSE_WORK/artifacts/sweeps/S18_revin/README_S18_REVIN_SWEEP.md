# S18 RevIN Sweep — Phase 40 (POST-TRAINING)

## Status

**PASS — empirical comparison complete**

## Comparison

| Condition | RevIN | Mode      | RMSE Wh                  | Run ID                  |
|-----------|-------|-----------|--------------------------|-------------------------|
| RN0       | OFF   | REUSE_REF | 57.696800           | `RUN_TR_S14_0023_A711A9B8` |
| RN1       | ON    | TRAIN_NEW | 62.286958           | `RUN_TR_S18_0031_A711A9B8` |

Strict BEST verification: PASS (canonical absolute tolerance 1e-09, comparison_rule=absolute_only, cpu_fallback=DISABLED, recorded_device=mps, verification_device=mps).

## Winner

**RN0** (EMPIRICAL_COMPARISON_RMSE_HIGHER_RN0_RETAINED)

## Files

All O40.1–O40.49 outputs are present in this directory. See
`s18_revin_sweep_report.md` for the canonical narrative.

## Generated

2026-08-24T15:09:06.661035+00:00
