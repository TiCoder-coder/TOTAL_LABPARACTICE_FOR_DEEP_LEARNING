# Phase 50 — Error-by-Regime Analysis Report

**Phase 50 status**: PASS  
**Ready for Phase 51**: YES  
**Authority**: Phase 49 → Phase 50 handoff (TRAIN_DERIVED_ONLY)  
**Residual convention**: y_true − y_pred (positive = UNDERPREDICTION)

Regime thresholds are derived from TRAIN only; no Test-derived threshold tuning is performed.

## 1. Frozen TRAIN-derived thresholds

- R1 Q25_y = 50.0 Wh (target_level: LOW/MID split)
- R1 Q75_y = 100.0 Wh (target_level: MID/HIGH split)
- R2 Q90_y = 210.0 Wh (extreme_high flag)
- R3 Q90_abs_delta = 80.0 Wh (rapid change flag)

All four quantiles derived from REGIME_REFERENCE_TRAIN-v1 (N=13,670)  
using numpy.quantile(method='linear', dtype=float64).

## 2. Six predeclared regime families

- R1 TARGET_LEVEL: TL_LOW / TL_MID / TL_HIGH
- R2 EXTREME_HIGH: EXTREME_HIGH (y ≥ Q90) / NON_EXTREME
- R3 CHANGE_MAGNITUDE: CHANGE_RAPID (|Δy| ≥ Q90_abs_delta) / CHANGE_NORMAL / CHANGE_UNCLASSIFIED
- R4 CHANGE_DIRECTION: DIR_UP / DIR_FLAT / DIR_DOWN / DIR_UNCLASSIFIED (EXACT_ZERO only; no epsilon)
- R5 TIME_OF_DAY: TOD_NIGHT / MORNING / AFTERNOON / EVENING
- R6 DAY_TYPE: DAY_WEEKDAY / DAY_WEEKEND

## 3. Test regime prevalence (frozen)

- TL_LOW: Test N=120 (4.1%) | Train N=2645 (19.3%) | delta=-15.30pp
- TL_MID: Test N=2036 (68.8%) | Train N=7405 (54.2%) | delta=+14.59pp
- TL_HIGH: Test N=805 (27.2%) | Train N=3620 (26.5%) | delta=+0.71pp

Note: TL_LOW (N=120) and EXTREME_HIGH (N=248) have relatively small Test support, so their regime-level estimates should be interpreted cautiously.

## 4. Per-seed global metrics (Transformer)

- seed=123: MAE=27.115 Wh, RMSE=61.986 Wh
- seed=2026: MAE=28.942 Wh, RMSE=64.560 Wh
- seed=42: MAE=29.529 Wh, RMSE=64.943 Wh

## 5. RMSE by target level (cross-seed mean)

- TL_LOW: mean RMSE = 20.730 Wh, sd = 0.727 Wh
- TL_MID: mean RMSE = 23.498 Wh, sd = 2.489 Wh
- TL_HIGH: mean RMSE = 116.257 Wh, sd = 2.912 Wh

## 6. Persistence comparison

Persistence global MAE = 26.738 Wh (vs Transformer mean MAE ≈ 28.5 Wh).  
Persistence global RMSE = 66.837 Wh (vs Transformer mean RMSE ≈ 63.8 Wh).  
Per Phase 47 global interpretation: Persistence better MAE globally; Transformer better RMSE globally.  Inside specific regimes this may differ — see figures and regime_metrics_persistence.csv.

## 7. Descriptives only

All findings in this Phase are descriptive (no causal claims).  
No threshold tuning, no best-seed selection, no ensemble, no 3N iid pooling.

## 8. Safety invariants verified

- training = false
- new_test_inference = false
- checkpoint_loading = false
- optimizer_steps = 0
- scaler_fit = false
- best_seed_selected = false
- ensemble = false
- three_n_iid_interpretation = false
- test_derived_threshold_used = false
- cartesian_regime_mining = false
- prediction_correction = false
- worst_error_ranking_executed = false
- attention_analysis_executed = false
- phase47_modified = false
- phase48_modified = false
- phase49_modified = false

## 9. Provenance (read-only, from frozen Phase 50 artifacts)

| Artifact | sha256 |
|---|---|
| phase_50_signoff | `cd9da0f8a0490f8874fbe5264d53e28e22b30ca4203c603b9710804281880c80` |
| phase50_findings.json | `f364649d570f3d9bfdef41f5c4e15bd0a902cd7c425dbaaca6533629bc78710c` |
| phase51_handoff.json | `e5ee74215a40ad3249c57906f6686f10ff42ebe3112f1df622f308a0204f8c33` |
| regime_thresholds_train_only.json | `2fe9ad4f873e3b3e42013fe3b2d630e377e4e568120769bd2234a76d6974b109` |
| regime_cross_seed_summary.csv | `caf29ec39f05d3145cdb2eb0629fa590bc08b68daaad1f306727dfe4312a2f12` |
| test_regime_assignment.csv | `e90553cfc747a3f15e0e9ec9e6868ae497e7ade797dc14a81999e416b74219ac` |
| test_population (N=2961) | `d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87` |
| Phase 51 handoff SHA | `e5ee74215a40ad3249c57906f6686f10ff42ebe3112f1df622f308a0204f8c33` |
