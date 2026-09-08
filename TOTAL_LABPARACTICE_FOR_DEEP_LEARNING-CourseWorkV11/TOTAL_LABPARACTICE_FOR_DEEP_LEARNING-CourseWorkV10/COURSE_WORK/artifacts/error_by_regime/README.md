# Phase 50 — Error-by-Regime Analysis

## Description

Phase 50 analyses the Transformer per-seed Test residuals broken down by six predeclared regime
families (TARGET_LEVEL, EXTREME_HIGH, CHANGE_MAGNITUDE, CHANGE_DIRECTION, TIME_OF_DAY, DAY_TYPE).
All thresholds are derived from REGIME_REFERENCE_TRAIN-v1 (N=13,670) using numpy.quantile(linear).

## Leakage firewall

- Stage A: Train-only thresholds (no Test/Validation values used).
- Stage B: Test regime assignment using ONLY target_id, y_true_wh, timestamp, continuity, frozen Train thresholds.
- Stage C: Test assignment freeze + checksum.
- Stage D: Phase 49 residual join (read-only).

## Artifacts

- regime_reference_train_manifest.json
- regime_thresholds_train_only.json
- test_regime_assignment.csv (frozen at Stage C)
- regime_metrics_long.csv (per-seed × per-regime metrics)
- regime_cross_seed_summary.csv (mean + SD with ddof=1)
- regime_rmse_lift.csv (rmse_lift_wh / rmse_lift_ratio / rmse_lift_pct)
- regime_pairwise_contrasts.csv (predeclared contrasts only)
- regime_train_vs_test_prevalence.csv
- regime_rank_stability.csv (Spearman)
- regime_metrics_persistence.csv
- regime_metrics_lstm.csv (NOT_APPLICABLE)
- regime_seed_spread.csv (CROSS_SEED_PREDICTION_SPREAD only)
- regime_sign_consensus.csv (Phase 49 → Phase 50 mapping frozen)
- phase50_findings.json
- phase50_discrepancies.json
- phase50_report.md
- phase51_handoff.json
- phase_50_signoff.json
- figures/

## Forbidden

- No training, no new Test inference, no checkpoints loaded.
- No best-seed selection, no ensemble, no 3N iid pooling.
- No Test-derived thresholds, no Cartesian regime mining.
- No worst-error ranking (deferred to Phase 51).
