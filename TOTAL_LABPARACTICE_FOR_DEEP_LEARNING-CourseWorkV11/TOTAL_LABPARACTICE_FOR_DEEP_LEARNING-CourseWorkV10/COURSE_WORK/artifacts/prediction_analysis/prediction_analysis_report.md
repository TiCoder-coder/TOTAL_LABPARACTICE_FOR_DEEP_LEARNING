# Phase 48 — Prediction Analysis Report

**Project:** UCI Appliances Energy Prediction
**Phase:** 48 — Prediction Analysis
**Version:** PREDICTION_ANALYSIS-v1
**Upstream:** FINAL_TEST_EVAL-v1 (Phase 47, PASS)
**Test population fingerprint:** `d7dbc0b3cde772dc…` (N_TEST = 2961)

---

## 1. Purpose

Phase 48 is a descriptive analysis of how the three frozen Phase 47 Transformer
prediction bundles (seeds 42 / 123 / 2026) and the Persistence baseline behave on
the held-out Test set. It does not re-run Test inference, train any model, fit
scalers, load checkpoints, select a best seed, compute an ensemble metric, or
shift/clip/recalibrate predictions.

## 2. Frozen prediction sources

| Bundle | Path | sha256 (prefix 12) |
|---|---|---|
| seed42 | `artifacts/final_test/predictions/final_test_predictions_seed42.csv` | `246ee0d725af` |
| seed123 | `artifacts/final_test/predictions/final_test_predictions_seed123.csv` | `1bb55c445ffe` |
| seed2026 | `artifacts/final_test/predictions/final_test_predictions_seed2026.csv` | `bfb575357dd6` |
| Persistence | `artifacts/final_test/predictions/final_test_predictions_persistence.csv` | `7115af1c479b` |

All four source bundles remain byte-identical to their Phase 47 frozen
counterparts (verified after every Phase 48 sub-slice).

## 3. Test population

- **Population ID:** `FINAL_TEST_POP-v1`
- **N_TEST:** 2961
- **Fingerprint:** `d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87`
- **Boundary protocol:** WB0 (context carry-over from observed history)
- **Cadence:** 10 minutes, perfect continuity across the entire Test period

## 4. Distribution behavior

See `prediction_distribution_summary.csv` for 5 series (ACTUAL, SEED42,
SEED123, SEED2026, SEED_MEAN_DESCRIPTIVE) × 12 stats. All values are
descriptive. **SEED_MEAN_DESCRIPTIVE is not an evaluated ensemble.**

## 5. Range / variance compression

See `prediction_range_compression.csv` for per-seed std/IQR/range ratios
plus mean shift. Interpretation language: descriptive compression only,
NOT calibration.

## 6. Change behavior (gap-safe)

See `prediction_change_summary.csv` — first differences computed only
across exact 10-minute transitions. Valid transition count = 2960,
gaps excluded = 0 (perfect cadence).

## 7. Direction-of-change agreement

See `prediction_direction_agreement.csv`. Three-class sign policy
(NEGATIVE / ZERO / POSITIVE), no epsilon threshold. Both exact 3-class
rate and nonzero-only rate reported per seed.

## 8. Lag diagnostics (diagnostic only)

See `prediction_lag_diagnostics.csv`. Lag range = -6..+6 steps. Lag
convention: `lag_k_positive_means_prediction_compared_to_truth_shifted_k_future_steps`. Predictions are **NOT**
shifted by the apparent best-correlation lag. Lag 0 remains the canonical
in-sample comparison; the apparent peak at lag = -1 is descriptive only.

## 9. Prediction ACF (NOT residual ACF)

See `prediction_acf_diagnostics.csv`. Registered lags:
[1, 6, 12, 36, 72, 144] (in steps). ACTUAL + 3 seeds + SEED_MEAN_DESCRIPTIVE
series. Residual ACF explicitly not computed (Phase 49).

## 10. Local extrema

See `prediction_local_extrema_summary.csv`. Fixed definition:
- Local max: `y_t > y_{t-1} AND y_t >= y_{t+1}`
- Local min: `y_t < y_{t-1} AND y_t <= y_{t+1}`
- Edges of contiguous segments NOT eligible.
- Both neighbours must be exactly 10 minutes apart.

## 11. Peak timing (window fixed at ±1 step)

See `prediction_peak_timing_summary.csv`. Window = ±1 step = ±10 minutes.
Window is **frozen**; never widened after inspecting results.

## 12. Cross-seed agreement

See `prediction_seed_pairwise_agreement.csv`. Three pairs:
42-123, 42-2026, 123-2026. Each row is labelled `SEED_AGREEMENT_DIAGNOSTIC`
(not a model-performance metric). Pearson + Spearman + mean_abs_diff +
RMSE_between_predictions + max_abs_diff.

## 13. Cross-seed spread (descriptive)

See `prediction_seed_spread.csv` (2961 rows). Columns:
seed_mean, seed_std (ddof=1), seed_min, seed_max, seed_range. The
`spread_semantics` column is `CROSS_SEED_PREDICTION_SPREAD`. **Never**
labelled confidence interval, uncertainty interval, predictive uncertainty,
or calibrated uncertainty.

## 14. Top-20 seed disagreement

See `prediction_top_seed_disagreement.csv` (K=20 exactly). Ranked by
`seed_range_prediction` descending with deterministic tie-break by
`target_id` ascending. This is **not** worst-error analysis; worst-error
ranking is deferred to Phase 51.

## 15. Rolling 24h tracking

See `prediction_rolling_tracking.csv` (8454 rows =
2818 windows × 3 seeds). Every emitted window has
`window_valid_count = 144`. Windows that cross a temporal gap
are skipped (no forward fill, no partial substitution).

## 16. Negative / saturation audits

- `prediction_negative_value_audit.csv` — 0 negative predictions across
  all 3 seeds. No clipping applied.
- `prediction_saturation_audit.csv` — no structural saturation signal
  (unique_prediction_count == N for all seeds).

## 17. Persistence baseline context

See `prediction_baseline_context.csv`. Two rows:
- `PERSISTENCE` — bundle sha256 verified, population match verified,
  descriptive stats populated. Phase 47 final comparison verdict
  preserved: **Transformer better on RMSE and R²; Persistence better on
  MAE**.
- `LSTM_TUNED_DEV` — `prediction_bundle_available = False`,
  `interpretation_label = NOT_ELIGIBLE_CONFIG_MISMATCH`. No LSTM
  predictions were fabricated.

## 18. Key scientific findings

See `prediction_analysis_findings.csv`. Findings are descriptive only.
Phrasing per plan §128.

## 19. Important limitations

- This is descriptive analysis over Test predictions. It does NOT establish
  causality.
- Seed spread is NOT a confidence interval.
- Lag diagnostics are NOT used to shift predictions.
- Peak/trough capture is NOT a calibration claim.
- Persistence vs Transformer comparison is per Phase 47 metrics only;
  no new ranking is performed.

## 20. What Phase 48 intentionally did NOT do

- No new Test inference
- No checkpoint loading
- No training / no optimizer / no backward
- No scaler fitting
- No best-seed selection
- No evaluated ensemble metric
- No prediction shifting
- No clipping / recalibration
- No residual analysis (Phase 49)
- No regime-specific error analysis (Phase 50)
- No worst-error ranking (Phase 51)
- No attention analysis (Phase 52+)
- No modification of any Phase 47 source bundle

## 21. Handoff to Phase 49 / 50 / 51

- `phase49_residual_analysis_handoff.json` — `residual_convention =
  "y_true - y_pred"`. Phase 48 did NOT execute residual analysis.
- `phase50_error_regime_context_handoff.json` — `phase50_threshold_policy
  = "TRAIN_DERIVED_ONLY"`. Any regime thresholds must come from Train, not
  from held-out Test after inspection.
- `phase51_worst_error_context_handoff.json` — `phase48_executed_worst_error_ranking
  = False`. Worst-error ranking is Phase 51's responsibility.

---

**End of report — Phase 48 PREDICTION_ANALYSIS-v1.**
