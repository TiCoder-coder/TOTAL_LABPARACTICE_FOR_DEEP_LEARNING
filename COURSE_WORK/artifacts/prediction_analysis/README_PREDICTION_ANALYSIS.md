# Phase 48 — Prediction Analysis

This directory contains the descriptive prediction-behavior analysis of the
three frozen Phase 47 Transformer prediction bundles (seeds 42 / 123 / 2026)
and the Persistence baseline on the held-out Test set.

## Source lineage

Source predictions are byte-identical to:
- `artifacts/final_test/predictions/final_test_predictions_seed42.csv`
- `artifacts/final_test/predictions/final_test_predictions_seed123.csv`
- `artifacts/final_test/predictions/final_test_predictions_seed2026.csv`
- `artifacts/final_test/predictions/final_test_predictions_persistence.csv`

Each source bundle's sha256 is recorded in
`prediction_analysis_manifest.json` and re-verified after every Phase 48
sub-slice.

## File meanings

| File | Meaning |
|---|---|
| `prediction_analysis_manifest.json` | O48.1 — analysis manifest, source SHA256s, scope flag |
| `prediction_analysis_contract.json` | O48.2 — frozen parameters (lag, ACF lags, peak window, etc.) |
| `phase48_preflight_audit.csv` | O48.3 — 16 preflight gates |
| `prediction_source_verification.csv` | O48.4 — per-source sha256 + row count verification |
| `prediction_alignment_audit.csv` | O48.5 — target_ids / timestamps / y_true alignment |
| `prediction_wide_table.csv` | O48.6 — 2961 × 11 aligned predictions table |
| `prediction_long_table.csv` | O48.7 — 11844 × 7 long table (Persistence normalized to "PERSISTENCE") |
| `prediction_distribution_summary.csv` | O48.8 — 5 series × 12 stats |
| `prediction_range_compression.csv` | O48.9 — per-seed compression ratios |
| `prediction_change_summary.csv` | O48.10 — gap-safe first-difference stats |
| `prediction_direction_agreement.csv` | O48.11 — 3-class direction agreement |
| `prediction_lag_diagnostics.csv` | O48.12 — lag -6..+6 Pearson correlations |
| `prediction_acf_diagnostics.csv` | O48.13 — registered-lag ACF (no residual ACF) |
| `prediction_local_extrema_summary.csv` | O48.14 — true local extrema + prediction at extrema |
| `prediction_peak_timing_summary.csv` | O48.15 — peak timing ±1 step |
| `prediction_seed_pairwise_agreement.csv` | O48.16 — 3 seed-pair agreement metrics |
| `prediction_seed_spread.csv` | O48.17 — per-target cross-seed spread |
| `prediction_top_seed_disagreement.csv` | O48.18 — top-20 by seed_range |
| `prediction_rolling_tracking.csv` | O48.19 — gap-safe 144-point rolling tracking |
| `prediction_negative_value_audit.csv` | O48.20 — negative prediction audit |
| `prediction_saturation_audit.csv` | O48.21 — structural saturation audit |
| `prediction_baseline_context.csv` | O48.22 — Persistence + LSTM ineligibility context |
| `prediction_analysis_findings.csv` | O48.27 — descriptive findings (with safe wording) |
| `phase49_residual_analysis_handoff.json` | O48.28 — Phase 49 handoff (residual_convention = y_true − y_pred) |
| `phase50_error_regime_context_handoff.json` | O48.29 — Phase 50 handoff (TRAIN-derived thresholds) |
| `phase51_worst_error_context_handoff.json` | O48.30 — Phase 51 handoff (no worst-error ranking in Phase 48) |
| `prediction_analysis_tests.csv` | O48.31 — focused test summary |
| `prediction_analysis_discrepancies.json` | O48.32 — discrepancy log |
| `prediction_analysis_summary.json` | O48.33 — cross-cutting summary |
| `prediction_analysis_report.md` | O48.34 — human-readable report |
| `README_PREDICTION_ANALYSIS.md` | O48.35 — this file |
| `phase_48_signoff.json` | O48.36 — strict phase signoff |
| `figures/` | O48.23–O48.26 — 20 deterministic PNG figures |

## Seed mean semantics

`seed_mean_prediction` is the arithmetic mean of the three Transformer
predictions at the same timestamp. It is **descriptive only**, NOT an
evaluated ensemble. It is never used as a primary metric.

## Seed spread semantics

`seed_std_prediction` (sample SD, ddof=1) and `seed_range_prediction`
(max − min) are **CROSS_SEED PREDICTION SPREAD** — they quantify
stochastic variability across the three predeclared seeds, NOT a
confidence interval, NOT a predictive uncertainty band, NOT a
calibrated uncertainty estimate.

## Lag convention

`lag_k_positive_means_prediction_compared_to_truth_shifted_k_future_steps`

Concretely, for k > 0 we correlate `pred[0..N-k-1]` with `true[k..N-1]`;
for k < 0 we correlate `pred[|k|..N-1]` with `true[0..N-|k|-1]`. The lag
yielding the highest correlation is reported as a **diagnostic only**;
predictions are NEVER shifted by that lag.

## Rolling semantics

Rolling window is exactly 144 samples (24h at 10-min cadence). Every
emitted window is verified to have all 143 within-window transitions be
exactly 10 minutes; otherwise the window is skipped (no forward fill, no
partial substitution). Only `window_valid_count == 144` rows are emitted.

## Reproducibility

All Phase 48 artifacts are derived from immutable source bundles. Re-running
`materialize_phase48()`, `materialize_phase48c()`, `materialize_phase48d()`,
`materialize_phase48e()` (in order) reproduces the canonical outputs.

## Forbidden interpretation

This directory does NOT contain:
- Best-seed selection (all 3 seeds treated symmetrically)
- Evaluated ensemble performance metrics
- Shifted / clipped / recalibrated predictions
- Residual analysis (deferred to Phase 49)
- Regime-specific error analysis (deferred to Phase 50)
- Worst-error ranking (deferred to Phase 51)
- Attention analysis (deferred to Phase 52+)
