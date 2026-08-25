# Phase 40 — S18 RevIN Sweep Report

## 1. Objective

Under the frozen Phase 39 GC1 configuration (FV=FS2_TF1, YS=YS1, L=36, pooling=LAST_STEP,
activation=GELU, batch=32, lr=0.0003, wd=0.001, dropout=0.1, D=64, H=4, HD=16, N=2,
F=256, loss=MSE, epochs=50, GC=1.0), evaluate whether adding **target-selective
Reversible Instance Normalization** improves Validation RMSE Wh.

## 2. Scientific Comparison

| Condition | RevIN | Mode      | Run ID                  | Validation RMSE (Wh) |
|-----------|-------|-----------|-------------------------|----------------------|
| RN0       | OFF   | REUSE_REF | `RUN_TR_S14_0023_A711A9B8` | 57.696800       |
| RN1       | ON    | TRAIN_NEW | `RUN_TR_S18_0031_A711A9B8` | 62.286958       |

RMSE margin (RN0 - RN1) = -4.590158 Wh

## 3. Strict BEST Verification

Tolerance: 1e-6 (Phase 39 convention; absolute & relative).
Verification device: mps.
Recorded device: mps.

| Metric | Stored | Recomputed | Δ |
|--------|--------|------------|---|
| RMSE Wh | 62.2869581617 | 62.2869581617 | 0.00e+00 |
| MAE Wh  | 28.5819273826  | 28.5819273826  | 0.00e+00 |
| R²      | 0.5440007380      | 0.5440007380      | 0.00e+00 |

Strict BEST verdict: **PASS**.
Wrapped-model shape match, population-fingerprint match, sample-order match all PASS.

## 4. RN1 Architecture

- Backbone: identical to RN0 (D64, H4, HD16, N2, F256, GELU, dropout=0.1, LAST_STEP)
- RevIN wrapper: `RevINWrappedTransformerRegressor` (TargetSelectiveRevIN + backbone)
- RevIN channels: 28
- Passthrough channels (time features): 5
- Target channel: Appliances (original_index=25, subset_index=25)
- Eps = 1e-05, affine = True, gamma init = 1.0, beta init = 0.0
- Statistics: per-sample per-feature, across TIME dim only, detached, no running stats
- Variances: population variance, unbiased=False

## 5. Parameter delta

- RN0 trainable: 102209
- RN1 trainable: 102265
- Expected delta: 2 * 28 = 56
- Observed delta: 56
- Delta pass: **True**

## 6. Channel scope and time-feature passthrough

- Time-feature set (passthrough, never normalized): hour_sin, hour_cos, dow_sin, dow_cos, weekend
- All other 28 signal channels (G1 historical Appliances + G2 raw continuous
  exogenous + G3 rv1/rv2 if present) are RevIN-normalized
- x_model feature order unchanged after RevIN normalization + scatter

## 7. X→Y coordinate bridge

- After backbone regression head (normalized coordinate)
- RevIN target denormalization (using historical Appliances statistics from window X_{t-L+1:t})
- → X-target coordinate
- → inverse frozen X-target transform
- → raw Wh
- → frozen Y target transform (YS1)
- → y_model (loss applied here)

No Validation/Test scaler refit. No future-target leakage.

## 8. Gradient diagnostics (RN1, GC1 with RevIN)

- clip_max_norm: 1.0
- clip_order: ZERO_GRAD_FORWARD_CRITERION_BACKWARD_CLIP_OPTIMIZER_STEP
- total_batches: 7704
- clipped_batches: 6794
- clipping_fraction: 0.8819
- mean_preclip_global_grad_norm: 2.4130
- max_preclip_global_grad_norm: 27.1190
- nonfinite_grad_events: 0

## 9. Convergence diagnostics

- best_epoch: 8
- epochs_executed: 18 (early-stopped; patience=10)
- stop_reason: EARLY_STOPPING

## 10. Hypothesis outcomes

- H-S18-01 (RevIN reduces local variation): **REJECTED**
- H-S18-02 (RN0 better if local scale is predictive): **ACCEPTED**
- H-S18-03 (RevIN redundant under global scaling): **ACCEPTED**
- H-S18-04 (benefit depends on FS/L): DOCUMENTED_LIMITATION (single-seed, single-config)

## 11. Winner

Under the canonical rule `argmin(full-precision Validation RMSE Wh)`:

- Selected RevIN: **RN0**
- Selection basis: **EMPIRICAL_COMPARISON_RMSE_HIGHER_RN0_RETAINED**
- Winner run_id: **RUN_TR_S14_0023_A711A9B8**
- Winner RMSE Wh: 57.696800
- Test access: FORBIDDEN throughout

## 12. Reporting statement

Adding the registered target-selective RevIN setup did not improve Validation RMSE under the current frozen configuration, so RN0 was retained.

## 13. Inherited warnings

- H4_SOURCE_ARTIFACT_RETENTION_INCOMPLETE (carried from Phase 39)

## 14. Phase 41 handoff

- Phase 41 (S19 Boundary protocol sweep) is approved.
- RevIN setting is preserved as selected.
- Boundary protocol remains WB0 (Phase 41 will compare WB0 vs WB1).

Generated at: 2026-08-24T15:09:06.660916+00:00
