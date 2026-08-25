# Phase 39 S17 Gradient-Clipping Sweep Report

## 1. Objective
Compare gradient clipping OFF (GC0) vs global L2 max_norm=1.0 (GC1) under frozen configuration.

## 2. Reference (GC1 reuse)
GC1 reuses RUN_TR_S14_0023_A711A9B8 from Phase 36 S14 F256 reference.
GC1 Validation RMSE: 57.69679988114431

## 3. Conditions
GC0: clipping OFF, finite guard ON, preclip telemetry ON
GC1: global L2 norm clipping, max_norm=1.0

## 4. Architecture
D64, H4, head_dim16, N2, F256, GELU, dropout=0.1, LAST_STEP pooling

## 5. Data
FS2_TF1, YS1, L36, seed=42

## 6. Optimizer
AdamW lr=0.0003 wd=0.001, max_epochs=50, patience=10

## 7. GC0 Run
RUN_TR_S17_0029_082F7FF5, epochs=22, best_epoch=12
GC0 Validation RMSE: 58.87856773160628

## 8. Strict BEST Verification
GC0 strict BEST: PASS (tolerance=1e-6)
- Stored RMSE: 58.87856773160628
- Recomputed RMSE: 58.878567640388276
- Delta: 9.12e-08

## 9. Selection
Winner: GC1
GC0 vs GC1 delta: -1.181768

## 10. Gradient Diagnostics
GC0 actual clipping fraction: 0.0
GC1 actual clipping fraction: 0.929375531011045
GC0 max preclip norm: 18.315928813829274
GC1 max preclip norm: 17.934526443481445
GC0 non-finite events: 0

## 11. Test Status
FORBIDDEN - no Test access

## 12. Inherited Warnings
H4_SOURCE_ARTIFACT_RETENTION_INCOMPLETE

## 13. Phase 40
GC1 wins, GC0 not selected for Phase 40
