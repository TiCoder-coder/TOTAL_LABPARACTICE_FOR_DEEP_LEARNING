# Phase 47 — Final Test Evaluation Report

## 1. Objective

Evaluate three FINAL_REFIT Transformer seeds on the held-out Test set for the
UCI Appliances Energy Prediction task. No model selection, no ensemble,
no best-seed reporting.

## 2. Test-Release and No-Leakage Gate

- Phase46 release verified: PASS
- Phase47 authorized: PHASE_47_FINAL_TEST_EVALUATION
- Test accessed for first time: Phase 47

## 3. Frozen Evaluation Contract

The evaluation contract was frozen before any Test target values were accessed.
Key parameters:
- Lookback: 72
- Features: 33
- Boundary protocol: WB0_CONTEXT_CARRY_OVER
- Metrics: MAE, RMSE, R²
- Seed aggregation: mean ± sample SD (ddof=1)

## 4. Final Test Population

- Population ID: FINAL_TEST_POP-v1
- N_test: 2961
- Boundary: WB0 (context carry-over from observed history)

## 5. Final Transformer Checkpoints

- Seed 42: RUN_TR_FSD_0254_2B11AC68
- Seed 123: RUN_TR_FSD_0254_3858DDA9
- Seed 2026: RUN_TR_FSD_0255_C7E123FB

## 6. Transformer Test Metrics by Seed

| Seed | MAE (Wh) | RMSE (Wh) | R² |
|------|-----------|-----------|-----|
| 42 | 29.5286 | 64.9428 | 0.4893 |
| 123 | 27.1149 | 61.9861 | 0.5347 |
| 2026 | 28.9423 | 64.5601 | 0.4953 |

## 7. Seed Mean ± Sample SD

- MAE: 28.5286 ± 1.2589
- RMSE: 63.8297 ± 1.6080
- R²: 0.5064 ± 0.0247

## 8. Persistence Test Baseline

- MAE: 26.7376
- RMSE: 66.8369
- R²: 0.4590

Transformer mean RMSE beats Persistence.

## 9. LSTM Test Eligibility

- Status: NOT EVALUATED BY PROTOCOL
- Note: LSTM_TUNED_DEV uses L36 lookback (not L72) and was not a FINAL_REFIT symmetric with Transformer. Per Phase47 plan §61, §143.

## 10. No Best-Seed / No Ensemble Statement

All three seeds are reported as predeclared materializations of the same
locked scientific configuration. No seed was selected based on Test performance.
No seed-averaged ensemble was created.

## Overall Status: PASS