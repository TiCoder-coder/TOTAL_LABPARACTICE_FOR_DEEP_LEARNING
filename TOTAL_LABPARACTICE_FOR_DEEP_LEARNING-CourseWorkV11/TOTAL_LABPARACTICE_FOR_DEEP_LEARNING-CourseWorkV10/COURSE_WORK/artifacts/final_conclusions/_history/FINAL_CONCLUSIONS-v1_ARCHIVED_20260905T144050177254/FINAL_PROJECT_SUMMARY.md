# Final Project Summary

## 1. Project Objective

Multivariate time-series regression for UCI Appliances Energy Prediction using a
Transformer Encoder, benchmarked against tuned LSTM and Persistence baselines,
with temporal attention analysis at map, head, error-conditioned, and cross-seed
levels.

## 2. Final Protocol

- Task: Sequence-to-One, one-step-ahead, multivariate regression (H=1, 10 min)
- Boundary: WB0 (previously observed targets assumed available)
- Final model: Transformer Encoder (FS2_TF1 feature variant, embedded time features)
- Seeds: 42, 123, 2026 (predeclared before final runs)
- Test population: FINAL_TEST_POP-v1 (N = 2961 targets)
- Lookback: 72 steps = 12 h

## 3. Final Model

Transformer Encoder with multi-head self-attention (L=2, H=4 per layer),
position embedding, time-feature embedding, and feed-forward sub-layers.

## 4. Final Performance

- Final Transformer (three-seed mean ± SD, ddof=1):
  MAE 28.53 Wh | RMSE 63.83 Wh | R² 0.506
- Persistence Baseline: MAE 26.74 Wh | RMSE 66.84 Wh | R² 0.459
- Tuned LSTM: Not evaluated on FINAL_TEST_POP-v1 (lookback mismatch L36 vs L72)

## 5. Baseline Comparison

Persistence and Tuned LSTM baselines are reported as required comparison
references. Both favorable and unfavorable comparisons are reported.

## 6. Robustness

Rolling-origin development robustness evidence is available (Phase 44/FT03).
This is development evidence only; it is not a second Held-Out Test.

## 7. Error Diagnostics

Error concentration in high-consumption and rapid-change regimes; worst errors
are valid frozen observations (not deleted).

## 8. Attention Findings

- Recent-history attention (last-query): substantial mass on recent lags
- Head profiles: non-identical within each layer
- Error-conditioned: associations observed (descriptive, non-causal)
- Cross-seed: layer head-mean more consistent than individual heads

## 9. Seed Stability

Three-seed evaluation provides limited stochastic robustness. Layer head-mean
(permutation-invariant) is the primary stability view. Matching ambiguity
and partial cycle consistency are reported.

## 10. Limitations

13 upstream caveats propagated from FA12. Key categories: single-house
dataset (L1), H=1 scope (L2), sequential tuning (L3), three seeds (L3),
time-series dependence (L4), attention as temporal diagnostic only (L5),
no deployment (L6).

## 11. Future Work

10 future-work items documented, each linked to a limitation. Priority:
P1 = external validation; P2 = multi-step + deployment; P3 = ablation;
P4 = more seeds; P5 = additional baselines.

## 12. Reproducibility Package

Final lock SHA: `585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24`
Final Test population SHA: `d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87`
Seeds: 42, 123, 2026
Final tables: FT01–FT10 (FINAL_TABLES-v1)

## 13. Completion Status

Phase 0–59 execution plan: COMPLETE
Scientific narrative frozen: TRUE
