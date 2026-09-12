# MODEL_IMPROVEMENT_V2 — Final Closure

## Status

`MODEL_IMPROVEMENT_V2_STATUS = COMPLETE`

The final locked prediction policy is the equal-weight mean of the three
Step 16 final-refit `TR_C2_ALT_LOOKBACK_E14_M1` models with seeds 42, 123,
and 2026. Weights remain `[1/3, 1/3, 1/3]`; no best seed was selected.

## Locked development evidence

| Metric | Value |
|---|---:|
| Pooled RMSE | 58.57180673355406 Wh |
| Pooled MAE | 25.545269885543682 Wh |
| Pooled R² | 0.5967752914420708 |
| Worst-fold RMSE | 67.53675009939884 Wh |
| Fold-RMSE SD | 8.879494624591965 Wh |

## Step 17 historical benchmark

The mandatory label is `POST_HOC_V2_BENCHMARK`. The old Test was accessed
once, with 2,961 ordered targets. This is not an unbiased unseen-Test result.

| Model/policy | RMSE (Wh) | MAE (Wh) | R² |
|---|---:|---:|---:|
| V2 equal-weight ensemble | 61.608936806100665 | 25.897129774085265 | 0.5403636015384808 |
| V1 historical Transformer mean | 63.8296583863937 | 28.52860338489352 | 0.5064220688328526 |
| Persistence last value | 66.83691534084765 | 26.73758865248227 | 0.459046692018256 |

V2 minus V1 historical mean is −2.220721580293038 Wh RMSE,
−2.6314736108082535 Wh MAE, and +0.033941532705628275 R². V2 minus
Persistence is −5.227978534746988 Wh RMSE, −0.8404588783970048 Wh MAE,
and +0.08131690952022486 R².

The V1 delta is contextual: V1 reports the mean of per-seed metrics whereas
V2 reports the metric of averaged predictions. It must not be presented as a
matched ensemble-policy comparison.

## Audit conclusion

The Step 16 lock, three checkpoint hashes, config fingerprints, X/Y scaler
hashes, Step 17 manifest chain, five prediction hashes, population identity,
ensemble arithmetic, and recomputed metrics all pass. Step 17 records no
training, scaler fitting, best-seed selection, weight change, or post-Test
retuning. Scientific conclusions and configuration were not changed after
Test access.

## Remaining reporting debt

- The old Test had already been observed during V1 and therefore provides
  post-hoc historical evidence only.
- The V1 mean-metric and V2 ensemble-metric policies are not symmetric.
- The generic E20 rolling-origin manifest carries seed 42 in the seed
  123/2026 branches; run-level provenance remains valid.
- No portable peak-memory contract was recorded.
- Phase 58/59 remain frozen V1 reporting; the V2 closure is appended as a
  separate notebook section instead of rewriting historical conclusions.

Canonical machine-readable closure:
`artifacts/model_improvement_v2/model_improvement_v2_final_closure.json`.
