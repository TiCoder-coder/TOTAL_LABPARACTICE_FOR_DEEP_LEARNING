## FT02 - Final Held-Out Test Performance

_Persistance vs Tuned LSTM vs Final Transformer (3 seeds + 3-seed summary)._

| model | seed | mae_wh | rmse_wh | r2 | evaluation_population | notes |
|---|---|---|---|---|---|---|
| Persistence Baseline | single | 26.74 | 66.84 | 0.459 | FINAL_TEST_POP-v1 (2961 targets) | single final Test realization; no fake SD |
| Tuned LSTM Baseline | single | N/A | N/A | N/A | FINAL_TEST_POP-v1 (NOT_EVALUATED_BY_PROTOCOL; L36 vs L72 mismatch) | Phase 47: LSTM not directly comparable on FINAL_TEST_POP-v1 (Tuned LSTM was trained with lookback=36; Final Transformer uses lookback=72). Per Phase 47 plan §141. |
| Final Transformer — Seed 42 | 42.000 | 29.53 | 64.94 | 0.489 | FINAL_TEST_POP-v1 (2961 targets) | checkpoint_sha256=c3cfad116aa91d47... |
| Final Transformer — Seed 123 | 123.000 | 27.11 | 61.99 | 0.535 | FINAL_TEST_POP-v1 (2961 targets) | checkpoint_sha256=8a134fec517be0df... |
| Final Transformer — Seed 2026 | 2026.000 | 28.94 | 64.56 | 0.495 | FINAL_TEST_POP-v1 (2961 targets) | checkpoint_sha256=8753800539f7a617... |
| Final Transformer — Three-Seed Summary | 42,123,2026 | 28.53 | 63.83 | 0.506 | FINAL_TEST_POP-v1 (2961 targets per seed) | mean ± sample SD (ddof=1). Descriptive run-variability; NOT an ensemble forecast. SD on R² refers to variability on raw seed R² values — not pooled over targets. |

> Frozen upstream source. See final_table_source_ledger.csv for cell lineage.
> Mean ± SD (ddof=1) of seed-level metrics; descriptive only; NOT an ensemble forecast.
