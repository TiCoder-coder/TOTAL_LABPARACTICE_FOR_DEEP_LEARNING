## FT01 - Final Experimental and Model Configuration

_Final scientific configuration of the Final Transformer (Phase 45 lock)._

| category | final_setting | source |
|---|---|---|
| Task formulation | Sequence-to-One, one-step-ahead, multivariate regression (target=Appliances Wh) | Phase45 final lock |
| Forecast horizon | 1.000 | Phase45 final lock |
| Sampling interval | 10 min | Phase45 final lock |
| Final feature variant | FS2_TF1 | Phase45 final lock |
| Time-feature setting | embedded time features (encoded via feature set FS2_TF1) | Phase45 final lock |
| Target scaling | YS1 | Phase45 final lock |
| Lookback | 72 steps (12h) | Phase45 final lock |
| Boundary protocol | WB0_CONTEXT_CARRY_OVER | Phase45 final lock |
| Window population policy | VALIDATION (WINDOWPOP-v1) | Phase45 final lock |
| Input feature count | 33.000 | Phase45 final lock |
| Train samples | 13670.000 | Phase45 final lock |
| Validation samples | 2960.000 | Phase45 final lock |
| Test samples | 2961.000 | Phase45 final lock |
| Transformer d_model | 64.000 | Phase45 final lock |
| Attention heads | 4.000 | Phase45 final lock |
| Head dimension | 16.000 | Phase45 final lock |
| Encoder layers | 2.000 | Phase45 final lock |
| FFN dimension | 256.000 | Phase45 final lock |
| Activation | GELU | Phase45 final lock |
| Dropout | 0.100 | Phase45 final lock |
| Positional encoding | SINUSOIDAL | Phase45 final lock |
| Normalization style | post-LN (norm_first = False) | Phase45 final lock |
| Pooling | LAST_STEP | Phase45 final lock |
| Loss name | MSE | Phase45 final lock |
| RevIN | False | Phase45 final lock |
| Final seeds | 42, 123, 2026 | Phase45 final lock |
| Deterministic dataloader seed | 42.000 | Phase45 final lock |
| cudnn.deterministic | True | Phase45 final lock |
| cudnn.benchmark | False | Phase45 final lock |

> Frozen upstream source. See final_table_source_ledger.csv for cell lineage.
> Mean ± SD (ddof=1) of seed-level metrics; descriptive only; NOT an ensemble forecast.
