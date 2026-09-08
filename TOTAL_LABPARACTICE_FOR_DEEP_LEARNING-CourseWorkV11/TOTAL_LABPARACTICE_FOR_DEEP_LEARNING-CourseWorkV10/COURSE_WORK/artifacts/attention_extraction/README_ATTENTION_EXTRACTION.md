# Phase 52 -- Attention Extraction (README)

## 1. Canonical attention layout

```
A[b, h, q, s]
b = batch sample
h = attention head
q = query temporal position (0 = oldest, L-1 = newest historical)
s = source/key temporal position
```

Last-query attention is the row at A[:, :, L-1, :] -- the attention of the
newest historical token over all source positions. It is NOT A[:, :, :, L-1].

## 2. Two-tier storage (frozen, float32)

Tier A -- Dense case attention (frozen Phase 51 case set):
  artifacts/attention_extraction/raw/dense_case_attention_seed{42,123,2026}.npz
  shape: [K, num_layers, num_heads, L, L] float32

Tier B -- All-Test last-query attention (every FINAL_TEST_POP-v1 target):
  artifacts/attention_extraction/raw/last_query_attention_seed{42,123,2026}.npz
  shape: [N_test, num_layers, num_heads, L] float32

## 3. Position / lag mapping

- Position 0 = oldest historical input (lag = L = 72 -> 720 min)
- Position L-1 = newest historical input (lag = 1 -> 10 min)
- Forecast target is NOT an attention token
- LagSteps_p = L - p (H = 1)
- LagMinutes_p = 10 * LagSteps_p

## 4. Masks

- causal_mask = None
- padding_mask = None

## 5. Pooling caveat

Pooling = LAST_STEP: last_query directly corresponds to pooled token.
If pooling were MEAN, dense maps would be more important than last-query.

## 6. RevIN caveat

RevIN = disabled. Attention came from the exact forward path used in Phase 47.

## 7. Interpretation limits (HARD)

- Attention is TEMPORAL POSITION, not raw feature dimension.
  Input projection Linear(F, D) mixes features before self-attention.
- Attention is NOT causal explanation. Internal allocation diagnostic only.
- Head numeric ID does NOT guarantee semantic alignment across seeds.
- No averaging / thresholding / smoothing in raw storage.

## 8. Phase 53-57 consumers

Phase 53 -- Attention Heatmaps
Phase 54 -- Last-Query Attention
Phase 55 -- Head Comparison
Phase 56 -- Error-Conditioned Attention
Phase 57 -- Seed-Stability Attention

Each requires a separate Human-approved governance gate.
