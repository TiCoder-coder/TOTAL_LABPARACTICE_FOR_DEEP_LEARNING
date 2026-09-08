# README — Phase 54 Last-Query Attention Analysis

**Version:** LAST_QUERY_ATTENTION-v1

## What `A[:,:,L-1,:]` means

For each test target, each layer, each head, we extract the attention vector that
the **newest** historical token (the last input time-step) distributes over all
72 historical source positions. This is the attention pattern of the token that
(under LAST_STEP pooling) is directly consumed by the regression head.

## Why source positions are historical only

The Transformer's source axis represents temporal token positions — i.e., time steps
in the input window — after feature projection. The forecast target is NOT an attention token.

## How source position maps to forecast lag

```
lag_steps_p = L - p, with H = 1
```

So position 71 → lag 1 (10 min before target), position 0 → lag 72 (12 h before target).

## Why raw order and recency order differ

- Raw position order: oldest → newest (position 0 oldest, position L-1 newest).
- Recency order (used for cumulative coverage): newest → oldest.
- Lag coverage radii (Lag50/80/90) and cumulative recency profiles MUST be computed in recency order.

## What entropy means

Shannon entropy in nats of the last-query vector: `H = -Σ a_p · log(a_p + ε)`, ε = 1e-12.
Higher entropy = more diffuse; lower entropy = more concentrated on fewer positions.

## What effective source count means

`N_eff = exp(H)` — the number of uniformly-weighted source positions that would
produce the same Shannon entropy. Bounds approximately 1..L.

## What expected lag means

`E[LagSteps] = Σ a_p · lag_steps_p`. With `E[LagMinutes] = 10 × E[LagSteps]`.
Caveat: bimodal distributions can have an expected lag where little mass exists;
interpret with entropy and profile.

## What recent 1h/6h/12h/24h mass means

Cumulative mass on the last-`h` (or last-`steps`) source positions, in recency order.
Because L = 72 (12h), the 24h window is **TRUNCATED**. `recent_24h_coverage_truncated = True`.

## What Lag50/Lag80/Lag90 mean

Smallest most-recent lag radius (in steps) containing at least 50% / 80% / 90%
of last-query attention mass. Smaller = more recency-concentrated; larger = more
history is needed. No quality judgment implied.

## How non-overlapping lag bins work

Bins 1–6 / 7–36 / 37–72 / 73–144 / 145–L are mutually exclusive. Masses sum to 1.
Only bins where the upper bound ≤ L are emitted (L=72 ⇒ 73–144 / 145–72 not applicable).

## Why mean profiles sum to 1

Each mean profile is the average over Test targets of normalized probability vectors.
The mean of normalized vectors is itself a normalized vector (sum = 1).

## Why heads are not ranked

Head ranking would require either predictive quality evidence (Phase 55) or
error-conditioned comparison (Phase 56). Phase 54 reports each head's distributional
profile without claiming superiority.

## Why same-index heads across seeds may differ semantically

The architecture does NOT guarantee that head index 0 in seed 42 has the same
learned semantic role as head index 0 in seed 123. Phase 54 displays heads by
architectural index only; cross-seed matching is allocated to Phase 57.

## Why error-conditioned analysis is deferred

Comparing high-error vs low-error attention is allocated to Phase 56. Phase 54
uses error labels only for deterministic case-level views on the frozen Phase 51
shared ranks 1–5.

## Why raw attention is temporal allocation, not feature importance

Attention weights describe temporal token-to-token allocation inside the Transformer
Encoder. They are NOT raw-feature importance, NOT causal attribution, NOT proof of
model reasoning. The model's output also depends on value projections, residual paths,
LayerNorm, FFN, later layers, pooling, and the regression head.

## How downstream phases consume numerical outputs

- **Phase 55** (head comparison): standardized head metrics + profiles (see `phase55_head_comparison_handoff.json`).
- **Phase 56** (error-conditioned): context handoff references Phase 52 raw last-query NPZ (NOT PNG).
- **Phase 57** (seed stability): context handoff pins numeric source to Phase 52 NPZ (NOT PNG).

All downstream phases MUST use the canonical Phase 52 raw last-query NPZ as numeric source.
PNG pixels are derived visualizations only and MUST NOT be used to recover attention values.
