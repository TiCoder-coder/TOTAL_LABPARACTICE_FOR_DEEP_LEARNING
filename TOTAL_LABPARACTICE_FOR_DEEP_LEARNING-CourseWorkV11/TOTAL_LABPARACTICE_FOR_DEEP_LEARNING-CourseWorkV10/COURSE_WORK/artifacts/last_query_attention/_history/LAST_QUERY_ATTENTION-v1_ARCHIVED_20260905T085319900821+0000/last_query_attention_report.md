# Phase 54 — Last-Query Attention Analysis Report

**Version:** LAST_QUERY_ATTENTION-v1
**Generated (UTC):** 2026-09-04T13:32:23+00:00
**Numeric source:** Phase 52 raw last-query NPZ (NOT PNG)
**Status:** PASS

---

## 1. Objective

Phase 54 quantitatively characterizes last-query attention `A[:,:,L-1,:]` over
the full FINAL_TEST_POP-v1 (N=2961) for the three frozen TransformerEncoder seeds
(42 / 123 / 2026) across 2 layers × 4 heads each.

## 2. Why last-query attention is analyzed

Because pooling is `LAST_STEP`, the newest encoded token is directly passed to the
regression head, making the last-query attention vector a particularly relevant
internal diagnostic for understanding which historical positions the model
considers when producing the forecast.

## 3. Pooling-dependent interpretation

- Pooling: **LAST_STEP**
- `last_query_directly_corresponds_to_pooled_token` = `True`
- This means last-query attention IS the attention for the token directly consumed by the regression head.

## 4. Frozen numerical sources

All Phase 54 metrics are computed from the frozen Phase 52 raw last-query NPZ:
- `last_query_attention_seed42.npz`
- `last_query_attention_seed123.npz`
- `last_query_attention_seed2026.npz`

Each NPZ carries shape `[2961, 2, 4, 72]` float32 — `N_TEST × layers × heads × L`.

## 5. Last-query and lag semantics

- Last query: `A[:, :, L-1, :]` (NOT `A[:, :, :, L-1]`).
- Source position `p` corresponds to lag steps `L - p` (H=1).
- Position 0 = oldest (12 h old); position L-1 = newest (10 min old).
- Forecast target is NOT an attention token.

## 6. Raw-source integrity verification

See `last_query_source_verification.csv` and `last_query_integrity_audit.csv`. All three NPZ files:
- SHA256 match frozen Phase 52 checksums.
- Shape: [2961, 2, 4, 72]
- Dtype: float32
- Per-vector probability: finite, nonnegative within tolerance, sum ≈ 1.

## 7. Phase 52 summary reconstruction

Phase 54 recomputes 10 canonical Phase 52 metrics from raw vectors and compares
against the frozen `attention_last_query_summary.csv`. Reconstruction status: PASS.

## 8. Concentration metrics

Per-seed/per-layer/per-head aggregates of:
- entropy (Shannon, ε = 1e-12)
- normalized entropy (H / log L)
- effective source count (exp(H))
- top1 weight, top1 tie count (NEWEST_SOURCE rule)
- top5 mass

See `last_query_metric_summary_by_head.csv`.

## 9. Expected-lag metrics

- expected_lag_steps = Σ a_p · lag_steps_p
- expected_lag_minutes = 10 × expected_lag_steps
- lag_sd_steps = √(Σ a_p · (lag_steps_p − E[lag])²)

## 10. Recent-history cumulative mass

Windows (steps): 1h≤6, 6h≤36, 12h≤72, 24h≤144. `effective_steps = min(requested, L)`.
L=72 means the 24h window is **TRUNCATED** — model only has 12h of input context.
See `last_query_recent_mass_summary.csv`.

## 11. Non-overlapping temporal allocation bins

Bins (lag steps): 1–6, 7–36, 37–72, 73–144, 145–L. Only supported bins are emitted.
See `last_query_lag_bin_mass.csv`.

## 12. Lag50/Lag80/Lag90 coverage radii

Computed on cumulative recency mass in newest→oldest order.
See `last_query_coverage_radius_summary.csv`.

## 13. Mean temporal profiles by head

Per seed/layer/head: mean, median, SD, p05/p25/p75/p95 attention weight by lag.
Mean profile audit: `Σ_lag mean_weight ≈ 1` per seed/layer/head.
See `last_query_profile_by_lag.csv`.

## 14. Layer head-mean profiles

Layer-level head-mean profile = mean over heads of mean profile.
Permutation-invariant summary; does NOT replace per-head profiles.
See `last_query_layer_head_mean_profile.csv`.

## 15. Top1 lag-frequency patterns

For each seed/layer/head: count + fraction of targets with each top1_lag_steps value.
NEWEST_SOURCE tie rule preserved. Fractions sum to 1 per seed/layer/head.
See `last_query_top1_lag_frequency.csv` and `last_query_top1_tie_summary.csv`.

## 16. Deterministic shared-worst case views

Deterministic Phase 51 W2 SHARED_WORST ranks 1–5. One grid per case×seed,
rows=layers, columns=heads. x=lag (minutes, recency order), y=raw last-query weight.
See `figures/report_cases/SHARED_R{01..05}_SEED{42,123,2026}_LAST_QUERY.png`.

## 17. Main descriptive findings

Restrained qualitative findings (descriptive only). No head ranking. No causal claim.
See `last_query_analysis_findings.csv`.

## 18. Why no head winner is selected

Head ranking would require either predictive quality data (deferred to Phase 55)
or error-conditioned groups (deferred to Phase 56). Phase 54 reports each head's
distributional profile only.

## 19. Why attention is not raw-feature importance

Attention source axes are TEMPORAL token positions, not raw feature dimensions.
The Transformer's input projection (Linear(33, D)) mixes features before self-attention.
Attention weight = temporal allocation diagnostic, not feature importance.

## 20. Why error-conditioning is deferred to Phase 56

Comparing high-error vs low-error attention requires Phase 49 residuals and
Phase 50 regime labels — explicitly allocated to Phase 56.

## 21. Why seed-stability is deferred to Phase 57

Same-index heads across seeds are NOT assumed semantically equivalent.
Cross-seed stability requires explicit head matching, allocated to Phase 57.

## 22. Limitations

- L=72 means 12h input context. 24h requested windows are TRUNCATED.
- Attention is one internal signal among many (residual paths, FFN, LayerNorm, etc.).
- Per-vector metrics describe distribution, not contribution magnitude.
- Empirical quantiles (p05–p95) are not confidence intervals.
- N=2961 vectors are temporally dependent, not iid.

## 23. Handoff to head comparison

Phase 55 receives:
- `last_query_metrics_long.csv`
- `last_query_metric_summary_by_head.csv`
- `last_query_profile_by_lag.csv`
- `last_query_lag_bin_mass.csv`
- `last_query_recent_mass_summary.csv`
- `last_query_coverage_radius_summary.csv`
- `last_query_top1_lag_frequency.csv`
- `phase55_head_comparison_handoff.json`

`ready_for_phase55 = true`, `phase55_authorized = false`.

## 24. Definition of Done

Phase 54 is PASS only when:
- All Phase 52 raw last-query NPZ verified.
- Last-query treated exactly as `A[:, :, L-1, :]`.
- Vectors finite, nonnegative, sum ≈ 1.
- Phase 52 summaries reconstructed within tolerance.
- Lag mapping exact (lag1↔L-1, lagL↔0).
- Per-target metrics include all required fields.
- 1h/6h/12h/24h masses recorded with truncation flags.
- Non-overlap lag-bin masses sum ≈ 1.
- Mean profiles sum ≈ 1 per seed/layer/head.
- Layer head-mean profiles produced without replacing per-head.
- Top1 frequency/tie computed with NEWEST_SOURCE rule.
- Phase 51 shared ranks 1–5 used as illustrative case views.
- No new extraction/inference/training; no head ranking/error conditioning/seed-stability claims.
- Phase 55 receives standardized head metrics; Phase 56/57 receive context only.

**Phase 54 status: PASS**
