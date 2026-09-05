# Phase 53 — Attention Heatmaps — Main Report
**Version:** ATTENTION_HEATMAPS-v1  
**Date:** 2026-09-04

## 1. Objective

Phase 53 visualizes the frozen Phase 52 raw dense temporal self-attention matrices for the canonical Phase 51 dense case set, using a frozen rendering contract that preserves query/source orientation, temporal order, layer/head identity, and a deterministic Mode A absolute-probability reference and a Mode B case-shared comparison scale.

## 2. Frozen Phase 52 attention sources

- Seed 42: `artifacts/attention_extraction/raw/dense_case_attention_seed42.npz` SHA256 `33d979a1b6b0aa6e1c2af11c88f1ee943a46b4489e849f6675e65a5036de1d3c`
- Seed 123: `artifacts/attention_extraction/raw/dense_case_attention_seed123.npz` SHA256 `29b8fdebcfe771f3b0c9e5e215417a37d6dec5ce16b8b6054a60aa21ee2cca25`
- Seed 2026: `artifacts/attention_extraction/raw/dense_case_attention_seed2026.npz` SHA256 `5d4824a5f2f8b46d1ee8e1b9ba5ad6da43159352e32be121dca4dc2abfbc78bc`

Dense shape per seed: `[44, 2, 4, 72, 72]` float32.

- Dense case order SHA256: `e4fce0fabb9bc264d76e2b3f57682a9cc9b30490a488d70ccb926b4c3b1b62d1`
- Relative position map SHA256: `948bc0dcb56674f7f8c117d7e7050490b391e39225754e8a21861d00e59e5063`

## 3. Query/source tensor semantics

For each case/layer/head, M[q,s] is the attention weight from query temporal position q to source/key temporal position s. Matrix shape: [L, L] = [72, 72]. Canonical orientation: rows = QUERY, columns = SOURCE. Plot raw M; no transpose.

## 4. Temporal position and lag mapping

Position 0 is the OLDEST historical input; position L-1 = 71 is the NEWEST. Lag-steps: LagSteps_p = L - p (H=1). For L=72:

- position 71 → lag 1 → 10 minutes before target
- position 66 → lag 6 → 60 minutes before target
- position 36 → lag 36 → 6 hours before target
- position 0  → lag 72 → 12 hours before target

Lag ticks are derived from the frozen Phase 52 relative-position map; axis tick audit covers 7 lag entries. The forecast target is NOT an attention token.

## 5. Rendering contract

Frozen rendering contract (`attention_heatmap_render_config.json`, SHA256 `cd437644a9d7de147b8cb5ed78f08e0e1da459e45b92c78a3cc55d771dbb2bb8`):

- matrix_orientation = QUERY_ROWS_SOURCE_COLUMNS
- transpose = false
- origin = ROW0_TOP
- x_order = OLDEST_TO_NEWEST (left → right)
- y_order = OLDEST_TO_NEWEST_TOP_TO_BOTTOM (top → bottom)
- interpolation = NONE / NEAREST
- aspect = SQUARE_PER_MATRIX
- colorbar_label = "Attention weight"
- colormap = sequential perceptually uniform (viridis)

## 6. Color-scale methodology

Two color-scale modes are used:

- **Mode A — FIXED_PROBABILITY**: vmin = 0, vmax = 1, no clipping. Provides an absolute probability reference comparable across cases, seeds, layers, and heads.
- **Mode B — CASE_SHARED_SCALE**: For each target case c, vmax_case = MAX attention over all 3 seeds, all layers, all heads, all q, all s for that case. All Mode B panels for the same case share this EXACT scale; no per-panel autoscaling.

## 7. FIXED_PROBABILITY vs CASE_SHARED_SCALE

Mode A is the global absolute probability reference. Mode B is the within-case comparison view. Both use the same perceptually uniform sequential colormap; the colorbar label is always "Attention weight".

## 8. Full case/seed heatmap coverage

V1 case/seed grids: **132** (= 44 cases × 3 seeds). Each V1 grid has rows = LAYERS (2) and columns = HEADS (4) → 8 panels per grid in Mode B scale.

## 9. Shared worst-case rank1 heatmaps

Highest-ranked shared worst-error case: `TGT_00019582` at `2016-05-26 16:40:00`. Rendered as V1 grid per seed (3 seeds × 8 panels), one V2 cross-seed grid per layer, one Mode A FIXED_PROBABILITY reference per layer, plus V3 individual per-head maps.

## 10. Shared worst-case ranks2–5 heatmaps

Additional shared worst-error cases (ranks 2–5):

- rank 2: `TGT_00018148` at `2016-05-16 17:40:00`
- rank 3: `TGT_00019552` at `2016-05-26 11:40:00`
- rank 4: `TGT_00019684` at `2016-05-27 09:40:00`
- rank 5: `TGT_00019581` at `2016-05-26 16:30:00`

## 11. Layer/head visual diversity

V1 grids include every layer (L1, L2) and every head (H1, H2, H3, H4). Visual differences between heads are reported only qualitatively. Per-head maps remain the primary scientific visualization.

## 12. Cross-seed visual comparison caveat

**Same numeric head index across seeds is NOT assumed to represent the same learned semantic role.** Cross-seed V2 grids display heads at matching architectural positions only. Head-index semantic permutation may exist.

## 13. Qualitative visual observations

Phase 53 emits ONLY restrained qualitative observations, per canonical plan §98–§100. No morphology labels are used as hard scientific outputs. No best/useless claims. Visual notes (`attention_heatmap_visual_notes.csv`) are scope-tagged QUALITATIVE_DESCRIPTIVE_ONLY and used_for_selection = false.

## 14. Why heatmaps are not feature importance

Attention weights are temporal token-to-token attention probabilities over the historical input window of L = 72 positions, NOT raw input-feature importances. The X-axis is source/key historical position; the Y-axis is query historical position. Attention is neither a per-feature weight nor a SHAP-like importance score.

## 15. Why heatmaps are not causal explanation

Attention is a learned allocation pattern, not a causal attribution. Bright cells do NOT identify causal predictors. Final encoder has no causal mask; upper-triangle values are valid historical token-to-token attention and are NOT hidden as 'leakage'.

## 16. Why images are not numerical source

Heatmap PNGs are DERIVED visualizations. Phase 54 quantitative analyses (expected lag, recent mass, top source) MUST use Phase 52 RAW LAST_QUERY NPZ, NOT PNG pixel digitization. Phase 54 handoff (`phase54_last_query_attention_context_handoff.json`) pins this contract.

## 17. Handoff to last-query attention analysis

Phase 53 emitted a Phase 54 context handoff that:

- sets `phase54_numeric_source = PHASE52_RAW_LAST_QUERY_NPZ`
- sets `do_not_extract_last_query_from_png = true`
- sets `phase54_authorized = false`

Phase 53 only supplies visualization context/index to Phase 54. Phase 54 implementation requires separate Human approval.

## 18. Limitations

- Dense 72×72 matrices are visually non-trivial to inspect; Mode A FIXED_PROBABILITY view provides the absolute probability reference; Mode B case-shared view provides within-case comparison.
- Same-index heads across seeds are not assumed semantically aligned.
- Optional individual per-head maps exist for shared top5 only.

## 19. Definition of Done

Phase 53 PASS condition is fully met when:

- All 3 Phase 52 raw dense NPZ files verified by SHA256 + dtype + shape + case order.
- Render config frozen and fingerprinted before any official render.
- Synthetic + real-source orientation tests PASS.
- Mode-B case-shared vmax computed once per case before individual rendering.
- All dense cases rendered as V1 case/seed grids (132 grids).
- Shared top5 cases rendered as V2 cross-seed grids (10) and Mode A FIXED_PROBABILITY references (10) plus optional V3 individual per-head maps.
- Catalog and provenance complete; image checksums present; QA PASS.
- Phase 54 context handoff emitted; phase54_authorized = false.
- Phase 47/48/49/50/51/52 frozen artifacts UNCHANGED.
- Notebook UNCHANGED.

