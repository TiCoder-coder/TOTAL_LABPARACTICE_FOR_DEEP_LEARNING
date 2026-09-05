# README — ATTENTION_HEATMAPS-v1

Phase 53 attention heatmap visualization package.

## Quick orientation

Heatmaps are DERIVED visualizations of frozen Phase 52 raw temporal self-attention. All scientific numerical source remains the float32 raw NPZ. PNG/SVG images are presentation-only.

## Canonical contract (frozen before official render)

```text
raw source      = Phase52 float32 dense NPZ only
matrix orient   = QUERY_ROWS_SOURCE_COLUMNS
transpose       = false
origin          = ROW0_TOP
x_order         = OLDEST_TO_NEWEST
y_order         = OLDEST_TO_NEWEST_TOP_TO_BOTTOM
x_label         = Source / key historical position
y_label         = Query historical position
interpolation   = NONE / NEAREST
aspect          = SQUARE_PER_MATRIX
color           = sequential perceptually uniform
colorbar label  = "Attention weight"
Mode A          = FIXED_PROBABILITY [0, 1]
Mode B          = CASE_SHARED_SCALE (vmax = max over 3 seeds × L × H × q × s for that case)
V1              = every dense case × every seed; rows=layers, columns=heads
V2              = Phase51 W2 SHARED_WORST ranks 1–5; rows=seeds, columns=heads
V3              = shared top5 × seed × layer × head × mode (deterministic only)
```

## Directory layout

```text
artifacts/attention_heatmaps/
├── case_grids/                 # V1 case/seed layer/head grids (PNG)
├── cross_seed/                 # V2 cross-seed report grids (PNG)
├── fixed_probability/          # Mode A FIXED_PROBABILITY references (PNG)
├── individual_maps/            # V3 individual per-head maps (PNG)
├── attention_heatmap_render_config.json
├── attention_heatmap_render_config_fingerprint.json
├── attention_heatmap_orientation_tests.csv
├── attention_heatmap_axis_tick_audit.csv
├── attention_heatmap_source_verification.csv
├── attention_heatmap_case_scale_manifest.csv
├── attention_heatmap_report_cases.csv
├── attention_heatmap_catalog.csv
├── attention_heatmap_case_index.csv
├── attention_heatmap_render_audit.csv
├── attention_heatmap_image_checksums.json
├── attention_heatmap_visual_notes.csv
├── attention_heatmap_findings.csv
├── attention_heatmap_discrepancies.json
├── attention_heatmaps_manifest.json
├── attention_heatmaps_contract.json
├── attention_heatmap_summary.json
├── attention_heatmap_report.md
├── README_ATTENTION_HEATMAPS.md
└── phase54_last_query_attention_context_handoff.json
```

## Image naming convention

- V1: `V1_CASE<row>_SEED<seed>.png`
- V2: `V2_CASE<row>_L<layer>.png`
- Mode A: `MODE_A_CASE<row>_L<layer>.png`
- V3: `V3_CASE<row>_SEED<seed>_L<layer>_H<head>_<mode>.png`

## Forbidden operations

Phase 53 production path NEVER executes:

- `torch.load` / `model.forward` / `return_attention`
- `materialize_phase52` / `extract_attention`
- `fit` / `fit_transform` / scaler fitting
- new Test inference / prediction generation
- training, optimizer.step, .backward
- best-head selection / best-seed selection / ensemble
- attention mislabeled as feature importance or causal attribution
- extracting last-query numeric metrics from PNG pixels

## Upstream immutability

Phase 47/48/49/50/51/52 canonical artifacts are UNCHANGED. Phase 53 may READ them; Phase 53 may NOT modify them.

## Phase 54 handoff

Phase 54 numeric source remains the Phase 52 RAW LAST_QUERY NPZ files. PNG pixels are NOT used as numerical input. Phase 54 implementation requires separate Human approval.

