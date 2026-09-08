# Phase 53 — Attention Heatmap Catalog
Catalog organized by CASE (not by best-looking head).

- V1 case/seed grids: 132
- V2 cross-seed report grids: 10
- Mode A FIXED_PROBABILITY views: 10
- V3 individual per-head maps: 240

## Image directories
- `artifacts/attention_heatmaps/case_grids/` — V1 case × seed layer/head grids
- `artifacts/attention_heatmaps/cross_seed/` — V2 cross-seed report grids (shared top5 × layer)
- `artifacts/attention_heatmaps/fixed_probability/` — Mode A FIXED_PROBABILITY [0,1] references (shared top5 × layer)
- `artifacts/attention_heatmaps/individual_maps/` — V3 individual per-head maps (shared top5)

## Image naming convention
- V1: `V1_CASE<row>_SEED<seed>.png`
- V2: `V2_CASE<row>_L<layer>.png`
- Mode A: `MODE_A_CASE<row>_L<layer>.png`
- V3: `V3_CASE<row>_SEED<seed>_L<layer>_H<head>_<mode>.png`

## Per-case organization

### Case row 0 — TGT_00019582 (2016-05-26 16:40:00)
- W2 SHARED_WORST rank 1
- Seed 42 V1: `artifacts/attention_heatmaps/case_grids/V1_CASE00_SEED42.png`
- Seed 123 V1: `artifacts/attention_heatmaps/case_grids/V1_CASE00_SEED123.png`
- Seed 2026 V1: `artifacts/attention_heatmaps/case_grids/V1_CASE00_SEED2026.png`
- Cross-seed V2: `artifacts/attention_heatmaps/cross_seed/V2_CASE00_L1.png`
- Cross-seed V2: `artifacts/attention_heatmaps/cross_seed/V2_CASE00_L2.png`
- Mode A: `artifacts/attention_heatmaps/fixed_probability/MODE_A_CASE00_L1.png`
- Mode A: `artifacts/attention_heatmaps/fixed_probability/MODE_A_CASE00_L2.png`

### Case row 1 — TGT_00018148 (2016-05-16 17:40:00)
- W2 SHARED_WORST rank 2
- Seed 42 V1: `artifacts/attention_heatmaps/case_grids/V1_CASE01_SEED42.png`
- Seed 123 V1: `artifacts/attention_heatmaps/case_grids/V1_CASE01_SEED123.png`
- Seed 2026 V1: `artifacts/attention_heatmaps/case_grids/V1_CASE01_SEED2026.png`
- Cross-seed V2: `artifacts/attention_heatmaps/cross_seed/V2_CASE01_L1.png`
- Cross-seed V2: `artifacts/attention_heatmaps/cross_seed/V2_CASE01_L2.png`
- Mode A: `artifacts/attention_heatmaps/fixed_probability/MODE_A_CASE01_L1.png`
- Mode A: `artifacts/attention_heatmaps/fixed_probability/MODE_A_CASE01_L2.png`

### Case row 3 — TGT_00019552 (2016-05-26 11:40:00)
- W2 SHARED_WORST rank 3
- Seed 42 V1: `artifacts/attention_heatmaps/case_grids/V1_CASE03_SEED42.png`
- Seed 123 V1: `artifacts/attention_heatmaps/case_grids/V1_CASE03_SEED123.png`
- Seed 2026 V1: `artifacts/attention_heatmaps/case_grids/V1_CASE03_SEED2026.png`
- Cross-seed V2: `artifacts/attention_heatmaps/cross_seed/V2_CASE03_L1.png`
- Cross-seed V2: `artifacts/attention_heatmaps/cross_seed/V2_CASE03_L2.png`
- Mode A: `artifacts/attention_heatmaps/fixed_probability/MODE_A_CASE03_L1.png`
- Mode A: `artifacts/attention_heatmaps/fixed_probability/MODE_A_CASE03_L2.png`

### Case row 6 — TGT_00019684 (2016-05-27 09:40:00)
- W2 SHARED_WORST rank 4
- Seed 42 V1: `artifacts/attention_heatmaps/case_grids/V1_CASE06_SEED42.png`
- Seed 123 V1: `artifacts/attention_heatmaps/case_grids/V1_CASE06_SEED123.png`
- Seed 2026 V1: `artifacts/attention_heatmaps/case_grids/V1_CASE06_SEED2026.png`
- Cross-seed V2: `artifacts/attention_heatmaps/cross_seed/V2_CASE06_L1.png`
- Cross-seed V2: `artifacts/attention_heatmaps/cross_seed/V2_CASE06_L2.png`
- Mode A: `artifacts/attention_heatmaps/fixed_probability/MODE_A_CASE06_L1.png`
- Mode A: `artifacts/attention_heatmaps/fixed_probability/MODE_A_CASE06_L2.png`

### Case row 2 — TGT_00019581 (2016-05-26 16:30:00)
- W2 SHARED_WORST rank 5
- Seed 42 V1: `artifacts/attention_heatmaps/case_grids/V1_CASE02_SEED42.png`
- Seed 123 V1: `artifacts/attention_heatmaps/case_grids/V1_CASE02_SEED123.png`
- Seed 2026 V1: `artifacts/attention_heatmaps/case_grids/V1_CASE02_SEED2026.png`
- Cross-seed V2: `artifacts/attention_heatmaps/cross_seed/V2_CASE02_L1.png`
- Cross-seed V2: `artifacts/attention_heatmaps/cross_seed/V2_CASE02_L2.png`
- Mode A: `artifacts/attention_heatmaps/fixed_probability/MODE_A_CASE02_L1.png`
- Mode A: `artifacts/attention_heatmaps/fixed_probability/MODE_A_CASE02_L2.png`

## Notes
- All heatmaps are DERIVED visualizations; raw attention NPZ remains scientific source.
- Same numeric head index across seeds is shown at matching architectural positions only; not assumed semantically aligned.
- Heatmaps are NOT feature importance, NOT causal explanation, NOT proof of model reasoning.
