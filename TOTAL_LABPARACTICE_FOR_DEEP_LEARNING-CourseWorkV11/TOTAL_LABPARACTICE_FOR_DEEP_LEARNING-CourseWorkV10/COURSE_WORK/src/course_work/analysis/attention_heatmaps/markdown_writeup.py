"""Phase 53 — Markdown writeup helpers (catalog Markdown, report, README)."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def write_catalog_markdown(
    project_root: Path,
    output_md: Path,
    report_cases: list[dict],
    catalog_summary: dict,
    case_grids_meta: dict,
    cross_seed_meta: dict,
    fixed_prob_meta: dict,
) -> None:
    """Write `attention_heatmap_catalog.md` (organized by case)."""
    n_v1 = catalog_summary.get("n_v1_grids", 0)
    n_v2 = catalog_summary.get("n_v2_grids", 0)
    n_mode_a = catalog_summary.get("n_mode_a_grids", 0)
    n_v3 = catalog_summary.get("n_v3_images", 0)

    lines: list[str] = []
    lines.append("# Phase 53 — Attention Heatmap Catalog\n")
    lines.append("Catalog organized by CASE (not by best-looking head).\n")
    lines.append(f"\n- V1 case/seed grids: {n_v1}\n")
    lines.append(f"- V2 cross-seed report grids: {n_v2}\n")
    lines.append(f"- Mode A FIXED_PROBABILITY views: {n_mode_a}\n")
    lines.append(f"- V3 individual per-head maps: {n_v3}\n")
    lines.append("\n## Image directories\n")
    lines.append("- `artifacts/attention_heatmaps/case_grids/` — V1 case × seed layer/head grids\n")
    lines.append("- `artifacts/attention_heatmaps/cross_seed/` — V2 cross-seed report grids (shared top5 × layer)\n")
    lines.append("- `artifacts/attention_heatmaps/fixed_probability/` — Mode A FIXED_PROBABILITY [0,1] references (shared top5 × layer)\n")
    lines.append("- `artifacts/attention_heatmaps/individual_maps/` — V3 individual per-head maps (shared top5)\n")
    lines.append("\n## Image naming convention\n")
    lines.append("- V1: `V1_CASE<row>_SEED<seed>.png`\n")
    lines.append("- V2: `V2_CASE<row>_L<layer>.png`\n")
    lines.append("- Mode A: `MODE_A_CASE<row>_L<layer>.png`\n")
    lines.append("- V3: `V3_CASE<row>_SEED<seed>_L<layer>_H<head>_<mode>.png`\n")
    lines.append("\n## Per-case organization\n")

    for rc in report_cases:
        cr = rc["case_row_idx0"]
        if cr < 0:
            continue
        lines.append(f"\n### Case row {cr} — {rc['target_id']} ({rc['target_timestamp']})\n")
        lines.append(f"- W2 SHARED_WORST rank {rc['shared_worst_rank']}\n")
        sg = case_grids_meta.get(cr, {})
        if sg.get("seed42_grid"):
            lines.append(f"- Seed 42 V1: `{sg['seed42_grid']}`\n")
        if sg.get("seed123_grid"):
            lines.append(f"- Seed 123 V1: `{sg['seed123_grid']}`\n")
        if sg.get("seed2026_grid"):
            lines.append(f"- Seed 2026 V1: `{sg['seed2026_grid']}`\n")
        cs = cross_seed_meta.get(cr, [])
        for p in cs:
            lines.append(f"- Cross-seed V2: `{p}`\n")
        fp = fixed_prob_meta.get(cr, [])
        for p in fp:
            lines.append(f"- Mode A: `{p}`\n")

    lines.append("\n## Notes\n")
    lines.append("- All heatmaps are DERIVED visualizations; raw attention NPZ remains scientific source.\n")
    lines.append("- Same numeric head index across seeds is shown at matching architectural positions only; not assumed semantically aligned.\n")
    lines.append("- Heatmaps are NOT feature importance, NOT causal explanation, NOT proof of model reasoning.\n")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("".join(lines))


def write_main_report(
    project_root: Path,
    output_md: Path,
    summary: dict,
    report_cases: list[dict],
    n_cases: int,
    n_seeds: int,
    n_layers: int,
    n_heads: int,
    lookback: int,
    raw_file_sha_per_seed: dict,
    render_config_sha256: str,
    case_order_sha: str,
    position_map_sha: str,
    catalog_summary: dict,
    axis_tick_count: int,
    preflight_status_summary: dict,
    discrepancy_status: str,
) -> None:
    """Write `attention_heatmap_report.md` (19-section structure per canonical plan §173)."""
    L: list[str] = []
    L.append("# Phase 53 — Attention Heatmaps — Main Report\n")
    L.append("**Version:** ATTENTION_HEATMAPS-v1  \n")
    L.append("**Date:** 2026-09-04\n\n")

    L.append("## 1. Objective\n\n")
    L.append(
        "Phase 53 visualizes the frozen Phase 52 raw dense temporal self-attention "
        "matrices for the canonical Phase 51 dense case set, using a frozen rendering "
        "contract that preserves query/source orientation, temporal order, layer/head "
        "identity, and a deterministic Mode A absolute-probability reference and a "
        "Mode B case-shared comparison scale.\n\n"
    )

    L.append("## 2. Frozen Phase 52 attention sources\n\n")
    for seed, sha in raw_file_sha_per_seed.items():
        L.append(f"- Seed {seed}: `artifacts/attention_extraction/raw/dense_case_attention_seed{seed}.npz` SHA256 `{sha}`\n")
    L.append(f"\nDense shape per seed: `[{n_cases}, {n_layers}, {n_heads}, {lookback}, {lookback}]` float32.\n\n")
    L.append(f"- Dense case order SHA256: `{case_order_sha}`\n")
    L.append(f"- Relative position map SHA256: `{position_map_sha}`\n\n")

    L.append("## 3. Query/source tensor semantics\n\n")
    L.append(
        "For each case/layer/head, M[q,s] is the attention weight from query temporal "
        "position q to source/key temporal position s. Matrix shape: [L, L] = [72, 72]. "
        "Canonical orientation: rows = QUERY, columns = SOURCE. Plot raw M; no transpose.\n\n"
    )

    L.append("## 4. Temporal position and lag mapping\n\n")
    L.append(
        "Position 0 is the OLDEST historical input; position L-1 = 71 is the NEWEST. "
        "Lag-steps: LagSteps_p = L - p (H=1). For L=72:\n\n"
        "- position 71 → lag 1 → 10 minutes before target\n"
        "- position 66 → lag 6 → 60 minutes before target\n"
        "- position 36 → lag 36 → 6 hours before target\n"
        "- position 0  → lag 72 → 12 hours before target\n\n"
    )
    L.append(
        "Lag ticks are derived from the frozen Phase 52 relative-position map; "
        f"axis tick audit covers {axis_tick_count} lag entries. The forecast target is NOT an attention token.\n\n"
    )

    L.append("## 5. Rendering contract\n\n")
    L.append(
        "Frozen rendering contract (`attention_heatmap_render_config.json`, "
        f"SHA256 `{render_config_sha256}`):\n\n"
        "- matrix_orientation = QUERY_ROWS_SOURCE_COLUMNS\n"
        "- transpose = false\n"
        "- origin = ROW0_TOP\n"
        "- x_order = OLDEST_TO_NEWEST (left → right)\n"
        "- y_order = OLDEST_TO_NEWEST_TOP_TO_BOTTOM (top → bottom)\n"
        "- interpolation = NONE / NEAREST\n"
        "- aspect = SQUARE_PER_MATRIX\n"
        "- colorbar_label = \"Attention weight\"\n"
        "- colormap = sequential perceptually uniform (viridis)\n\n"
    )

    L.append("## 6. Color-scale methodology\n\n")
    L.append(
        "Two color-scale modes are used:\n\n"
        "- **Mode A — FIXED_PROBABILITY**: vmin = 0, vmax = 1, no clipping. Provides an "
        "absolute probability reference comparable across cases, seeds, layers, and heads.\n"
        "- **Mode B — CASE_SHARED_SCALE**: For each target case c, vmax_case = MAX attention over "
        "all 3 seeds, all layers, all heads, all q, all s for that case. All Mode B panels for "
        "the same case share this EXACT scale; no per-panel autoscaling.\n\n"
    )

    L.append("## 7. FIXED_PROBABILITY vs CASE_SHARED_SCALE\n\n")
    L.append(
        "Mode A is the global absolute probability reference. Mode B is the within-case "
        "comparison view. Both use the same perceptually uniform sequential colormap; "
        "the colorbar label is always \"Attention weight\".\n\n"
    )

    L.append("## 8. Full case/seed heatmap coverage\n\n")
    L.append(
        f"V1 case/seed grids: **{catalog_summary.get('n_v1_grids', 0)}** (= 44 cases × 3 seeds). "
        "Each V1 grid has rows = LAYERS (2) and columns = HEADS (4) → 8 panels per grid in Mode B scale.\n\n"
    )

    L.append("## 9. Shared worst-case rank1 heatmaps\n\n")
    rank1 = [rc for rc in report_cases if rc.get("shared_worst_rank") == 1]
    if rank1:
        rc = rank1[0]
        L.append(
            f"Highest-ranked shared worst-error case: `{rc['target_id']}` at `{rc['target_timestamp']}`. "
            "Rendered as V1 grid per seed (3 seeds × 8 panels), one V2 cross-seed grid per layer, "
            "one Mode A FIXED_PROBABILITY reference per layer, plus V3 individual per-head maps.\n\n"
        )

    L.append("## 10. Shared worst-case ranks2–5 heatmaps\n\n")
    other_ranks = [rc for rc in report_cases if rc.get("shared_worst_rank", 0) >= 2]
    if other_ranks:
        L.append("Additional shared worst-error cases (ranks 2–5):\n\n")
        for rc in other_ranks:
            L.append(f"- rank {rc['shared_worst_rank']}: `{rc['target_id']}` at `{rc['target_timestamp']}`\n")
        L.append("\n")

    L.append("## 11. Layer/head visual diversity\n\n")
    L.append(
        "V1 grids include every layer (L1, L2) and every head (H1, H2, H3, H4). "
        "Visual differences between heads are reported only qualitatively. "
        "Per-head maps remain the primary scientific visualization.\n\n"
    )

    L.append("## 12. Cross-seed visual comparison caveat\n\n")
    L.append(
        "**Same numeric head index across seeds is NOT assumed to represent the same "
        "learned semantic role.** Cross-seed V2 grids display heads at matching architectural "
        "positions only. Head-index semantic permutation may exist.\n\n"
    )

    L.append("## 13. Qualitative visual observations\n\n")
    L.append(
        "Phase 53 emits ONLY restrained qualitative observations, per canonical plan §98–§100. "
        "No morphology labels are used as hard scientific outputs. No best/useless claims. "
        "Visual notes (`attention_heatmap_visual_notes.csv`) are scope-tagged "
        "QUALITATIVE_DESCRIPTIVE_ONLY and used_for_selection = false.\n\n"
    )

    L.append("## 14. Why heatmaps are not feature importance\n\n")
    L.append(
        "Attention weights are temporal token-to-token attention probabilities over the "
        "historical input window of L = 72 positions, NOT raw input-feature importances. "
        "The X-axis is source/key historical position; the Y-axis is query historical position. "
        "Attention is neither a per-feature weight nor a SHAP-like importance score.\n\n"
    )

    L.append("## 15. Why heatmaps are not causal explanation\n\n")
    L.append(
        "Attention is a learned allocation pattern, not a causal attribution. "
        "Bright cells do NOT identify causal predictors. Final encoder has no causal mask; "
        "upper-triangle values are valid historical token-to-token attention and are NOT "
        "hidden as 'leakage'.\n\n"
    )

    L.append("## 16. Why images are not numerical source\n\n")
    L.append(
        "Heatmap PNGs are DERIVED visualizations. Phase 54 quantitative analyses "
        "(expected lag, recent mass, top source) MUST use Phase 52 RAW LAST_QUERY NPZ, "
        "NOT PNG pixel digitization. Phase 54 handoff (`phase54_last_query_attention_"
        "context_handoff.json`) pins this contract.\n\n"
    )

    L.append("## 17. Handoff to last-query attention analysis\n\n")
    L.append(
        "Phase 53 emitted a Phase 54 context handoff that:\n\n"
        "- sets `phase54_numeric_source = PHASE52_RAW_LAST_QUERY_NPZ`\n"
        "- sets `do_not_extract_last_query_from_png = true`\n"
        "- sets `phase54_authorized = false`\n\n"
        "Phase 53 only supplies visualization context/index to Phase 54. Phase 54 "
        "implementation requires separate Human approval.\n\n"
    )

    L.append("## 18. Limitations\n\n")
    L.append(
        "- Dense 72×72 matrices are visually non-trivial to inspect; Mode A FIXED_PROBABILITY "
        "view provides the absolute probability reference; Mode B case-shared view provides "
        "within-case comparison.\n"
        "- Same-index heads across seeds are not assumed semantically aligned.\n"
        "- Optional individual per-head maps exist for shared top5 only.\n\n"
    )

    L.append("## 19. Definition of Done\n\n")
    L.append(
        "Phase 53 PASS condition is fully met when:\n\n"
        "- All 3 Phase 52 raw dense NPZ files verified by SHA256 + dtype + shape + case order.\n"
        "- Render config frozen and fingerprinted before any official render.\n"
        "- Synthetic + real-source orientation tests PASS.\n"
        "- Mode-B case-shared vmax computed once per case before individual rendering.\n"
        "- All dense cases rendered as V1 case/seed grids (132 grids).\n"
        "- Shared top5 cases rendered as V2 cross-seed grids (10) and Mode A FIXED_PROBABILITY "
        "references (10) plus optional V3 individual per-head maps.\n"
        "- Catalog and provenance complete; image checksums present; QA PASS.\n"
        "- Phase 54 context handoff emitted; phase54_authorized = false.\n"
        "- Phase 47/48/49/50/51/52 frozen artifacts UNCHANGED.\n"
        "- Notebook UNCHANGED.\n\n"
    )

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("".join(L))


def write_readme(
    project_root: Path,
    output_md: Path,
    summary: dict,
) -> None:
    """Write `README_ATTENTION_HEATMAPS.md`."""
    L: list[str] = []
    L.append("# README — ATTENTION_HEATMAPS-v1\n\n")
    L.append("Phase 53 attention heatmap visualization package.\n\n")
    L.append("## Quick orientation\n\n")
    L.append(
        "Heatmaps are DERIVED visualizations of frozen Phase 52 raw temporal self-attention. "
        "All scientific numerical source remains the float32 raw NPZ. PNG/SVG images are "
        "presentation-only.\n\n"
    )

    L.append("## Canonical contract (frozen before official render)\n\n")
    L.append(
        "```text\n"
        "raw source      = Phase52 float32 dense NPZ only\n"
        "matrix orient   = QUERY_ROWS_SOURCE_COLUMNS\n"
        "transpose       = false\n"
        "origin          = ROW0_TOP\n"
        "x_order         = OLDEST_TO_NEWEST\n"
        "y_order         = OLDEST_TO_NEWEST_TOP_TO_BOTTOM\n"
        "x_label         = Source / key historical position\n"
        "y_label         = Query historical position\n"
        "interpolation   = NONE / NEAREST\n"
        "aspect          = SQUARE_PER_MATRIX\n"
        "color           = sequential perceptually uniform\n"
        "colorbar label  = \"Attention weight\"\n"
        "Mode A          = FIXED_PROBABILITY [0, 1]\n"
        "Mode B          = CASE_SHARED_SCALE (vmax = max over 3 seeds × L × H × q × s for that case)\n"
        "V1              = every dense case × every seed; rows=layers, columns=heads\n"
        "V2              = Phase51 W2 SHARED_WORST ranks 1–5; rows=seeds, columns=heads\n"
        "V3              = shared top5 × seed × layer × head × mode (deterministic only)\n"
        "```\n\n"
    )

    L.append("## Directory layout\n\n")
    L.append(
        "```text\n"
        "artifacts/attention_heatmaps/\n"
        "├── case_grids/                 # V1 case/seed layer/head grids (PNG)\n"
        "├── cross_seed/                 # V2 cross-seed report grids (PNG)\n"
        "├── fixed_probability/          # Mode A FIXED_PROBABILITY references (PNG)\n"
        "├── individual_maps/            # V3 individual per-head maps (PNG)\n"
        "├── attention_heatmap_render_config.json\n"
        "├── attention_heatmap_render_config_fingerprint.json\n"
        "├── attention_heatmap_orientation_tests.csv\n"
        "├── attention_heatmap_axis_tick_audit.csv\n"
        "├── attention_heatmap_source_verification.csv\n"
        "├── attention_heatmap_case_scale_manifest.csv\n"
        "├── attention_heatmap_report_cases.csv\n"
        "├── attention_heatmap_catalog.csv\n"
        "├── attention_heatmap_case_index.csv\n"
        "├── attention_heatmap_render_audit.csv\n"
        "├── attention_heatmap_image_checksums.json\n"
        "├── attention_heatmap_visual_notes.csv\n"
        "├── attention_heatmap_findings.csv\n"
        "├── attention_heatmap_discrepancies.json\n"
        "├── attention_heatmaps_manifest.json\n"
        "├── attention_heatmaps_contract.json\n"
        "├── attention_heatmap_summary.json\n"
        "├── attention_heatmap_report.md\n"
        "├── README_ATTENTION_HEATMAPS.md\n"
        "└── phase54_last_query_attention_context_handoff.json\n"
        "```\n\n"
    )

    L.append("## Image naming convention\n\n")
    L.append(
        "- V1: `V1_CASE<row>_SEED<seed>.png`\n"
        "- V2: `V2_CASE<row>_L<layer>.png`\n"
        "- Mode A: `MODE_A_CASE<row>_L<layer>.png`\n"
        "- V3: `V3_CASE<row>_SEED<seed>_L<layer>_H<head>_<mode>.png`\n\n"
    )

    L.append("## Forbidden operations\n\n")
    L.append(
        "Phase 53 production path NEVER executes:\n\n"
        "- `torch.load` / `model.forward` / `return_attention`\n"
        "- `materialize_phase52` / `extract_attention`\n"
        "- `fit` / `fit_transform` / scaler fitting\n"
        "- new Test inference / prediction generation\n"
        "- training, optimizer.step, .backward\n"
        "- best-head selection / best-seed selection / ensemble\n"
        "- attention mislabeled as feature importance or causal attribution\n"
        "- extracting last-query numeric metrics from PNG pixels\n\n"
    )

    L.append("## Upstream immutability\n\n")
    L.append(
        "Phase 47/48/49/50/51/52 canonical artifacts are UNCHANGED. "
        "Phase 53 may READ them; Phase 53 may NOT modify them.\n\n"
    )

    L.append("## Phase 54 handoff\n\n")
    L.append(
        "Phase 54 numeric source remains the Phase 52 RAW LAST_QUERY NPZ files. "
        "PNG pixels are NOT used as numerical input. "
        "Phase 54 implementation requires separate Human approval.\n\n"
    )

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("".join(L))
