"""Phase 53 — frozen render-config + render-config fingerprint.

Render-config MUST be FROZEN BEFORE any official rendering.

`attention_heatmap_render_config.json` is the canonical contract frozen up front;
its SHA-256 is `attention_heatmap_render_config_fingerprint.json`.

Frozen contract fields per canonical Phase 53 detail §130 + §18.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .sources import (
    ASPECT,
    COLORBAR_LABEL,
    COLORMAP,
    COLORMAP_POLICY,
    DPI,
    INCLUDE_OLDEST_TICK,
    INTERPOLATION,
    LAG_TICK_STEPS_REQUESTED,
    LOOKBACK,
    MATRIX_ORIENTATION,
    MAX_REPORT_CASES,
    MODE_A_NAME,
    MODE_A_VMAX,
    MODE_A_VMIN,
    MODE_B_NAME,
    MODE_B_VMAX_FORMULA,
    MODE_B_VMIN,
    NUM_HEADS,
    NUM_LAYERS,
    ORIGIN,
    RAW_DTYPE,
    REPORT_CASE_RANK_RANGE,
    REPORT_CASE_RULE,
    SEEDS,
    TRANSPOSE,
    X_LABEL,
    X_ORDER,
    Y_LABEL,
    Y_ORDER,
)


def build_render_config() -> dict:
    """Construct frozen render-config (canonical Phase 53 contract)."""
    cfg = {
        "version": "ATTENTION_HEATMAPS-render-config-v1",
        "phase": 53,
        "matrix_orientation": MATRIX_ORIENTATION,
        "transpose": TRANSPOSE,
        "origin": ORIGIN,
        "x_order": X_ORDER,
        "y_order": Y_ORDER,
        "x_label": X_LABEL,
        "y_label": Y_LABEL,
        "lag_tick_steps_requested": list(LAG_TICK_STEPS_REQUESTED),
        "lag_tick_steps_included": [int(lag) for lag in LAG_TICK_STEPS_REQUESTED if 1 <= lag <= LOOKBACK],
        "include_oldest_tick": INCLUDE_OLDEST_TICK,
        "interpolation": INTERPOLATION,
        "aspect": ASPECT,
        "colorbar_label": COLORBAR_LABEL,
        "colormap": COLORMAP,
        "colormap_policy": COLORMAP_POLICY,
        "mode_A": {
            "name": MODE_A_NAME,
            "vmin": MODE_A_VMIN,
            "vmax": MODE_A_VMAX,
            "purpose": "absolute probability reference; comparable across cases / seeds / layers / heads; no clipping.",
        },
        "mode_B": {
            "name": MODE_B_NAME,
            "vmin": MODE_B_VMIN,
            "vmax_formula": MODE_B_VMAX_FORMULA,
            "purpose": "case-shared comparison; one vmax per (target case) computed across all 3 seeds × all layers × all heads × all q × all s; all panels of that case share identical scale.",
        },
        "V1_layout": {
            "name": "V1_CASE_BY_SEED_GRID",
            "rows": "LAYERS",
            "columns": "HEADS",
            "panel_count": NUM_LAYERS * NUM_HEADS,
            "mode": MODE_B_NAME,
            "scope": "every dense case × every seed",
            "expected_total_grids": "n_cases × n_seeds",
        },
        "V2_layout": {
            "name": "V2_CROSS_SEED_REPORT_GRID",
            "rows": "SEEDS",
            "columns": "HEADS",
            "panel_count": len(SEEDS) * NUM_HEADS,
            "mode": MODE_B_NAME,
            "scope": "Phase51 W2 SHARED_WORST ranks 1-5 × each layer",
            "expected_total_grids": "n_report_cases × n_layers",
        },
        "V3_layout": {
            "name": "V3_INDIVIDUAL_PER_HEAD_MAP",
            "mode": MODE_B_NAME,
            "scope": "shared top5 report cases (deterministic only; no visual selection)",
        },
        "fixed_probability_layout": {
            "name": "FIXED_PROBABILITY_REPORT",
            "vmin": MODE_A_VMIN,
            "vmax": MODE_A_VMAX,
            "scope": "shared top5 report cases × each seed × each layer × each head",
        },
        "report_case_rule": REPORT_CASE_RULE,
        "report_case_rank_range": list(REPORT_CASE_RANK_RANGE),
        "max_report_cases": MAX_REPORT_CASES,
        "sepal_columns_rank1_to_5": True,
        "resolution_policy": {
            "primary_dpi": DPI,
            "vector_output": "svg_if_requested_but_canonical_png",
        },
        "figure_size_policy": {
            "V1_per_panel": [4.0, 4.0],
            "V2_per_panel": [3.5, 3.5],
            "V3_per_panel": [4.5, 4.5],
            "fixed_probability_per_panel": [4.0, 4.0],
        },
        "annotation_policy": {
            "row_labels": ["L1", "L2"][:NUM_LAYERS],
            "column_labels": [f"H{h+1}" for h in range(NUM_HEADS)],
            "x_axis_tick_interval_lag_steps": 6,
            "y_axis_tick_interval_lag_steps": 6,
        },
        "raw_dtype": RAW_DTYPE,
        "created_before_render": True,
        "created_at_utc": "2026-09-04",
    }
    return cfg


def freeze_render_config(output_dir: Path) -> tuple[Path, dict]:
    """Write frozen render config + fingerprint. Returns (cfg_fp, fingerprint_dict)."""
    cfg = build_render_config()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_fp = output_dir / "attention_heatmap_render_config.json"
    cfg_fp.write_text(json.dumps(cfg, indent=2, sort_keys=True))

    fp_data = {
        "render_config_path": str(cfg_fp.relative_to(cfg_fp.parent.parent.parent)),
        "render_config_sha256": hashlib.sha256(cfg_fp.read_bytes()).hexdigest(),
        "frozen_at_utc": "2026-09-04",
        "created_before_render": True,
        "status": "FROZEN",
    }
    fp_fp = output_dir / "attention_heatmap_render_config_fingerprint.json"
    fp_fp.write_text(json.dumps(fp_data, indent=2, sort_keys=True))
    return cfg_fp, fp_data
