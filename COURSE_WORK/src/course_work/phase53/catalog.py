"""Phase 53 — catalog writer + case visualization index.

`attention_heatmap_catalog.csv` per canonical schema §135.
`attention_heatmap_case_index.csv` per canonical schema §136.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .sources import (
    NUM_HEADS,
    NUM_LAYERS,
    SEEDS,
)


def write_heatmap_catalog(
    catalog_rows: list[dict],
    output_csv: Path,
) -> None:
    """Write `attention_heatmap_catalog.csv`."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not catalog_rows:
        # Write headers only
        with output_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "image_id", "path", "image_type",
                "case_row_idx0", "target_id", "target_timestamp",
                "seed", "layer_idx0_if_specific", "layer_display_if_specific",
                "head_idx0_if_specific", "head_display_if_specific",
                "render_mode", "vmin", "vmax",
                "grid_rows", "grid_columns", "panel_count",
                "raw_source_file", "raw_source_sha256",
                "case_order_sha256", "position_map_sha256", "render_config_sha256",
                "report_selected", "image_sha256_if_available", "status",
            ])
            w.writeheader()
        return
    with output_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(catalog_rows[0].keys()))
        w.writeheader()
        w.writerows(catalog_rows)


def build_case_index(
    case_rows: list[dict],
    seed_grids: dict,
    cross_seed_grids: dict,
    fixed_prob_views: dict,
    output_csv: Path,
) -> None:
    """Write `attention_heatmap_case_index.csv`.

    case_rows: list of dicts per case_row_idx0 with target_id, target_timestamp,
               selection_roles, shared_rank, six regime labels.
    seed_grids: {case_row: {"seed42_grid": path, ...}}
    cross_seed_grids: {case_row: [paths]}
    fixed_prob_views: {case_row: [paths]}
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for cr in case_rows:
        cr_idx = cr["case_row_idx0"]
        sg = seed_grids.get(cr_idx, {})
        out_rows.append({
            "case_row_idx0": cr_idx,
            "target_id": cr.get("target_id", "UNKNOWN"),
            "target_timestamp": cr.get("target_timestamp", "UNKNOWN"),
            "selection_roles": cr.get("selection_roles", ""),
            "shared_rank": cr.get("shared_rank", ""),
            "target_level_regime": cr.get("R1_TARGET_LEVEL", ""),
            "extreme_high_regime": cr.get("R2_EXTREME_HIGH", ""),
            "change_magnitude_regime": cr.get("R3_CHANGE_MAGNITUDE", ""),
            "change_direction_regime": cr.get("R4_CHANGE_DIRECTION", ""),
            "time_of_day_regime": cr.get("R5_TIME_OF_DAY", ""),
            "day_type_regime": cr.get("R6_DAY_TYPE", ""),
            "seed42_grid": sg.get("seed42_grid", ""),
            "seed123_grid": sg.get("seed123_grid", ""),
            "seed2026_grid": sg.get("seed2026_grid", ""),
            "cross_seed_layer_grids": "|".join(cross_seed_grids.get(cr_idx, [])),
            "fixed_probability_view": "|".join(fixed_prob_views.get(cr_idx, [])),
            "status": "PASS" if (sg.get("seed42_grid") and sg.get("seed123_grid") and sg.get("seed2026_grid")) else "FAIL",
        })
    with output_csv.open("w", newline="") as f:
        if not out_rows:
            return
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)


if __name__ == "__main__":
    raise SystemExit("Phase 53 catalog is a library — import it from orchestrator.")
