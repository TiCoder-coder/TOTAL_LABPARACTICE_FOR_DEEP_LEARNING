"""Phase 53 — Mode-B case-shared scale computation.

For each target case c, compute vmax_case as the maximum attention weight
across all 3 seeds × all layers × all heads × all q × all s for THAT case.

This guarantees:

* Mode B scale is computed ONCE per case;
* all Mode B panels for the same case share an EXACT scale;
* no per-panel autoscale.

Output: `attention_heatmap_case_scale_manifest.csv` per canonical schema §133.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .sources import (
    MODE_B_NAME,
    MODE_B_VMIN,
    NUM_HEADS,
    NUM_LAYERS,
    SEEDS,
)


def compute_case_scales(loaded: dict) -> dict:
    """Compute Mode B case-wide vmax per case."""
    seed42 = loaded[42]["attention"]   # [K, L, H, L, L]
    seed123 = loaded[123]["attention"]
    seed2026 = loaded[2026]["attention"]
    K = seed42.shape[0]
    scales = {}
    for c in range(K):
        # Concatenate all (seeds × layers × heads) for this case
        per_seed_layer_head = []
        for arr in (seed42, seed123, seed2026):
            # arr[c] -> [L, H, L, L]
            per_seed_layer_head.append(arr[c])
        stacked = np.stack(per_seed_layer_head, axis=0)  # [3, L, H, L, L]
        case_max = float(np.max(stacked))
        case_min = float(np.min(stacked))
        scales[c] = {
            "case_row_idx0": c,
            "case_raw_min": case_min,
            "case_raw_max": case_max,
            "vmin": MODE_B_VMIN,
            "vmax": case_max,
            "all_panels_share_scale": True,
            "mode_B_name": MODE_B_NAME,
            "case_attention_max_positive": case_max > 0,
        }
    return scales


def get_seed_specific_max(loaded: dict, case_row: int) -> dict:
    """Return per-seed vmax for a given case row (used in case-scale manifest)."""
    out = {}
    for seed in SEEDS:
        arr = loaded[seed]["attention"]
        out[seed] = float(np.max(arr[case_row]))
    return out


def write_case_scale_manifest(
    scales: dict,
    loaded: dict,
    target_ids: list[str],
    target_timestamps: dict[str, str],
    output_csv: Path,
) -> None:
    """Write `attention_heatmap_case_scale_manifest.csv`."""
    rows = []
    for case_row, scale in sorted(scales.items()):
        tid = target_ids[case_row] if case_row < len(target_ids) else "UNKNOWN"
        ts = target_timestamps.get(tid, "UNKNOWN")
        per_seed = get_seed_specific_max(loaded, case_row)
        rows.append({
            "case_row_idx0": case_row,
            "target_id": tid,
            "target_timestamp": ts,
            "case_raw_min": f"{scale['case_raw_min']:.6e}",
            "case_raw_max": f"{scale['case_raw_max']:.6e}",
            "seed42_max": f"{per_seed[42]:.6e}",
            "seed123_max": f"{per_seed[123]:.6e}",
            "seed2026_max": f"{per_seed[2026]:.6e}",
            "mode_B_vmin": scale["vmin"],
            "mode_B_vmax": f"{scale['vmax']:.6e}",
            "all_panels_share_scale": scale["all_panels_share_scale"],
            "case_attention_max_positive": scale["case_attention_max_positive"],
            "status": "PASS" if scale["case_attention_max_positive"] else "FAIL",
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit("Phase 53 scales is a library — import it from orchestrator.")
