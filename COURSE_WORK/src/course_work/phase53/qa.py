"""Phase 53 — render QA + image checksums.

`attention_heatmap_render_audit.csv` per canonical schema §137.
`attention_heatmap_image_checksums.json` per canonical schema §139.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

from .sources import (
    COLORBAR_LABEL,
    INTERPOLATION,
    MATRIX_ORIENTATION,
    MODE_A_NAME,
    MODE_A_VMAX,
    MODE_A_VMIN,
    MODE_B_NAME,
    ORIGIN,
    TRANSPOSE,
)


def write_render_qa(
    render_audit_rows: list[dict],
    output_csv: Path,
) -> None:
    """Write `attention_heatmap_render_audit.csv`."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not render_audit_rows:
        # headers only
        with output_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "image_id", "file_exists", "file_size_bytes",
                "expected_panel_count", "observed_panel_count",
                "orientation_verified", "axis_semantics_verified",
                "scale_mode", "vmin", "vmax", "raw_max_within_scale",
                "colorbar_configured", "interpolation_policy_match",
                "render_complete", "status",
            ])
            w.writeheader()
        return
    with output_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(render_audit_rows[0].keys()))
        w.writeheader()
        w.writerows(render_audit_rows)


def hash_image(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_image_checksums(
    image_shas: dict,
    render_config_sha256: str,
    output_json: Path,
) -> None:
    """Write `attention_heatmap_image_checksums.json`."""
    payload = {
        "render_config_sha256": render_config_sha256,
        "images": image_shas,
        "n_images": len(image_shas),
        "status": "PASS" if all(image_shas.values()) else "FAIL",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit("Phase 53 qa is a library — import it from orchestrator.")
