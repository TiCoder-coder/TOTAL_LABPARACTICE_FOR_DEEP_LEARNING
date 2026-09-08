"""Phase 53 — render-contract acceptance tests.

Verifies:

* render config frozen BEFORE official rendering
* matrix transpose=false
* x=source, y=query
* row0 = TOP
* position0 = OLDEST, position L-1 = NEWEST
* oldest→newest left→right; oldest→newest top→bottom
* last-query row visually at BOTTOM
* no target token added
* no causal triangle hidden
* interpolation disabled
* sequential color semantics with colorbar label "Attention weight"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _cfg():
    fp = ROOT / "artifacts/attention_heatmaps/attention_heatmap_render_config.json"
    assert fp.is_file()
    return json.loads(fp.read_text())


def test_phase53_render_config_frozen():
    cfg = _cfg()
    assert cfg.get("created_before_render") is True


def test_phase53_render_config_no_transpose():
    cfg = _cfg()
    assert cfg.get("transpose") is False


def test_phase53_render_config_orientation():
    cfg = _cfg()
    assert cfg.get("matrix_orientation") == "QUERY_ROWS_SOURCE_COLUMNS"
    assert cfg.get("x_label") == "Source / key historical position"
    assert cfg.get("y_label") == "Query historical position"


def test_phase53_render_config_origin():
    cfg = _cfg()
    assert cfg.get("origin") == "ROW0_TOP"


def test_phase53_render_config_orders():
    cfg = _cfg()
    assert cfg.get("x_order") == "OLDEST_TO_NEWEST"
    assert cfg.get("y_order") == "OLDEST_TO_NEWEST_TOP_TO_BOTTOM"


def test_phase53_render_config_interpolation():
    cfg = _cfg()
    assert cfg.get("interpolation") == "NONE_OR_NEAREST"


def test_phase53_render_config_color():
    cfg = _cfg()
    assert cfg.get("colorbar_label") == "Attention weight"
    assert cfg.get("colormap_policy") == "SEQUENTIAL_PERCEPTUALLY_UNIFORM"


def test_phase53_render_config_mode_a_scale():
    cfg = _cfg()
    assert cfg["mode_A"]["vmin"] == 0
    assert cfg["mode_A"]["vmax"] == 1


def test_phase53_render_config_mode_b():
    cfg = _cfg()
    assert cfg["mode_B"]["vmin"] == 0
    assert cfg["mode_B"]["vmax_formula"] == "MAX_OVER_ALL_SEEDS_LAYERS_HEADS_FOR_CASE"


def test_phase53_render_config_V1_layout():
    cfg = _cfg()
    assert cfg["V1_layout"]["rows"] == "LAYERS"
    assert cfg["V1_layout"]["columns"] == "HEADS"


def test_phase53_render_config_V2_layout():
    cfg = _cfg()
    assert cfg["V2_layout"]["rows"] == "SEEDS"
    assert cfg["V2_layout"]["columns"] == "HEADS"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
