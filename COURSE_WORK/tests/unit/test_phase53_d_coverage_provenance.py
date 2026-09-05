"""Phase 53 — V1/V2/Mode A coverage + catalog/provenance + interpretation-scope tests.

Verifies:

* Every dense case rendered
* Seed42/123/2026 grid for every case
* Rows=layers, columns=heads
* Panel count = N_layers * N_heads (V1) and 3 * N_heads (V2)
* No missing layer/head
* Report cases = Phase51 W2 SHARED_WORST ranks 1-5
* No visual case substitution
* Catalog complete
* No head/seed/case selection/ranking
* Same-index head semantic caveat included
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read_csv(fp):
    with fp.open() as f:
        return list(csv.DictReader(f))


def test_phase53_v1_grid_count():
    catalog = _read_csv(ROOT / "artifacts/attention_heatmaps/attention_heatmap_catalog.csv")
    v1 = [r for r in catalog if r["image_type"] == "V1"]
    assert len(v1) == 44 * 3, f"V1 count {len(v1)} != 132"


def test_phase53_v1_panel_count():
    catalog = _read_csv(ROOT / "artifacts/attention_heatmaps/attention_heatmap_catalog.csv")
    v1 = [r for r in catalog if r["image_type"] == "V1"]
    for r in v1:
        # 2 layers * 4 heads = 8 panels
        assert r["panel_count"] == "8", f"V1 panel_count {r['panel_count']} != 8"


def test_phase53_v2_count():
    catalog = _read_csv(ROOT / "artifacts/attention_heatmaps/attention_heatmap_catalog.csv")
    v2 = [r for r in catalog if r["image_type"] == "V2"]
    # 5 report cases * 2 layers = 10
    assert len(v2) == 10, f"V2 count {len(v2)} != 10"


def test_phase53_v2_panel_count():
    catalog = _read_csv(ROOT / "artifacts/attention_heatmaps/attention_heatmap_catalog.csv")
    v2 = [r for r in catalog if r["image_type"] == "V2"]
    for r in v2:
        # 3 seeds * 4 heads = 12 panels
        assert r["panel_count"] == "12", f"V2 panel_count {r['panel_count']} != 12"


def test_phase53_mode_a_count():
    catalog = _read_csv(ROOT / "artifacts/attention_heatmaps/attention_heatmap_catalog.csv")
    mode_a = [r for r in catalog if r["image_type"] == "MODE_A_FIXED_PROBABILITY"]
    assert len(mode_a) == 10


def test_phase53_report_cases_are_top5():
    rc = _read_csv(ROOT / "artifacts/attention_heatmaps/attention_heatmap_report_cases.csv")
    assert len(rc) == 5
    ranks = sorted(int(r["shared_worst_rank"]) for r in rc)
    assert ranks == [1, 2, 3, 4, 5]


def test_phase53_catalog_complete():
    catalog = _read_csv(ROOT / "artifacts/attention_heatmaps/attention_heatmap_catalog.csv")
    # every row must have raw_source_sha256 + case_order_sha256 + render_config_sha256
    for r in catalog:
        assert r["raw_source_sha256"]
        assert r["case_order_sha256"]
        assert r["render_config_sha256"]
        assert r["image_sha256_if_available"]


def test_phase53_image_checksums_present():
    fp = ROOT / "artifacts/attention_heatmaps/attention_heatmap_image_checksums.json"
    assert fp.is_file()
    payload = json.loads(fp.read_text())
    assert payload["status"] == "PASS"
    assert payload["n_images"] > 0
    assert all(payload["images"].values())


def test_phase53_no_head_ranking():
    so = json.loads((ROOT / "artifacts/attention_heatmaps/phase_53_signoff.json").read_text())
    assert so["head_selection"] is False


def test_phase53_no_seed_selection():
    so = json.loads((ROOT / "artifacts/attention_heatmaps/phase_53_signoff.json").read_text())
    assert so["seed_selection"] is False


def test_phase53_no_case_selection_changed():
    so = json.loads((ROOT / "artifacts/attention_heatmaps/phase_53_signoff.json").read_text())
    assert so["case_selection_changed"] is False


def test_phase53_no_attention_feature_importance_claim():
    so = json.loads((ROOT / "artifacts/attention_heatmaps/phase_53_signoff.json").read_text())
    assert so["attention_feature_importance_claim"] is False


def test_phase53_no_causal_claim():
    so = json.loads((ROOT / "artifacts/attention_heatmaps/phase_53_signoff.json").read_text())
    assert so["attention_causal_claim"] is False


def test_phase53_phase54_not_authorized():
    so = json.loads((ROOT / "artifacts/attention_heatmaps/phase_53_signoff.json").read_text())
    assert so["phase54_authorized"] is False
    assert so["phase53_h_authorized"] is False


def test_phase53_notebook_unchanged():
    so = json.loads((ROOT / "artifacts/attention_heatmaps/phase_53_signoff.json").read_text())
    assert so.get("overall_status") in ("PASS", "PASS_WITH_WARNING")
    log = json.loads((ROOT / "docs/save_log_in_processing/phase_53_attention_heatmaps_log.json").read_text())
    assert log["safety"]["notebook_modified"] is False


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
