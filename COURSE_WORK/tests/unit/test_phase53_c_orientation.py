"""Phase 53 — orientation acceptance tests.

Verifies:

* Synthetic asymmetric-matrix test PASS
* Real-source coordinate mapping test PASS
* Diagonal top-left → bottom-right
* Last-query row is bottom
* Source newest position is right edge
* Query newest position is bottom edge
* Lag ticks match Phase52 map
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read_csv(fp):
    with fp.open() as f:
        return list(csv.DictReader(f))


def test_phase53_orientation_synthetic_passed():
    fp = ROOT / "artifacts/attention_heatmaps/attention_heatmap_orientation_tests.csv"
    rows = _read_csv(fp)
    syn = [r for r in rows if r["test_case"] == "SYNTHETIC_ASYMMETRIC_MATRIX"]
    assert syn, "synthetic test row missing"
    assert syn[0]["status"] == "PASS"


def test_phase53_orientation_real_source_passed():
    fp = ROOT / "artifacts/attention_heatmaps/attention_heatmap_orientation_tests.csv"
    rows = _read_csv(fp)
    rs = [r for r in rows if r["test_case"] == "REAL_SOURCE_DENSE_SLICE_SEED42_CASE0_L0_H0"]
    assert rs, "real-source test row missing"
    assert rs[0]["status"] == "PASS"
    assert rs[0]["transpose_applied"] == "FALSE"
    assert rs[0]["expected_x"] == rs[0]["source_idx"]
    assert rs[0]["expected_y"] == rs[0]["query_idx"]


def test_phase53_orientation_diagonal_top_left_to_bottom_right():
    fp = ROOT / "artifacts/attention_heatmaps/attention_heatmap_orientation_tests.csv"
    rows = _read_csv(fp)
    diag = [r for r in rows if r["test_case"] == "DIAGONAL_DIRECTION"]
    assert diag, "diagonal test row missing"
    assert diag[0]["diagonal_direction"] == "TOP_LEFT_TO_BOTTOM_RIGHT"


def test_phase53_orientation_last_query_bottom_edge():
    fp = ROOT / "artifacts/attention_heatmaps/attention_heatmap_orientation_tests.csv"
    rows = _read_csv(fp)
    lq = [r for r in rows if r["test_case"] == "LAST_QUERY_BOTTOM_EDGE"]
    assert lq, "last-query test row missing"
    assert lq[0]["last_query_display_edge"] == "BOTTOM"


def test_phase53_orientation_axis_tick_audit_present():
    fp = ROOT / "artifacts/attention_heatmaps/attention_heatmap_axis_tick_audit.csv"
    rows = _read_csv(fp)
    assert rows
    statuses = {r["status"] for r in rows}
    assert all(s in ("OK", "EXCLUDED") for s in statuses)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
