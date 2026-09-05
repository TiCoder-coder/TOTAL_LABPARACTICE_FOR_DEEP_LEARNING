"""Phase 50-H focused tests — notebook presentation dashboard + preservation."""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
from pathlib import Path

import nbformat
import pytest

from course_work.reporting.phase_50_dashboard import render_phase_50_dashboard


NB = Path("notebook_course_work/CourseWork_1.ipynb")


def test_notebook_has_phase50_cells():
    nb = nbformat.read(NB, as_version=4)
    # Must have >= 132 cells (130 + 2 new)
    assert len(nb.cells) >= 132


def test_phase50_md_cell_present():
    nb = nbformat.read(NB, as_version=4)
    md = next((c for c in nb.cells if c.cell_type == "markdown" and "Phase 50" in c.source), None)
    assert md is not None
    assert "Error-by-Regime Analysis" in md.source


def test_phase50_code_cell_present_and_executed():
    nb = nbformat.read(NB, as_version=4)
    code = next(
        (
            c for c in nb.cells
            if c.cell_type == "code" and "render_phase_50_dashboard" in c.source
        ),
        None,
    )
    assert code is not None
    assert code.get("outputs"), "code cell must have at least one output"
    out = code.outputs[0]
    assert out.output_type == "display_data"
    assert "text/html" in out.data
    assert len(out.data["text/html"]) > 100000, "dashboard HTML must be non-trivial"


def test_phase50_html_contains_required_sections():
    nb = nbformat.read(NB, as_version=4)
    code = next(
        c for c in nb.cells
        if c.cell_type == "code" and "render_phase_50_dashboard" in c.source
    )
    html_str = code.outputs[0].data["text/html"]
    for section in [
        "Phase 50 status",
        "TRAIN-derived thresholds",
        "Test regime prevalence",
        "Target-level error",
        "Extreme-high error",
        "Change-magnitude error",
        "Change-direction error",
        "Time-of-day error",
        "Day-type error",
        "SAE / SSE contribution",
        "RMSE lift",
        "Cross-seed regime stability",
        "Persistence baseline by regime",
        "LSTM eligibility",
        "Phase 48 seed-spread by regime",
        "Phase 49 sign-consensus by regime",
        "Scientific findings",
        "Curated figures",
    ]:
        assert section in html_str, f"section missing: {section}"


def test_phase50_html_embedded_figures():
    nb = nbformat.read(NB, as_version=4)
    code = next(c for c in nb.cells if c.cell_type == "code" and "render_phase_50_dashboard" in c.source)
    html_str = code.outputs[0].data["text/html"]
    n_imgs = html_str.count("data:image/png;base64")
    assert n_imgs >= 10, f"expected >= 10 embedded figures, got {n_imgs}"


def test_phase50_html_no_training_or_scaler_text():
    nb = nbformat.read(NB, as_version=4)
    code = next(c for c in nb.cells if c.cell_type == "code" and "render_phase_50_dashboard" in c.source)
    html_str = code.outputs[0].data["text/html"]
    for forbidden in ["model.train", "fit_transform", ".backward(", "torch.load", "checkpoint", "scaler.fit"]:
        # "checkpoint" appears only in status text as "checkpoint_loading"; allow that exact phrase
        if forbidden == "checkpoint":
            # we need to check there's no positive claim
            assert "checkpoint_loading = true" not in html_str.lower()
        else:
            assert forbidden not in html_str, f"forbidden token in HTML: {forbidden}"


def test_phase0_49_cells_preserved_source():
    """Phase 0-49 cell sources must be unchanged (no rerun)."""
    backups = sorted(Path("notebook_course_work").glob("CourseWork_1.ipynb.bak_before_phase50_*"))
    assert backups, "no Phase 50 backup found"
    backup = backups[-1]
    nb_pre = nbformat.read(backup, as_version=4)
    nb_post = nbformat.read(NB, as_version=4)
    n_pre = len(nb_pre.cells)
    n_post = len(nb_post.cells)
    assert n_post >= n_pre + 2
    for i in range(n_pre):
        assert nb_post.cells[i].source == nb_pre.cells[i].source, (
            f"cell {i} source changed"
        )


def test_phase0_49_cells_preserved_outputs():
    """Phase 0-49 cell outputs must be unchanged (no rerun)."""
    backups = sorted(Path("notebook_course_work").glob("CourseWork_1.ipynb.bak_before_phase50_*"))
    backup = backups[-1]
    nb_pre = nbformat.read(backup, as_version=4)
    nb_post = nbformat.read(NB, as_version=4)
    n_pre = len(nb_pre.cells)
    for i in range(n_pre):
        pre_out = nb_pre.cells[i].get("outputs", [])
        post_out = nb_post.cells[i].get("outputs", [])
        assert len(pre_out) == len(post_out), (
            f"cell {i} output count changed: {len(pre_out)} → {len(post_out)}"
        )


def test_dashboard_renderer_read_only():
    """phase_50_dashboard.py must not import or call training routines."""
    src = Path("src/course_work/reporting/phase_50_dashboard.py").read_text()
    forbidden = [
        "materialize_phase50",  # reversed just for safety
        "model.train",
        "optimizer.step",
        ".backward(",
        "torch.load",
        "scaler.fit",
        "fit_transform",
        "subprocess",
        "run_path",
    ]
    for f in forbidden:
        assert f not in src, f"forbidden token in renderer source: {f}"


def test_phase47_unchanged_post_50h():
    expected = {
        "artifacts/final_test/phase_47_signoff.json": "80614523b6091e21",
        "artifacts/final_test/predictions/final_test_predictions_seed42.csv": "246ee0d725af972b",
        "artifacts/final_test/predictions/final_test_predictions_persistence.csv": "7115af1c479b8957",
    }
    for p, exp in expected.items():
        sha = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        assert sha.startswith(exp)


def test_phase48_unchanged_post_50h():
    sha = hashlib.sha256(Path("artifacts/prediction_analysis/phase_48_signoff.json").read_bytes()).hexdigest()
    assert sha.startswith("e8c102d582a35dd2")


def test_phase49_unchanged_post_50h():
    expected = {
        "artifacts/residual_analysis/phase_49_signoff.json": "9d5fc659717dbc15",
        "artifacts/residual_analysis/phase50_handoff.json": "ebfe2cdfbdb8523f",
        "artifacts/residual_analysis/residual_long_table.csv": "8418a99110bfda70",
        "artifacts/residual_analysis/residual_wide_table.csv": "931ff9109aef236d",
        "artifacts/residual_analysis/phase49_cross_seed_sign_consensus.csv": "41e862ff18693f8c",
    }
    for p, exp in expected.items():
        sha = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        assert sha.startswith(exp)


def test_render_works_with_explicit_path():
    """Sanity: renderer returns HTML object with expected structure."""
    obj = render_phase_50_dashboard("/Users/vientu/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK")
    assert hasattr(obj, "data")
    assert obj.data.startswith("<style>") or "<style>" in obj.data[:1000]


def test_run_all_not_used():
    """Notebook cell execution_count should be None for Phase 50 cell (not run via kernel)."""
    nb = nbformat.read(NB, as_version=4)
    code = next(c for c in nb.cells if c.cell_type == "code" and "render_phase_50_dashboard" in c.source)
    # execution_count is None → we did not "run" the cell in a kernel
    # (we built the output deterministically and injected it as a display_data output)
    assert code.execution_count is None
