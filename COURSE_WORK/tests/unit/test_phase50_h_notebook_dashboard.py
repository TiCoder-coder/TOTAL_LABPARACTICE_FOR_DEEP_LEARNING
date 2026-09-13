"""Read-only regression checks for the current Phase 50 presentation."""
from __future__ import annotations

import json
from pathlib import Path

from course_work.reporting.results_rebuild import render_verified_phase_result


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebook_course_work/CourseWork.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _phase50_cell() -> dict:
    return next(
        cell
        for cell in _notebook()["cells"]
        if cell.get("cell_type") == "code"
        and "render_verified_phase_result(50, PROJECT_ROOT)" in "".join(cell.get("source", []))
    )


def test_notebook_has_current_phase50_markdown_and_code_cells():
    notebook = _notebook()
    assert any(
        cell.get("cell_type") == "markdown"
        and "Phase 50" in "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    assert _phase50_cell()


def test_phase50_saved_output_matches_current_verified_renderer():
    cell = _phase50_cell()
    assert len(cell.get("outputs", [])) == 1
    output = cell["outputs"][0]
    assert output["output_type"] == "display_data"
    stored = "".join(output["data"]["text/html"])
    current = render_verified_phase_result(50, ROOT).data
    assert stored == current
    assert current.count("<table") == 3
    assert "Phase status" in current
    assert "Configuration / analysis" in current
    assert "Results / decision" in current


def test_phase50_presentation_contains_no_execution_tokens_or_evidence_paths():
    html = render_verified_phase_result(50, ROOT).data
    for forbidden in (
        "model.train",
        "optimizer.step",
        ".backward(",
        "torch.load",
        "scaler.fit",
        "fit_transform",
        "Evidence:",
        "artifacts/",
    ):
        assert forbidden not in html


def test_phase50_renderer_sources_are_read_only():
    for relative in (
        "src/course_work/reporting/results_rebuild.py",
        "src/course_work/reporting/phase_50_dashboard.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        for forbidden in ("model.train", "optimizer.step", ".backward(", "scaler.fit"):
            assert forbidden not in source


def test_phase47_50_current_source_artifacts_exist():
    for relative in (
        "artifacts/final_test/phase_47_signoff.json",
        "artifacts/prediction_analysis/phase_48_signoff.json",
        "artifacts/residual_analysis/phase49_summary.json",
        "artifacts/error_by_regime/phase50_findings.json",
        "artifacts/error_by_regime/phase_50_signoff.json",
    ):
        assert (ROOT / relative).is_file(), relative
