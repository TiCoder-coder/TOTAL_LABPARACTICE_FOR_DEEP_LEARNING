import hashlib
import json
from pathlib import Path

import pytest

from course_work.reporting.frozen_evidence import (
    render_frozen_analysis_summary,
    render_frozen_phase_evidence,
    render_historical_analysis_summary,
    render_verified_courseworkv10_dashboard,
)


ROOT = Path(__file__).resolve().parents[2]


def test_missing_phase2_source_uses_frozen_log_table():
    output = render_frozen_phase_evidence(2, ROOT).data
    assert "Data Acquisition" in output
    assert "<table" in output
    assert "NOT_AVAILABLE" not in output


def test_phase36_uses_current_read_only_recovery_audit():
    output = render_frozen_phase_evidence(36, ROOT).data
    assert "Phase 36" in output
    assert "VALID_REUSABLE" in output
    assert "RENDER_ONLY" in output
    assert "BLOCKED" not in output
    assert "FROZEN_EVIDENCE" not in output
    assert "<table" in output


def test_phase6_adds_tabular_inventory_without_recomputing_eda():
    output = render_frozen_phase_evidence(6, ROOT).data
    assert "Exploratory Data Analysis" in output
    assert "<table" in output
    assert "eda_manifest.json" in output


def test_phase6_notebook_block_is_preserved_exactly_from_coursework_10():
    current = json.loads(
        (ROOT / "notebook_course_work/CourseWork.ipynb").read_text(encoding="utf-8")
    )
    reference_path = ROOT / "notebook_course_work/CourseWork_10.ipynb"
    if not reference_path.is_file():
        # The retained read-only recovery source uses the canonical notebook
        # name.  Never copy or mutate it merely to satisfy this presentation
        # regression check.
        reference_path = ROOT.parent / "COURSE_WORK_2/notebook_course_work/CourseWork.ipynb"
    if not reference_path.is_file():
        pytest.skip("Phase 6 recovery-source notebook is not present in this workspace")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))

    def phase6_cells(notebook):
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        start = next(index for index, source in enumerate(sources) if source.startswith("## Phase 6 "))
        end = next(index for index, source in enumerate(sources[start + 1 :], start + 1) if source.startswith("## Phase 7 "))
        return notebook["cells"][start:end]

    assert phase6_cells(current) == phase6_cells(reference)


def test_all_retained_phase6_tables_and_calendar_figures_match_signoff():
    signoff = json.loads(
        (ROOT / "artifacts/eda/phase_6_signoff.json").read_text(encoding="utf-8")
    )
    table_entries = {
        path: checksum
        for path, checksum in signoff["output_checksums"].items()
        if path.startswith("artifacts/eda/tables/")
    }
    assert len(table_entries) == 16

    for relative_path, expected_checksum in table_entries.items():
        artifact_path = ROOT / relative_path
        assert artifact_path.is_file()
        assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == expected_checksum

    calendar_figures = [
        "EDA_06_hourly_profile.png",
        "EDA_07_weekday_profile.png",
        "EDA_08_weekday_weekend.png",
        "EDA_09_hour_weekday_heatmap.png",
    ]
    for filename in calendar_figures:
        relative_path = f"artifacts/eda/figures/{filename}"
        artifact_path = ROOT / relative_path
        assert artifact_path.is_file()
        assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == signoff["output_checksums"][relative_path]


def test_phase54_summary_fallback_is_tabular():
    output = render_historical_analysis_summary(
        54,
        ROOT,
        "artifacts/last_query_attention/last_query_attention_summary.json",
        "artifacts/last_query_attention/phase_54_signoff.json",
    ).data
    assert "HISTORICAL_SUMMARY_ONLY" in output
    assert "FROZEN_EVIDENCE" not in output
    assert "<table" in output
    assert "last_query_attention_summary.json" in output


def test_historical_summary_backward_compatible_alias():
    output = render_frozen_analysis_summary(
        54,
        ROOT,
        "artifacts/last_query_attention/last_query_attention_summary.json",
        "artifacts/last_query_attention/phase_54_signoff.json",
    ).data
    assert "HISTORICAL_SUMMARY_ONLY" in output


def test_courseworkv10_phase54_57_dashboards_are_checksum_verified():
    for phase_id in range(54, 58):
        output = render_verified_courseworkv10_dashboard(phase_id, ROOT).data
        assert output.count("<table") == 4
        assert "HISTORICAL_SUMMARY_ONLY" not in output
        assert "FROZEN_EVIDENCE" not in output
        assert "<script" not in output.lower()
