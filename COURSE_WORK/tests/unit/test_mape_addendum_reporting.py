from pathlib import Path

from course_work.reporting.mape_addendum import render_mape_addendum


def test_real_mape_addendum_renderer_is_static_and_complete() -> None:
    project_root = Path(__file__).resolve().parents[2]
    rendered = render_mape_addendum(project_root).data
    assert "Supplementary MAPE Metric Addendum" in rendered
    assert "Validation MAPE by available prediction artifact" in rendered
    assert "Machine-learning pipeline compliance audit" in rendered
    assert "BLOCKED_SOURCE_UNAVAILABLE" in rendered
    assert "Cross-config aggregation" in rendered
    assert "PROHIBITED" in rendered
    assert "<script" not in rendered.lower()
    assert "widget" not in rendered.lower()
