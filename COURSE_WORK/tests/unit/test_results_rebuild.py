import json
import hashlib
from pathlib import Path

from course_work.reporting.results_rebuild import (
    render_verified_phase_result,
    render_verified_v2_results,
)


ROOT = Path(__file__).resolve().parents[2]


def test_every_phase_43_59_renders_nonempty_table_without_evidence_metadata():
    for phase_id in range(43, 60):
        html = render_verified_phase_result(phase_id, ROOT).data
        assert html.count("<table") == 3
        assert "Phase status" in html
        assert "Configuration / analysis" in html
        assert "Results / decision" in html
        assert "Evidence:" not in html
        assert "artifacts/" not in html
        assert "<script" not in html.lower()
        assert "raw json" not in html.lower()


def test_phase47_uses_current_verified_per_seed_and_baseline_metrics():
    html = render_verified_phase_result(47, ROOT).data
    for value in ("64.942754", "61.986086", "64.560136", "66.836915"):
        assert value in html
    assert "Best seed" not in html


def test_v2_result_chain_is_complete_without_evidence_metadata():
    html = render_verified_v2_results(ROOT).data
    for label in ("E01", "E20", "Step 14A", "Step 14B", "Step 16", "Step 17", "Closure"):
        assert label in html
    assert "61.608937" in html
    assert "COMPLETE" in html
    assert html.count("<table") == 3
    assert "Phase status" in html
    assert "Configuration / analysis" in html
    assert "Results / decision" in html
    assert "Evidence:" not in html
    assert "artifacts/" not in html


def test_notebook_phase_1_42_source_matches_locked_rebuild_hash():
    notebook = json.loads((ROOT / "notebook_course_work/CourseWork.ipynb").read_text(encoding="utf-8"))
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    phase43_index = next(i for i, source in enumerate(sources) if source.startswith("## Phase 43"))
    assert phase43_index == 117
    # Output and execution metadata may be refreshed read-only.  Lock the
    # scientific narrative and executable source, not transient UI metadata.
    prefix = json.dumps(
        [
            {
                "cell_type": cell.get("cell_type"),
                "id": cell.get("id"),
                "source": cell.get("source", []),
            }
            for cell in notebook["cells"][:phase43_index]
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert hashlib.sha256(prefix).hexdigest() == (
        "f3d57cb71a38d40d73362a87930314ea75dbdb2ed9411024c72386a0adc0fad9"
    )
