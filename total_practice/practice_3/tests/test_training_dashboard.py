import json
from pathlib import Path

from processing_own_phase.training_dashboard import render_training_dashboard


def test_dashboard_reads_only_aggregate_and_saves_html(tmp_path: Path) -> None:
    payload = {
        "protocol_version": "practice_3_v2.3", "generated_at": "now", "experiments": [],
        "training_history": {}, "experiment_comparison": {"ranking": []},
        "winner": {}, "final_holdout": {"available": False},
        "error_analysis": {"available": False}, "custom_inference": {"available": False},
        "save_reload_verification": {"available": False}, "sources": {},
    }
    source, destination = tmp_path / "visualization_data.json", tmp_path / "dashboard.html"
    source.write_text(json.dumps(payload), encoding="utf-8")
    dashboard = render_training_dashboard(source, destination)
    assert destination.is_file()
    assert "Practice 3 — Training & Evaluation Dashboard" in dashboard.html
    assert dashboard.html.count("Artifact not generated yet") >= 4
    assert "Trainer" not in dashboard.html
