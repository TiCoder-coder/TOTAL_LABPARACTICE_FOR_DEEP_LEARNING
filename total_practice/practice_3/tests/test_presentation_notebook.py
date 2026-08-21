from __future__ import annotations

import json
from pathlib import Path

from processing_own_phase.visualization_data import load_live_training_snapshot, load_presentation_data


NOTEBOOK = Path(__file__).parents[1] / "notebook_practice_3" / "practice_3.ipynb"


def test_notebook_has_no_execution_entrypoints() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code")
    prohibited = (
        "Trainer.train", "trainer.train", "optimizer.step", "run_experiment(",
        "experiment_runner_v2_3", "final_holdout_evaluation", "from_pretrained(",
        "load_dataset(", "while True", "run_exercise_1_pipeline(",
    )
    assert not [token for token in prohibited if token in source]
    assert "build_visualization_data" in source
    assert "render_training_dashboard" in source
    assert "practice_3_dashboard.html" in source


def test_missing_optional_artifacts_are_non_raising(tmp_path: Path) -> None:
    data = load_presentation_data(tmp_path)
    assert data["final_holdout"]["metrics"]["message"] == "Artifact not generated yet"
    assert data["error_analysis"]["errors"]["message"] == "Artifact not generated yet"
    assert data["custom_inference"]["message"] == "Artifact not generated yet"
    assert data["save_reload"]["message"] == "Artifact not generated yet"


def test_live_snapshot_never_waits(tmp_path: Path) -> None:
    snapshot = load_live_training_snapshot(tmp_path)
    assert snapshot["active"] is False
    assert snapshot["message"] == "No active training session."
