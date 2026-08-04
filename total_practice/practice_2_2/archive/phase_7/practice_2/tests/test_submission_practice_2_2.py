"""Submission consistency and presentation-safety contracts."""

import json
from pathlib import Path


PRACTICE_ROOT = Path(__file__).resolve().parents[2] / "practice_2_2"
NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "practice_2_2.ipynb"
CANONICAL = (
    PRACTICE_ROOT
    / "outputs"
    / "practice_2_2"
    / "canonical_26dc4625_52aaf974_s42_v1"
)


def test_official_notebook_is_report_only_and_has_required_flow():
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["metadata"]["practice_2_2"]["mode"] == "canonical_report_only"
    assert notebook["metadata"]["practice_2_2"]["training_allowed"] is False
    assert notebook["metadata"]["practice_2_2"]["final_test_evaluation_allowed"] is False
    code_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    forbidden = [
        "train_model(",
        "create_test_dataset(",
        "allow_test=True",
        "optimizer.step(",
        "scheduler.step(",
        "run_final_test(",
    ]
    assert not any(token in code_source for token in forbidden)
    markdown_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    for section in range(1, 13):
        assert f"## Phase {section} -" in markdown_source


def test_readme_notebook_and_final_artifacts_are_consistent():
    readme = (PRACTICE_ROOT / "README.md").read_text()
    notebook = NOTEBOOK.read_text()
    summary = json.loads((CANONICAL / "final_test" / "final_test_summary.json").read_text())
    selection = json.loads((CANONICAL / "final_test" / "final_selection.json").read_text())
    guard = json.loads((CANONICAL / "final_test" / "FINAL_TEST_COMPLETED.json").read_text())
    assert selection["selected_experiment_id"] == "E2_partial_finetune"
    assert abs(summary["test_accuracy"] - 76.36363636363636) < 1e-12
    assert abs(summary["macro_f1"] - 0.7617081319378308) < 1e-12
    assert summary["total_predictions"] == 440
    assert guard["FINAL_TEST_COMPLETED"] is True
    assert guard["final_test_evaluation_count"] == 1
    for text in (readme, notebook):
        assert "76.36" in text
        assert "0.7617" in text
        assert "canonical_26dc4625_52aaf974_s42_v1" in text


def test_legacy_outputs_are_explicitly_noncanonical():
    marker = (PRACTICE_ROOT / "LEGACY_ARTIFACTS.md").read_text()
    assert "LEGACY / NON-CANONICAL / DO NOT REPORT" in marker
    assert "78.57%" in marker
    assert "81.80%" in marker
