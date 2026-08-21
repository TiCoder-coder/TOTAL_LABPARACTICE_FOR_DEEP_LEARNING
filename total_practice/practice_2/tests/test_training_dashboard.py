"""Tests for the artifact-only Practice 2 HTML dashboard."""

import csv
import json
from pathlib import Path

import pytest

from processing_own_phase.training_dashboard import (
    build_training_dashboard_html,
    load_training_dashboard_data,
    render_training_dashboard,
)
from processing_own_phase.visualization_data import build_visualization_data, read_metrics_jsonl


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture_project(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "practice_2"
    run_id = "E2_resnet18_partial_fixture"
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "best_val_loss.pt"
    checkpoint.touch()
    for name in ("latest.pt", "best.pt", "best_val_accuracy.pt"):
        (run_dir / name).touch()
    _write_json(root / "outputs" / "hyperparameter_selection_locked.json", {
        "selected_experiment": run_id,
        "selected_checkpoint": str(checkpoint),
        "best_epoch": 2,
        "best_val_loss": 0.4,
        "best_val_accuracy": 0.91,
        "selection_source": "validation_only",
    })
    metrics = [
        {"epoch": 1, "train_loss": 0.8, "val_loss": 0.6, "train_accuracy": 75.0, "val_accuracy": 82.0, "generalization_gap": -7.0, "learning_rates": {"head": 0.001, "backbone": 0.0001}, "epoch_seconds": 10},
        {"epoch": 2, "train_loss": 0.5, "val_loss": 0.4, "train_accuracy": 90.0, "val_accuracy": 91.0, "generalization_gap": -1.0, "learning_rates": {"head": 0.0005, "backbone": 0.00005}, "epoch_seconds": 11},
    ]
    (run_dir / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in metrics) + "\n", encoding="utf-8")
    _write_json(run_dir / f"{run_id}_summary.json", {
        "metadata": {"run_id": run_id, "exp_id": "E2_resnet18_partial", "model_name": "resnet18", "freeze_strategy": "partial_finetune", "epochs_trained": 2, "training_time": 21},
        "config": {"model_name": "resnet18", "training_mode": "partial_finetune"},
    })
    _write_json(run_dir / "checkpoint_manifest.json", {"checkpoints": {"latest.pt": {"epoch": 2, "bytes": 1024, "sha256": "abc123", "val_loss": 0.4, "val_acc": 91.0}}})
    (run_dir / f"{run_id}.log").write_text("2026-01-01 | INFO | Using device: mps\n", encoding="utf-8")
    _write_csv(root / "outputs" / "hyperparameter_ranking.csv", [{"rank": "1", "stage": "learning_rate", "candidate": "lr", "run_id": run_id, "head_learning_rate": "0.001", "backbone_learning_rate": "0.0001", "hidden_layers": "[]", "best_epoch": "2", "best_val_loss": "0.4", "best_val_accuracy": "0.91"}])
    _write_json(root / "outputs" / "summary.json", {"test_accuracy": 0.9, "test_loss": 0.3, "macro_precision": 0.9, "macro_recall": 0.89, "macro_f1": 0.895, "device": "mps"})
    _write_csv(root / "outputs" / "confusion_matrix.csv", [{"true_label": "cat", "cat": "8", "dog": "2"}, {"true_label": "dog", "cat": "1", "dog": "9"}])
    _write_csv(root / "outputs" / "classification_report.csv", [{"class": "cat", "precision": "0.8", "recall": "0.8", "f1-score": "0.8", "support": "10"}])
    _write_csv(root / "outputs" / "predictions.csv", [{"true_label": "cat", "predicted_label": "dog", "confidence": "0.95", "is_correct": "False"}])
    return root, run_dir


def test_metrics_jsonl_parser_ignores_only_partial_final_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"epoch": 1}\n{"epoch": 2', encoding="utf-8")
    warnings = []
    assert read_metrics_jsonl(path, warnings) == [{"epoch": 1}]
    assert warnings
    path.write_text('{"epoch": 1}\nBROKEN\n{"epoch": 3}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="row 2"):
        read_metrics_jsonl(path)


def test_selected_run_detection_uses_locked_checkpoint(tmp_path):
    root, run_dir = _fixture_project(tmp_path)
    path = root / "outputs" / "visualization_data.json"
    build_visualization_data(root, path)
    data = load_training_dashboard_data(path)
    assert data["selected_run"]["run_id"] == run_dir.name
    assert data["selected_run"]["device"] == "mps"
    assert [row["epoch"] for row in data["training_history"]] == [1, 2]


def test_missing_files_render_without_training_or_evaluation(tmp_path):
    root = tmp_path / "practice_2"
    root.mkdir()
    data = build_visualization_data(root, root / "outputs" / "visualization_data.json")
    dashboard = build_training_dashboard_html(data)
    assert "Practice 2 — Training Dashboard" in dashboard
    assert "No saved data available" in dashboard


def test_ranking_selection_and_all_dashboard_sections_render(tmp_path):
    root, run_dir = _fixture_project(tmp_path)
    path = root / "outputs" / "visualization_data.json"
    build_visualization_data(root, path)
    dashboard = build_training_dashboard_html(load_training_dashboard_data(path))
    assert run_dir.name in dashboard
    assert "p2d-selected" in dashboard
    for heading in (
        "Training curves",
        "Hyperparameter search",
        "Checkpoint status",
        "Saved Final Evaluation",
        "Error analysis",
    ):
        assert heading in dashboard
    assert "Saved Final Test artifacts only" in dashboard
    assert "regenerate_final_artifacts" not in dashboard


def test_render_dashboard_saves_only_requested_report_path(tmp_path):
    root, _ = _fixture_project(tmp_path)
    data_path = root / "outputs" / "visualization_data.json"
    build_visualization_data(root, data_path)
    destination = tmp_path / "report-target" / "dashboard.html"
    dashboard = render_training_dashboard(data_path, save_path=destination)
    assert dashboard.saved_path == destination.resolve()
    assert destination.is_file()
    assert destination.read_text(encoding="utf-8") == dashboard.html
    assert "<script" not in dashboard.html


def test_aggregate_schema_and_sources_are_compact(tmp_path):
    root, run_dir = _fixture_project(tmp_path)
    destination = root / "outputs" / "visualization_data.json"
    payload = build_visualization_data(root, destination)
    assert destination.is_file()
    assert set(payload) >= {"generated_at", "selected_run", "training_history", "hyperparameter_search", "checkpoints", "final_evaluation", "error_analysis", "sources"}
    assert payload["sources"]["metrics_jsonl"] == f"runs/{run_dir.name}/metrics.jsonl"
    assert payload["training_history"][0]["head_lr"] == 0.001
    assert payload["checkpoints"]["latest"]["exists"] is True
    assert payload["final_evaluation"]["test_accuracy"] == 0.9


def test_renderer_requires_single_valid_aggregate_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_training_dashboard_data(tmp_path / "missing.json")
    malformed = tmp_path / "bad.json"
    malformed.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing sections"):
        load_training_dashboard_data(malformed)
