"""Tests for the experiment module."""

import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from processing_own_phase.experiment import (
    CONTROLLED_EXPERIMENT_IDS,
    CONTROLLED_FIELDS,
    _append_to_csv,
    run_controlled_experiments,
    run_experiment,
    select_experiment_by_validation,
    validate_controlled_experiment_configs,
)
from configs import CONFIG, EXPERIMENTS


def test_append_to_csv():
    """Test the CSV appending logic to ensure no duplicates and correct columns."""
    result1 = {"exp_id": "exp1", "run_id": "run1", "accuracy": 0.9}
    result2 = {"exp_id": "exp2", "run_id": "run2", "accuracy": 0.8}
    result1_updated = {"exp_id": "exp1", "run_id": "run3", "accuracy": 0.95}

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_results.csv")
        
        # Write first
        _append_to_csv(result1, csv_path)
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert df["exp_id"].iloc[0] == "exp1"
        
        # Write second
        _append_to_csv(result2, csv_path)
        df = pd.read_csv(csv_path)
        assert len(df) == 2
        
        # Write updated first (should overwrite exp1)
        _append_to_csv(result1_updated, csv_path)
        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert df[df["exp_id"] == "exp1"]["run_id"].iloc[0] == "run3"
        assert df[df["exp_id"] == "exp1"]["accuracy"].iloc[0] == 0.95


def test_e1_e2_are_controlled_experiments():
    validate_controlled_experiment_configs()
    e1 = EXPERIMENTS[CONTROLLED_EXPERIMENT_IDS[0]]
    e2 = EXPERIMENTS[CONTROLLED_EXPERIMENT_IDS[1]]

    assert all(e1[field] == e2[field] for field in CONTROLLED_FIELDS)
    assert e1["training_mode"] == "head_only"
    assert e2["training_mode"] == "partial_finetune"
    assert e1["epochs"] >= 5
    assert e1["early_stopping_patience"] == 4
    assert e1["best_model_metric"] == "val_loss"
    assert e1["early_stopping_metric"] == "val_loss"
    assert e1["optimizer"] == "AdamW"
    assert e1["dropout"] == 0.20
    assert e1["label_smoothing"] == 0.05
    differing_fields = {
        field
        for field in set(e1).union(e2)
        if e1.get(field) != e2.get(field)
    }
    assert differing_fields == {"training_mode"}


def test_selection_uses_validation_accuracy_and_rejects_test_metrics():
    results = [
        {"metadata": {"exp_id": "E1", "accuracy": 0.81}},
        {"metadata": {"exp_id": "E2", "accuracy": 0.87}},
    ]
    selected = select_experiment_by_validation(results)
    assert selected["metadata"]["exp_id"] == "E2"

    results[0]["metadata"]["test_accuracy"] = 0.99
    with pytest.raises(ValueError, match="Test metrics are forbidden"):
        select_experiment_by_validation(results)


@patch("processing_own_phase.experiment.train_model")
@patch("processing_own_phase.experiment.eval_full")
@patch("processing_own_phase.experiment.sanity_check_model")
@patch("processing_own_phase.experiment.build_model")
@patch("processing_own_phase.experiment.load_model_from_checkpoint")
@patch("processing_own_phase.experiment._make_selection_dataloaders")
@patch("processing_own_phase.experiment.load_datasets")
def test_run_experiment_outputs(
    mock_load_ds,
    mock_make_selection_dl,
    mock_load_checkpoint,
    mock_build,
    mock_sanity,
    mock_eval,
    mock_train,
):
    """Test that run_experiment correctly builds summary and saves files."""
    
    # Mocks
    mock_load_ds.return_value = (["train"], ["val"], ["test"])
    mock_make_selection_dl.return_value = (MagicMock(), MagicMock())
    mock_build.return_value = MagicMock()
    mock_load_checkpoint.return_value = mock_build.return_value
    mock_sanity.return_value = {"forward_shape": (4, 10)}

    def fake_train_model(*args, **kwargs):
        output_dir = kwargs["config"]["output_dir"]
        Path(output_dir, "best.pt").touch()
        return {
            "epoch": [1],
            "train_loss": [0.6],
            "train_acc": [75.0],
            "val_loss": [0.5],
            "val_acc": [80.0],
            "lr": [0.001],
            "epoch_time": [1.0],
            "best_epoch": 1,
            "best_metric": 80.0,
            "best_metric_name": "val_acc",
            "stopped_early": False,
            "epochs_planned": 1,
        }

    mock_train.side_effect = fake_train_model
    
    mock_eval.return_value = {
        "loss": 0.4,
        "accuracy": 0.85,
        "predictions": [0, 1, 0],
        "labels": [0, 1, 1],
        "probabilities": [[0.8, 0.2], [0.1, 0.9], [0.6, 0.4]],
        "inference_time": 0.1,
        "inference_fps": 30.0
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Patch config directories
        with patch.dict(CONFIG, {"runs_dir": tmpdir, "output_dir": tmpdir}):
            with patch("processing_own_phase.experiment.EXPERIMENT_RESULTS_CSV", os.path.join(tmpdir, "results.csv")):
                
                summary = run_experiment("E1_resnet18_head", use_quick_run=True)
                
                # Check summary contents
                assert "metadata" in summary
                assert "metrics" in summary
                
                m = summary["metadata"]
                assert m["exp_id"] == "E1_resnet18_head"
                assert m["accuracy"] == 0.85
                assert m["best_val_acc"] == 0.8
                assert Path(m["checkpoint_path"]).is_file()
                assert "f1_score" in m
                assert m["training_time"] >= 0
                assert m["best_epoch"] == 1
                assert m["selection_source"] == "validation_only"
                assert m["test_data_used"] is False
                assert not any(key.startswith("test") for key in m if key != "test_data_used")
                
                # Check files created
                run_dir = os.path.join(tmpdir, m["run_id"])
                assert os.path.exists(run_dir)
                
                json_path = os.path.join(run_dir, f"{m['run_id']}_summary.json")
                assert os.path.exists(json_path)
                with open(json_path, "r") as f:
                    data = json.load(f)
                    assert data["metadata"]["run_id"] == m["run_id"]
                    
                csv_path = os.path.join(tmpdir, "results.csv")
                assert os.path.exists(csv_path)
                df = pd.read_csv(csv_path)
                assert len(df) == 1
                assert df["exp_id"].iloc[0] == "E1_resnet18_head"


@patch("processing_own_phase.experiment.run_experiment")
def test_controlled_artifact_has_multi_epoch_history_and_no_test_metrics(
    mock_run_experiment,
    tmp_path,
):
    def fake_result(experiment_id, use_quick_run=False):
        accuracy = 0.82 if experiment_id.startswith("E1") else 0.88
        strategy = "head_only" if experiment_id.startswith("E1") else "partial_finetune"
        return {
            "metadata": {
                "exp_id": experiment_id,
                "freeze_strategy": strategy,
                "epochs_trained": 5,
                "best_epoch": 4,
                "accuracy": accuracy,
                "best_val_loss": 0.4,
                "f1_score": accuracy - 0.01,
                "training_time": 10.0,
                "checkpoint_path": str(tmp_path / f"{experiment_id}.pt"),
                "early_stopping_triggered": False,
                "test_data_used": False,
            },
            "history": {
                "epoch": [1, 2, 3, 4, 5],
                "train_loss": [1.0, 0.8, 0.6, 0.5, 0.4],
                "train_acc": [60, 70, 78, 82, 84],
                "val_loss": [0.9, 0.7, 0.5, 0.4, 0.42],
                "val_acc": [62, 72, 80, 88, 87],
                "lr": [0.001] * 5,
                "epoch_time": [2.0] * 5,
                "best_epoch": 4,
                "best_metric": 88,
                "best_metric_name": "val_acc",
                "stopped_early": False,
                "epochs_planned": 5,
            },
        }

    mock_run_experiment.side_effect = fake_result
    result = run_controlled_experiments(
        output_dir=str(tmp_path / "outputs"),
        reports_dir=str(tmp_path / "reports"),
    )

    selection = result["selection"]
    assert selection["selected_experiment"] == "E2_resnet18_partial"
    assert selection["selection_metric"] == "val_accuracy"
    assert selection["selection_source"] == "validation_only"
    assert selection["test_data_used"] is False
    assert selection["epochs_trained"] > 1
    assert 1 <= selection["best_epoch"] <= selection["epochs_trained"]

    selection_text = Path(result["selection_path"]).read_text().lower()
    assert "test_accuracy" not in selection_text
    assert "test_loss" not in selection_text

    comparison = pd.read_csv(result["comparison_path"])
    assert "test_accuracy" not in comparison.columns
    assert "test_loss" not in comparison.columns

    histories = json.loads(Path(result["history_path"]).read_text())
    assert set(histories) == {"E1_resnet18_head", "E2_resnet18_partial"}
    assert all(len(history["epoch"]) > 1 for history in histories.values())
