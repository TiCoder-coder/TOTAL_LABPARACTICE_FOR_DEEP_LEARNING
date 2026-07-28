"""Tests for the experiment module."""

import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd

from processing_own_phase.experiment import run_experiment, _append_to_csv
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


@patch("processing_own_phase.experiment.train_model")
@patch("processing_own_phase.experiment.eval_full")
@patch("processing_own_phase.experiment.sanity_check_model")
@patch("processing_own_phase.experiment.build_model")
@patch("processing_own_phase.experiment.make_dataloaders")
@patch("processing_own_phase.experiment.load_datasets")
def test_run_experiment_outputs(mock_load_ds, mock_make_dl, mock_build, mock_sanity, mock_eval, mock_train):
    """Test that run_experiment correctly builds summary and saves files."""
    
    # Mocks
    mock_load_ds.return_value = (["train"], ["val"], ["test"])
    mock_make_dl.return_value = (MagicMock(), MagicMock(), MagicMock())
    mock_build.return_value = MagicMock()
    mock_sanity.return_value = {"forward_shape": (4, 10)}
    mock_train.return_value = {"val_loss": [0.5], "val_acc": [80.0]}
    
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
                assert "f1_score" in m
                assert m["training_time"] >= 0
                
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
