import json
from pathlib import Path

from processing_own_phase.visualization_data import RUN_IDS, build_visualization_data


def test_saved_artifact_aggregation_is_training_free(tmp_path: Path) -> None:
    root = tmp_path / "docs/result/practice_3_v2_3"
    root.mkdir(parents=True)
    (root / "experiment_registry.json").write_text(json.dumps({
        "protocol_version": "practice_3_v2.3", "status": "COMPLETED", "holdout_status": "SEALED",
        "experiments": [{"experiment_id": item, "status": "PLANNED"} for item in RUN_IDS],
    }))
    run = root / "experiments" / RUN_IDS[0]
    run.mkdir(parents=True)
    (run / "experiment_config.json").write_text(json.dumps({"learning_rate": 1e-5, "controlled_variable": "learning_rate"}))
    (run / "run_summary.json").write_text(json.dumps({"status": "COMPLETED", "best_epoch": 1, "best_val_loss": 0.4, "best_val_f1": 0.8}))
    (run / "training_history.json").write_text(json.dumps({"records": [{"epoch": 1, "train_loss": .5, "eval_loss": .4, "eval_accuracy": .8}]}))
    destination = root / "visualization_data.json"
    data = build_visualization_data(tmp_path, destination)
    assert len(data["experiments"]) == 6
    assert data["validation_leader"] == RUN_IDS[0]
    assert data["holdout_status"] == "SEALED"
    assert data["training_history"][RUN_IDS[0]][0]["validation_loss"] == .4
    assert data["experiments"][0]["learning_rate"] == 1e-5
    assert destination.is_file()
    assert not (tmp_path / "outputs").exists()
    assert not (tmp_path / "reports").exists()


def test_missing_post_training_sections_are_explicit(tmp_path: Path) -> None:
    data = build_visualization_data(tmp_path)
    for section in ("final_holdout", "error_analysis", "custom_inference", "save_reload_verification"):
        assert data[section]["available"] is False
        assert data[section]["message"] == "Artifact not generated yet"
