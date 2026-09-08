import json
from pathlib import Path

import pandas as pd

from course_work.metric_addendum.materialize import materialize_mape_addendum


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_predictions(path: Path, run_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "run_id": [run_id, run_id, run_id],
            "sample_idx": [0, 1, 2],
            "y_true_wh": [100.0, 200.0, 400.0],
            "y_pred_wh": [90.0, 220.0, 360.0],
            "residual_wh": [10.0, -20.0, 40.0],
        }
    ).to_csv(path, index=False)


def test_materializer_uses_existing_validation_predictions_and_blocks_missing_test(tmp_path: Path) -> None:
    persistence_prediction = tmp_path / "artifacts/baselines/persistence/persistence_validation_predictions.csv"
    persistence_metrics = tmp_path / "artifacts/baselines/persistence/persistence_validation_metrics.json"
    transformer_prediction = tmp_path / "artifacts/runs/RUN_TR_TEST/predictions/best_validation_predictions.csv"
    transformer_metrics = tmp_path / "artifacts/runs/RUN_TR_TEST/metrics/best_validation_metrics.json"
    _write_predictions(persistence_prediction, "RUN_PS_TEST")
    _write_predictions(transformer_prediction, "RUN_TR_TEST")
    _write_json(
        persistence_metrics,
        {"metric_result": {"model_id": "PERSISTENCE_LAST_VALUE", "split_id": "VALIDATION", "population_fingerprint": "POP"}},
    )
    _write_json(
        transformer_metrics,
        {"metric_result": {"model_id": "Transformer Encoder Regressor", "split_id": "VALIDATION", "population_fingerprint": "POP"}},
    )
    _write_json(
        tmp_path / "artifacts/final_test/prediction_checksums.json",
        {
            "predictions": {
                "seed_42": {
                    "path": "artifacts/final_test/predictions/final_test_predictions_seed42.csv",
                    "sha256": "abc",
                    "rows": 3,
                }
            }
        },
    )
    first = materialize_mape_addendum(tmp_path)
    second = materialize_mape_addendum(tmp_path)
    results = pd.read_csv(tmp_path / "artifacts/metric_addendum/mape/validation_mape_by_run.csv")
    test_status = json.loads(
        (tmp_path / "artifacts/metric_addendum/mape/final_test_mape_status.json").read_text(encoding="utf-8")
    )
    validation_summary = json.loads(
        (tmp_path / "artifacts/metric_addendum/mape/validation_mape_summary.json").read_text(encoding="utf-8")
    )
    assert first == second
    assert len(results) == 2
    assert set(results["mape_status"]) == {"DEFINED"}
    assert set(results["mape_pct"].round(12)) == {10.0}
    assert test_status["status"] == "BLOCKED_SOURCE_UNAVAILABLE"
    assert test_status["test_inference_executed"] is False
    assert test_status["test_mape_computed"] is False
    assert validation_summary["cross_configuration_mape_aggregation"] == "PROHIBITED"
    assert validation_summary["seed_mean_sd_status"] == "BLOCKED_SOURCE_UNAVAILABLE"


def test_materializer_does_not_import_training_or_inference_modules() -> None:
    source_root = Path(__file__).resolve().parents[2] / "src/course_work/metric_addendum"
    content = "\n".join(path.read_text(encoding="utf-8") for path in sorted(source_root.glob("*.py")))
    forbidden = (
        "course_work.training",
        "course_work.phase47.evaluation",
        "torch.load",
        "model.forward",
        "optimizer.step",
        "scaler.fit",
    )
    assert all(token not in content for token in forbidden)
