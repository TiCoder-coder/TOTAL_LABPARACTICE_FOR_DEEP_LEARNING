from pathlib import Path
from typing import Any

import pandas as pd

from course_work.utils.artifacts import read_json, sha256_file


REQUIRED_PREDICTION_COLUMNS = ("run_id", "sample_idx", "y_true_wh", "y_pred_wh")


def _relative(project_root: Path, path: Path) -> str:
    return path.relative_to(project_root).as_posix()


def discover_validation_sources(project_root: Path) -> list[dict[str, Any]]:
    root = Path(project_root).resolve()
    sources: list[dict[str, Any]] = []
    persistence = root / "artifacts/baselines/persistence/persistence_validation_predictions.csv"
    if persistence.exists():
        sources.append(
            {
                "prediction_path": _relative(root, persistence),
                "metrics_path": "artifacts/baselines/persistence/persistence_validation_metrics.json",
                "model_family": "PERSISTENCE",
            }
        )
    for prediction_path in sorted((root / "artifacts/runs").glob("*/predictions/best_validation_predictions.csv")):
        run_id = prediction_path.parents[1].name
        model_family = "LSTM" if run_id.startswith("RUN_LS_") else "TRANSFORMER"
        metrics_path = prediction_path.parents[1] / "metrics/best_validation_metrics.json"
        sources.append(
            {
                "prediction_path": _relative(root, prediction_path),
                "metrics_path": _relative(root, metrics_path),
                "model_family": model_family,
            }
        )
    return sources


def load_validated_prediction_frame(project_root: Path, relative_path: str) -> pd.DataFrame:
    path = Path(project_root).resolve() / relative_path
    frame = pd.read_csv(path)
    missing = [name for name in REQUIRED_PREDICTION_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"Prediction artifact is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Prediction artifact is empty")
    if frame["sample_idx"].duplicated().any():
        raise ValueError("Prediction artifact contains duplicate sample_idx values")
    if frame["run_id"].nunique(dropna=False) != 1:
        raise ValueError("Prediction artifact contains multiple run_id values")
    return frame


def load_metric_evidence(project_root: Path, relative_path: str) -> dict[str, Any]:
    path = Path(project_root).resolve() / relative_path
    if not path.exists():
        return {}
    payload = read_json(path)
    return payload.get("metric_result", payload)


def inspect_final_test_sources(project_root: Path) -> list[dict[str, Any]]:
    root = Path(project_root).resolve()
    registry_path = root / "artifacts/final_test/prediction_checksums.json"
    if not registry_path.exists():
        return [
            {
                "source_id": "prediction_checksums",
                "path": "artifacts/final_test/prediction_checksums.json",
                "expected_sha256": "",
                "actual_sha256": "",
                "expected_rows": "",
                "actual_rows": "",
                "status": "MISSING_REGISTRY",
            }
        ]
    registry = read_json(registry_path)
    rows: list[dict[str, Any]] = []
    for source_id, item in sorted(registry.get("predictions", {}).items()):
        relative_path = str(item["path"])
        path = root / relative_path
        expected_sha = str(item["sha256"])
        expected_rows = int(item["rows"])
        if not path.exists():
            rows.append(
                {
                    "source_id": source_id,
                    "path": relative_path,
                    "expected_sha256": expected_sha,
                    "actual_sha256": "",
                    "expected_rows": expected_rows,
                    "actual_rows": "",
                    "status": "MISSING",
                }
            )
            continue
        actual_sha = sha256_file(path)
        actual_rows = len(pd.read_csv(path))
        status = "PASS" if actual_sha == expected_sha and actual_rows == expected_rows else "MISMATCH"
        rows.append(
            {
                "source_id": source_id,
                "path": relative_path,
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
                "expected_rows": expected_rows,
                "actual_rows": actual_rows,
                "status": status,
            }
        )
    return rows
