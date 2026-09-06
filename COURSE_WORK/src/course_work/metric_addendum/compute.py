from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from course_work.evaluation.metrics import compute_mape_pct
from course_work.metric_addendum.sources import (
    discover_validation_sources,
    load_metric_evidence,
    load_validated_prediction_frame,
)
from course_work.utils.artifacts import sha256_file


def compute_validation_mape_records(project_root: Path) -> list[dict[str, Any]]:
    root = Path(project_root).resolve()
    records: list[dict[str, Any]] = []
    for source in discover_validation_sources(root):
        prediction_path = str(source["prediction_path"])
        metrics_path = str(source["metrics_path"])
        frame = load_validated_prediction_frame(root, prediction_path)
        result = compute_mape_pct(frame["y_true_wh"], frame["y_pred_wh"])
        evidence = load_metric_evidence(root, metrics_path)
        run_id = str(frame["run_id"].iloc[0])
        model_id = str(evidence.get("model_id", source["model_family"]))
        record = asdict(result)
        record.update(
            {
                "run_id": run_id,
                "model_id": model_id,
                "model_family": source["model_family"],
                "split_id": str(evidence.get("split_id", "VALIDATION")),
                "population_fingerprint": str(evidence.get("population_fingerprint", "")),
                "target_min_wh": float(np.min(frame["y_true_wh"].to_numpy(dtype=np.float64))),
                "target_max_wh": float(np.max(frame["y_true_wh"].to_numpy(dtype=np.float64))),
                "prediction_path": prediction_path,
                "prediction_sha256": sha256_file(root / prediction_path),
                "metrics_path": metrics_path,
                "metrics_sha256": sha256_file(root / metrics_path) if (root / metrics_path).exists() else "",
            }
        )
        records.append(record)
    return sorted(records, key=lambda item: (str(item["model_family"]), str(item["run_id"])))


def summarize_validation_mape(records: list[dict[str, Any]]) -> dict[str, Any]:
    defined = [record for record in records if record["mape_status"] == "DEFINED"]
    transformer_count = sum(record["model_family"] == "TRANSFORMER" for record in defined)
    return {
        "validation_sources_discovered": len(records),
        "validation_mape_defined": len(defined),
        "validation_mape_undefined": len(records) - len(defined),
        "model_families": sorted({str(record["model_family"]) for record in records}),
        "transformer_run_count": transformer_count,
        "cross_configuration_mape_aggregation": "PROHIBITED",
        "seed_mean_sd_status": "BLOCKED_SOURCE_UNAVAILABLE",
        "selection_metric": "rmse_wh",
        "mape_role": "SUPPLEMENTARY_REPORTING_ONLY",
    }
