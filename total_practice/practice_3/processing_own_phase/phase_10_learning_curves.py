"""Phase 10: artifact-only learning curves and training analysis."""

import json
import math
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt

from .config import RESULT_DIR


PHASE_09_DIR = RESULT_DIR / "phase_09_training"
LOSS_FIGURE_PATH = RESULT_DIR / "phase_10_loss_curves.png"
METRICS_FIGURE_PATH = RESULT_DIR / "phase_10_validation_metrics.png"
ANALYSIS_PATH = RESULT_DIR / "phase_10_learning_curve_analysis.json"
SOURCE_PATHS = {
    "training_history": PHASE_09_DIR / "training_history.json",
    "validation_metrics": PHASE_09_DIR / "validation_metrics_by_epoch.json",
    "checkpoint_records": PHASE_09_DIR / "checkpoint_records.json",
    "selected_checkpoint": PHASE_09_DIR / "selected_checkpoint.json",
    "manifest": PHASE_09_DIR / "phase_09_training_manifest.json",
}
REFERENCE = {
    1: {"eval_loss": 0.4152307808, "eval_f1": 0.8379513014},
    2: {"eval_loss": 0.3904653192, "eval_f1": 0.8579387187},
    3: {"eval_loss": 0.4926925004, "eval_f1": 0.8576675849},
}


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required Phase 9 artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_phase_09_artifacts() -> Dict[str, Any]:
    """Load only Phase 9 JSON artifacts; no model or dataset is touched."""
    return {name: _load_json(path) for name, path in SOURCE_PATHS.items()}


def build_epoch_records(artifacts: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Join one Train-loss and one Validation record by exact epoch/step."""
    history = artifacts["training_history"]
    validation = artifacts["validation_metrics"]
    train_records = [record for record in history if "loss" in record]
    if len(train_records) != 3 or len(validation) != 3:
        raise ValueError("Expected exactly three Train-loss and Validation records")
    train_by_key = {(float(r["epoch"]), int(r["step"])): r for r in train_records}
    joined = []
    for record in validation:
        key = (float(record["epoch"]), int(record["step"]))
        if key not in train_by_key:
            raise ValueError(f"No Train-loss record matches Validation record {key}")
        joined.append({
            "epoch": key[0],
            "step": key[1],
            "train_loss": float(train_by_key[key]["loss"]),
            "validation_loss": float(record["eval_loss"]),
            "validation_accuracy": float(record["eval_accuracy"]),
            "validation_f1": float(record["eval_f1"]),
        })
    return sorted(joined, key=lambda row: (row["epoch"], row["step"]))


def _plot_loss(records: list[Dict[str, Any]], selected: Dict[str, Any]) -> None:
    epochs = [row["epoch"] for row in records]
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(epochs, [row["train_loss"] for row in records], marker="o", linewidth=2, label="Train loss")
    axis.plot(epochs, [row["validation_loss"] for row in records], marker="o", linewidth=2, label="Validation loss")
    selected_epoch = float(selected["epoch"])
    axis.axvline(selected_epoch, color="tab:green", linestyle="--", alpha=0.75)
    axis.scatter([selected_epoch], [float(selected["eval_loss"])], color="tab:green", s=90, zorder=5)
    axis.annotate(
        f"Selected: epoch {selected_epoch:g} / {Path(selected['checkpoint']).name}",
        (selected_epoch, float(selected["eval_loss"])), xytext=(8, 15),
        textcoords="offset points", color="tab:green",
    )
    axis.set(title="Training and Validation Loss", xlabel="Epoch", ylabel="Loss", xticks=[1, 2, 3])
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(LOSS_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_validation_metrics(records: list[Dict[str, Any]], selected: Dict[str, Any]) -> None:
    epochs = [row["epoch"] for row in records]
    figure, axis = plt.subplots(figsize=(9, 5.5))
    series = (
        ("Validation Accuracy", "validation_accuracy", "tab:blue"),
        ("Validation F1", "validation_f1", "tab:orange"),
    )
    for label, key, color in series:
        values = [row[key] for row in records]
        axis.plot(epochs, values, marker="o", linewidth=2, label=label, color=color)
        for epoch, value in zip(epochs, values):
            axis.annotate(f"{value:.4f}", (epoch, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    selected_epoch = float(selected["epoch"])
    selected_row = next(row for row in records if row["epoch"] == selected_epoch)
    axis.axvline(selected_epoch, color="tab:green", linestyle="--", alpha=0.75)
    axis.scatter([selected_epoch], [selected_row["validation_f1"]], color="tab:green", s=90, zorder=5)
    axis.annotate(
        f"Selected: {Path(selected['checkpoint']).name}",
        (selected_epoch, selected_row["validation_f1"]), xytext=(10, -25),
        textcoords="offset points", color="tab:green",
    )
    axis.set(title="Validation Accuracy and F1", xlabel="Epoch", ylabel="Score", xticks=[1, 2, 3], ylim=(0, 1))
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(METRICS_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _figure_check(path: Path) -> Dict[str, Any]:
    image = plt.imread(path)
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "non_empty": path.stat().st_size > 0,
        "decodable": image.ndim in (2, 3),
        "pixel_shape": list(image.shape),
    }


def analyze_learning_curves() -> Dict[str, Any]:
    """Validate Phase 9 evidence, create figures, and save Phase 10 analysis."""
    artifacts = load_phase_09_artifacts()
    manifest = artifacts["manifest"]
    selected = artifacts["selected_checkpoint"]
    records = build_epoch_records(artifacts)
    epochs = [row["epoch"] for row in records]
    steps = [row["step"] for row in records]
    finite = all(
        math.isfinite(row[key])
        for row in records
        for key in ("train_loss", "validation_loss", "validation_accuracy", "validation_f1")
    )
    scores_in_range = all(
        0.0 <= row[key] <= 1.0
        for row in records for key in ("validation_accuracy", "validation_f1")
    )
    references_match = all(
        math.isclose(row["validation_loss"], REFERENCE[int(row["epoch"])]["eval_loss"], abs_tol=1e-9)
        and math.isclose(row["validation_f1"], REFERENCE[int(row["epoch"])]["eval_f1"], abs_tol=1e-9)
        for row in records
    )
    selected_row = next(row for row in records if row["epoch"] == float(selected["epoch"]))
    selected_matches = (
        float(selected["epoch"]) == 2.0
        and int(selected["step"]) == 1068
        and Path(selected["checkpoint"]).name == "checkpoint-1068"
        and math.isclose(float(selected["eval_loss"]), selected_row["validation_loss"])
        and math.isclose(float(selected["eval_f1"]), selected_row["validation_f1"])
    )
    deltas = {}
    for left, right in ((0, 1), (1, 2)):
        label = f"epoch_{int(records[left]['epoch'])}_to_{int(records[right]['epoch'])}"
        deltas[label] = {
            key: records[right][key] - records[left][key]
            for key in ("train_loss", "validation_loss", "validation_accuracy", "validation_f1")
        }
    conditions = {
        "epoch_1_to_2_train_loss_decreased": deltas["epoch_1_to_2"]["train_loss"] < 0,
        "epoch_1_to_2_validation_loss_decreased": deltas["epoch_1_to_2"]["validation_loss"] < 0,
        "epoch_1_to_2_accuracy_improved": deltas["epoch_1_to_2"]["validation_accuracy"] > 0,
        "epoch_1_to_2_f1_improved": deltas["epoch_1_to_2"]["validation_f1"] > 0,
        "epoch_2_to_3_train_loss_decreased": deltas["epoch_2_to_3"]["train_loss"] < 0,
        "epoch_2_to_3_validation_loss_increased": deltas["epoch_2_to_3"]["validation_loss"] > 0,
        "epoch_2_to_3_accuracy_not_improved": deltas["epoch_2_to_3"]["validation_accuracy"] <= 0,
        "epoch_2_to_3_f1_not_improved": deltas["epoch_2_to_3"]["validation_f1"] <= 0,
    }
    divergence = conditions["epoch_2_to_3_train_loss_decreased"] and conditions["epoch_2_to_3_validation_loss_increased"]
    interpretation = (
        "Epoch 1 to 2 improves both fit and Validation behavior. From epoch 2 to 3, "
        "Train loss continues to decrease while Validation loss increases and Validation "
        "Accuracy/F1 do not improve. This indicates generalization begins to worsen after "
        "epoch 2 and is consistent with the onset of overfitting. With only three epochs "
        "and one run, this is limited evidence rather than a definitive broad conclusion."
        if divergence else
        "The artifact trajectory does not satisfy the planned Train/Validation divergence condition; no overfitting interpretation is asserted."
    )

    _plot_loss(records, selected)
    _plot_validation_metrics(records, selected)
    figures = {
        "loss_curves": _figure_check(LOSS_FIGURE_PATH),
        "validation_metrics": _figure_check(METRICS_FIGURE_PATH),
    }
    checks = {
        "source_readback_pass": True,
        "phase_09_manifest_pass": manifest.get("status") == "PASS",
        "phase_09_test_not_accessed": manifest.get("test_accessed") is False,
        "epochs_exact_and_ordered": epochs == [1.0, 2.0, 3.0],
        "steps_exact_and_ordered": steps == [534, 1068, 1602],
        "checkpoint_record_count_three": len(artifacts["checkpoint_records"]) == 3,
        "finite_metrics": finite,
        "scores_in_range": scores_in_range,
        "reference_values_match": references_match,
        "selected_checkpoint_matches": selected_matches,
        "planned_trajectory_conditions_pass": all(conditions.values()),
        "figures_verified": all(all(item[key] for key in ("exists", "non_empty", "decodable")) for item in figures.values()),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "source_artifacts": {name: str(path.resolve()) for name, path in SOURCE_PATHS.items()},
        "epoch_records": records,
        "selected_checkpoint": {
            "epoch": selected["epoch"], "step": selected["step"],
            "checkpoint": selected["checkpoint"], "eval_loss": selected["eval_loss"],
            "eval_f1": selected["eval_f1"],
        },
        "validation_checks": checks,
        "epoch_deltas": deltas,
        "generalization_conditions": conditions,
        "generalization_interpretation": interpretation,
        "figures": figures,
        "training_performed": False,
        "model_loaded": False,
        "model_evaluated": False,
        "dataset_loaded": False,
        "test_accessed": False,
        "checkpoint_changed": False,
        "phase_11_started": False,
    }
    ANALYSIS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if json.loads(ANALYSIS_PATH.read_text(encoding="utf-8")) != result:
        raise RuntimeError("Phase 10 analysis JSON read-back failed")
    if result["status"] != "PASS":
        raise RuntimeError(f"Phase 10 verification failed: {checks}")
    return result
