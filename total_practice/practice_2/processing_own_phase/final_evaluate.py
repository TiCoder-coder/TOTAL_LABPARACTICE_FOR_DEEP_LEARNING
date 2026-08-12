"""Verify the Validation-selected checkpoint and generate final Test artifacts."""

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from configs import CLASS_NAMES, CONFIG, OUTPUT_DIR, REPORTS_DIR
from .data import get_transforms
from .evaluate import (
    compute_classification_metrics,
    evaluate,
    export_predictions,
    generate_classification_report,
)
from .save_load import load_model_from_checkpoint
from .train import get_criterion
from .utils import get_device, setup_reproducibility
from .visualize import (
    _compute_confusion_matrix,
    plot_confidence_distribution,
    plot_confusion_matrix,
    plot_metrics_bar,
    plot_prediction_gallery,
)


DEFAULT_SELECTION_PATH = OUTPUT_DIR / "controlled_experiment_selection.json"
CONFUSION_MATRIX_ARTIFACT = "confusion_matrix.csv"


def _load_selection_record(selection_path: str) -> Dict[str, Any]:
    """Load and validate the Validation-only experiment-selection record."""
    path = Path(selection_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Selection artifact does not exist: {path}")

    selection = json.loads(path.read_text())
    required = {
        "selection_metric",
        "selection_source",
        "selected_experiment",
        "selected_checkpoint",
        "best_val_accuracy",
        "test_data_used",
    }
    missing = sorted(required.difference(selection))
    if missing:
        raise ValueError(
            "Selection artifact is missing required fields: " + ", ".join(missing)
        )
    if selection["selection_metric"] not in {"val_accuracy", "val_loss"}:
        raise ValueError(
            "Final evaluation requires a Validation-only accuracy/loss metric; "
            f"received {selection['selection_metric']!r}."
        )
    if selection["selection_source"] != "validation_only":
        raise ValueError(
            "Final evaluation requires selection_source='validation_only'."
        )
    if selection["test_data_used"] is not False:
        raise ValueError("Selection artifact indicates that Test data was used.")

    checkpoint = Path(selection["selected_checkpoint"]).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Validation-selected checkpoint does not exist: {checkpoint}"
        )
    selection["selected_checkpoint"] = str(checkpoint)
    selection["best_val_accuracy"] = float(selection["best_val_accuracy"])
    if selection.get("best_val_loss") is not None:
        selection["best_val_loss"] = float(selection["best_val_loss"])
    expected_sha256 = selection.get("checkpoint_sha256")
    if expected_sha256 and _checkpoint_sha256(checkpoint) != expected_sha256:
        raise RuntimeError("Selected checkpoint SHA256 does not match locked selection.")
    return selection


def _assert_validation_match(
    expected_accuracy: float,
    actual_accuracy: float,
    tolerance: float,
) -> float:
    """Raise before Test access when reloaded Validation accuracy does not match."""
    delta = abs(float(actual_accuracy) - float(expected_accuracy))
    if delta > tolerance:
        raise RuntimeError(
            "Checkpoint verification failed; Final Test was not run: "
            f"expected Validation accuracy={expected_accuracy:.8f}, "
            f"reloaded={actual_accuracy:.8f}, delta={delta:.8f}, "
            f"tolerance={tolerance:.8f}."
        )
    return delta


def _make_evaluation_loader(dataset, batch_size: int) -> DataLoader:
    """Build a deterministic loader for an already transformed dataset view."""
    num_workers = CONFIG["num_workers"]
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        # Pinned host memory accelerates CUDA transfers but is unsupported on MPS.
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _load_validation_subset():
    """Rebuild the saved Validation split without constructing official Test."""
    raw_train_dataset = datasets.CIFAR10(
        root=CONFIG["data_dir"],
        train=True,
        download=True,
        transform=None,
    )
    n_total = len(raw_train_dataset)
    n_val = round(n_total * (1.0 - CONFIG["train_split_ratio"]))
    generator = torch.Generator().manual_seed(CONFIG["seed"])
    indices = torch.randperm(n_total, generator=generator).tolist()
    val_indices = indices[n_total - n_val:]
    _, eval_transform = get_transforms()
    validation_view = datasets.CIFAR10(
        root=CONFIG["data_dir"],
        train=True,
        download=False,
        transform=eval_transform,
    )
    return Subset(validation_view, val_indices)


def _load_official_test_dataset():
    """Construct official Test only after checkpoint verification has passed."""
    _, eval_transform = get_transforms()
    return datasets.CIFAR10(
        root=CONFIG["data_dir"],
        train=False,
        download=True,
        transform=eval_transform,
    )


def _verify_checkpoint_context(
    selection_path: str,
    batch_size: int,
    validation_tolerance: float,
    device: Optional[torch.device],
) -> Dict[str, Any]:
    """Reload the selected checkpoint and verify it on Validation only."""
    selection = _load_selection_record(selection_path)
    setup_reproducibility(CONFIG["seed"])
    resolved_device = device or get_device()

    val_subset = _load_validation_subset()
    val_loader = _make_evaluation_loader(val_subset, batch_size)
    model = load_model_from_checkpoint(
        selection["selected_checkpoint"],
        device=resolved_device,
    )
    checkpoint_payload = torch.load(
        selection["selected_checkpoint"], map_location="cpu", weights_only=False
    )
    criterion = get_criterion(checkpoint_payload.get("config", {}))
    validation_result = evaluate(
        model,
        val_loader,
        criterion,
        resolved_device,
    )
    validation_delta = _assert_validation_match(
        selection["best_val_accuracy"],
        validation_result["accuracy"],
        validation_tolerance,
    )
    loss_delta = None
    if selection.get("best_val_loss") is not None:
        loss_delta = abs(validation_result["loss"] - selection["best_val_loss"])
        if loss_delta > validation_tolerance:
            raise RuntimeError(
                "Checkpoint verification failed; Final Test was not run: "
                f"Validation loss delta={loss_delta:.8f}, "
                f"tolerance={validation_tolerance:.8f}."
            )
    return {
        "selection": selection,
        "model": model,
        "device": resolved_device,
        "validation_result": validation_result,
        "validation_delta": validation_delta,
        "validation_loss_delta": loss_delta,
    }


def verify_selected_checkpoint(
    selection_path: str = str(DEFAULT_SELECTION_PATH),
    batch_size: int = 128,
    validation_tolerance: float = 1e-6,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Return a presentation-safe PASS record without accessing the Test loader."""
    context = _verify_checkpoint_context(
        selection_path=selection_path,
        batch_size=batch_size,
        validation_tolerance=validation_tolerance,
        device=device,
    )
    selection = context["selection"]
    validation_result = context["validation_result"]
    return {
        "status": "PASS",
        "selected_experiment": selection["selected_experiment"],
        "checkpoint": selection["selected_checkpoint"],
        "selection_metric": selection["selection_metric"],
        "selection_source": selection["selection_source"],
        "recorded_val_accuracy": selection["best_val_accuracy"],
        "verified_val_accuracy": validation_result["accuracy"],
        "validation_loss": validation_result["loss"],
        "validation_delta": context["validation_delta"],
        "validation_loss_delta": context["validation_loss_delta"],
        "test_accessed": False,
    }


def _top_prediction_indices(
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    top_k: int = 10,
) -> np.ndarray:
    confidences = probabilities[np.arange(len(predictions)), predictions]
    correct_mask = predictions == labels

    selected = []
    for mask in (correct_mask, ~correct_mask):
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            continue
        order = np.argsort(confidences[indices])[::-1]
        selected.extend(indices[order[:top_k]].tolist())
    return np.asarray(selected, dtype=np.int64)


def _validate_confusion_matrix(
    confusion_matrix: np.ndarray,
    test_samples: int,
    expected_accuracy: float,
    tolerance: float = 1e-12,
) -> float:
    """Check that the official confusion matrix matches Test size and accuracy."""
    matrix = np.asarray(confusion_matrix)
    if matrix.shape != (len(CLASS_NAMES), len(CLASS_NAMES)):
        raise RuntimeError(
            "Confusion matrix has invalid shape: "
            f"expected {(len(CLASS_NAMES), len(CLASS_NAMES))}, got {matrix.shape}."
        )
    matrix_samples = int(matrix.sum())
    if matrix_samples != int(test_samples):
        raise RuntimeError(
            "Confusion matrix sample count mismatch: "
            f"matrix={matrix_samples}, summary={test_samples}."
        )
    matrix_accuracy = float(np.trace(matrix) / max(matrix_samples, 1))
    if abs(matrix_accuracy - float(expected_accuracy)) > tolerance:
        raise RuntimeError(
            "Confusion matrix accuracy mismatch: "
            f"matrix={matrix_accuracy:.8f}, summary={expected_accuracy:.8f}."
        )
    return matrix_accuracy


def validate_final_artifacts(
    output_dir: str = str(OUTPUT_DIR),
    tolerance: float = 1e-12,
) -> Dict[str, Any]:
    """Validate summary, predictions and machine-readable confusion matrix."""
    output_path = Path(output_dir)
    summary = json.loads((output_path / "summary.json").read_text())
    confusion_frame = pd.read_csv(
        output_path / CONFUSION_MATRIX_ARTIFACT,
        index_col=0,
    )
    confusion_matrix = confusion_frame.to_numpy(dtype=np.int64)
    matrix_accuracy = _validate_confusion_matrix(
        confusion_matrix,
        summary["test_samples"],
        summary["test_accuracy"],
        tolerance=tolerance,
    )

    predictions = pd.read_csv(output_path / "predictions.csv")
    if len(predictions) != int(summary["test_samples"]):
        raise RuntimeError(
            "Predictions sample count mismatch: "
            f"predictions={len(predictions)}, summary={summary['test_samples']}."
        )
    predictions_accuracy = float(predictions["is_correct"].astype(bool).mean())
    if abs(predictions_accuracy - float(summary["test_accuracy"])) > tolerance:
        raise RuntimeError(
            "Predictions accuracy does not match summary.json: "
            f"predictions={predictions_accuracy:.8f}, "
            f"summary={summary['test_accuracy']:.8f}."
        )
    probability_columns = [f"probability_{class_name}" for class_name in CLASS_NAMES]
    missing_probability_columns = sorted(
        set(probability_columns).difference(predictions.columns)
    )
    if missing_probability_columns:
        raise RuntimeError(
            "Predictions artifact is missing probability columns: "
            + ", ".join(missing_probability_columns)
        )
    probabilities = predictions[probability_columns].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Predictions artifact contains non-finite probabilities.")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise RuntimeError("Predictions artifact contains probabilities outside [0, 1].")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("Prediction probabilities do not sum to one.")
    predicted_label_ids = predictions["predicted_label_id"].to_numpy(dtype=int)
    if not np.array_equal(probabilities.argmax(axis=1), predicted_label_ids):
        raise RuntimeError("Predicted labels do not match probability argmax values.")
    expected_confidence = probabilities[
        np.arange(len(predictions)),
        predicted_label_ids,
    ]
    if not np.allclose(
        predictions["confidence"].to_numpy(dtype=float),
        expected_confidence,
        atol=1e-12,
    ):
        raise RuntimeError("Prediction confidence does not match class probability.")
    return {
        "status": "PASS",
        "test_samples": int(summary["test_samples"]),
        "confusion_matrix_accuracy": matrix_accuracy,
        "predictions_accuracy": predictions_accuracy,
    }


def _checkpoint_sha256(checkpoint: Path) -> str:
    digest = hashlib.sha256()
    with checkpoint.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regenerate_final_artifacts(
    selection_path: str = str(DEFAULT_SELECTION_PATH),
    output_dir: str = str(OUTPUT_DIR),
    reports_dir: str = str(REPORTS_DIR),
    batch_size: int = 128,
    validation_tolerance: float = 1e-6,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Verify Validation, run one Test pass, and overwrite final artifacts."""
    started_at = time.time()
    context = _verify_checkpoint_context(
        selection_path=selection_path,
        batch_size=batch_size,
        validation_tolerance=validation_tolerance,
        device=device,
    )
    selection = context["selection"]
    validation_result = context["validation_result"]
    checkpoint = Path(selection["selected_checkpoint"])
    checkpoint_hash = _checkpoint_sha256(checkpoint)
    output_path = Path(output_dir)
    receipt_path = output_path / f"final_test_receipt_{checkpoint_hash[:12]}.json"
    if receipt_path.exists():
        raise RuntimeError(
            "Final Test is already recorded for this checkpoint; refusing rerun: "
            f"{receipt_path}"
        )

    # Official Test is intentionally constructed only after Validation verification.
    test_dataset = _load_official_test_dataset()
    test_loader = _make_evaluation_loader(test_dataset, batch_size)
    test_evaluation_count = 0
    if test_evaluation_count != 0:
        raise RuntimeError("Final Test evaluation may run only once.")
    test_result = evaluate(
        context["model"],
        test_loader,
        nn.CrossEntropyLoss(),
        context["device"],
    )
    test_evaluation_count += 1
    if test_evaluation_count != 1:
        raise RuntimeError("Final Test evaluation count is not exactly one.")

    reports_path = Path(reports_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)

    metrics = compute_classification_metrics(
        test_result["predictions"],
        test_result["labels"],
        len(CLASS_NAMES),
        CLASS_NAMES,
    )
    confusion_matrix = _compute_confusion_matrix(
        test_result["predictions"],
        test_result["labels"],
        len(CLASS_NAMES),
    )
    matrix_accuracy = _validate_confusion_matrix(
        confusion_matrix,
        len(test_result["labels"]),
        test_result["accuracy"],
    )
    confusion_frame = pd.DataFrame(
        confusion_matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )
    confusion_frame.index.name = "true_label"
    confusion_frame.to_csv(output_path / CONFUSION_MATRIX_ARTIFACT)

    generate_classification_report(
        metrics,
        CLASS_NAMES,
        str(output_path / "classification_report.txt"),
        str(output_path / "classification_report.csv"),
    )
    export_predictions(
        test_result["predictions"],
        test_result["labels"],
        test_result["probabilities"],
        CLASS_NAMES,
        str(output_path / "predictions.csv"),
    )

    for destination in (reports_path, output_path):
        plot_metrics_bar(
            metrics,
            CLASS_NAMES,
            save_path=str(destination / "metrics_bar.png"),
        )
        plot_confusion_matrix(
            test_result["predictions"],
            test_result["labels"],
            CLASS_NAMES,
            save_dir=str(destination),
        )
        plot_confidence_distribution(
            test_result["probabilities"],
            test_result["labels"],
            test_result["predictions"],
            save_path=str(destination / "confidence_distribution.png"),
        )

    selected_indices = _top_prediction_indices(
        test_result["predictions"],
        test_result["labels"],
        test_result["probabilities"],
    )
    selected_images = torch.stack(
        [test_dataset[int(index)][0] for index in selected_indices]
    )
    selected_predictions = test_result["predictions"][selected_indices]
    selected_labels = test_result["labels"][selected_indices]
    selected_probabilities = test_result["probabilities"][selected_indices]
    for destination in (reports_path, output_path):
        plot_prediction_gallery(
            selected_images,
            selected_predictions,
            selected_labels,
            selected_probabilities,
            CLASS_NAMES,
            top_k=10,
            save_dir=str(destination),
        )

    shutil.copyfile(
        output_path / "prediction_gallery_correct.png",
        output_path / "predictions_grid.png",
    )
    shutil.copyfile(
        output_path / "prediction_gallery_incorrect.png",
        output_path / "top_misclassified.png",
    )

    checkpoint_data = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    summary = {
        "validation_verification": "PASS",
        "selection_metric": selection["selection_metric"],
        "selection_source": selection["selection_source"],
        "selected_experiment": selection["selected_experiment"],
        "device": str(context["device"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": checkpoint_data.get("epoch"),
        "model_name": checkpoint_data.get(
            "model_name",
            checkpoint_data.get("config", {}).get("model_name"),
        ),
        "training_mode": checkpoint_data.get(
            "training_mode",
            checkpoint_data.get("config", {}).get("training_mode"),
        ),
        "recorded_val_accuracy": selection["best_val_accuracy"],
        "verified_val_accuracy": validation_result["accuracy"],
        "validation_accuracy_delta": context["validation_delta"],
        "validation_loss": validation_result["loss"],
        "test_accuracy": test_result["accuracy"],
        "test_loss": test_result["loss"],
        "macro_precision": metrics["macro_avg"]["precision"],
        "macro_recall": metrics["macro_avg"]["recall"],
        "macro_f1": metrics["macro_avg"]["f1-score"],
        "confusion_matrix_accuracy": matrix_accuracy,
        "inference_time": test_result["inference_time"],
        "inference_fps": test_result["inference_fps"],
        "test_samples": len(test_result["labels"]),
        "test_evaluation_count": test_evaluation_count,
        "total_evaluation_time": time.time() - started_at,
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2))
    validate_final_artifacts(output_dir=str(output_path))
    receipt = {
        "status": "FINAL_TEST_COMPLETE",
        "checkpoint_sha256": checkpoint_hash,
        "selection_artifact": str(Path(selection_path).resolve()),
        "test_evaluation_count": 1,
        "test_samples": summary["test_samples"],
        "test_accuracy": summary["test_accuracy"],
    }
    receipt_path.write_text(json.dumps(receipt, indent=2))
    return summary
