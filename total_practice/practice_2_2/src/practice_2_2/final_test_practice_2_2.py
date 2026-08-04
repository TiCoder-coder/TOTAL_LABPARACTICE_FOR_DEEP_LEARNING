"""One-time canonical Final Test evaluation for Practice 2.2."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader

from .canonical_train_practice_2_2 import (
    DATASET_FINGERPRINT,
    EXPECTED_MODEL_COUNTS,
    RUN_ID,
    SPLIT_FINGERPRINT,
    _state_dict_hash,
    verify_canonical_input,
)
from .data_practice_2_2 import (
    create_test_dataset,
    file_sha256,
    get_practice_2_2_transforms,
    load_split_manifest,
)
from .save_load import load_checkpoint, load_model_from_checkpoint


EXPERIMENT_ID = "E2_partial_finetune"
BEST_EPOCH = 14
VALIDATION_ACCURACY = 78.31050228310502
VALIDATION_MACRO_F1 = 0.7820960879325867
CHECKPOINT_SHA256 = "4f65bec200d023c51f95493833d057029b83a92729c3b88dd9159158dd949ae3"
CONFIG_FINGERPRINT = "25a4bc76ad5f58152e5a0af3e4fa79566f2191545a060c50fef9a4f9627c6c8d"
CLASS_ORDER = [
    "body_wash",
    "face_mask",
    "facial_cleanser",
    "lipstick",
    "moisturizer",
    "perfume",
    "serum",
    "shampoo",
    "sunscreen",
    "toner",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def claim_final_test_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json_exclusive(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def freeze_selection(final_dir, checkpoint_path):
    payload = {
        "immutable": True,
        "canonical_run_id": RUN_ID,
        "selected_experiment_id": EXPERIMENT_ID,
        "selection_metric": "validation_accuracy",
        "tie_breaker": "validation_loss",
        "best_epoch": BEST_EPOCH,
        "validation_accuracy": VALIDATION_ACCURACY,
        "validation_macro_f1": VALIDATION_MACRO_F1,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "dataset_fingerprint_sha256": DATASET_FINGERPRINT,
        "split_fingerprint_sha256": SPLIT_FINGERPRINT,
        "config_fingerprint_sha256": CONFIG_FINGERPRINT,
        "selection_timestamp_utc": utc_now(),
        "test_used_for_selection": False,
        "rejected_experiments": {
            "E3_layer4_1_head": "Validation performance decreased materially",
            "E4_moderate_augmentation": (
                "Generalization gap decreased but Validation performance decreased materially"
            ),
        },
        "winner_change_allowed": False,
    }
    write_json_exclusive(Path(final_dir) / "final_selection.json", payload)
    return payload


def pre_test_gate(
    dataset_root,
    manifest_path,
    split_summary_path,
    checkpoint_path,
    baseline_lineage_path,
    final_dir,
):
    selection_path = Path(final_dir) / "final_selection.json"
    if not selection_path.is_file():
        raise RuntimeError("Final selection must be frozen before the Test gate")
    if (Path(final_dir) / "FINAL_TEST_COMPLETED.json").exists():
        raise RuntimeError("Final Test has already completed")
    if (Path(final_dir) / "final_test_summary.json").exists():
        raise RuntimeError("A Final Test result already exists")

    verification = verify_canonical_input(dataset_root, manifest_path, split_summary_path)
    if verification["dataset_fingerprint_sha256"] != DATASET_FINGERPRINT:
        raise RuntimeError("Dataset fingerprint mismatch")
    if verification["split_fingerprint_sha256"] != SPLIT_FINGERPRINT:
        raise RuntimeError("Split fingerprint mismatch")
    if verification["model_use_counts"] != EXPECTED_MODEL_COUNTS:
        raise RuntimeError("Canonical model-use counts changed")
    if file_sha256(checkpoint_path) != CHECKPOINT_SHA256:
        raise RuntimeError("Selected E2 checkpoint SHA-256 mismatch")

    checkpoint = load_checkpoint(str(checkpoint_path), device=torch.device("cpu"))
    config = checkpoint.get("config", {})
    required_checkpoint = {
        "epoch": BEST_EPOCH,
        "model_name": "resnet18",
        "training_mode": "partial_finetune",
        "num_classes": 10,
    }
    for key, expected in required_checkpoint.items():
        if checkpoint.get(key) != expected:
            raise RuntimeError(f"Checkpoint {key} mismatch")
    required_config = {
        "run_id": RUN_ID,
        "experiment_id": EXPERIMENT_ID,
        "dataset_fingerprint_sha256": DATASET_FINGERPRINT,
        "split_fingerprint_sha256": SPLIT_FINGERPRINT,
        "config_fingerprint_sha256": CONFIG_FINGERPRINT,
        "image_size": 224,
        "dropout": 0.2,
        "test_data_used": False,
        "test_loader_constructed": False,
        "test_evaluated": False,
    }
    for key, expected in required_config.items():
        if config.get(key) != expected:
            raise RuntimeError(f"Checkpoint config {key} mismatch")

    baseline = json.loads(Path(baseline_lineage_path).read_text())
    if baseline.get("run_id") != RUN_ID:
        raise RuntimeError("Canonical lineage run ID mismatch")
    if any(
        baseline.get(key) is not False
        for key in ("test_loader_constructed", "test_evaluated", "test_metrics_recorded")
    ):
        raise RuntimeError("Canonical lineage indicates pre-existing Test use")

    manifest = load_split_manifest(manifest_path)
    test_rows = manifest.loc[
        manifest["use_for_model"] & manifest["split"].eq("Test")
    ].copy()
    if len(test_rows) != EXPECTED_MODEL_COUNTS["Test"]:
        raise RuntimeError("Canonical Test count is not 440")
    if test_rows["is_generated"].any() or test_rows["quarantined"].any():
        raise RuntimeError("Generated or quarantined files entered Test")
    mapping = (
        manifest.loc[manifest["use_for_model"], ["class_name", "label"]]
        .drop_duplicates()
        .sort_values("label")
    )
    if mapping["class_name"].tolist() != CLASS_ORDER:
        raise RuntimeError("Canonical class mapping mismatch")

    assigned = manifest.loc[manifest["split"].isin(["Train", "Validation", "Test"])]
    if assigned.groupby("duplicate_cluster_id")["split"].nunique().max() != 1:
        raise RuntimeError("Duplicate cluster crosses canonical split")
    if assigned.groupby("file_hash")["split"].nunique().max() != 1:
        raise RuntimeError("Exact hash crosses canonical split")

    gate = {
        "status": "PASS",
        "verified_at_utc": utc_now(),
        "final_selection_frozen": True,
        "dataset_fingerprint_sha256": DATASET_FINGERPRINT,
        "split_fingerprint_sha256": SPLIT_FINGERPRINT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_identity": required_checkpoint,
        "checkpoint_config_identity": required_config,
        "class_order": CLASS_ORDER,
        "model_use_counts": EXPECTED_MODEL_COUNTS,
        "generated_test_files": 0,
        "quarantined_test_files": 0,
        "duplicate_clusters_cross_split": 0,
        "exact_hashes_cross_split": 0,
        "preexisting_test_metrics": False,
        "test_loader_constructed": False,
    }
    write_json_exclusive(Path(final_dir) / "pre_test_verification.json", gate)
    return gate, manifest, test_rows


def _save_confusion_png(matrix, path):
    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(CLASS_ORDER)), CLASS_ORDER, rotation=45, ha="right")
    axis.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Canonical E2 — Final Test confusion matrix")
    threshold = matrix.max() / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def evaluate_final_test(
    dataset_root,
    manifest_path,
    test_rows,
    checkpoint_path,
    final_dir,
    batch_size=32,
    num_workers=0,
    device=None,
):
    selected_device = device or torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    test_dataset = create_test_dataset(
        dataset_root, manifest_path, image_size=224, allow_test=True
    )
    if len(test_dataset) != 440:
        raise RuntimeError("Test dataset count changed after construction")
    if test_dataset.dataset.classes != CLASS_ORDER:
        raise RuntimeError("ImageFolder class order mismatch")
    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )

    model = load_model_from_checkpoint(str(checkpoint_path), device=selected_device)
    model.eval()
    before_hash = _state_dict_hash(model)
    train_rows = load_split_manifest(manifest_path).loc[
        lambda frame: frame["use_for_model"] & frame["split"].eq("Train")
    ]
    counts = np.bincount(train_rows["label"].astype(int), minlength=10)
    class_weights = torch.tensor(
        len(train_rows) / (10 * counts.astype(np.float64)),
        dtype=torch.float32,
        device=selected_device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    targets = []
    predictions = []
    probabilities = []
    total_loss = 0.0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(selected_device)
            labels = labels.to(selected_device)
            logits = model(images)
            loss = criterion(logits, labels)
            probs = logits.softmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            targets.extend(labels.cpu().tolist())
            predictions.extend(probs.argmax(dim=1).cpu().tolist())
            probabilities.extend(probs.cpu().tolist())
    after_hash = _state_dict_hash(model)
    if after_hash != before_hash:
        raise RuntimeError("Frozen model weights or BatchNorm state changed during Test")
    if len(targets) != 440:
        raise RuntimeError("Final Test did not produce exactly 440 predictions")

    targets_array = np.asarray(targets)
    predictions_array = np.asarray(predictions)
    probabilities_array = np.asarray(probabilities)
    matrix = confusion_matrix(targets_array, predictions_array, labels=range(10))
    per_precision, per_recall, per_f1, support = precision_recall_fscore_support(
        targets_array, predictions_array, labels=range(10), zero_division=0
    )
    macro = precision_recall_fscore_support(
        targets_array, predictions_array, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        targets_array, predictions_array, average="weighted", zero_division=0
    )
    correct = int((targets_array == predictions_array).sum())

    metrics_by_class = pd.DataFrame(
        {
            "class_name": CLASS_ORDER,
            "precision": per_precision,
            "recall": per_recall,
            "f1": per_f1,
            "support": support.astype(int),
        }
    )
    metrics_by_class.to_csv(Path(final_dir) / "final_test_metrics_by_class.csv", index=False)
    pd.DataFrame(matrix, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(
        Path(final_dir) / "final_test_confusion_matrix.csv", index_label="true_class"
    )
    _save_confusion_png(matrix, Path(final_dir) / "final_test_confusion_matrix.png")

    rows = test_rows.reset_index(drop=True)
    prediction_data = {
        "stable_image_id": rows["record_id"],
        "relative_path": rows["relative_path"],
        "true_class": [CLASS_ORDER[index] for index in targets],
        "predicted_class": [CLASS_ORDER[index] for index in predictions],
        "correct": targets_array == predictions_array,
        "confidence": probabilities_array.max(axis=1),
        "duplicate_cluster_id": rows["duplicate_cluster_id"],
        "source_group": rows["source_group"],
    }
    for index, class_name in enumerate(CLASS_ORDER):
        prediction_data[f"prob_{class_name}"] = probabilities_array[:, index]
    prediction_frame = pd.DataFrame(prediction_data)
    prediction_frame.to_csv(Path(final_dir) / "final_test_predictions.csv", index=False)
    prediction_frame.loc[~prediction_frame["correct"]].to_csv(
        Path(final_dir) / "misclassified_samples.csv", index=False
    )

    pairs = []
    for true_index, true_name in enumerate(CLASS_ORDER):
        for predicted_index, predicted_name in enumerate(CLASS_ORDER):
            if true_index != predicted_index and matrix[true_index, predicted_index] > 0:
                pairs.append(
                    {
                        "true_class": true_name,
                        "predicted_class": predicted_name,
                        "count": int(matrix[true_index, predicted_index]),
                    }
                )
    pairs.sort(key=lambda item: (-item["count"], item["true_class"], item["predicted_class"]))
    high_confidence_errors = (
        prediction_frame.loc[~prediction_frame["correct"]]
        .sort_values("confidence", ascending=False)
        .head(20)[
            ["stable_image_id", "relative_path", "true_class", "predicted_class", "confidence"]
        ]
        .to_dict(orient="records")
    )
    summary = {
        "canonical_run_id": RUN_ID,
        "experiment_id": EXPERIMENT_ID,
        "evaluated_at_utc": utc_now(),
        "test_loss": total_loss / len(targets),
        "test_accuracy": 100.0 * correct / len(targets),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "total_predictions": len(targets),
        "correct_predictions": correct,
        "incorrect_predictions": len(targets) - correct,
        "class_order": CLASS_ORDER,
        "test_dataloader": {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "device": str(selected_device),
        },
        "test_transform": repr(get_practice_2_2_transforms(224, "base")[1]),
        "model_state_sha256_before": before_hash,
        "model_state_sha256_after": after_hash,
        "weight_updates_performed": False,
        "optimizer_steps": 0,
        "scheduler_steps": 0,
        "five_lowest_recall_classes": (
            metrics_by_class.sort_values(["recall", "class_name"])
            .head(5)[["class_name", "recall"]]
            .to_dict(orient="records")
        ),
        "top_confusion_pairs": pairs[:10],
        "high_confidence_errors": high_confidence_errors,
    }
    write_json_exclusive(Path(final_dir) / "final_test_summary.json", summary)
    return summary


def run_final_test(
    dataset_root,
    manifest_path,
    split_summary_path,
    checkpoint_path,
    config_snapshot_path,
    baseline_lineage_path,
    final_dir,
):
    final_dir = claim_final_test_directory(final_dir)
    selection = freeze_selection(final_dir, checkpoint_path)
    gate, _, test_rows = pre_test_gate(
        dataset_root,
        manifest_path,
        split_summary_path,
        checkpoint_path,
        baseline_lineage_path,
        final_dir,
    )
    summary = evaluate_final_test(
        dataset_root,
        manifest_path,
        test_rows,
        checkpoint_path,
        final_dir,
    )

    artifact_names = [
        "final_selection.json",
        "pre_test_verification.json",
        "final_test_summary.json",
        "final_test_metrics_by_class.csv",
        "final_test_predictions.csv",
        "final_test_confusion_matrix.csv",
        "final_test_confusion_matrix.png",
        "misclassified_samples.csv",
    ]
    hashes = {name: file_sha256(final_dir / name) for name in artifact_names}
    hashes["selected_checkpoint"] = file_sha256(checkpoint_path)
    hashes["config_snapshot"] = file_sha256(config_snapshot_path)
    hashes["canonical_manifest"] = file_sha256(manifest_path)
    lineage = {
        "FINAL_TEST_COMPLETED": True,
        "canonical_run_id": RUN_ID,
        "selected_experiment_id": EXPERIMENT_ID,
        "dataset_fingerprint_sha256": DATASET_FINGERPRINT,
        "split_fingerprint_sha256": SPLIT_FINGERPRINT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_fingerprint_sha256": CONFIG_FINGERPRINT,
        "selection": selection,
        "pre_test_gate_status": gate["status"],
        "output_artifact_sha256": hashes,
        "completed_at_utc": utc_now(),
        "final_test_evaluation_count": 1,
        "winner_change_allowed": False,
        "retraining_after_test_allowed": False,
    }
    write_json_exclusive(final_dir / "final_test_lineage.json", lineage)
    lineage_hash = file_sha256(final_dir / "final_test_lineage.json")
    guard = {
        "FINAL_TEST_COMPLETED": True,
        "completed_at_utc": lineage["completed_at_utc"],
        "canonical_run_id": RUN_ID,
        "final_test_evaluation_count": 1,
        "summary_sha256": hashes["final_test_summary.json"],
        "lineage_sha256": lineage_hash,
        "maintenance_override_used": False,
        "repeat_evaluation_allowed": False,
    }
    write_json_exclusive(final_dir / "FINAL_TEST_COMPLETED.json", guard)
    return {"summary": summary, "lineage": lineage, "guard": guard}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split-summary", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config-snapshot", required=True, type=Path)
    parser.add_argument("--baseline-lineage", required=True, type=Path)
    parser.add_argument("--final-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run_final_test(
        args.dataset_root,
        args.manifest,
        args.split_summary,
        args.checkpoint,
        args.config_snapshot,
        args.baseline_lineage,
        args.final_dir,
    ), indent=2))


if __name__ == "__main__":
    main()
