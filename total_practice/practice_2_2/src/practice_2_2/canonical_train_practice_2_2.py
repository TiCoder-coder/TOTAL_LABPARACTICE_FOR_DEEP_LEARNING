"""Canonical, Test-locked E1/E2 training lineage for Practice 2.2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision

from .data_practice_2_2 import (
    _stable_fingerprint,
    compute_train_class_weights,
    create_train_validation_datasets,
    file_sha256,
    get_practice_2_2_transforms,
    load_split_manifest,
    make_train_validation_loaders,
    verify_artifact_fingerprints,
)
from .model import build_model
from .save_load import load_model_from_checkpoint
from .train import evaluate, get_criterion, train_model
from .utils import get_device, setup_reproducibility


DATASET_FINGERPRINT = "26dc4625f96c7cb86bc6a4df0fbed34bc8b22fb6472ef0d008df341f99f71fd6"
SPLIT_FINGERPRINT = "52aaf97499ede5dd4689c2bb36dc0679fe04b8ec8139aae7e8fdcde88042043d"
SEED = 42
RUN_ID = f"canonical_{DATASET_FINGERPRINT[:8]}_{SPLIT_FINGERPRINT[:8]}_s{SEED}_v1"
EXPECTED_MODEL_COUNTS = {"Train": 2016, "Validation": 438, "Test": 440}

CANONICAL_BASE_CONFIG = {
    "seed": SEED,
    "model_name": "resnet18",
    "pretrained_weights": "ResNet18_Weights.DEFAULT",
    "num_classes": 10,
    "image_size": 224,
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std": [0.229, 0.224, 0.225],
    "augment_strength": "base",
    "batch_size": 32,
    "epochs": 15,
    "optimizer": "AdamW",
    "learning_rate": 1e-3,
    "head_learning_rate": 1e-3,
    "backbone_learning_rate": 1e-4,
    "weight_decay": 2e-4,
    "scheduler": "ReduceLROnPlateau",
    "early_stopping_patience": 3,
    "early_stopping_metric": "val_loss",
    "best_model_metric": "val_acc",
    "label_smoothing": 0.05,
    "dropout": 0.20,
    "grad_clip": 1.0,
    "selection_source": "validation_only",
    "test_data_used": False,
    "test_loader_constructed": False,
    "test_evaluated": False,
    "offline_generated_files_used": False,
    "resume_from_legacy": False,
}

EXPERIMENTS = {
    "E1_head_only": {**CANONICAL_BASE_CONFIG, "training_mode": "head_only"},
    "E2_partial_finetune": {
        **CANONICAL_BASE_CONFIG,
        "training_mode": "partial_finetune",
    },
}


def _json_hash(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _state_dict_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_controlled_configs():
    e1 = EXPERIMENTS["E1_head_only"]
    e2 = EXPERIMENTS["E2_partial_finetune"]
    differing = {
        key for key in set(e1).union(e2) if e1.get(key) != e2.get(key)
    }
    if differing != {"training_mode"}:
        raise RuntimeError(
            "Canonical E1/E2 must differ only by training_mode; found "
            + ", ".join(sorted(differing))
        )
    if e1["training_mode"] != "head_only":
        raise RuntimeError("E1 must train the classifier head only")
    if e2["training_mode"] != "partial_finetune":
        raise RuntimeError("E2 must train layer4 and the classifier head")
    if any(config["test_data_used"] for config in EXPERIMENTS.values()):
        raise RuntimeError("Test data is forbidden during canonical training")
    return True


def verify_canonical_input(dataset_root, manifest_path, summary_path):
    """Re-hash the dataset and verify the immutable canonical assignment."""
    root = Path(dataset_root).resolve()
    manifest = load_split_manifest(manifest_path)
    persisted = verify_artifact_fingerprints(manifest_path, summary_path)
    if persisted["dataset_fingerprint_sha256"] != DATASET_FINGERPRINT:
        raise RuntimeError("Unexpected canonical dataset fingerprint")
    if persisted["split_fingerprint_sha256"] != SPLIT_FINGERPRINT:
        raise RuntimeError("Unexpected canonical split fingerprint")

    manifest_paths = set(manifest["relative_path"])
    actual_paths = {
        str(path.resolve().relative_to(root))
        for path in root.glob("*/*")
        if path.is_file()
    }
    if actual_paths != manifest_paths:
        raise RuntimeError(
            "Dataset files differ from the canonical manifest: "
            f"missing={sorted(manifest_paths - actual_paths)[:5]}, "
            f"unexpected={sorted(actual_paths - manifest_paths)[:5]}"
        )

    live_rows = []
    for row in manifest.sort_values("relative_path").itertuples():
        path = root / row.relative_path
        digest = file_sha256(path)
        if digest != row.file_hash:
            raise RuntimeError(f"File hash changed: {row.relative_path}")
        live_rows.append(
            {
                "relative_path": row.relative_path,
                "class_name": row.class_name,
                "file_hash": digest,
            }
        )
    live_fingerprint = _stable_fingerprint(
        pd.DataFrame(live_rows),
        ["relative_path", "class_name", "file_hash"],
    )
    if live_fingerprint != DATASET_FINGERPRINT:
        raise RuntimeError("Live dataset fingerprint does not match canonical input")

    used = manifest.loc[manifest["use_for_model"]]
    counts = used["split"].value_counts().to_dict()
    if counts != EXPECTED_MODEL_COUNTS:
        raise RuntimeError(f"Unexpected canonical model counts: {counts}")
    if used["is_generated"].any() or used["quarantined"].any():
        raise RuntimeError("Generated or quarantined rows entered model-use data")
    assigned = manifest.loc[
        manifest["split"].isin(["Train", "Validation", "Test"])
    ]
    if assigned.groupby("duplicate_cluster_id")["split"].nunique().max() != 1:
        raise RuntimeError("A duplicate cluster crosses split boundaries")
    if assigned.groupby("source_group")["split"].nunique().max() != 1:
        raise RuntimeError("A generated family crosses split boundaries")
    if assigned.groupby("file_hash")["split"].nunique().max() != 1:
        raise RuntimeError("An exact hash crosses split boundaries")
    summary = json.loads(Path(summary_path).read_text())
    if summary.get("test_loader_constructed") is not False:
        raise RuntimeError("Canonical split summary does not prove Test is locked")
    if summary.get("test_evaluated") is not False:
        raise RuntimeError("Canonical split summary indicates Test access")
    return {
        "dataset_fingerprint_sha256": live_fingerprint,
        "split_fingerprint_sha256": persisted["split_fingerprint_sha256"],
        "model_use_counts": counts,
        "quarantine_files": int(manifest["quarantined"].sum()),
        "generated_files_excluded": int(manifest["is_generated"].sum()),
        "test_loader_constructed": False,
        "test_evaluated": False,
    }


def _environment_snapshot(device):
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "device": str(device),
        "deterministic_algorithms": True,
        "deterministic_warn_only": True,
    }


def _snapshot(run_id, dataset_root, manifest_path, device):
    train_transform, validation_transform = get_practice_2_2_transforms(
        image_size=CANONICAL_BASE_CONFIG["image_size"],
        augment_strength=CANONICAL_BASE_CONFIG["augment_strength"],
    )
    snapshot = {
        "schema_version": 1,
        "run_id": run_id,
        "dataset_root": str(Path(dataset_root).resolve()),
        "canonical_manifest": str(Path(manifest_path).resolve()),
        "dataset_fingerprint_sha256": DATASET_FINGERPRINT,
        "split_fingerprint_sha256": SPLIT_FINGERPRINT,
        "model_use_counts": EXPECTED_MODEL_COUNTS,
        "train_transform": repr(train_transform),
        "validation_transform": repr(validation_transform),
        "test_transform": "LOCKED_NOT_CONSTRUCTED",
        "loss_function": (
            "CrossEntropyLoss(inverse-frequency Train weights, "
            "label_smoothing=0.05)"
        ),
        "selection_policy": {
            "primary": "validation_accuracy",
            "tie_breaker": "validation_loss",
            "test_data_used": False,
        },
        "experiments": EXPERIMENTS,
        "environment": _environment_snapshot(device),
        "reproducibility": {
            "python_seed": SEED,
            "numpy_seed": SEED,
            "torch_seed": SEED,
            "cuda_seed_if_available": SEED,
            "mps_seed_if_available": SEED,
        },
    }
    snapshot["config_fingerprint_sha256"] = _json_hash(snapshot)
    return snapshot


def run_canonical_training(
    dataset_root,
    manifest_path,
    summary_path,
    runs_root,
    outputs_root,
    device=None,
):
    validate_controlled_configs()
    verification = verify_canonical_input(
        dataset_root, manifest_path, summary_path
    )
    selected_device = device or get_device()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    run_root = Path(runs_root) / RUN_ID
    output_root = Path(outputs_root) / RUN_ID
    if run_root.exists() or output_root.exists():
        raise FileExistsError(
            f"Canonical run lineage already exists; refusing overwrite: {RUN_ID}"
        )
    run_root.mkdir(parents=True)
    output_root.mkdir(parents=True)

    snapshot = _snapshot(
        RUN_ID, dataset_root, manifest_path, selected_device
    )
    (output_root / "config_snapshot.json").write_text(
        json.dumps(snapshot, indent=2)
    )

    results = []
    histories = {}
    initial_hashes = {}
    checkpoint_verifications = {}
    for experiment_id, experiment_config in EXPERIMENTS.items():
        setup_reproducibility(SEED)
        train_dataset, validation_dataset = create_train_validation_datasets(
            dataset_root,
            manifest_path,
            image_size=experiment_config["image_size"],
            augment_strength=experiment_config["augment_strength"],
        )
        train_loader, validation_loader = make_train_validation_loaders(
            train_dataset,
            validation_dataset,
            batch_size=experiment_config["batch_size"],
            num_workers=0,
            seed=SEED,
        )
        if len(train_dataset) != EXPECTED_MODEL_COUNTS["Train"]:
            raise RuntimeError("Train count changed after DataLoader preparation")
        if len(validation_dataset) != EXPECTED_MODEL_COUNTS["Validation"]:
            raise RuntimeError("Validation count changed after DataLoader preparation")

        model = build_model(
            experiment_config["model_name"],
            experiment_config["training_mode"],
            num_classes=experiment_config["num_classes"],
            dropout=experiment_config["dropout"],
        ).to(selected_device)
        initial_hashes[experiment_id] = _state_dict_hash(model)
        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        experiment_snapshot = {
            **experiment_config,
            "run_id": RUN_ID,
            "experiment_id": experiment_id,
            "dataset_fingerprint_sha256": DATASET_FINGERPRINT,
            "split_fingerprint_sha256": SPLIT_FINGERPRINT,
            "canonical_config_fingerprint_sha256": snapshot[
                "config_fingerprint_sha256"
            ],
            "initial_model_state_sha256": initial_hashes[experiment_id],
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
            "output_dir": str((run_root / experiment_id).resolve()),
        }
        experiment_snapshot["config_fingerprint_sha256"] = _json_hash(
            experiment_snapshot
        )

        class_weights = compute_train_class_weights(train_dataset)
        started = time.perf_counter()
        history = train_model(
            model,
            train_loader,
            validation_loader,
            experiment_snapshot,
            selected_device,
            class_weights=class_weights,
        )
        training_seconds = time.perf_counter() - started
        histories[experiment_id] = history
        best_checkpoint = run_root / experiment_id / "best.pt"
        latest_checkpoint = run_root / experiment_id / "latest.pt"
        if not best_checkpoint.is_file() or not latest_checkpoint.is_file():
            raise FileNotFoundError(f"Missing canonical checkpoint for {experiment_id}")

        reloaded = load_model_from_checkpoint(
            str(best_checkpoint), device=selected_device
        )
        criterion = get_criterion(
            experiment_snapshot, class_weights.to(selected_device)
        )
        val_loss, val_accuracy, val_macro_f1 = evaluate(
            reloaded,
            validation_loader,
            criterion,
            selected_device,
            return_macro_f1=True,
        )
        best_epoch = int(history["best_epoch"])
        best_index = best_epoch - 1
        recorded_accuracy = float(history["val_acc"][best_index])
        recorded_loss = float(history["val_loss"][best_index])
        accuracy_delta = abs(val_accuracy - recorded_accuracy)
        loss_delta = abs(val_loss - recorded_loss)
        if accuracy_delta > 1e-6 or loss_delta > 1e-6:
            raise RuntimeError(
                f"Checkpoint reload verification failed for {experiment_id}: "
                f"accuracy_delta={accuracy_delta}, loss_delta={loss_delta}"
            )
        checkpoint_hash = file_sha256(best_checkpoint)
        checkpoint_verifications[experiment_id] = {
            "status": "PASS",
            "checkpoint": str(best_checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "recorded_val_accuracy": recorded_accuracy,
            "reloaded_val_accuracy": val_accuracy,
            "accuracy_delta": accuracy_delta,
            "recorded_val_loss": recorded_loss,
            "reloaded_val_loss": val_loss,
            "loss_delta": loss_delta,
            "test_loader_constructed": False,
            "test_evaluated": False,
        }
        results.append(
            {
                "run_id": RUN_ID,
                "experiment": experiment_id,
                "strategy": experiment_config["training_mode"],
                "best_epoch": best_epoch,
                "epochs_trained": len(history["epoch"]),
                "train_accuracy_at_best": float(
                    history["train_acc"][best_index]
                ),
                "best_val_accuracy": val_accuracy,
                "best_val_loss": val_loss,
                "best_val_macro_f1": val_macro_f1,
                "generalization_gap": float(
                    history["train_acc"][best_index] - val_accuracy
                ),
                "trainable_parameters": trainable_parameters,
                "total_parameters": total_parameters,
                "training_seconds": training_seconds,
                "early_stopping_triggered": bool(history["stopped_early"]),
                "checkpoint": str(best_checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_hash,
                "config_fingerprint_sha256": experiment_snapshot[
                    "config_fingerprint_sha256"
                ],
                "validation_reload_status": "PASS",
                "test_loader_constructed": False,
                "test_evaluated": False,
            }
        )
        del model, reloaded, train_loader, validation_loader
        if selected_device.type == "cuda":
            torch.cuda.empty_cache()

    if len(set(initial_hashes.values())) != 1:
        raise RuntimeError("E1/E2 did not start from identical model weights")
    comparison = pd.DataFrame(results).sort_values(
        ["best_val_accuracy", "best_val_loss"],
        ascending=[False, True],
    ).reset_index(drop=True)
    winner = comparison.iloc[0]
    selection = {
        "run_id": RUN_ID,
        "status": "PROVISIONAL_VALIDATION_ONLY",
        "selection_metric": "validation_accuracy",
        "tie_breaker": "validation_loss",
        "selected_experiment": winner["experiment"],
        "selected_checkpoint": winner["checkpoint"],
        "best_val_accuracy": float(winner["best_val_accuracy"]),
        "best_val_loss": float(winner["best_val_loss"]),
        "dataset_fingerprint_sha256": DATASET_FINGERPRINT,
        "split_fingerprint_sha256": SPLIT_FINGERPRINT,
        "test_loader_constructed": False,
        "test_evaluated": False,
        "test_metrics_recorded": False,
    }
    comparison.to_csv(output_root / "validation_comparison.csv", index=False)
    (output_root / "training_histories.json").write_text(
        json.dumps(histories, indent=2)
    )
    (output_root / "checkpoint_verification.json").write_text(
        json.dumps(checkpoint_verifications, indent=2)
    )
    (output_root / "provisional_selection.json").write_text(
        json.dumps(selection, indent=2)
    )
    lineage = {
        "run_id": RUN_ID,
        "created_unix_time": time.time(),
        "config_snapshot": str((output_root / "config_snapshot.json").resolve()),
        "dataset_verification": verification,
        "initial_model_state_sha256": next(iter(initial_hashes.values())),
        "experiments": results,
        "provisional_selection": selection,
        "test_loader_constructed": False,
        "test_evaluated": False,
        "test_metrics_recorded": False,
    }
    (output_root / "lineage_manifest.json").write_text(
        json.dumps(lineage, indent=2)
    )
    return lineage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split-summary", required=True, type=Path)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--outputs-root", required=True, type=Path)
    args = parser.parse_args()
    result = run_canonical_training(
        args.dataset_root,
        args.manifest,
        args.split_summary,
        args.runs_root,
        args.outputs_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
