"""Leakage-safe E1/E2 training runner for Practice 2.2.

This module intentionally imports no Test dataset/loader builder. Model
selection is performed exclusively with the duplicate-aware Validation split.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import torch

from configs.experiment_config import EXPERIMENTS
from processing_own_phase.data_practice_2_2 import (
    compute_train_class_weights,
    create_train_validation_datasets,
    make_train_validation_loaders,
)
from processing_own_phase.model import build_model
from processing_own_phase.save_load import load_model_from_checkpoint
from processing_own_phase.train import evaluate, get_criterion, train_model
from processing_own_phase.utils import get_device, setup_reproducibility


RUN_SPECS = (
    ("E1_head_only", "E1_resnet18_head"),
    ("E2_partial_finetune", "E2_resnet18_partial"),
)


def validate_run_specs():
    """Prove that the approved experiments differ only by training_mode."""
    configurations = [EXPERIMENTS[config_id] for _, config_id in RUN_SPECS]
    differing = {
        key
        for key in set(configurations[0]).union(configurations[1])
        if configurations[0].get(key) != configurations[1].get(key)
    }
    if differing != {"training_mode"}:
        raise RuntimeError(
            "E1/E2 must differ only by training_mode; found "
            + ", ".join(sorted(differing))
        )
    if [config["training_mode"] for config in configurations] != [
        "head_only",
        "partial_finetune",
    ]:
        raise RuntimeError("Unexpected E1/E2 training modes")
    if any(config.get("test_data_used") for config in configurations):
        raise RuntimeError("Test data is forbidden during controlled training")
    return True


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _train_one_experiment(
    run_name,
    config_id,
    dataset_root,
    manifest_path,
    staging_runs_root,
    device,
):
    config = dict(EXPERIMENTS[config_id])
    config["experiment"] = run_name
    config["config_id"] = config_id
    config["dataset_root"] = str(Path(dataset_root).resolve())
    config["split_manifest"] = str(Path(manifest_path).resolve())
    config["test_loader_constructed"] = False
    config["test_evaluated"] = False
    run_dir = Path(staging_runs_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    config["output_dir"] = str(run_dir.resolve())

    setup_reproducibility(config["seed"])
    train_dataset, validation_dataset = create_train_validation_datasets(
        dataset_root,
        manifest_path,
        image_size=config["image_size"],
    )
    train_loader, validation_loader = make_train_validation_loaders(
        train_dataset,
        validation_dataset,
        batch_size=config["batch_size"],
        num_workers=0,
        seed=config["seed"],
    )
    class_weights = compute_train_class_weights(train_dataset)

    model = build_model(
        config["model_name"],
        config["training_mode"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
    ).to(device)
    started = time.time()
    history = train_model(
        model,
        train_loader,
        validation_loader,
        config,
        device,
        class_weights=class_weights,
    )
    training_time = time.time() - started
    if len(history["epoch"]) < 2:
        raise RuntimeError("Controlled training must contain multiple epochs")

    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing best checkpoint: {checkpoint_path}")
    best_model = load_model_from_checkpoint(str(checkpoint_path), device=device)
    criterion = get_criterion(config, class_weights.to(device))
    val_loss, val_accuracy, val_macro_f1 = evaluate(
        best_model,
        validation_loader,
        criterion,
        device,
        return_macro_f1=True,
    )
    stored_accuracy = float(history["best_metric"])
    verification_delta = abs(val_accuracy - stored_accuracy)
    if verification_delta > 1e-6:
        raise RuntimeError(
            f"Validation checkpoint verification failed for {run_name}: "
            f"stored={stored_accuracy:.8f}, reloaded={val_accuracy:.8f}"
        )

    best_epoch = int(history["best_epoch"])
    best_index = best_epoch - 1
    result = {
        "experiment": run_name,
        "strategy": config["training_mode"],
        "epochs_trained": len(history["epoch"]),
        "epochs_planned": config["epochs"],
        "best_epoch": best_epoch,
        "best_val_accuracy": val_accuracy,
        "best_val_loss": val_loss,
        "best_val_macro_f1": val_macro_f1,
        "train_accuracy_at_best": history["train_acc"][best_index],
        "generalization_gap_at_best": (
            history["train_acc"][best_index] - val_accuracy
        ),
        "training_time": training_time,
        "early_stopping_triggered": bool(history["stopped_early"]),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "validation_verification_passed": True,
        "validation_verification_delta": verification_delta,
        "selection_source": "validation_only",
        "test_loader_constructed": False,
        "test_evaluated": False,
        "test_access_count": 0,
    }
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    (run_dir / "validation_result.json").write_text(
        json.dumps(result, indent=2)
    )
    return result, history, config


def run_controlled_training(
    dataset_root,
    manifest_path,
    staging_root,
    device=None,
):
    """Train E1/E2 into staging and save Validation-only artifacts."""
    validate_run_specs()
    selected_device = device or get_device()
    if selected_device.type == "cpu":
        raise RuntimeError(
            "GPU/MPS is unavailable. Refusing to silently run the approved "
            "15-epoch experiments on CPU."
        )
    staging_root = Path(staging_root)
    staging_runs_root = staging_root / "runs"
    staging_outputs_root = staging_root / "outputs"
    staging_runs_root.mkdir(parents=True, exist_ok=False)
    staging_outputs_root.mkdir(parents=True, exist_ok=False)

    results = []
    histories = {}
    configs = {}
    run_directories = {}
    for run_name, config_id in RUN_SPECS:
        result, history, config = _train_one_experiment(
            run_name,
            config_id,
            dataset_root,
            manifest_path,
            staging_runs_root,
            selected_device,
        )
        results.append(result)
        histories[run_name] = history
        configs[run_name] = {
            key: config[key] for key in EXPERIMENTS[config_id]
        }
        run_directories[run_name] = str(
            (staging_runs_root / run_name).resolve()
        )

    selected = max(
        results,
        key=lambda row: (row["best_val_accuracy"], -row["best_val_loss"]),
    )
    comparison = pd.DataFrame(results)
    comparison.to_csv(
        staging_outputs_root / "controlled_experiment_comparison.csv",
        index=False,
    )
    (staging_outputs_root / "controlled_training_history.json").write_text(
        json.dumps(histories, indent=2)
    )
    selection = {
        "selection_metric": "val_accuracy",
        "tie_breaker": "val_loss",
        "selection_source": "validation_only",
        "selected_experiment": selected["experiment"],
        "selected_checkpoint": selected["checkpoint_path"],
        "best_epoch": selected["best_epoch"],
        "best_val_accuracy": selected["best_val_accuracy"],
        "best_val_loss": selected["best_val_loss"],
        "best_val_macro_f1": selected["best_val_macro_f1"],
        "epochs_trained": selected["epochs_trained"],
        "early_stopping_triggered": selected["early_stopping_triggered"],
        "validation_verification_passed": True,
        "test_loader_constructed": False,
        "test_evaluated": False,
        "test_access_count": 0,
        "experiments": [row["experiment"] for row in results],
    }
    (staging_outputs_root / "controlled_experiment_selection.json").write_text(
        json.dumps(selection, indent=2)
    )
    run_manifest = {
        "device": str(selected_device),
        "dataset_root": str(Path(dataset_root).resolve()),
        "split_manifest": str(Path(manifest_path).resolve()),
        "experiment_configurations": configs,
        "run_directories": run_directories,
        "test_loader_constructed": False,
        "test_evaluated": False,
        "test_access_count": 0,
    }
    (staging_outputs_root / "controlled_run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2)
    )
    return {
        "results": results,
        "selection": selection,
        "staging_root": str(staging_root.resolve()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--staging-root", required=True, type=Path)
    args = parser.parse_args()
    result = run_controlled_training(
        args.dataset_root,
        args.manifest,
        args.staging_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
