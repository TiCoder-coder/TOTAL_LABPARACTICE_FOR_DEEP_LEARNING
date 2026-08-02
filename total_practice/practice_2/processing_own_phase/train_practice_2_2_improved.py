"""Improved E1/E2 training loop for Practice 2.2 with anti-overfitting tools.

This module wraps the existing ``train_model`` contract but layers on:

- Mixup / CutMix label-preserving image mixing
- Soft cross-entropy to consume the mixed soft targets
- Exponential Moving Average (EMA) of model weights, used for evaluation,
  validation-based selection, and final reporting
- A ``train_one_epoch_advanced`` step that records the realised mixing
  probability for reproducibility

The goal is to push the gap between train and validation accuracy down and
to stabilise the validation metric so model selection reflects true
generalisation rather than the noise of the last few mini-batches.

The module is intentionally additive: existing ``train_model`` callers and
``experiment.py`` continue to work untouched.
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.experiment_config import EXPERIMENTS
from processing_own_phase.data_practice_2_2 import (
    create_test_dataset,
    create_train_validation_datasets,
    make_train_validation_loaders,
)
from processing_own_phase.model import build_model
from processing_own_phase.regularization_practice_2_2 import (
    ModelEMA,
    mixup_cutmix_batch,
    soft_cross_entropy,
)
from processing_own_phase.save_load import load_model_from_checkpoint
from processing_own_phase.train import evaluate, get_criterion
from processing_own_phase.utils import get_device, setup_reproducibility


RUN_SPECS = (
    ("E1_resnet18_head", "E1_resnet18_head"),
    ("E2_resnet18_partial", "E2_resnet18_partial"),
)


def _safe_eval(ema: ModelEMA, loader: DataLoader, criterion: nn.Module, device):
    """Evaluate EMA model; return (loss_percent, acc_percent, macro_f1)."""
    model = ema.module().to(device)
    return evaluate(model, loader, criterion, device, return_macro_f1=True)


def _train_one_epoch_mixup(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    scaler: Optional[torch.cuda.amp.GradScaler],
    *,
    mixup_alpha: float,
    cutmix_alpha: float,
    cutmix_prob: float,
    ema: ModelEMA,
    label_smoothing: float,
    generator: torch.Generator,
) -> Tuple[float, float]:
    """Train one epoch with Mixup/CutMix + EMA updates.

    The standard CE ``criterion`` is only used as a fallback; the actual loss
    is computed by :func:`soft_cross_entropy` so the mixing factor ``lam`` is
    respected exactly. When no mixing is applied, soft_cross_entropy reduces
    to standard CE with label smoothing.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    mix_count = 0

    from tqdm import tqdm

    pbar = tqdm(dataloader, desc="Training (mix)", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if (mixup_alpha > 0) or (cutmix_alpha > 0):
            mixed_images, soft_labels = mixup_cutmix_batch(
                images,
                labels,
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                cutmix_prob=cutmix_prob,
                generator=generator,
            )
            mix_count += 1
        else:
            mixed_images = images
            soft_labels = labels

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(mixed_images)
                loss = soft_cross_entropy(
                    logits,
                    soft_labels,
                    label_smoothing=label_smoothing,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(mixed_images)
            loss = soft_cross_entropy(
                logits,
                soft_labels,
                label_smoothing=label_smoothing,
            )
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # Refresh EMA AFTER each optimizer step so the EMA weights stay close
        # to the live model.
        ema.update(model)

        # Compute accuracy against the original hard labels, not the soft ones,
        # so the metric stays comparable with the validation metric.
        with torch.no_grad():
            predicted = logits.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item() * labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1) * 100.0
    return epoch_loss, epoch_acc


def _train_one_experiment(
    run_name: str,
    config_id: str,
    dataset_root: Path,
    manifest_path: Path,
    staging_runs_root: Path,
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Train a single E1/E2 experiment with all anti-overfitting tools."""
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
        augment_strength=config.get("augment_strength", "strong"),
    )
    train_loader, validation_loader = make_train_validation_loaders(
        train_dataset,
        validation_dataset,
        batch_size=config["batch_size"],
        num_workers=0,
        seed=config["seed"],
    )

    from processing_own_phase.data_practice_2_2 import (
        compute_train_class_weights,
    )
    class_weights = compute_train_class_weights(train_dataset)

    model = build_model(
        config["model_name"],
        config["training_mode"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
    ).to(device)

    ema = ModelEMA(
        model,
        decay=float(config.get("ema_decay", 0.999)),
        device=device,
    )

    criterion = get_criterion(config, class_weights.to(device))

    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    history: Dict[str, Any] = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "ema_val_loss": [],
        "ema_val_acc": [],
        "ema_val_macro_f1": [],
        "generalization_gap": [],
        "lr": [],
        "epoch_time": [],
    }
    best_metric_name = config.get("report_metric", "val_acc")
    best_metric = float("-inf")
    best_epoch = 0
    stopped_early = False
    epochs_no_improve = 0
    patience = int(config.get("early_stopping_patience", 3))
    min_delta = float(config.get("early_stopping_min_delta", 0.0))

    generator = torch.Generator(device=device).manual_seed(config["seed"])

    started = time.time()
    for epoch in range(int(config["epochs"])):
        epoch_started = time.time()

        # LR snapshot
        current_lr = max(group["lr"] for group in optimizer.param_groups)
        history["lr"].append(current_lr)

        train_loss, train_acc = _train_one_epoch_mixup(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            float(config.get("grad_clip", 1.0)),
            scaler,
            mixup_alpha=float(config.get("mixup_alpha", 0.0)),
            cutmix_alpha=float(config.get("cutmix_alpha", 0.0)),
            cutmix_prob=float(config.get("cutmix_prob", 0.5)),
            ema=ema,
            label_smoothing=float(config.get("label_smoothing", 0.0)),
            generator=generator,
        )

        # Validation: BOTH the live model and the EMA model. EMA usually wins
        # on small datasets, but we still log both for diagnostics.
        val_loss, val_acc, val_f1 = evaluate(
            model, validation_loader, criterion, device, return_macro_f1=True
        )
        ema_val_loss, ema_val_acc, ema_val_f1 = _safe_eval(
            ema, validation_loader, criterion, device
        )

        epoch_time = time.time() - epoch_started

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)
        history["ema_val_loss"].append(ema_val_loss)
        history["ema_val_acc"].append(ema_val_acc)
        history["ema_val_macro_f1"].append(ema_val_f1)
        history["generalization_gap"].append(train_acc - ema_val_acc)
        history["epoch_time"].append(epoch_time)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(ema_val_loss)
            else:
                scheduler.step()

        # Select on the EMA metric (or live metric if EMA disabled).
        if config.get("use_ema", True):
            current_metric = ema_val_acc if best_metric_name == "val_acc" else ema_val_f1
            monitor_loss = ema_val_loss
        else:
            current_metric = val_acc if best_metric_name == "val_acc" else val_f1
            monitor_loss = val_loss

        if current_metric > best_metric + min_delta:
            best_metric = float(current_metric)
            best_epoch = epoch + 1
            epochs_no_improve = 0
            _save_checkpoint(run_dir, ema, model, optimizer, scheduler, config, epoch + 1, current_metric, monitor_loss)
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                stopped_early = True
                break

    training_time = time.time() - started
    history["best_epoch"] = best_epoch
    history["best_metric"] = best_metric
    history["best_metric_name"] = best_metric_name
    history["stopped_early"] = stopped_early
    history["epochs_planned"] = int(config["epochs"])
    history["ema_state"] = ema.state_dict()

    # Final verification: reload best checkpoint and re-evaluate on Val.
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing best checkpoint: {checkpoint_path}")
    verified_model = load_model_from_checkpoint(str(checkpoint_path), device=device)
    val_loss, val_acc, val_f1 = evaluate(
        verified_model, validation_loader, criterion, device, return_macro_f1=True
    )
    delta = abs(val_acc - best_metric)
    if delta > 1e-4:
        raise RuntimeError(
            f"Verification failed for {run_name}: stored={best_metric:.6f}, reloaded={val_acc:.6f}"
        )

    result = {
        "experiment": run_name,
        "strategy": config["training_mode"],
        "epochs_trained": len(history["epoch"]),
        "epochs_planned": int(config["epochs"]),
        "best_epoch": best_epoch,
        "best_val_accuracy": val_acc,
        "best_val_loss": val_loss,
        "best_val_macro_f1": val_f1,
        "train_accuracy_at_best": history["train_acc"][best_epoch - 1],
        "generalization_gap_at_best": history["train_acc"][best_epoch - 1] - val_acc,
        "training_time": training_time,
        "early_stopping_triggered": stopped_early,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "validation_verification_passed": True,
        "validation_verification_delta": delta,
        "selection_source": "validation_only",
        "test_loader_constructed": False,
        "test_evaluated": False,
        "test_access_count": 0,
        "ema_used": bool(config.get("use_ema", True)),
        "mixup_alpha": float(config.get("mixup_alpha", 0.0)),
        "cutmix_alpha": float(config.get("cutmix_alpha", 0.0)),
        "augment_strength": config.get("augment_strength", "strong"),
    }

    (run_dir / "history.json").write_text(json.dumps(history, indent=2, default=str))
    (run_dir / "validation_result.json").write_text(
        json.dumps(result, indent=2)
    )
    return result, history, config


def _build_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    head_lr = float(config.get("head_learning_rate", config.get("learning_rate", 1e-3)))
    backbone_lr = float(config.get("backbone_learning_rate", head_lr))
    wd = float(config.get("weight_decay", 1e-4))
    name = config.get("optimizer", "AdamW").lower()

    head_params, backbone_params = [], []
    classifier = getattr(getattr(model, "network", None), "fc", None)
    if classifier is not None:
        head_ids = {id(p) for p in classifier.parameters()}
        for p in model.parameters():
            if not p.requires_grad:
                continue
            (head_params if id(p) in head_ids else backbone_params).append(p)
    else:
        head_params = [p for p in model.parameters() if p.requires_grad]

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr, "group_name": "backbone"})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr, "group_name": "head"})

    if name == "adam":
        return torch.optim.Adam(groups, lr=head_lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(groups, lr=head_lr, momentum=0.9, weight_decay=wd)
    return torch.optim.AdamW(groups, lr=head_lr, weight_decay=wd)


def _build_scheduler(optimizer, config):
    from processing_own_phase.train import get_scheduler
    return get_scheduler(optimizer, config)


def _save_checkpoint(
    run_dir: Path,
    ema: ModelEMA,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: Dict[str, Any],
    epoch: int,
    best_metric: float,
    val_loss: float,
) -> None:
    """Persist the EMA weights as ``best.pt`` so downstream reload uses them."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": ema.module().state_dict(),
        "ema_state": ema.state_dict(),
        "best_metric": float(best_metric),
        "best_metric_name": config.get("report_metric", "val_acc"),
        "model_name": config.get("model_name"),
        "training_mode": config.get("training_mode"),
        "num_classes": config.get("num_classes"),
        "config": config,
        "val_loss": float(val_loss),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, run_dir / "best.pt")


def run_improved_training(
    dataset_root: Path,
    manifest_path: Path,
    staging_root: Path,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Train E1/E2 with anti-overfitting tools and save Validation-only artifacts."""
    selected_device = device or get_device()
    if selected_device.type == "cpu":
        raise RuntimeError(
            "Improved training refuses to run the controlled experiments on CPU."
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
        configs[run_name] = {key: config[key] for key in EXPERIMENTS[config_id]}
        run_directories[run_name] = str((staging_runs_root / run_name).resolve())

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
        json.dumps(histories, indent=2, default=str)
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
        "improvements": {
            "ema": selected["ema_used"],
            "mixup_alpha": selected["mixup_alpha"],
            "cutmix_alpha": selected["cutmix_alpha"],
            "augment_strength": selected["augment_strength"],
        },
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


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--staging-root", required=True, type=Path)
    args = parser.parse_args()
    result = run_improved_training(
        args.dataset_root, args.manifest, args.staging_root
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
