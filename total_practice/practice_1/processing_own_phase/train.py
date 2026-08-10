import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .data import make_train_loader
from .evaluate import evaluate_loader
from .model import build_model
from .utils import count_parameters, setup_reproducibility


def build_optimizer(
    model: nn.Module,
    config: Dict,
) -> torch.optim.Optimizer:
    optimizer_name = config["optimizer"].lower()
    learning_rate = config["learning_rate"]
    weight_decay = config.get("weight_decay", 0.0)

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=config.get("momentum", 0.9),
            weight_decay=weight_decay,
        )
    raise ValueError(
        f"Unsupported optimizer: {config['optimizer']}"
    )


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    sample_count = 0
    started_at = time.perf_counter()

    for batch_index, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss at batch {batch_index}"
            )

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (
            logits.argmax(dim=1)
            .eq(labels)
            .sum()
            .item()
        )
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError("Training loader is empty")

    return {
        "loss": loss_sum / sample_count,
        "accuracy": correct / sample_count,
        "sample_count": sample_count,
        "elapsed_seconds": (
            time.perf_counter() - started_at
        ),
    }


def run_training(
    config: Dict,
    training_dataset,
    validation_loader: DataLoader,
    device: torch.device,
    log_directory: Optional[Path] = None,
    verbose: bool = True,
) -> Dict:
    setup_reproducibility(config.get("seed", 42))
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    train_loader = make_train_loader(
        training_dataset,
        batch_size=config["batch_size"],
        seed=config.get("seed", 42),
        num_workers=config.get("num_workers", 0),
    )
    writer = (
        SummaryWriter(log_dir=str(log_directory))
        if log_directory is not None
        else None
    )
    history = []
    best_state = None
    best_epoch = 0
    best_validation_accuracy = -1.0
    best_validation_loss = float("inf")
    started_at = time.perf_counter()

    try:
        for epoch in range(1, config["epochs"] + 1):
            train_metrics = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )
            validation_metrics = evaluate_loader(
                model,
                validation_loader,
                criterion,
                device,
            )
            learning_rate = optimizer.param_groups[0]["lr"]
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": (
                    validation_metrics["accuracy"]
                ),
                "learning_rate": learning_rate,
                "elapsed_seconds": (
                    train_metrics["elapsed_seconds"]
                ),
            }
            history.append(epoch_record)

            validation_is_better = (
                validation_metrics["accuracy"]
                > best_validation_accuracy
                or (
                    validation_metrics["accuracy"]
                    == best_validation_accuracy
                    and validation_metrics["loss"]
                    < best_validation_loss
                )
            )
            if validation_is_better:
                best_epoch = epoch
                best_validation_accuracy = (
                    validation_metrics["accuracy"]
                )
                best_validation_loss = (
                    validation_metrics["loss"]
                )
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value
                    in model.state_dict().items()
                }

            if writer is not None:
                writer.add_scalar(
                    "Loss/Train",
                    train_metrics["loss"],
                    epoch,
                )
                writer.add_scalar(
                    "Loss/Validation",
                    validation_metrics["loss"],
                    epoch,
                )
                writer.add_scalar(
                    "Accuracy/Train",
                    train_metrics["accuracy"],
                    epoch,
                )
                writer.add_scalar(
                    "Accuracy/Validation",
                    validation_metrics["accuracy"],
                    epoch,
                )
                writer.add_scalar(
                    "LearningRate",
                    learning_rate,
                    epoch,
                )

            if verbose:
                print(
                    f"Epoch {epoch:02d}/{config['epochs']} | "
                    f"Train Loss {train_metrics['loss']:.4f} | "
                    f"Train Accuracy "
                    f"{train_metrics['accuracy']:.2%} | "
                    f"Validation Loss "
                    f"{validation_metrics['loss']:.4f} | "
                    f"Validation Accuracy "
                    f"{validation_metrics['accuracy']:.2%}"
                )
    finally:
        if writer is not None:
            writer.close()

    if best_state is None:
        raise RuntimeError(
            "Training did not produce a valid checkpoint"
        )

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    return {
        "experiment_id": config["experiment_id"],
        "config": dict(config),
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_accuracy": (
            best_validation_accuracy
        ),
        "best_validation_loss": best_validation_loss,
        "parameter_count": count_parameters(model),
        "training_sample_count": len(training_dataset),
        "total_seconds": (
            time.perf_counter() - started_at
        ),
    }


def train_final_model(
    config: Dict,
    training_dataset,
    device: torch.device,
    log_directory: Optional[Path] = None,
    verbose: bool = True,
) -> Dict:
    setup_reproducibility(config.get("seed", 42))
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    train_loader = make_train_loader(
        training_dataset,
        batch_size=config["batch_size"],
        seed=config.get("seed", 42),
        num_workers=config.get("num_workers", 0),
    )
    writer = (
        SummaryWriter(log_dir=str(log_directory))
        if log_directory is not None
        else None
    )
    history = []
    started_at = time.perf_counter()

    try:
        for epoch in range(1, config["epochs"] + 1):
            train_metrics = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "learning_rate": (
                    optimizer.param_groups[0]["lr"]
                ),
                "elapsed_seconds": (
                    train_metrics["elapsed_seconds"]
                ),
            }
            history.append(epoch_record)

            if writer is not None:
                writer.add_scalar(
                    "Loss/Train",
                    train_metrics["loss"],
                    epoch,
                )
                writer.add_scalar(
                    "Accuracy/Train",
                    train_metrics["accuracy"],
                    epoch,
                )
                writer.add_scalar(
                    "LearningRate",
                    optimizer.param_groups[0]["lr"],
                    epoch,
                )

            if verbose:
                print(
                    f"Final Epoch "
                    f"{epoch:02d}/{config['epochs']} | "
                    f"Train Loss {train_metrics['loss']:.4f} | "
                    f"Train Accuracy "
                    f"{train_metrics['accuracy']:.2%}"
                )
    finally:
        if writer is not None:
            writer.close()

    model.eval()
    return {
        "model": model,
        "history": history,
        "config": dict(config),
        "parameter_count": count_parameters(model),
        "training_sample_count": len(training_dataset),
        "total_seconds": (
            time.perf_counter() - started_at
        ),
    }
