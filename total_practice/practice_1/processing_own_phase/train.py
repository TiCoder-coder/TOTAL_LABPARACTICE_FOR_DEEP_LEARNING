"""Training functions for FashionMNIST classification."""

import time
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import FashionMLP
from .utils import format_time


def train_one_epoch(
    model: FashionMLP,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Train the model for one epoch.

    Returns:
        dict with keys: loss, accuracy, time_seconds.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    start = time.time()

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_time = time.time() - start
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "time_seconds": epoch_time,
    }


def fit(
    model: FashionMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 10,
    save_best_callback=None,
    verbose: bool = True,
) -> Dict:
    """Train the model for multiple epochs with validation.

    Args:
        save_best_callback: optional callable(model, epoch, val_metrics) called
            when validation accuracy improves. Used to save checkpoints.

    Returns:
        dict containing:
            - history: list of per-epoch metrics
            - best_epoch: epoch with highest val accuracy
            - best_val_acc: highest val accuracy
            - best_val_loss: val loss at the best epoch
            - total_time: total training time in seconds
    """
    from .evaluate import evaluate

    history: List[Dict] = []
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    start = time.time()

    for epoch in range(epochs):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            if save_best_callback is not None:
                save_best_callback(model, epoch + 1, val_metrics)

        if verbose:
            print(
                f"Epoch {epoch + 1:02d}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} | "
                f"Time: {format_time(train_metrics['time_seconds'])}"
            )

    total_time = time.time() - start
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "total_time": total_time,
    }
