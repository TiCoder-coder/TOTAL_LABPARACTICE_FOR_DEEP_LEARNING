from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_loader(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    sample_count = 0

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite evaluation loss"
                )

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
        raise ValueError("Evaluation loader is empty")

    return {
        "loss": loss_sum / sample_count,
        "accuracy": correct / sample_count,
        "sample_count": sample_count,
    }


def evaluate_classifier(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict:
    model.eval()
    loss_sum = 0.0
    correct = 0
    sample_count = 0
    prediction_batches = []
    target_batches = []
    probability_batches = []

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite test loss")

            predictions = logits.argmax(dim=1)
            probabilities = torch.softmax(logits, dim=1)
            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            correct += predictions.eq(labels).sum().item()
            sample_count += batch_size
            prediction_batches.append(predictions.cpu())
            target_batches.append(labels.cpu())
            probability_batches.append(probabilities.cpu())

    if sample_count == 0:
        raise ValueError("Test loader is empty")

    return {
        "loss": loss_sum / sample_count,
        "accuracy": correct / sample_count,
        "sample_count": sample_count,
        "predictions": torch.cat(
            prediction_batches
        ).numpy(),
        "targets": torch.cat(target_batches).numpy(),
        "probabilities": torch.cat(
            probability_batches
        ).numpy(),
    }


def per_class_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 10,
) -> Dict[int, float]:
    accuracies = {}
    for class_index in range(num_classes):
        class_mask = targets == class_index
        if class_mask.sum() == 0:
            accuracies[class_index] = 0.0
        else:
            accuracies[class_index] = float(
                (
                    predictions[class_mask]
                    == targets[class_mask]
                ).mean()
            )
    return accuracies
