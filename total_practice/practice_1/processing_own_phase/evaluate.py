"""Evaluation utilities for FashionMNIST classification."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import FashionMLP


def evaluate(
    model: FashionMLP,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Run the model on a dataloader and compute loss/accuracy/predictions.

    Returns:
        dict with keys: loss, accuracy, predictions, labels, probabilities.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            probabilities = F.softmax(logits, dim=1)
            _, predicted = logits.max(1)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_predictions.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())

    predictions = np.concatenate(all_predictions)
    labels = np.concatenate(all_labels)
    probabilities = np.concatenate(all_probabilities)

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "predictions": predictions,
        "labels": labels,
        "probabilities": probabilities,
    }


def per_class_accuracy(predictions: np.ndarray, labels: np.ndarray, num_classes: int = 10) -> dict:
    """Compute per-class accuracy."""
    accuracies = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            accuracies[c] = 0.0
            continue
        accuracies[c] = float((predictions[mask] == labels[mask]).mean())
    return accuracies
