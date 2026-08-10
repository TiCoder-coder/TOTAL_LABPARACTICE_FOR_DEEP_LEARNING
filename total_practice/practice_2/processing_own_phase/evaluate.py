"""Evaluation utilities for Pre-trained Neural Networks."""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Run the model on a dataloader and compute loss/accuracy/predictions.

    Returns:
        dict with keys: loss, accuracy, predictions, labels, probabilities, inference_time, inference_fps
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []

    start_time = time.time()

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
            
    end_time = time.time()
    inference_time = end_time - start_time
    inference_fps = total / inference_time if inference_time > 0 else 0.0

    predictions = np.concatenate(all_predictions)
    labels = np.concatenate(all_labels)
    probabilities = np.concatenate(all_probabilities)

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "predictions": predictions,
        "labels": labels,
        "probabilities": probabilities,
        "inference_time": inference_time,
        "inference_fps": inference_fps
    }


def compute_classification_metrics(predictions: np.ndarray, labels: np.ndarray, num_classes: int, class_names: list) -> dict:
    """Compute Precision, Recall, F1-score (Macro, Weighted, Per-class)."""
    
    per_class_metrics = {}
    total_samples = len(labels)
    
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0
    
    for c in range(num_classes):
        tp = np.sum((predictions == c) & (labels == c))
        fp = np.sum((predictions == c) & (labels != c))
        fn = np.sum((predictions != c) & (labels == c))
        support = np.sum(labels == c)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = tp / support if support > 0 else 0.0
        
        per_class_metrics[class_names[c]] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "accuracy": accuracy,
            "support": support
        }
        
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        
        weight = support / total_samples if total_samples > 0 else 0.0
        weighted_precision += precision * weight
        weighted_recall += recall * weight
        weighted_f1 += f1 * weight
        
    macro_precision /= num_classes
    macro_recall /= num_classes
    macro_f1 /= num_classes
    
    correct = np.sum(predictions == labels)
    overall_accuracy = correct / total_samples if total_samples > 0 else 0.0
    
    return {
        "accuracy": overall_accuracy,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1-score": macro_f1,
            "support": total_samples
        },
        "weighted_avg": {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1-score": weighted_f1,
            "support": total_samples
        },
        "per_class": per_class_metrics
    }


def generate_classification_report(metrics_dict: dict, class_names: list, txt_path: str = None, csv_path: str = None) -> str:
    """Format and save the classification report."""
    
    # Format TXT
    lines = []
    lines.append(f"{'Class':>15} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>10}")
    lines.append("-" * 60)
    
    rows = []
    
    for c_name in class_names:
        m = metrics_dict["per_class"][c_name]
        lines.append(f"{c_name:>15} {m['precision']:10.4f} {m['recall']:10.4f} {m['f1-score']:10.4f} {m['support']:10d}")
        
        rows.append({
            "class": c_name,
            "precision": m['precision'],
            "recall": m['recall'],
            "f1-score": m['f1-score'],
            "support": m['support']
        })
        
    lines.append("-" * 60)
    
    # Overall Accuracy
    acc = metrics_dict["accuracy"]
    support = metrics_dict["macro_avg"]["support"]
    lines.append(f"{'accuracy':>15} {'':>10} {'':>10} {acc:10.4f} {support:10d}")
    
    # Macro avg
    mac = metrics_dict["macro_avg"]
    lines.append(f"{'macro avg':>15} {mac['precision']:10.4f} {mac['recall']:10.4f} {mac['f1-score']:10.4f} {mac['support']:10d}")
    
    # Weighted avg
    wt = metrics_dict["weighted_avg"]
    lines.append(f"{'weighted avg':>15} {wt['precision']:10.4f} {wt['recall']:10.4f} {wt['f1-score']:10.4f} {wt['support']:10d}")
    
    rows.extend([
        {"class": "accuracy", "precision": None, "recall": None, "f1-score": acc, "support": support},
        {"class": "macro avg", "precision": mac['precision'], "recall": mac['recall'], "f1-score": mac['f1-score'], "support": mac['support']},
        {"class": "weighted avg", "precision": wt['precision'], "recall": wt['recall'], "f1-score": wt['f1-score'], "support": wt['support']}
    ])
    
    report_str = "\n".join(lines)
    
    if txt_path:
        with open(txt_path, 'w') as f:
            f.write(report_str)
            
    if csv_path:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        
    return report_str


def export_predictions(predictions: np.ndarray, labels: np.ndarray, probabilities: np.ndarray, class_names: list, csv_path: str):
    """Export predictions and confidences to CSV."""
    confidences = probabilities[np.arange(len(predictions)), predictions]
    
    df = pd.DataFrame({
        "true_label_id": labels,
        "true_label": [class_names[l] for l in labels],
        "predicted_label_id": predictions,
        "predicted_label": [class_names[p] for p in predictions],
        "confidence": confidences,
        "is_correct": predictions == labels
    })
    for class_index, class_name in enumerate(class_names):
        df[f"probability_{class_name}"] = probabilities[:, class_index]
    
    df.to_csv(csv_path, index=False)
    return df
