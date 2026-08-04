"""Reload canonical E2 on Validation only; never constructs a Test dataset."""

from __future__ import annotations

import json

import torch
from torch.utils.data import DataLoader

from practice_2_2.data_practice_2_2 import (
    compute_train_class_weights,
    create_train_validation_datasets,
)
from practice_2_2.resources import (
    resolve_best_checkpoint,
    resolve_dataset_root,
    resolve_split_manifest,
)
from practice_2_2.save_load import load_model_from_checkpoint
from practice_2_2.train import evaluate, get_criterion


def main() -> None:
    train_dataset, validation_dataset = create_train_validation_datasets(
        resolve_dataset_root(),
        resolve_split_manifest(),
        image_size=224,
        augment_strength="base",
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=32, shuffle=False, num_workers=0
    )
    device = torch.device("cpu")
    model = load_model_from_checkpoint(str(resolve_best_checkpoint()), device=device)
    criterion = get_criterion(
        {"label_smoothing": 0.05},
        compute_train_class_weights(train_dataset).to(device),
    )
    loss, accuracy, macro_f1 = evaluate(
        model, validation_loader, criterion, device, return_macro_f1=True
    )
    result = {
        "validation_samples": len(validation_dataset),
        "validation_accuracy": accuracy,
        "validation_loss": loss,
        "validation_macro_f1": macro_f1,
        "test_loader_constructed": False,
        "test_evaluated": False,
    }
    if len(validation_dataset) != 438:
        raise RuntimeError("Canonical Validation sample count changed")
    if abs(accuracy - 78.31050228310502) > 1e-9:
        raise RuntimeError("Canonical Validation accuracy changed")
    if abs(macro_f1 - 0.7820960879325867) > 1e-9:
        raise RuntimeError("Canonical Validation Macro F1 changed")
    if abs(loss - 1.021770360262971) > 1e-5:
        raise RuntimeError("Canonical Validation loss changed beyond tolerance")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
