"""Model saving and loading utilities."""

from pathlib import Path
from typing import Dict, Optional

import torch

from configs import CLASS_NAMES, MODEL_SAVE_PATH, NUM_CLASSES
from .model import PretrainedClassifier


def save_checkpoint(
    model: PretrainedClassifier,
    path: str = str(MODEL_SAVE_PATH),
    config: Optional[Dict] = None,
    class_names=CLASS_NAMES,
    best_epoch: Optional[int] = None,
    best_val_acc: Optional[float] = None,
    best_val_loss: Optional[float] = None,
    extra: Optional[Dict] = None,
) -> None:
    """Save a model checkpoint with metadata."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_name": model.model_name,
        "training_mode": getattr(
            model,
            "training_mode",
            (config or {}).get("training_mode", "head_only"),
        ),
        "num_classes": NUM_CLASSES,
        "class_names": list(class_names),
    }
    if config is not None:
        checkpoint["config"] = dict(config)
    if best_epoch is not None:
        checkpoint["best_epoch"] = best_epoch
    if best_val_acc is not None:
        checkpoint["best_val_acc"] = best_val_acc
    if best_val_loss is not None:
        checkpoint["best_val_loss"] = best_val_loss
    if extra is not None:
        checkpoint["extra"] = dict(extra)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str = str(MODEL_SAVE_PATH),
    device: Optional[torch.device] = None,
) -> Dict:
    """Load a checkpoint file and return its contents."""
    map_location = device if device is not None else "cpu"
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    return checkpoint


def load_model_from_checkpoint(
    path: str = str(MODEL_SAVE_PATH),
    device: Optional[torch.device] = None,
) -> PretrainedClassifier:
    """Build a PretrainedClassifier from a checkpoint and return it on the right device."""
    checkpoint = load_checkpoint(path, device=device)
    model = PretrainedClassifier(
        model_name=checkpoint.get("model_name", checkpoint.get("config", {}).get("model_name", "resnet18")),
        num_classes=checkpoint.get("num_classes", NUM_CLASSES),
        dropout=checkpoint.get("config", {}).get("dropout", 0.0),
    )
    training_mode = checkpoint.get(
        "training_mode",
        checkpoint.get("config", {}).get("training_mode", "head_only"),
    )
    model.set_training_mode(training_mode)
    model.load_state_dict(checkpoint["model_state_dict"])
    if device is not None:
        model.to(device)
    model.eval()
    return model


def verify_loaded_model(
    original_model: PretrainedClassifier,
    loaded_model: PretrainedClassifier,
    test_input: torch.Tensor,
) -> Dict:
    """Verify that the loaded model produces the same predictions as the original."""
    original_model.eval()
    loaded_model.eval()
    with torch.inference_mode():
        out_orig = original_model(test_input)
        out_loaded = loaded_model(test_input)
    same_predictions = torch.equal(out_orig.argmax(1), out_loaded.argmax(1))
    max_logit_diff = (out_orig - out_loaded).abs().max().item()
    return {
        "same_predictions": bool(same_predictions),
        "max_logit_diff": float(max_logit_diff),
    }
