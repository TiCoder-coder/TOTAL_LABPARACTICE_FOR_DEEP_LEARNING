"""Phase 7: construct and verify the generic DistilBERT classifier."""

import json
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoModelForSequenceClassification

from .config import MAX_TOKEN_LENGTH, RESULT_DIR
from .phase_01_environment import get_device


MODEL_CHECKPOINT = "distilbert/distilbert-base-uncased"
ID2LABEL = {0: "NEGATIVE", 1: "POSITIVE"}
LABEL2ID = {"NEGATIVE": 0, "POSITIVE": 1}


def build_model(checkpoint: str = MODEL_CHECKPOINT):
    """Load an unfine-tuned binary sequence classifier without changing gradients."""
    return AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )


def _same_distilbert_family(tokenizer: Any, model: Any) -> bool:
    tokenizer_name = str(getattr(tokenizer, "name_or_path", "")).lower()
    return (
        "distilbert-base-uncased" in tokenizer_name
        and getattr(model.config, "model_type", None) == "distilbert"
    )


def _deterministic_binary_train_samples(tokenized_dataset: Any) -> list[dict]:
    """Select the first Train example of each binary label, deterministically."""
    train_split = tokenized_dataset["train"]
    indices: Dict[int, int] = {}
    for index, label in enumerate(train_split["labels"]):
        label = int(label)
        if label in (0, 1) and label not in indices:
            indices[label] = index
        if len(indices) == 2:
            break
    if set(indices) != {0, 1}:
        raise ValueError("Train sanity batch requires both labels 0 and 1")
    return [train_split[indices[0]], train_split[indices[1]]]


def verify_model_construction(
    model: Any,
    tokenizer: Any,
    tokenized_dataset: Any,
    data_collator: Any,
) -> Dict[str, Any]:
    """Run one no-gradient, Train-only forward pass and return real evidence."""
    samples = _deterministic_binary_train_samples(tokenized_dataset)
    observed_labels = sorted({int(sample["labels"]) for sample in samples})
    labels_valid = set(observed_labels).issubset({0, 1})
    required_fields_present = all(
        all(field in sample for field in ("input_ids", "attention_mask", "labels"))
        for sample in samples
    )
    batch = data_collator(samples)
    batch_size = int(batch["input_ids"].shape[0])
    batch_tensor_shapes = {
        key: list(value.shape) for key, value in batch.items() if torch.is_tensor(value)
    }
    batch_shapes_compatible = all(shape[0] == batch_size for shape in batch_tensor_shapes.values())

    device = torch.device(get_device())
    model = model.to(device)
    device_batch = {key: value.to(device) for key, value in batch.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**device_batch)

    logits_shape = list(outputs.logits.shape)
    expected_logits_shape = [batch_size, 2]
    logits_shape_pass = logits_shape == expected_logits_shape
    logits_finite = bool(torch.isfinite(outputs.logits).all().item())
    loss_present = outputs.loss is not None
    loss_finite = bool(loss_present and torch.isfinite(outputs.loss).all().item())
    sanity_loss = float(outputs.loss.detach().cpu().item()) if loss_present else None

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    frozen_parameters = total_parameters - trainable_parameters
    all_parameters_trainable = trainable_parameters == total_parameters
    label_mapping_pass = (
        model.config.num_labels == 2
        and model.config.id2label == ID2LABEL
        and model.config.label2id == LABEL2ID
    )
    tokenizer_model_family_match = _same_distilbert_family(tokenizer, model)
    checks = [
        required_fields_present, batch_shapes_compatible, labels_valid,
        logits_shape_pass, logits_finite, loss_present, loss_finite,
        label_mapping_pass, tokenizer_model_family_match, all_parameters_trainable,
    ]

    return {
        "status": "PASS" if all(checks) else "FAIL",
        "checkpoint": MODEL_CHECKPOINT,
        "model_class": model.__class__.__name__,
        "model_type": model.config.model_type,
        "num_labels": model.config.num_labels,
        "id2label": model.config.id2label,
        "label2id": model.config.label2id,
        "label_mapping_pass": label_mapping_pass,
        "tokenizer_name_or_path": str(tokenizer.name_or_path),
        "tokenizer_model_family_match": tokenizer_model_family_match,
        "max_token_length": MAX_TOKEN_LENGTH,
        "device": str(device),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "frozen_parameters": frozen_parameters,
        "trainable_percentage": round(100 * trainable_parameters / total_parameters, 6),
        "all_parameters_trainable": all_parameters_trainable,
        "batch_source": "train",
        "batch_size": batch_size,
        "required_fields_present": required_fields_present,
        "batch_tensor_shapes": batch_tensor_shapes,
        "batch_shapes_compatible": batch_shapes_compatible,
        "observed_label_set": observed_labels,
        "labels_valid": labels_valid,
        "expected_logits_shape": expected_logits_shape,
        "logits_shape": logits_shape,
        "logits_shape_pass": logits_shape_pass,
        "logits_finite": logits_finite,
        "loss_present": loss_present,
        "loss_finite": loss_finite,
        "sanity_loss": sanity_loss,
        "model_in_eval_mode_during_sanity": not model.training,
        "no_grad_forward": True,
        "backward_called": False,
        "optimizer_created": False,
        "scheduler_created": False,
        "training_performed": False,
        "validation_accessed": False,
        "test_accessed": False,
        "model_checkpoint_saved": False,
    }


def save_model_verification(
    verification: Dict[str, Any], path: Path | None = None
) -> Path:
    """Save Phase 7 evidence and fail if the JSON read-back differs."""
    output_path = path or RESULT_DIR / "phase_07_model_verification.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    readback = json.loads(output_path.read_text(encoding="utf-8"))
    if readback != json.loads(json.dumps(verification)):
        raise RuntimeError("Phase 7 artifact read-back verification failed")
    return output_path
